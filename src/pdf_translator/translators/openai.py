# SPDX-License-Identifier: Apache-2.0
"""OpenAI GPT translation backend."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any

from pdf_translator.translators.base import (
    ArrayLengthMismatchError,
    ConfigurationError,
    TranslationError,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from pydantic import BaseModel
    from tiktoken import Encoding

logger = logging.getLogger(__name__)


# Language code to full name mapping for prompts
LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "auto": "the source language",
}

# Enhanced system prompt for reliable array-length matching
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional translator. Translate the given texts accurately "
    "while preserving the original meaning, tone, and formatting. "
    "IMPORTANT: You must return EXACTLY the same number of translations as input texts. "
    "Each input text must have exactly one corresponding translation. "
    "Do not merge, split, or skip any texts."
)

# Model context sizes for calculating token limits
#
# Sources:
# - OpenAI API documentation: https://platform.openai.com/docs/models
# - Price comparison: https://pricepertoken.com (aggregated model data)
#
# These values may change as OpenAI updates their models.
# Users can override limits using the `max_tokens` parameter.
#
# Last verified: 2026-01-06
MODEL_CONTEXT_SIZES: dict[str, int] = {
    # GPT-5 series (400K context)
    "gpt-5": 400_000,
    "gpt-5-nano": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-codex": 400_000,
    "gpt-5-image": 400_000,
    "gpt-5-pro": 400_000,
    # GPT-5 chat variants (128K context)
    "gpt-5-chat": 128_000,
    # GPT-5.1 series (400K context)
    "gpt-5.1": 400_000,
    "gpt-5.1-codex": 400_000,
    "gpt-5.1-codex-mini": 400_000,
    "gpt-5.1-codex-max": 400_000,
    "gpt-5.1-chat": 128_000,
    # GPT-5.2 series (400K context)
    "gpt-5.2": 400_000,
    "gpt-5.2-pro": 400_000,
    "gpt-5.2-chat": 128_000,
    # GPT-4.1 series (1M context)
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    # GPT-4o series (128K context)
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    # o3 series (200K context)
    "o3": 200_000,
    "o3-mini": 200_000,
    "o3-pro": 200_000,
    # o1 series (200K context)
    "o1": 200_000,
    "o1-mini": 200_000,
    "o1-pro": 200_000,
}

# Model-specific token limits for text splitting (per single text)
# Uses context_size / 4 as safe buffer for translation (input + output + overhead)
# Unknown models fall back to DEFAULT_MAX_TOKENS (conservative)
MODEL_TOKEN_LIMITS: dict[str, int] = {
    # GPT-5 series (400K context → 100K safe for translation)
    "gpt-5": 100_000,
    "gpt-5-nano": 100_000,
    "gpt-5-mini": 100_000,
    "gpt-5-codex": 100_000,
    "gpt-5-image": 100_000,
    "gpt-5-pro": 100_000,
    # GPT-5 chat variants (128K context → 32K safe)
    "gpt-5-chat": 32_000,
    # GPT-5.1 series (400K context → 100K safe)
    "gpt-5.1": 100_000,
    "gpt-5.1-codex": 100_000,
    "gpt-5.1-codex-mini": 100_000,
    "gpt-5.1-codex-max": 100_000,
    "gpt-5.1-chat": 32_000,
    # GPT-5.2 series (400K context → 100K safe)
    "gpt-5.2": 100_000,
    "gpt-5.2-pro": 100_000,
    "gpt-5.2-chat": 32_000,
    # GPT-4.1 series (1M context → 250K safe)
    "gpt-4.1": 250_000,
    "gpt-4.1-mini": 250_000,
    "gpt-4.1-nano": 250_000,
    # GPT-4o series (128K context → 32K safe)
    "gpt-4o": 32_000,
    "gpt-4o-mini": 32_000,
    # o3 series (200K context → 50K safe)
    "o3": 50_000,
    "o3-mini": 50_000,
    "o3-pro": 50_000,
    # o1 series (200K context → 50K safe)
    "o1": 50_000,
    "o1-mini": 50_000,
    "o1-pro": 50_000,
}


class OpenAITranslator:
    """OpenAI GPT translation backend.

    This backend uses OpenAI's GPT models with Structured Outputs
    to ensure reliable array-based translation responses.

    Supports custom system prompts for specialized translation needs.

    Attributes:
        name: Backend identifier ("openai").
    """

    DEFAULT_MODEL = "gpt-5-nano"

    # Fallback token limit for unknown models (conservative)
    # Known models use MODEL_TOKEN_LIMITS for optimal settings
    DEFAULT_MAX_TOKENS = 8_000

    # Conservative character-to-token ratio for estimation
    # CJK: ~1-2 chars/token, English: ~4 chars/token
    # Use 1.0 for worst-case CJK handling (1 char = 1 token)
    CHARS_PER_TOKEN = 1.0

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize OpenAITranslator.

        Args:
            api_key: OpenAI API key.
            model: Model to use. Priority: argument > OPENAI_MODEL env > default.
            system_prompt: Custom system prompt for translation.
            max_tokens: Maximum tokens for input texts (for splitting).
                If None, auto-detected from MODEL_TOKEN_LIMITS based on model.
                Set to 0 to disable splitting.

        Raises:
            ConfigurationError: If API key is not provided.
            ImportError: If openai package is not installed.
        """
        import os

        if not api_key:
            raise ConfigurationError("OpenAI API key is required")

        # Lazy import openai and pydantic
        try:
            from openai import AsyncOpenAI as _AsyncOpenAI

            self._AsyncOpenAI = _AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI backend. "
                "Install with: pip install pdf-translator[openai]"
            ) from None

        try:
            from pydantic import BaseModel as _BaseModel
            from pydantic import Field as _Field

            self._BaseModel = _BaseModel
            self._Field = _Field
        except ImportError:
            raise ImportError(
                "pydantic is required for OpenAI backend. "
                "Install with: pip install pdf-translator[openai]"
            ) from None

        # Lazy import tiktoken (optional)
        self._tiktoken_available = False
        self._encoding: Encoding | None = None
        self._tiktoken: Any = None  # Module or None
        try:
            import tiktoken

            self._tiktoken = tiktoken
            self._tiktoken_available = True
        except ImportError:
            logger.debug(
                "tiktoken not installed, using character-based estimation. "
                "Install with: pip install pdf-translator[openai]"
            )

        self._api_key = api_key
        env_model = os.environ.get("OPENAI_MODEL")
        self._model = model or env_model or self.DEFAULT_MODEL
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_tokens = (
            max_tokens if max_tokens is not None else self._get_model_max_tokens()
        )
        self._client: AsyncOpenAI | None = None

    @property
    def name(self) -> str:
        """Return backend name."""
        return "openai"

    def _get_model_max_tokens(self) -> int:
        """Get max tokens limit for the current model.

        Looks up MODEL_TOKEN_LIMITS for known models, falls back to
        DEFAULT_MAX_TOKENS for unknown models.

        Returns:
            Token limit for text splitting.
        """
        # Direct lookup
        if self._model in MODEL_TOKEN_LIMITS:
            return MODEL_TOKEN_LIMITS[self._model]

        # Try base model name (e.g., "gpt-5-nano-2025-01" → "gpt-5-nano")
        # This handles dated model versions
        # Sort by key length descending to match longer (more specific) keys first
        # e.g., "gpt-5-chat" should match before "gpt-5"
        for known_model in sorted(MODEL_TOKEN_LIMITS.keys(), key=len, reverse=True):
            if self._model.startswith(known_model):
                logger.debug(
                    "Using token limit for base model '%s' (matched from '%s')",
                    known_model,
                    self._model,
                )
                return MODEL_TOKEN_LIMITS[known_model]

        # Fallback to conservative default
        logger.debug(
            "Unknown model '%s', using conservative default %d tokens",
            self._model,
            self.DEFAULT_MAX_TOKENS,
        )
        return self.DEFAULT_MAX_TOKENS

    @property
    def max_text_length(self) -> int | None:
        """Maximum text length for OpenAI (character-based approximation).

        OpenAI uses token-based limits, not character limits. This property
        returns a character-based approximation for pipeline text splitting.

        Important: The pipeline splits texts based on this character limit,
        NOT actual token counts. The count_tokens() method provides accurate
        token counting via tiktoken, but is not used for split decisions.

        The approximation uses CHARS_PER_TOKEN (1.0) which is intentionally
        conservative to handle worst-case CJK scenarios where 1 char ≈ 1 token.
        For English (~4 chars/token), this results in more splits than
        strictly necessary, but ensures we never exceed token limits.

        Returns:
            Approximate character limit based on max_tokens setting.
            Returns None if max_tokens is 0 (splitting disabled).
        """
        if self._max_tokens == 0:
            return None
        return int(self._max_tokens * self.CHARS_PER_TOKEN)

    @property
    def max_batch_tokens(self) -> int:
        """Maximum total tokens for a single batch request.

        This limit applies to the sum of all input texts in a batch,
        ensuring the total request doesn't exceed the model's context window.

        Uses context_size / 2 as the limit, reserving half for:
        - Output translations (similar length to input)
        - System prompt overhead
        - Safety margin

        The pipeline should use this to chunk texts by total tokens,
        not just by count.

        Returns:
            Maximum tokens for a batch request.
        """
        context_size = self._get_model_context_size()
        # Use half the context for input (other half for output + overhead)
        return context_size // 2

    def _get_model_context_size(self) -> int:
        """Get context window size for the current model.

        Returns:
            Context size in tokens. Falls back to conservative 32K for unknown models.
        """
        # Direct lookup
        if self._model in MODEL_CONTEXT_SIZES:
            return MODEL_CONTEXT_SIZES[self._model]

        # Try prefix matching (longer keys first)
        for known_model in sorted(MODEL_CONTEXT_SIZES.keys(), key=len, reverse=True):
            if self._model.startswith(known_model):
                return MODEL_CONTEXT_SIZES[known_model]

        # Conservative fallback (smallest common context size)
        return 32_000

    def _ensure_client(self) -> AsyncOpenAI:
        """Ensure OpenAI client exists.

        Returns:
            Active OpenAI async client.
        """
        if self._client is None:
            self._client = self._AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _get_language_name(self, lang_code: str) -> str:
        """Convert language code to full name.

        Args:
            lang_code: Language code (e.g., "en", "ja").

        Returns:
            Full language name.
        """
        return LANGUAGE_NAMES.get(lang_code.lower(), lang_code)

    def _get_encoding(self) -> "Encoding | None":
        """Get tiktoken encoding for the current model.

        Returns:
            tiktoken Encoding object, or None if tiktoken is not available.
        """
        if not self._tiktoken_available or self._tiktoken is None:
            return None

        if self._encoding is not None:
            return self._encoding

        try:
            # Try to get encoding for the specific model
            self._encoding = self._tiktoken.encoding_for_model(self._model)
        except KeyError:
            # Fall back to cl100k_base (used by GPT-4, GPT-3.5-turbo, etc.)
            logger.debug(
                "No specific encoding for model '%s', using cl100k_base",
                self._model,
            )
            self._encoding = self._tiktoken.get_encoding("cl100k_base")

        return self._encoding

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text using tiktoken.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens. If tiktoken is not available, returns an
            estimate based on character count and CHARS_PER_TOKEN ratio.
            Returns 0 for empty text, otherwise at least 1.
        """
        if not text:
            return 0

        encoding = self._get_encoding()
        if encoding is not None:
            return len(encoding.encode(text))

        # Fallback to character-based estimation
        # Use max(1, ...) to avoid underestimation for short texts
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))

    def _create_response_model(self, n: int) -> type["BaseModel"]:
        """Create a Pydantic model with fixed-length translations array.

        This ensures OpenAI Structured Outputs enforces the exact array length,
        preventing ArrayLengthMismatchError.

        Args:
            n: Expected number of translations.

        Returns:
            Pydantic model class with length-constrained translations field.
        """

        class TranslationResult(self._BaseModel):  # type: ignore[misc, name-defined]
            translations: Annotated[list[str], self._Field(min_length=n, max_length=n)]

        return TranslationResult

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate a single text using OpenAI.

        Args:
            text: Text to translate.
            source_lang: Source language code ("en", "ja", "auto").
            target_lang: Target language code ("en", "ja").

        Returns:
            Translated text.

        Raises:
            TranslationError: On translation failure.
            ConfigurationError: On authentication failure.
        """
        # Early return for empty or whitespace-only text
        if not text or not text.strip():
            return text

        results = await self.translate_batch([text], source_lang, target_lang)
        return results[0]

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate multiple texts in batch using OpenAI Structured Outputs.

        Args:
            texts: List of texts to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of translated texts (same order and length as input).

        Raises:
            TranslationError: On translation failure.
            ConfigurationError: On authentication failure.
        """
        if not texts:
            return []

        # Track empty/whitespace indices for restoration
        results: list[str] = [""] * len(texts)
        non_empty_indices: list[int] = []
        non_empty_texts: list[str] = []

        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)
            else:
                results[i] = text  # Preserve original empty/whitespace

        if not non_empty_texts:
            return results

        # Translate non-empty texts
        translated_texts = await self._translate_with_structured_output(
            non_empty_texts, source_lang, target_lang
        )

        # Restore translations to original positions
        for i, translated in zip(non_empty_indices, translated_texts):
            results[i] = translated

        return results

    async def _translate_with_structured_output(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate texts using Structured Outputs with fixed-length schema.

        Uses dynamic Pydantic schema with min_length/max_length constraints
        and JSON-formatted input to ensure reliable array-length matching.

        Args:
            texts: List of non-empty texts to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of translated texts.

        Raises:
            TranslationError: On translation failure.
            ConfigurationError: On authentication failure.
        """
        client = self._ensure_client()
        n = len(texts)

        source_name = self._get_language_name(source_lang)
        target_name = self._get_language_name(target_lang)

        # Create dynamic response model with fixed array length
        response_model = self._create_response_model(n)

        # Use indexed JSON format for clear text boundaries (compact for token efficiency)
        indexed_texts = {str(i): text for i, text in enumerate(texts)}
        texts_json = json.dumps(indexed_texts, ensure_ascii=False)

        user_content = (
            f"Translate {n} texts from {source_name} to {target_name}.\n"
            f"CRITICAL: Return exactly {n} translations, one for each indexed input.\n\n"
            f"Input texts (indexed JSON):\n{texts_json}\n\n"
            f"Return translations array with {n} elements in index order (0 to {n - 1})."
        )

        try:
            response = await client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format=response_model,
            )

            result = response.choices[0].message.parsed
            if result is None:
                raise TranslationError("OpenAI returned empty response")

            # Validate response length (should be guaranteed by schema, but verify)
            translations: list[str] = result.translations  # type: ignore[attr-defined]
            if len(translations) != n:
                raise ArrayLengthMismatchError(
                    expected=n,
                    actual=len(translations),
                )

            return translations

        except self._get_openai_errors() as e:
            self._handle_openai_error(e)
            raise  # Should not reach here

    def _get_openai_errors(self) -> tuple[type[Exception], ...]:
        """Get OpenAI exception types for error handling.

        Catches OpenAIError (base class) to handle all API errors including:
        - AuthenticationError
        - RateLimitError
        - BadRequestError
        - APIConnectionError
        - APITimeoutError

        Returns:
            Tuple of exception types to catch.
        """
        try:
            from openai import OpenAIError

            return (OpenAIError,)
        except ImportError:
            return (Exception,)

    def _handle_openai_error(self, error: Any) -> None:
        """Handle OpenAI API errors.

        Args:
            error: The caught exception.

        Raises:
            ConfigurationError: On authentication or model access failure.
            TranslationError: On other API errors.
        """
        try:
            from openai import AuthenticationError, NotFoundError, RateLimitError
        except ImportError:
            raise TranslationError(f"OpenAI API error: {error}") from error

        if isinstance(error, AuthenticationError):
            raise ConfigurationError("Invalid OpenAI API key") from error
        elif isinstance(error, RateLimitError):
            raise TranslationError(
                "OpenAI rate limit exceeded, please retry later"
            ) from error
        elif isinstance(error, NotFoundError):
            # NotFoundError is raised when model is not found
            raise ConfigurationError(
                f"Model '{self._model}' is not available. "
                f"Set OPENAI_MODEL environment variable to use a different model "
                f"(e.g., 'gpt-4o-mini', 'gpt-4o')."
            ) from error

        # Check for model not found/access error via error code or message
        error_code = getattr(error, "code", None) or ""
        error_str = str(error).lower()
        is_model_error = (
            error_code in ("model_not_found", "invalid_model", "not_found")
            or (
                "model" in error_str
                and (
                    "not found" in error_str
                    or "invalid" in error_str
                    or "does not exist" in error_str
                    or "do not have access" in error_str
                )
            )
        )
        if is_model_error:
            raise ConfigurationError(
                f"Model '{self._model}' is not available. "
                f"Set OPENAI_MODEL environment variable to use a different model "
                f"(e.g., 'gpt-4o-mini', 'gpt-4o')."
            ) from error

        raise TranslationError(f"OpenAI API error: {error}") from error

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
