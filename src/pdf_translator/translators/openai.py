# SPDX-License-Identifier: Apache-2.0
"""OpenAI GPT translation backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

from pdf_translator.translators.base import (
    ArrayLengthMismatchError,
    ConfigurationError,
    TranslationError,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from pydantic import BaseModel


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


class OpenAITranslator:
    """OpenAI GPT translation backend.

    This backend uses OpenAI's GPT models with Structured Outputs
    to ensure reliable array-based translation responses.

    Supports custom system prompts for specialized translation needs.

    Attributes:
        name: Backend identifier ("openai").
    """

    DEFAULT_MODEL = "gpt-5-nano"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize OpenAITranslator.

        Args:
            api_key: OpenAI API key.
            model: Model to use. Priority: argument > OPENAI_MODEL env > default.
            system_prompt: Custom system prompt for translation.

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

        self._api_key = api_key
        env_model = os.environ.get("OPENAI_MODEL")
        self._model = model or env_model or self.DEFAULT_MODEL
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._client: AsyncOpenAI | None = None

    @property
    def name(self) -> str:
        """Return backend name."""
        return "openai"

    @property
    def max_text_length(self) -> None:
        """Maximum text length for OpenAI.

        OpenAI uses token-based limits, not character limits.
        Returns None because:
        1. Token limits vary by model (GPT-4o: 128K, GPT-4.1: 1M tokens)
        2. Character-to-token ratio varies by language and content
        3. tiktoken dependency would add complexity

        Token limit errors are handled by the pipeline's retry logic:
        - TranslationError triggers retry with exponential backoff
        - ArrayLengthMismatchError triggers batch splitting

        For very long texts, consider using a model with larger context.
        """
        return None

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
