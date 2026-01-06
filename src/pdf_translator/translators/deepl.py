# SPDX-License-Identifier: Apache-2.0
"""DeepL translation backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pdf_translator.translators.base import ConfigurationError, TranslationError

if TYPE_CHECKING:
    import aiohttp


class DeepLTranslator:
    """DeepL translation backend.

    This backend uses DeepL API for high-quality translation.
    Requires an API key (free or pro).

    Supports batch translation with multiple text parameters in a single request.

    Attributes:
        name: Backend identifier ("deepl").
    """

    DEFAULT_API_URL = "https://api-free.deepl.com/v2/translate"
    MAX_TEXTS_PER_REQUEST = 50
    MAX_REQUEST_SIZE = 128 * 1024  # 128KB

    def __init__(
        self,
        api_key: str,
        api_url: str | None = None,
    ) -> None:
        """Initialize DeepLTranslator.

        Args:
            api_key: DeepL API key.
            api_url: API URL (default: free API endpoint).

        Raises:
            ConfigurationError: If API key is not provided.
            ImportError: If aiohttp is not installed.
        """
        if not api_key:
            raise ConfigurationError("DeepL API key is required")

        # Lazy import aiohttp
        try:
            import aiohttp as _aiohttp

            self._aiohttp = _aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for DeepL backend. "
                "Install with: pip install pdf-translator[deepl]"
            ) from None

        self._api_key = api_key
        self._api_url = api_url or self.DEFAULT_API_URL
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        """Return backend name."""
        return "deepl"

    @property
    def max_text_length(self) -> int:
        """Maximum text length for DeepL.

        DeepL has a 128KB request limit. With UTF-8 encoding,
        this allows approximately 50,000 characters safely.
        """
        return 50000

    async def __aenter__(self) -> DeepLTranslator:
        """Enter async context manager."""
        self._session = self._aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists.

        Returns:
            Active aiohttp session.
        """
        if self._session is None:
            self._session = self._aiohttp.ClientSession()
        return self._session

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate a single text using DeepL.

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
        """Translate multiple texts in batch using DeepL.

        Uses multiple 'text' parameters in a single request for efficiency.
        Automatically chunks requests if they exceed API limits.

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

        # Chunk texts if necessary
        chunks = self._chunk_texts(non_empty_texts)
        translated_texts: list[str] = []

        for chunk in chunks:
            chunk_results = await self._translate_chunk(
                chunk, source_lang, target_lang
            )
            translated_texts.extend(chunk_results)

        # Restore translations to original positions
        for i, translated in zip(non_empty_indices, translated_texts):
            results[i] = translated

        return results

    def _chunk_texts(self, texts: list[str]) -> list[list[str]]:
        """Split texts into chunks respecting API limits.

        Args:
            texts: List of texts to chunk.

        Returns:
            List of text chunks.
        """
        chunks: list[list[str]] = []
        current_chunk: list[str] = []
        current_size = 0

        for text in texts:
            text_size = len(text.encode("utf-8"))

            # Check if adding this text would exceed limits
            if (
                len(current_chunk) >= self.MAX_TEXTS_PER_REQUEST
                or current_size + text_size > self.MAX_REQUEST_SIZE
            ):
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [text]
                current_size = text_size
            else:
                current_chunk.append(text)
                current_size += text_size

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def _translate_chunk(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate a chunk of texts.

        Args:
            texts: List of texts to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of translated texts.

        Raises:
            TranslationError: On translation failure.
            ConfigurationError: On authentication failure.
        """
        session = await self._ensure_session()

        # Build request parameters with multiple 'text' entries
        params: list[tuple[str, str]] = [("text", t) for t in texts]
        params.append(("auth_key", self._api_key))
        params.append(("target_lang", target_lang.upper()))

        # DeepL doesn't support "auto" - omit source_lang for auto-detection
        if source_lang.lower() != "auto":
            params.append(("source_lang", source_lang.upper()))

        try:
            async with session.post(self._api_url, data=params) as response:
                if response.status == 200:
                    data = await response.json()
                    translations = [t["text"] for t in data["translations"]]
                    # Validate response length matches input
                    if len(translations) != len(texts):
                        raise TranslationError(
                            f"DeepL returned {len(translations)} translations "
                            f"for {len(texts)} texts"
                        )
                    return translations
                elif response.status == 403:
                    raise ConfigurationError("Invalid DeepL API key")
                elif response.status == 429:
                    raise TranslationError(
                        "DeepL rate limit exceeded, please retry later"
                    )
                elif response.status >= 500:
                    raise TranslationError(
                        f"DeepL server error (status {response.status})"
                    )
                else:
                    error_text = await response.text()
                    raise TranslationError(
                        f"DeepL API error (status {response.status}): {error_text}"
                    )
        except self._aiohttp.ClientError as e:
            raise TranslationError(f"DeepL request failed: {e}") from e

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
