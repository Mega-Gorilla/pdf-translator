# SPDX-License-Identifier: Apache-2.0
"""Google Translate backend using deep-translator."""

import asyncio

from deep_translator import GoogleTranslator as DeepGoogleTranslator  # type: ignore[import-untyped]

from pdf_translator.translators.base import TranslationError


class GoogleTranslator:
    """Google Translate backend.

    This backend uses Google Translate via deep-translator library.
    No API key is required (uses free web API).

    Attributes:
        name: Backend identifier ("google").
    """

    def __init__(self, max_concurrent: int = 5) -> None:
        """Initialize GoogleTranslator.

        Args:
            max_concurrent: Maximum concurrent translation requests.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def name(self) -> str:
        """Return backend name."""
        return "google"

    @property
    def max_text_length(self) -> int:
        """Maximum text length for Google Translate.

        Google Translate web API has a 5,000 character limit.
        """
        return 5000

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate a single text using Google Translate.

        Args:
            text: Text to translate.
            source_lang: Source language code ("en", "ja", "auto").
            target_lang: Target language code ("en", "ja").

        Returns:
            Translated text.

        Raises:
            TranslationError: On translation failure.
        """
        # Early return for empty or whitespace-only text
        if not text or not text.strip():
            return text

        async with self._semaphore:
            return await asyncio.to_thread(
                self._translate_sync, text, source_lang, target_lang
            )

    def _translate_sync(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Synchronous translation implementation.

        Args:
            text: Text to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            Translated text.

        Raises:
            TranslationError: On translation failure.
        """
        try:
            translator = DeepGoogleTranslator(source=source_lang, target=target_lang)
            result = translator.translate(text)
            return result if result is not None else text
        except Exception as e:
            raise TranslationError(f"Google Translate failed: {e}") from e

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate multiple texts in batch using parallel execution.

        Args:
            texts: List of texts to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of translated texts (same order and length as input).

        Raises:
            TranslationError: On translation failure.
        """
        if not texts:
            return []

        tasks = [self.translate(t, source_lang, target_lang) for t in texts]
        return list(await asyncio.gather(*tasks))
