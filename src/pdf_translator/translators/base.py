# SPDX-License-Identifier: Apache-2.0
"""Base classes and protocols for translation backends."""

from typing import Protocol, runtime_checkable


class TranslatorError(Exception):
    """Base exception for translator module."""

    pass


class TranslationError(TranslatorError):
    """Error during translation (API call failure, rate limit, etc.).

    This error type is potentially retryable.
    """

    pass


class ConfigurationError(TranslatorError):
    """Configuration error (missing API key, invalid parameters, etc.).

    This error type is NOT retryable - fix the configuration first.
    """

    pass


@runtime_checkable
class TranslatorBackend(Protocol):
    """Protocol definition for translation backends.

    All translator implementations must conform to this protocol.
    """

    @property
    def name(self) -> str:
        """Backend name ("google", "deepl", "openai")."""
        ...

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate a single text.

        Args:
            text: Text to translate.
            source_lang: Source language code ("en", "ja", "auto").
            target_lang: Target language code ("en", "ja").

        Returns:
            Translated text.

        Raises:
            TranslationError: On translation failure.
        """
        ...

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """Translate multiple texts in batch.

        Args:
            texts: List of texts to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of translated texts (same order and length as input).

        Raises:
            TranslationError: On translation failure.
        """
        ...
