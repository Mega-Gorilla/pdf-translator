# SPDX-License-Identifier: Apache-2.0
"""Translation backend modules.

This module provides translation backends for Google Translate, DeepL, and OpenAI.

Google Translate is always available (no API key required).
DeepL and OpenAI require optional dependencies and API keys.

Usage:
    # Google Translate (always available)
    from pdf_translator.translators import GoogleTranslator
    translator = GoogleTranslator()
    result = await translator.translate("Hello", "en", "ja")

    # DeepL (requires aiohttp and API key)
    from pdf_translator.translators import get_deepl_translator
    DeepLTranslator = get_deepl_translator()
    translator = DeepLTranslator(api_key="your-api-key")

    # OpenAI (requires openai package and API key)
    from pdf_translator.translators import get_openai_translator
    OpenAITranslator = get_openai_translator()
    translator = OpenAITranslator(api_key="your-api-key")
"""

from pdf_translator.translators.base import (
    ConfigurationError,
    QuotaExceededError,
    TranslationError,
    TranslatorBackend,
    TranslatorError,
)
from pdf_translator.translators.google import GoogleTranslator

__all__ = [
    # Protocol and exceptions
    "TranslatorBackend",
    "TranslatorError",
    "TranslationError",
    "ConfigurationError",
    "QuotaExceededError",
    # Always available
    "GoogleTranslator",
    # Lazy import functions
    "get_deepl_translator",
    "get_openai_translator",
]


def get_deepl_translator() -> type:
    """Get DeepLTranslator class with lazy import.

    This function imports DeepLTranslator only when called,
    avoiding import errors when aiohttp is not installed.

    Returns:
        DeepLTranslator class.

    Raises:
        ImportError: If aiohttp is not installed.
    """
    from pdf_translator.translators.deepl import DeepLTranslator

    return DeepLTranslator


def get_openai_translator() -> type:
    """Get OpenAITranslator class with lazy import.

    This function imports OpenAITranslator only when called,
    avoiding import errors when openai package is not installed.

    Returns:
        OpenAITranslator class.

    Raises:
        ImportError: If openai package is not installed.
    """
    from pdf_translator.translators.openai import OpenAITranslator

    return OpenAITranslator
