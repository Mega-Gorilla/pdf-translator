# SPDX-License-Identifier: Apache-2.0
"""Tests for translation backends."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pdf_translator.translators import (
    ConfigurationError,
    GoogleTranslator,
    TranslationError,
    TranslatorBackend,
    TranslatorError,
    get_deepl_translator,
    get_openai_translator,
)


class TestTranslatorBackendProtocol:
    """Test TranslatorBackend protocol."""

    def test_google_translator_implements_protocol(self) -> None:
        """GoogleTranslator should implement TranslatorBackend protocol."""
        translator = GoogleTranslator()
        assert isinstance(translator, TranslatorBackend)

    def test_protocol_has_name(self) -> None:
        """TranslatorBackend should have name property."""
        translator = GoogleTranslator()
        assert hasattr(translator, "name")
        assert translator.name == "google"


class TestExceptions:
    """Test exception hierarchy."""

    def test_translation_error_inherits_from_translator_error(self) -> None:
        """TranslationError should inherit from TranslatorError."""
        assert issubclass(TranslationError, TranslatorError)

    def test_configuration_error_inherits_from_translator_error(self) -> None:
        """ConfigurationError should inherit from TranslatorError."""
        assert issubclass(ConfigurationError, TranslatorError)

    def test_translator_error_inherits_from_exception(self) -> None:
        """TranslatorError should inherit from Exception."""
        assert issubclass(TranslatorError, Exception)


class TestGoogleTranslator:
    """Test GoogleTranslator."""

    def test_name(self) -> None:
        """GoogleTranslator should have name 'google'."""
        translator = GoogleTranslator()
        assert translator.name == "google"

    @pytest.mark.asyncio
    async def test_translate_empty_string(self) -> None:
        """Empty string should return as-is."""
        translator = GoogleTranslator()
        result = await translator.translate("", "en", "ja")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_whitespace_only(self) -> None:
        """Whitespace-only string should return as-is."""
        translator = GoogleTranslator()
        result = await translator.translate("   ", "en", "ja")
        assert result == "   "

    @pytest.mark.asyncio
    async def test_translate_batch_empty_list(self) -> None:
        """Empty list should return empty list."""
        translator = GoogleTranslator()
        result = await translator.translate_batch([], "en", "ja")
        assert result == []

    @pytest.mark.asyncio
    async def test_translate_batch_with_empty_strings(self) -> None:
        """Batch with empty strings should preserve them."""
        translator = GoogleTranslator()

        with patch.object(
            translator, "_translate_sync", return_value="translated"
        ):
            result = await translator.translate_batch(
                ["hello", "", "  ", "world"], "en", "ja"
            )

        assert len(result) == 4
        assert result[1] == ""  # Empty string preserved
        assert result[2] == "  "  # Whitespace preserved

    @pytest.mark.asyncio
    async def test_translate_mocked(self) -> None:
        """Test translate with mocked sync function."""
        translator = GoogleTranslator()

        with patch.object(
            translator, "_translate_sync", return_value="こんにちは"
        ):
            result = await translator.translate("Hello", "en", "ja")
            assert result == "こんにちは"

    def test_translate_sync_error_handling(self) -> None:
        """_translate_sync should wrap exceptions in TranslationError."""
        translator = GoogleTranslator()

        with patch(
            "pdf_translator.translators.google.DeepGoogleTranslator"
        ) as mock_class:
            mock_class.return_value.translate.side_effect = Exception(
                "API error"
            )
            with pytest.raises(TranslationError) as exc_info:
                translator._translate_sync("Hello", "en", "ja")
            assert "Google Translate failed" in str(exc_info.value)


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Integration tests disabled (set RUN_INTEGRATION=1 to run)",
)
class TestGoogleTranslatorIntegration:
    """Integration tests for GoogleTranslator (real API).

    These tests require network access and call the real Google Translate API.
    Run with: RUN_INTEGRATION=1 pytest tests/test_translators.py
    """

    @pytest.mark.asyncio
    async def test_real_translation_en_to_ja(self) -> None:
        """Test real translation from English to Japanese."""
        translator = GoogleTranslator()
        result = await translator.translate("Hello", "en", "ja")
        # Should return Japanese text, not the original
        assert result != "Hello"
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_real_translation_ja_to_en(self) -> None:
        """Test real translation from Japanese to English."""
        translator = GoogleTranslator()
        result = await translator.translate("こんにちは", "ja", "en")
        # Should return English text
        assert result != "こんにちは"
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_real_batch_translation(self) -> None:
        """Test real batch translation."""
        translator = GoogleTranslator()
        texts = ["Hello", "World", "Good morning"]
        results = await translator.translate_batch(texts, "en", "ja")
        # Should return same number of translations
        assert len(results) == len(texts)
        # Each should be translated
        for original, translated in zip(texts, results):
            assert translated != original


class TestDeepLTranslatorUnit:
    """Unit tests for DeepLTranslator (mocked)."""

    def test_requires_api_key(self) -> None:
        """DeepLTranslator should require API key."""
        DeepLTranslator = get_deepl_translator()
        with pytest.raises(ConfigurationError) as exc_info:
            DeepLTranslator(api_key="")
        assert "API key is required" in str(exc_info.value)

    def test_name(self) -> None:
        """DeepLTranslator should have name 'deepl'."""
        DeepLTranslator = get_deepl_translator()
        translator = DeepLTranslator(api_key="test-key")
        assert translator.name == "deepl"

    def test_custom_api_url(self) -> None:
        """DeepLTranslator should accept custom API URL."""
        DeepLTranslator = get_deepl_translator()
        translator = DeepLTranslator(
            api_key="test-key",
            api_url="https://api.deepl.com/v2/translate",
        )
        assert translator._api_url == "https://api.deepl.com/v2/translate"

    @pytest.mark.asyncio
    async def test_translate_empty_string(self) -> None:
        """Empty string should return as-is."""
        DeepLTranslator = get_deepl_translator()
        translator = DeepLTranslator(api_key="test-key")
        result = await translator.translate("", "en", "ja")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_batch_empty_list(self) -> None:
        """Empty list should return empty list."""
        DeepLTranslator = get_deepl_translator()
        translator = DeepLTranslator(api_key="test-key")
        result = await translator.translate_batch([], "en", "ja")
        assert result == []

    @pytest.mark.asyncio
    async def test_translate_batch_mocked(self) -> None:
        """Test batch translation with mocked API."""
        DeepLTranslator = get_deepl_translator()
        translator = DeepLTranslator(api_key="test-key")

        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "translations": [
                    {"text": "こんにちは"},
                    {"text": "世界"},
                ]
            }
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock())
        mock_session.post.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        translator._session = mock_session

        result = await translator.translate_batch(
            ["Hello", "World"], "en", "ja"
        )
        assert result == ["こんにちは", "世界"]

    @pytest.mark.asyncio
    async def test_translate_batch_with_empty_strings(self) -> None:
        """Batch with empty strings should preserve them."""
        DeepLTranslator = get_deepl_translator()
        translator = DeepLTranslator(api_key="test-key")

        # Mock only non-empty translation
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "translations": [
                    {"text": "こんにちは"},
                    {"text": "世界"},
                ]
            }
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock())
        mock_session.post.return_value.__aenter__ = AsyncMock(
            return_value=mock_response
        )
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        translator._session = mock_session

        result = await translator.translate_batch(
            ["Hello", "", "  ", "World"], "en", "ja"
        )
        assert len(result) == 4
        assert result[0] == "こんにちは"
        assert result[1] == ""  # Empty preserved
        assert result[2] == "  "  # Whitespace preserved
        assert result[3] == "世界"


class TestOpenAITranslatorUnit:
    """Unit tests for OpenAITranslator (mocked)."""

    def test_requires_api_key(self) -> None:
        """OpenAITranslator should require API key."""
        OpenAITranslator = get_openai_translator()
        with pytest.raises(ConfigurationError) as exc_info:
            OpenAITranslator(api_key="")
        assert "API key is required" in str(exc_info.value)

    def test_name(self) -> None:
        """OpenAITranslator should have name 'openai'."""
        OpenAITranslator = get_openai_translator()
        translator = OpenAITranslator(api_key="test-key")
        assert translator.name == "openai"

    def test_custom_model(self) -> None:
        """OpenAITranslator should accept custom model."""
        OpenAITranslator = get_openai_translator()
        translator = OpenAITranslator(api_key="test-key", model="gpt-4o")
        assert translator._model == "gpt-4o"

    def test_custom_system_prompt(self) -> None:
        """OpenAITranslator should accept custom system prompt."""
        OpenAITranslator = get_openai_translator()
        custom_prompt = "You are a specialized technical translator."
        translator = OpenAITranslator(
            api_key="test-key", system_prompt=custom_prompt
        )
        assert translator._system_prompt == custom_prompt

    @pytest.mark.asyncio
    async def test_translate_empty_string(self) -> None:
        """Empty string should return as-is."""
        OpenAITranslator = get_openai_translator()
        translator = OpenAITranslator(api_key="test-key")
        result = await translator.translate("", "en", "ja")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_batch_empty_list(self) -> None:
        """Empty list should return empty list."""
        OpenAITranslator = get_openai_translator()
        translator = OpenAITranslator(api_key="test-key")
        result = await translator.translate_batch([], "en", "ja")
        assert result == []

    @pytest.mark.asyncio
    async def test_translate_batch_all_empty(self) -> None:
        """Batch with all empty strings should return them as-is."""
        OpenAITranslator = get_openai_translator()
        translator = OpenAITranslator(api_key="test-key")
        result = await translator.translate_batch(["", "  ", ""], "en", "ja")
        assert result == ["", "  ", ""]


class TestLazyImports:
    """Test lazy import functions."""

    def test_get_deepl_translator_returns_class(self) -> None:
        """get_deepl_translator should return DeepLTranslator class."""
        DeepLTranslator = get_deepl_translator()
        assert DeepLTranslator.__name__ == "DeepLTranslator"

    def test_get_openai_translator_returns_class(self) -> None:
        """get_openai_translator should return OpenAITranslator class."""
        OpenAITranslator = get_openai_translator()
        assert OpenAITranslator.__name__ == "OpenAITranslator"
