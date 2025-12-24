# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI translator and batch splitting fallback."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pdf_translator.translators.base import (
    ArrayLengthMismatchError,
    ConfigurationError,
    TranslationError,
)
from pdf_translator.translators.openai import OpenAITranslator


class TestOpenAITranslatorModel:
    """Tests for model configuration."""

    def test_default_model_is_gpt5_nano(self) -> None:
        """Default model should be gpt-5-nano."""
        assert OpenAITranslator.DEFAULT_MODEL == "gpt-5-nano"

    @patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o"}, clear=False)
    def test_model_from_env_variable(self) -> None:
        """Model can be set via OPENAI_MODEL environment variable."""
        with patch("openai.AsyncOpenAI"):
            translator = OpenAITranslator(api_key="test-key")
            assert translator._model == "gpt-4o"

    @patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o"}, clear=False)
    def test_model_constructor_overrides_env(self) -> None:
        """Constructor argument should override environment variable."""
        with patch("openai.AsyncOpenAI"):
            translator = OpenAITranslator(api_key="test-key", model="gpt-5-mini")
            assert translator._model == "gpt-5-mini"

    def test_model_default_when_no_env(self) -> None:
        """Default model is used when no env var or constructor arg."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("openai.AsyncOpenAI"),
        ):
            # Remove OPENAI_MODEL if it exists
            os.environ.pop("OPENAI_MODEL", None)
            translator = OpenAITranslator(api_key="test-key")
            assert translator._model == "gpt-5-nano"


class TestArrayLengthMismatchError:
    """Tests for ArrayLengthMismatchError."""

    def test_error_message(self) -> None:
        """Error message should include expected and actual counts."""
        error = ArrayLengthMismatchError(expected=10, actual=9)
        assert str(error) == "Expected 10 translations but got 9"
        assert error.expected == 10
        assert error.actual == 9

    def test_inherits_from_translation_error(self) -> None:
        """ArrayLengthMismatchError should inherit from TranslationError."""
        error = ArrayLengthMismatchError(expected=5, actual=3)
        assert isinstance(error, TranslationError)


class TestOpenAITranslatorErrorHandling:
    """Tests for OpenAI translator error handling."""

    @pytest.fixture
    def mock_translator(self) -> OpenAITranslator:
        """Create a translator with mocked OpenAI client."""
        with patch("openai.AsyncOpenAI"):
            translator = OpenAITranslator(api_key="test-key")
            translator._client = AsyncMock()
            return translator

    @pytest.mark.asyncio
    async def test_array_length_mismatch_raises_error(
        self, mock_translator: OpenAITranslator
    ) -> None:
        """Should raise ArrayLengthMismatchError on length mismatch."""
        # Create mock response with wrong number of translations
        mock_result = MagicMock()
        mock_result.translations = ["translation1", "translation2"]  # 2 instead of 3

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_result

        mock_translator._client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        with pytest.raises(ArrayLengthMismatchError) as exc_info:
            await mock_translator.translate_batch(
                ["text1", "text2", "text3"],
                "en",
                "ja",
            )

        assert exc_info.value.expected == 3
        assert exc_info.value.actual == 2

    @pytest.mark.asyncio
    async def test_model_not_found_error_guidance(
        self, mock_translator: OpenAITranslator
    ) -> None:
        """Should provide guidance when model is not found."""
        # Simulate model not found error
        from openai import OpenAIError

        error_msg = "The model 'gpt-5-nano' does not exist or you do not have access"
        mock_translator._client.beta.chat.completions.parse = AsyncMock(
            side_effect=OpenAIError(error_msg)
        )

        with pytest.raises(ConfigurationError) as exc_info:
            await mock_translator.translate_batch(
                ["test text"],
                "en",
                "ja",
            )

        error_msg = str(exc_info.value)
        assert "not available" in error_msg
        assert "OPENAI_MODEL" in error_msg


class TestBatchSplittingFallback:
    """Tests for batch splitting fallback in translation pipeline."""

    @pytest.fixture
    def mock_translator(self) -> MagicMock:
        """Create a mock translator."""
        translator = MagicMock()
        translator.name = "openai"
        return translator

    @pytest.mark.asyncio
    async def test_array_length_mismatch_triggers_split(
        self, mock_translator: MagicMock
    ) -> None:
        """ArrayLengthMismatchError should trigger batch splitting."""
        from pdf_translator.pipeline.translation_pipeline import (
            PipelineConfig,
            TranslationPipeline,
        )

        call_count = 0

        async def mock_translate_batch(
            texts: list[str], source: str, target: str
        ) -> list[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1 and len(texts) == 4:
                # First call with full batch fails
                raise ArrayLengthMismatchError(expected=4, actual=3)
            # Subsequent calls succeed
            return [f"translated_{t}" for t in texts]

        mock_translator.translate_batch = mock_translate_batch
        mock_translator.translate = AsyncMock(
            side_effect=lambda t, s, d: f"translated_{t}"
        )

        config = PipelineConfig()
        pipeline = TranslationPipeline(mock_translator, config)

        result = await pipeline._translate_with_retry(
            ["text1", "text2", "text3", "text4"]
        )

        assert len(result) == 4
        assert all(r.startswith("translated_") for r in result)

    @pytest.mark.asyncio
    async def test_split_to_individual_translation(
        self, mock_translator: MagicMock
    ) -> None:
        """Should fall back to individual translation for single texts."""
        from pdf_translator.pipeline.translation_pipeline import (
            PipelineConfig,
            TranslationPipeline,
        )

        async def mock_translate(text: str, source: str, target: str) -> str:
            return f"translated_{text}"

        mock_translator.translate = mock_translate

        config = PipelineConfig()
        pipeline = TranslationPipeline(mock_translator, config)

        result = await pipeline._translate_with_split(["single_text"])

        assert result == ["translated_single_text"]

    @pytest.mark.asyncio
    async def test_strict_mode_raises_on_failure(
        self, mock_translator: MagicMock
    ) -> None:
        """Strict mode should raise error on single text translation failure."""
        from pdf_translator.pipeline.translation_pipeline import (
            PipelineConfig,
            TranslationPipeline,
        )

        async def mock_translate(text: str, source: str, target: str) -> str:
            raise TranslationError("API error")

        mock_translator.translate = mock_translate

        config = PipelineConfig(strict_mode=True)
        pipeline = TranslationPipeline(mock_translator, config)

        with pytest.raises(TranslationError):
            await pipeline._translate_with_split(["failing_text"])

    @pytest.mark.asyncio
    async def test_lenient_mode_returns_original(
        self, mock_translator: MagicMock
    ) -> None:
        """Lenient mode should return original text on failure."""
        from pdf_translator.pipeline.translation_pipeline import (
            PipelineConfig,
            TranslationPipeline,
        )

        async def mock_translate(text: str, source: str, target: str) -> str:
            raise TranslationError("API error")

        mock_translator.translate = mock_translate

        config = PipelineConfig(strict_mode=False)  # Default lenient mode
        pipeline = TranslationPipeline(mock_translator, config)

        result = await pipeline._translate_with_split(["failing_text"])

        assert result == ["failing_text"]

    @pytest.mark.asyncio
    async def test_split_recursive(self, mock_translator: MagicMock) -> None:
        """Should recursively split batch on continued failures."""
        from pdf_translator.pipeline.translation_pipeline import (
            PipelineConfig,
            TranslationPipeline,
        )

        call_count = 0

        async def mock_translate_batch(
            texts: list[str], source: str, target: str
        ) -> list[str]:
            nonlocal call_count
            call_count += 1
            if len(texts) > 2:
                # Fail for batches larger than 2
                raise ArrayLengthMismatchError(expected=len(texts), actual=len(texts) - 1)
            return [f"translated_{t}" for t in texts]

        mock_translator.translate_batch = mock_translate_batch
        mock_translator.translate = AsyncMock(
            side_effect=lambda t, s, d: f"translated_{t}"
        )

        config = PipelineConfig()
        pipeline = TranslationPipeline(mock_translator, config)

        result = await pipeline._translate_with_retry(
            ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
        )

        assert len(result) == 8
        assert all(r.startswith("translated_") for r in result)

    @pytest.mark.asyncio
    async def test_split_empty_list(self, mock_translator: MagicMock) -> None:
        """Should handle empty list."""
        from pdf_translator.pipeline.translation_pipeline import (
            PipelineConfig,
            TranslationPipeline,
        )

        config = PipelineConfig()
        pipeline = TranslationPipeline(mock_translator, config)

        result = await pipeline._translate_with_split([])

        assert result == []
