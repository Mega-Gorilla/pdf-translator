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


def _has_openai() -> bool:
    """Check if openai is available."""
    try:
        import openai  # noqa: F401

        return True
    except ImportError:
        return False


# Skip all tests in this module if openai is not installed
pytestmark = pytest.mark.skipif(not _has_openai(), reason="openai not installed")

# Import OpenAITranslator only if openai is available
if _has_openai():
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
        # Create mock response with wrong number of indexed translations
        mock_item1 = MagicMock()
        mock_item1.index = 0
        mock_item1.translation = "translation1"

        mock_item2 = MagicMock()
        mock_item2.index = 1
        mock_item2.translation = "translation2"

        mock_result = MagicMock()
        mock_result.translations = [mock_item1, mock_item2]  # 2 instead of 3

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

        result_msg = str(exc_info.value)
        assert "not available" in result_msg
        assert "OPENAI_MODEL" in result_msg

    @pytest.mark.asyncio
    async def test_model_not_found_via_notfounderror(
        self, mock_translator: OpenAITranslator
    ) -> None:
        """Should handle NotFoundError for model not found."""
        from openai import NotFoundError

        # Create a NotFoundError-like exception
        error = NotFoundError(
            message="The model 'gpt-5-nano' does not exist",
            response=MagicMock(status_code=404),
            body={"error": {"code": "model_not_found"}},
        )
        mock_translator._client.beta.chat.completions.parse = AsyncMock(
            side_effect=error
        )

        with pytest.raises(ConfigurationError) as exc_info:
            await mock_translator.translate_batch(
                ["test text"],
                "en",
                "ja",
            )

        result_msg = str(exc_info.value)
        assert "not available" in result_msg
        assert "OPENAI_MODEL" in result_msg

    @pytest.mark.asyncio
    async def test_model_not_found_via_error_code(
        self, mock_translator: OpenAITranslator
    ) -> None:
        """Should detect model_not_found via error.code attribute."""
        from openai import OpenAIError

        # Create an error with code attribute
        error = OpenAIError("API error")
        error.code = "model_not_found"  # type: ignore[attr-defined]
        mock_translator._client.beta.chat.completions.parse = AsyncMock(
            side_effect=error
        )

        with pytest.raises(ConfigurationError) as exc_info:
            await mock_translator.translate_batch(
                ["test text"],
                "en",
                "ja",
            )

        result_msg = str(exc_info.value)
        assert "not available" in result_msg


class TestOrderPreservation:
    """Tests for order preservation with large batches (> 9 items).

    These tests verify the fix for the string key sorting issue where
    dictionary keys like "10" sort before "2" in lexicographic order,
    causing translation misalignment.
    """

    @pytest.fixture
    def mock_translator(self) -> OpenAITranslator:
        """Create a translator with mocked OpenAI client."""
        with patch("openai.AsyncOpenAI"):
            translator = OpenAITranslator(api_key="test-key")
            translator._client = AsyncMock()
            return translator

    @pytest.mark.asyncio
    async def test_order_preserved_with_15_items(
        self, mock_translator: OpenAITranslator
    ) -> None:
        """Order should be preserved for 15 items (tests index > 9)."""
        # Create 15 indexed translation items in shuffled order
        # This simulates the model returning items out of order
        indices = list(range(15))
        import random

        random.seed(42)  # Reproducible shuffle
        shuffled = indices.copy()
        random.shuffle(shuffled)

        mock_items = []
        for idx in shuffled:
            item = MagicMock()
            item.index = idx
            item.translation = f"翻訳テキスト{idx}"
            mock_items.append(item)

        mock_result = MagicMock()
        mock_result.translations = mock_items

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_result

        mock_translator._client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        texts = [f"text{i}" for i in range(15)]
        result = await mock_translator.translate_batch(texts, "en", "ja")

        # Verify order is preserved (sorted by index, not insertion order)
        assert len(result) == 15
        for i, translation in enumerate(result):
            assert translation == f"翻訳テキスト{i}", f"Index {i} mismatch"

    @pytest.mark.asyncio
    async def test_order_preserved_with_25_items(
        self, mock_translator: OpenAITranslator
    ) -> None:
        """Order should be preserved for 25 items (tests indices 10-24)."""
        # Create 25 indexed translation items in reverse order
        mock_items = []
        for idx in reversed(range(25)):
            item = MagicMock()
            item.index = idx
            item.translation = f"翻訳{idx}"
            mock_items.append(item)

        mock_result = MagicMock()
        mock_result.translations = mock_items

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_result

        mock_translator._client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        texts = [f"text{i}" for i in range(25)]
        result = await mock_translator.translate_batch(texts, "en", "ja")

        # Verify order is preserved
        assert len(result) == 25
        for i, translation in enumerate(result):
            assert translation == f"翻訳{i}", f"Index {i} mismatch"

    @pytest.mark.asyncio
    async def test_missing_index_raises_error(
        self, mock_translator: OpenAITranslator
    ) -> None:
        """Missing index should raise ArrayLengthMismatchError."""
        # Create items with a gap (missing index 1)
        mock_items = []
        for idx in [0, 2]:  # Skip index 1
            item = MagicMock()
            item.index = idx
            item.translation = f"翻訳{idx}"
            mock_items.append(item)

        mock_result = MagicMock()
        mock_result.translations = mock_items

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_result

        mock_translator._client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        with pytest.raises(ArrayLengthMismatchError):
            await mock_translator.translate_batch(
                ["text0", "text1", "text2"], "en", "ja"
            )

    @pytest.mark.asyncio
    async def test_duplicate_index_raises_error(
        self, mock_translator: OpenAITranslator
    ) -> None:
        """Duplicate index should raise ArrayLengthMismatchError."""
        # Create items with duplicate index 0
        mock_items = []
        for idx in [0, 0, 2]:  # Duplicate index 0
            item = MagicMock()
            item.index = idx
            item.translation = f"翻訳{idx}"
            mock_items.append(item)

        mock_result = MagicMock()
        mock_result.translations = mock_items

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_result

        mock_translator._client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        with pytest.raises(ArrayLengthMismatchError):
            await mock_translator.translate_batch(
                ["text0", "text1", "text2"], "en", "ja"
            )


class TestBatchSplittingFallback:
    """Tests for batch splitting fallback in translation pipeline."""

    @pytest.fixture
    def mock_translator(self) -> MagicMock:
        """Create a mock translator."""
        translator = MagicMock()
        translator.name = "openai"
        translator.max_text_length = 100000  # Default max text length for tests
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


class TestParallelTranslation:
    """Tests for parallel batch translation."""

    @pytest.fixture
    def mock_translator(self) -> MagicMock:
        """Create a mock translator."""
        translator = MagicMock()
        translator.name = "openai"
        translator.max_text_length = 100000  # Default max text length for tests
        # Disable token-aware chunking for these tests
        translator.max_batch_tokens = None
        translator.count_tokens = None
        return translator

    @pytest.mark.asyncio
    async def test_parallel_translation_preserves_order(
        self, mock_translator: MagicMock
    ) -> None:
        """Parallel translation should preserve original order."""
        import asyncio

        from pdf_translator.core.models import BBox, Paragraph
        from pdf_translator.pipeline.translation_pipeline import (
            PipelineConfig,
            TranslationPipeline,
        )

        # Create mock with varying delays to test order preservation
        async def mock_translate_batch(
            texts: list[str], source: str, target: str
        ) -> list[str]:
            # Add varying delays to simulate real API behavior
            delay = 0.01 * (hash(texts[0]) % 5)
            await asyncio.sleep(delay)
            return [f"translated_{t}" for t in texts]

        mock_translator.translate_batch = mock_translate_batch

        config = PipelineConfig(
            translation_batch_size=2,
            translation_max_concurrent=5,
        )
        pipeline = TranslationPipeline(mock_translator, config)

        # Create test paragraphs
        paragraphs = [
            Paragraph(
                id=f"p{i}",
                text=f"text{i}",
                page_number=0,
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
                original_font_size=12.0,
                category="text",
            )
            for i in range(10)
        ]

        result = await pipeline._stage_translate(paragraphs)

        # Verify order is preserved
        assert len(result) == 10
        for i, para in enumerate(result):
            assert para.translated_text == f"translated_text{i}"

    @pytest.mark.asyncio
    async def test_parallel_translation_progress_callback(
        self, mock_translator: MagicMock
    ) -> None:
        """Progress callback should be called during parallel translation."""
        from pdf_translator.core.models import BBox, Paragraph
        from pdf_translator.pipeline.translation_pipeline import (
            PipelineConfig,
            TranslationPipeline,
        )

        async def mock_translate_batch(
            texts: list[str], source: str, target: str
        ) -> list[str]:
            return [f"translated_{t}" for t in texts]

        mock_translator.translate_batch = mock_translate_batch

        progress_calls: list[tuple[str, int, int]] = []

        def progress_callback(
            stage: str, current: int, total: int, message: str = ""
        ) -> None:
            progress_calls.append((stage, current, total))

        config = PipelineConfig(
            translation_batch_size=3,
            translation_max_concurrent=5,
        )
        pipeline = TranslationPipeline(
            mock_translator, config, progress_callback=progress_callback
        )

        paragraphs = [
            Paragraph(
                id=f"p{i}",
                text=f"text{i}",
                page_number=0,
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
                original_font_size=12.0,
                category="text",
            )
            for i in range(9)
        ]

        await pipeline._stage_translate(paragraphs)

        # Verify progress was reported
        translate_calls = [c for c in progress_calls if c[0] == "translate"]
        assert len(translate_calls) == 3  # 9 paragraphs / 3 batch size = 3 chunks
        # Final call should show total completed
        assert translate_calls[-1][1] == 9
        assert translate_calls[-1][2] == 9

    @pytest.mark.asyncio
    async def test_max_concurrent_zero_is_clamped_to_one(
        self, mock_translator: MagicMock
    ) -> None:
        """translation_max_concurrent=0 should be clamped to 1."""
        from pdf_translator.core.models import BBox, Paragraph
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
            return [f"translated_{t}" for t in texts]

        mock_translator.translate_batch = mock_translate_batch

        # This should not hang due to Semaphore(0)
        config = PipelineConfig(
            translation_batch_size=2,
            translation_max_concurrent=0,  # Should be clamped to 1
        )
        pipeline = TranslationPipeline(mock_translator, config)

        paragraphs = [
            Paragraph(
                id=f"p{i}",
                text=f"text{i}",
                page_number=0,
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
                original_font_size=12.0,
                category="text",
            )
            for i in range(4)
        ]

        result = await pipeline._stage_translate(paragraphs)

        assert len(result) == 4
        assert call_count == 2  # 4 paragraphs / 2 batch size = 2 calls

    @pytest.mark.asyncio
    async def test_max_concurrent_negative_is_clamped_to_one(
        self, mock_translator: MagicMock
    ) -> None:
        """translation_max_concurrent=-1 should be clamped to 1."""
        from pdf_translator.core.models import BBox, Paragraph
        from pdf_translator.pipeline.translation_pipeline import (
            PipelineConfig,
            TranslationPipeline,
        )

        async def mock_translate_batch(
            texts: list[str], source: str, target: str
        ) -> list[str]:
            return [f"translated_{t}" for t in texts]

        mock_translator.translate_batch = mock_translate_batch

        config = PipelineConfig(
            translation_batch_size=2,
            translation_max_concurrent=-1,  # Should be clamped to 1
        )
        pipeline = TranslationPipeline(mock_translator, config)

        paragraphs = [
            Paragraph(
                id="p0",
                text="text0",
                page_number=0,
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
                original_font_size=12.0,
                category="text",
            )
        ]

        result = await pipeline._stage_translate(paragraphs)

        assert len(result) == 1
        assert result[0].translated_text == "translated_text0"
