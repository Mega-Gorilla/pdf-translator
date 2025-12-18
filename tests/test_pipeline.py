# SPDX-License-Identifier: Apache-2.0
"""Tests for the translation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pdf_translator.core.models import BBox, Font, Metadata, Page, PDFDocument, TextObject
from pdf_translator.pipeline import (
    ExtractionError,
    FontAdjustmentError,
    LayoutAnalysisError,
    MergeError,
    PipelineConfig,
    PipelineError,
    ProgressCallback,
    TranslationPipeline,
    TranslationResult,
)
from pdf_translator.translators.base import ConfigurationError, TranslationError

# Test fixtures path
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample_llama.pdf"


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.source_lang == "en"
        assert config.target_lang == "ja"
        assert config.use_layout_analysis is True
        assert config.layout_containment_threshold == 0.5
        assert config.line_y_tolerance == 3.0
        assert config.min_font_size == 6.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PipelineConfig(
            source_lang="ja",
            target_lang="en",
            use_layout_analysis=False,
            max_retries=5,
        )

        assert config.source_lang == "ja"
        assert config.target_lang == "en"
        assert config.use_layout_analysis is False
        assert config.max_retries == 5


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic result creation."""
        pdf_bytes = b"%PDF-1.4 test content"
        result = TranslationResult(pdf_bytes=pdf_bytes)

        assert result.pdf_bytes == pdf_bytes
        assert result.stats is None

    def test_result_with_stats(self) -> None:
        """Test result with statistics."""
        pdf_bytes = b"%PDF-1.4 test content"
        stats = {"pages": 5, "translated_objects": 100}
        result = TranslationResult(pdf_bytes=pdf_bytes, stats=stats)

        assert result.pdf_bytes == pdf_bytes
        assert result.stats == stats
        assert result.stats["pages"] == 5


class TestPipelineErrors:
    """Tests for pipeline error classes."""

    def test_pipeline_error_basic(self) -> None:
        """Test basic PipelineError."""
        error = PipelineError("Test error", stage="extract")

        assert str(error) == "[extract] Test error"
        assert error.stage == "extract"
        assert error.cause is None

    def test_pipeline_error_with_cause(self) -> None:
        """Test PipelineError with cause."""
        cause = ValueError("Original error")
        error = PipelineError("Test error", stage="translate", cause=cause)

        assert "caused by" in str(error)
        assert error.cause is cause

    def test_extraction_error(self) -> None:
        """Test ExtractionError defaults to extract stage."""
        error = ExtractionError("Failed to extract")

        assert error.stage == "extract"
        assert "[extract]" in str(error)

    def test_layout_analysis_error(self) -> None:
        """Test LayoutAnalysisError defaults to analyze stage."""
        error = LayoutAnalysisError("Analysis failed")

        assert error.stage == "analyze"
        assert "[analyze]" in str(error)

    def test_merge_error(self) -> None:
        """Test MergeError defaults to merge stage."""
        error = MergeError("Merge failed")

        assert error.stage == "merge"
        assert "[merge]" in str(error)

    def test_font_adjustment_error(self) -> None:
        """Test FontAdjustmentError defaults to font_adjust stage."""
        error = FontAdjustmentError("Adjustment failed")

        assert error.stage == "font_adjust"
        assert "[font_adjust]" in str(error)


class TestProgressCallback:
    """Tests for ProgressCallback protocol."""

    def test_protocol_compliance(self) -> None:
        """Test that a function can be used as ProgressCallback."""
        calls: list[tuple[str, int, int, str]] = []

        def my_callback(stage: str, current: int, total: int, message: str = "") -> None:
            calls.append((stage, current, total, message))

        # Verify it's recognized as ProgressCallback
        assert isinstance(my_callback, ProgressCallback)

        # Test calling it
        my_callback("extract", 1, 1, "Done")
        assert calls == [("extract", 1, 1, "Done")]


class TestTranslationPipelineInit:
    """Tests for TranslationPipeline initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default config."""
        translator = MagicMock()
        translator.name = "mock"

        pipeline = TranslationPipeline(translator)

        assert pipeline._translator is translator
        assert pipeline._config.source_lang == "en"
        assert pipeline._config.target_lang == "ja"
        assert pipeline._progress_callback is None

    def test_init_with_config(self) -> None:
        """Test initialization with custom config."""
        translator = MagicMock()
        config = PipelineConfig(source_lang="de", target_lang="en")

        pipeline = TranslationPipeline(translator, config=config)

        assert pipeline._config.source_lang == "de"
        assert pipeline._config.target_lang == "en"

    def test_init_with_progress_callback(self) -> None:
        """Test initialization with progress callback."""
        translator = MagicMock()

        def my_callback(stage: str, current: int, total: int, message: str = "") -> None:
            pass

        pipeline = TranslationPipeline(translator, progress_callback=my_callback)

        assert pipeline._progress_callback is my_callback


class TestTranslationPipelineStages:
    """Tests for individual pipeline stages."""

    @pytest.fixture
    def mock_translator(self) -> MagicMock:
        """Create a mock translator."""
        translator = MagicMock()
        translator.name = "mock"
        translator.translate_batch = AsyncMock(return_value=["翻訳済み"])
        return translator

    @pytest.fixture
    def sample_pdf_doc(self) -> PDFDocument:
        """Create a sample PDF document for testing."""
        text_obj = TextObject(
            id="text_p0_i0",
            bbox=BBox(x0=72, y0=700, x1=300, y1=720),
            text="Hello world",
            font=Font(name="Helvetica", size=12.0),
        )
        page = Page(
            page_number=0,
            width=612,
            height=792,
            text_objects=[text_obj],
        )
        metadata = Metadata(source_file="test.pdf", created_at="2025-01-01T00:00:00", page_count=1)
        return PDFDocument(pages=[page], metadata=metadata)

    async def test_stage_extract(self, mock_translator: MagicMock) -> None:
        """Test extraction stage."""
        if not SAMPLE_PDF.exists():
            pytest.skip("Sample PDF not found")

        pipeline = TranslationPipeline(mock_translator)
        pdf_doc, processor = await pipeline._stage_extract(SAMPLE_PDF)

        try:
            assert isinstance(pdf_doc, PDFDocument)
            assert len(pdf_doc.pages) > 0
        finally:
            processor.close()

    async def test_stage_extract_error(self, mock_translator: MagicMock) -> None:
        """Test extraction stage with invalid file."""
        pipeline = TranslationPipeline(mock_translator)

        with pytest.raises(ExtractionError) as exc_info:
            await pipeline._stage_extract(Path("/nonexistent/file.pdf"))

        assert exc_info.value.stage == "extract"

    async def test_stage_merge(
        self, mock_translator: MagicMock, sample_pdf_doc: PDFDocument
    ) -> None:
        """Test merge stage."""
        from pdf_translator.core.models import ProjectCategory

        pipeline = TranslationPipeline(mock_translator)
        categories = {"text_p0_i0": ProjectCategory.TEXT}

        result = pipeline._stage_merge(sample_pdf_doc, categories)

        assert 0 in result
        assert len(result[0]) == 1
        assert result[0][0].text == "Hello world"

    async def test_stage_translate(self, mock_translator: MagicMock) -> None:
        """Test translation stage."""
        pipeline = TranslationPipeline(mock_translator)

        text_obj = TextObject(
            id="test",
            bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            text="Hello",
        )
        sorted_objects = {0: [text_obj]}

        await pipeline._stage_translate(sorted_objects)

        # Text should be updated in-place
        assert text_obj.text == "翻訳済み"
        mock_translator.translate_batch.assert_called_once()

    async def test_stage_translate_empty(self, mock_translator: MagicMock) -> None:
        """Test translation stage with no objects."""
        pipeline = TranslationPipeline(mock_translator)

        await pipeline._stage_translate({})

        # Should not call translator
        mock_translator.translate_batch.assert_not_called()

    def test_stage_font_adjust(self, mock_translator: MagicMock) -> None:
        """Test font adjustment stage."""
        pipeline = TranslationPipeline(mock_translator)

        text_obj = TextObject(
            id="test",
            bbox=BBox(x0=0, y0=0, x1=50, y1=20),  # Narrow box
            text="This is a very long text that needs adjustment",
            font=Font(name="Helvetica", size=12.0),
        )
        sorted_objects = {0: [text_obj]}

        pipeline._stage_font_adjust(sorted_objects)

        # Font size should be reduced
        assert text_obj.font is not None
        assert text_obj.font.size < 12.0


class TestTranslationPipelineRetry:
    """Tests for retry logic."""

    async def test_retry_on_translation_error(self) -> None:
        """Test that translation retries on TranslationError."""
        translator = MagicMock()
        translator.name = "mock"
        translator.translate_batch = AsyncMock(
            side_effect=[
                TranslationError("Temporary error"),
                ["翻訳済み"],
            ]
        )

        config = PipelineConfig(max_retries=3, retry_delay=0.01)
        pipeline = TranslationPipeline(translator, config=config)

        result = await pipeline._translate_with_retry(["Hello"])

        assert result == ["翻訳済み"]
        assert translator.translate_batch.call_count == 2

    async def test_no_retry_on_configuration_error(self) -> None:
        """Test that ConfigurationError is not retried."""
        translator = MagicMock()
        translator.name = "mock"
        translator.translate_batch = AsyncMock(
            side_effect=ConfigurationError("Invalid API key")
        )

        config = PipelineConfig(max_retries=3, retry_delay=0.01)
        pipeline = TranslationPipeline(translator, config=config)

        with pytest.raises(ConfigurationError):
            await pipeline._translate_with_retry(["Hello"])

        # Should only be called once (no retry)
        assert translator.translate_batch.call_count == 1

    async def test_max_retries_exceeded(self) -> None:
        """Test PipelineError after max retries."""
        translator = MagicMock()
        translator.name = "mock"
        translator.translate_batch = AsyncMock(
            side_effect=TranslationError("Persistent error")
        )

        config = PipelineConfig(max_retries=2, retry_delay=0.01)
        pipeline = TranslationPipeline(translator, config=config)

        with pytest.raises(PipelineError) as exc_info:
            await pipeline._translate_with_retry(["Hello"])

        assert exc_info.value.stage == "translate"
        assert "after 2 retries" in str(exc_info.value)
        assert translator.translate_batch.call_count == 3  # Initial + 2 retries


class TestTranslationPipelineProgress:
    """Tests for progress reporting."""

    async def test_progress_callback_called(self) -> None:
        """Test that progress callback is called during pipeline."""
        translator = MagicMock()
        translator.name = "mock"
        translator.translate_batch = AsyncMock(return_value=["翻訳済み"])

        calls: list[tuple[str, int, int, str]] = []

        def my_callback(stage: str, current: int, total: int, message: str = "") -> None:
            calls.append((stage, current, total, message))

        config = PipelineConfig(use_layout_analysis=False)
        pipeline = TranslationPipeline(translator, config=config, progress_callback=my_callback)

        # Just test _report_progress directly
        pipeline._report_progress("test", 1, 10, "Testing")

        assert len(calls) == 1
        assert calls[0] == ("test", 1, 10, "Testing")


class TestTranslationPipelineLayoutDisabled:
    """Tests for pipeline with layout analysis disabled."""

    @pytest.fixture
    def mock_translator(self) -> MagicMock:
        """Create a mock translator."""
        translator = MagicMock()
        translator.name = "mock"
        translator.translate_batch = AsyncMock(side_effect=lambda texts, *args: texts)
        return translator

    async def test_all_objects_translated_without_layout(
        self, mock_translator: MagicMock
    ) -> None:
        """Test that all objects are treated as TEXT when layout disabled."""
        from pdf_translator.core.models import ProjectCategory

        config = PipelineConfig(use_layout_analysis=False)
        pipeline = TranslationPipeline(mock_translator, config=config)

        # Create a sample PDF document
        text_obj = TextObject(
            id="text_p0_i0",
            bbox=BBox(x0=72, y0=700, x1=300, y1=720),
            text="Hello",
        )
        page = Page(page_number=0, width=612, height=792, text_objects=[text_obj])
        metadata = Metadata(source_file="test.pdf", created_at="2025-01-01T00:00:00", page_count=1)
        pdf_doc = PDFDocument(pages=[page], metadata=metadata)

        categories = await pipeline._stage_analyze(Path("dummy.pdf"), pdf_doc)

        assert "text_p0_i0" in categories
        assert categories["text_p0_i0"] == ProjectCategory.TEXT


class TestTranslationPipelineIntegration:
    """Integration tests for the full pipeline (optional, with real PDF)."""

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not found")
    async def test_full_pipeline_without_layout(self) -> None:
        """Test full pipeline without layout analysis."""
        translator = MagicMock()
        translator.name = "mock"
        # Return same text (identity translation for testing)
        translator.translate_batch = AsyncMock(
            side_effect=lambda texts, *args: [f"[ja] {t}" for t in texts]
        )

        config = PipelineConfig(use_layout_analysis=False)
        pipeline = TranslationPipeline(translator, config=config)

        result = await pipeline.translate(SAMPLE_PDF)

        assert isinstance(result, TranslationResult)
        assert len(result.pdf_bytes) > 0
        assert result.pdf_bytes.startswith(b"%PDF-")
        assert result.stats is not None
        assert result.stats["pages"] > 0
