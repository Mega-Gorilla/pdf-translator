# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline markdown integration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pdf_translator.core.models import BBox, Paragraph
from pdf_translator.output.markdown_writer import MarkdownOutputMode
from pdf_translator.pipeline.translation_pipeline import (
    PipelineConfig,
    TranslationPipeline,
    TranslationResult,
)


class TestPipelineConfigMarkdown:
    """Test markdown-related PipelineConfig fields."""

    def test_default_values(self) -> None:
        """Test default values for markdown options."""
        config = PipelineConfig()

        assert config.markdown_output is False
        assert config.markdown_mode == MarkdownOutputMode.TRANSLATED_ONLY
        assert config.markdown_include_metadata is True
        assert config.markdown_include_page_breaks is True
        assert config.markdown_heading_offset == 0
        assert config.save_intermediate is False

    def test_custom_values(self) -> None:
        """Test custom values for markdown options."""
        config = PipelineConfig(
            markdown_output=True,
            markdown_mode=MarkdownOutputMode.PARALLEL,
            markdown_include_metadata=False,
            markdown_include_page_breaks=False,
            markdown_heading_offset=2,
            save_intermediate=True,
        )

        assert config.markdown_output is True
        assert config.markdown_mode == MarkdownOutputMode.PARALLEL
        assert config.markdown_include_metadata is False
        assert config.markdown_include_page_breaks is False
        assert config.markdown_heading_offset == 2
        assert config.save_intermediate is True


class TestTranslationResultMarkdown:
    """Test markdown-related TranslationResult fields."""

    def test_default_values(self) -> None:
        """Test default values for markdown fields."""
        result = TranslationResult(pdf_bytes=b"test")

        assert result.markdown is None
        assert result.paragraphs is None

    def test_with_markdown(self) -> None:
        """Test result with markdown content."""
        paragraphs = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Hello",
                block_bbox=BBox(0, 0, 100, 50),
                line_count=1,
                translated_text="こんにちは",
            ),
        ]
        result = TranslationResult(
            pdf_bytes=b"test",
            markdown="# Hello\n\nこんにちは\n",
            paragraphs=paragraphs,
        )

        assert result.markdown == "# Hello\n\nこんにちは\n"
        assert result.paragraphs is not None
        assert len(result.paragraphs) == 1
        assert result.paragraphs[0].translated_text == "こんにちは"


class TestPipelineMarkdownStage:
    """Test _stage_markdown method."""

    def test_stage_markdown_translated_only(self) -> None:
        """Test markdown generation in TRANSLATED_ONLY mode."""
        mock_translator = MagicMock()
        config = PipelineConfig(
            markdown_output=True,
            markdown_mode=MarkdownOutputMode.TRANSLATED_ONLY,
            source_lang="en",
            target_lang="ja",
            extract_images=False,  # Disable for unit test
            extract_tables=False,  # Disable for unit test
        )
        pipeline = TranslationPipeline(mock_translator, config)

        paragraphs = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Hello",
                block_bbox=BBox(0, 0, 100, 50),
                line_count=1,
                category="text",
                translated_text="こんにちは",
            ),
        ]
        pdf_path = Path("/tmp/test.pdf")

        markdown = pipeline._stage_markdown(paragraphs, pdf_path, {}, None)

        assert "こんにちは" in markdown
        # Should NOT contain original text in TRANSLATED_ONLY mode
        # (unless fallback is triggered, but we have translated_text)

    def test_stage_markdown_parallel(self) -> None:
        """Test markdown generation in PARALLEL mode."""
        mock_translator = MagicMock()
        config = PipelineConfig(
            markdown_output=True,
            markdown_mode=MarkdownOutputMode.PARALLEL,
            source_lang="en",
            target_lang="ja",
            extract_images=False,  # Disable for unit test
            extract_tables=False,  # Disable for unit test
        )
        pipeline = TranslationPipeline(mock_translator, config)

        paragraphs = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Hello",
                block_bbox=BBox(0, 0, 100, 50),
                line_count=1,
                category="text",
                translated_text="こんにちは",
            ),
        ]
        pdf_path = Path("/tmp/test.pdf")

        markdown = pipeline._stage_markdown(paragraphs, pdf_path, {}, None)

        # Should contain both original and translation
        assert "Hello" in markdown
        assert "こんにちは" in markdown

    def test_stage_markdown_with_metadata(self) -> None:
        """Test markdown includes metadata when enabled."""
        mock_translator = MagicMock()
        config = PipelineConfig(
            markdown_output=True,
            markdown_include_metadata=True,
            source_lang="en",
            target_lang="ja",
            extract_images=False,  # Disable for unit test
            extract_tables=False,  # Disable for unit test
        )
        pipeline = TranslationPipeline(mock_translator, config)

        paragraphs = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Test",
                block_bbox=BBox(0, 0, 100, 50),
                line_count=1,
                translated_text="テスト",
            ),
        ]
        pdf_path = Path("/tmp/sample.pdf")

        markdown = pipeline._stage_markdown(paragraphs, pdf_path, {}, None)

        # Check YAML frontmatter
        assert "---" in markdown
        assert "source_lang: en" in markdown
        assert "target_lang: ja" in markdown
        assert "title: sample.pdf" in markdown

    def test_stage_markdown_without_metadata(self) -> None:
        """Test markdown excludes metadata when disabled."""
        mock_translator = MagicMock()
        config = PipelineConfig(
            markdown_output=True,
            markdown_include_metadata=False,
            source_lang="en",
            target_lang="ja",
            extract_images=False,  # Disable for unit test
            extract_tables=False,  # Disable for unit test
        )
        pipeline = TranslationPipeline(mock_translator, config)

        paragraphs = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Test",
                block_bbox=BBox(0, 0, 100, 50),
                line_count=1,
                translated_text="テスト",
            ),
        ]
        pdf_path = Path("/tmp/sample.pdf")

        markdown = pipeline._stage_markdown(paragraphs, pdf_path, {}, None)

        # Should NOT have YAML frontmatter
        assert "source_lang:" not in markdown
        assert "target_lang:" not in markdown


class TestPipelineSaveIntermediate:
    """Test _save_intermediate method."""

    def test_save_intermediate_creates_json(self) -> None:
        """Test intermediate JSON files are created (base + translation)."""
        from pdf_translator.output.base_document import BaseDocument, BaseDocumentMetadata
        from pdf_translator.output.translation_document import TranslationDocument

        mock_translator = MagicMock()
        mock_translator.__class__.__name__ = "GoogleTranslator"

        config = PipelineConfig(
            save_intermediate=True,
            source_lang="en",
            target_lang="ja",
        )
        pipeline = TranslationPipeline(mock_translator, config)
        # Simulate _stage_extract setting the page count
        pipeline._page_count = 2

        paragraphs = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Hello",
                block_bbox=BBox(0, 0, 100, 50),
                line_count=1,
                translated_text="こんにちは",
            ),
            Paragraph(
                id="p2",
                page_number=1,
                text="World",
                block_bbox=BBox(0, 0, 100, 50),
                line_count=1,
                translated_text="世界",
            ),
        ]

        # Create BaseDocument
        metadata = BaseDocumentMetadata(
            source_file="input.pdf",
            source_lang="en",
            page_count=2,
            paragraph_count=len(paragraphs),
        )
        base_document = BaseDocument(
            metadata=metadata,
            paragraphs=paragraphs,
            summary=None,
        )

        # Create TranslationDocument
        trans_paragraphs = {p.id: p.translated_text for p in paragraphs if p.translated_text}
        translation_document = TranslationDocument.from_pipeline_result(
            paragraphs=trans_paragraphs,
            target_lang="ja",
            base_file="input.json",
            translator_backend="google",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output.ja.pdf"

            pipeline._save_intermediate(
                base_document, translation_document, pdf_path, output_path
            )

            # Check base JSON was created
            base_json_path = Path(tmpdir) / "output.json"
            assert base_json_path.exists()

            with base_json_path.open() as f:
                base_data = json.load(f)

            assert base_data["metadata"]["source_file"] == "input.pdf"
            assert base_data["metadata"]["source_lang"] == "en"
            assert base_data["metadata"]["page_count"] == 2
            assert len(base_data["paragraphs"]) == 2

            # Check translation JSON was created
            trans_json_path = Path(tmpdir) / "output.ja.json"
            assert trans_json_path.exists()

            with trans_json_path.open() as f:
                trans_data = json.load(f)

            assert trans_data["target_lang"] == "ja"
            assert trans_data["translator_backend"] == "google"
            assert trans_data["translated_count"] == 2

    def test_save_intermediate_empty_paragraphs(self) -> None:
        """Test intermediate JSON with empty paragraphs."""
        from pdf_translator.output.base_document import BaseDocument, BaseDocumentMetadata
        from pdf_translator.output.translation_document import TranslationDocument

        mock_translator = MagicMock()
        mock_translator.__class__.__name__ = "DeepLTranslator"

        config = PipelineConfig(
            save_intermediate=True,
            source_lang="en",
            target_lang="ja",
        )
        pipeline = TranslationPipeline(mock_translator, config)

        # Create empty BaseDocument
        metadata = BaseDocumentMetadata(
            source_file="empty.pdf",
            source_lang="en",
            page_count=0,
            paragraph_count=0,
        )
        base_document = BaseDocument(
            metadata=metadata,
            paragraphs=[],
            summary=None,
        )

        # Create empty TranslationDocument
        translation_document = TranslationDocument.from_pipeline_result(
            paragraphs={},
            target_lang="ja",
            base_file="empty.json",
            translator_backend="deepl",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "empty.pdf"
            output_path = Path(tmpdir) / "output.ja.pdf"

            pipeline._save_intermediate(
                base_document, translation_document, pdf_path, output_path
            )

            base_json_path = Path(tmpdir) / "output.json"
            assert base_json_path.exists()

            with base_json_path.open() as f:
                data = json.load(f)

            assert data["metadata"]["page_count"] == 0
            assert len(data["paragraphs"]) == 0


class TestPipelineIntegrationMarkdown:
    """Integration tests for markdown pipeline."""

    def _create_mock_translator(self) -> MagicMock:
        """Create a properly configured mock translator."""
        mock_translator = MagicMock()
        mock_translator.__class__.__name__ = "GoogleTranslator"
        mock_translator.max_text_length = None
        # Use AsyncMock for async methods only
        mock_translator.translate_batch = AsyncMock(return_value=["翻訳されたテキスト"])
        mock_translator.translate = AsyncMock(return_value="翻訳されたテキスト")
        # These should NOT be AsyncMock - they are regular attributes/methods
        mock_translator.max_batch_tokens = None
        mock_translator.count_tokens = None
        return mock_translator

    @pytest.mark.asyncio
    async def test_translate_with_markdown_output(self) -> None:
        """Test translate method generates markdown when enabled."""
        mock_translator = self._create_mock_translator()

        config = PipelineConfig(
            markdown_output=True,
            source_lang="en",
            target_lang="ja",
            # Note: markdown_output=True forces layout analysis, so we mock it
        )
        pipeline = TranslationPipeline(mock_translator, config)

        # Mock ParagraphExtractor
        test_paragraph = Paragraph(
            id="test_1",
            page_number=0,
            text="Test text",
            block_bbox=BBox(10, 10, 200, 50),
            line_count=1,
            category="text",
        )

        with (
            patch(
                "pdf_translator.pipeline.translation_pipeline.ParagraphExtractor"
            ) as mock_extractor,
            patch(
                "pdf_translator.pipeline.translation_pipeline.PDFProcessor"
            ) as mock_processor,
            patch.object(
                pipeline._analyzer, "analyze_all", return_value={}
            ),
            tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f,
        ):
            f.write(b"%PDF-1.4 dummy")
            f.flush()
            pdf_path = Path(f.name)

            mock_extractor.extract_from_pdf.return_value = [test_paragraph]

            mock_proc_instance = MagicMock()
            mock_proc_instance.page_count = 1
            mock_proc_instance.to_bytes.return_value = b"translated pdf"
            mock_proc_instance.__enter__ = MagicMock(return_value=mock_proc_instance)
            mock_proc_instance.__exit__ = MagicMock(return_value=False)
            mock_processor.return_value = mock_proc_instance

            result = await pipeline.translate(pdf_path)

            assert result.markdown is not None
            assert "翻訳されたテキスト" in result.markdown
            assert result.paragraphs is not None
            assert len(result.paragraphs) == 1

            pdf_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_translate_without_markdown_output(self) -> None:
        """Test translate method does not generate markdown when disabled."""
        mock_translator = self._create_mock_translator()

        config = PipelineConfig(
            markdown_output=False,  # Disabled
            source_lang="en",
            target_lang="ja",
            layout_analysis=False,
        )
        pipeline = TranslationPipeline(mock_translator, config)

        test_paragraph = Paragraph(
            id="test_1",
            page_number=0,
            text="Test text",
            block_bbox=BBox(10, 10, 200, 50),
            line_count=1,
            category="text",
        )

        with (
            patch(
                "pdf_translator.pipeline.translation_pipeline.ParagraphExtractor"
            ) as mock_extractor,
            patch(
                "pdf_translator.pipeline.translation_pipeline.PDFProcessor"
            ) as mock_processor,
            tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f,
        ):
            f.write(b"%PDF-1.4 dummy")
            f.flush()
            pdf_path = Path(f.name)

            mock_extractor.extract_from_pdf.return_value = [test_paragraph]

            mock_proc_instance = MagicMock()
            mock_proc_instance.page_count = 1
            mock_proc_instance.to_bytes.return_value = b"translated pdf"
            mock_proc_instance.__enter__ = MagicMock(return_value=mock_proc_instance)
            mock_proc_instance.__exit__ = MagicMock(return_value=False)
            mock_processor.return_value = mock_proc_instance

            result = await pipeline.translate(pdf_path)

            # Markdown should be None when disabled
            assert result.markdown is None
            # But paragraphs should still be returned
            assert result.paragraphs is not None

            pdf_path.unlink(missing_ok=True)
