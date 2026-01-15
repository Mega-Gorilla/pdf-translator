# SPDX-License-Identifier: Apache-2.0
"""Tests for SummaryExtractor."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pdf_translator.core.models import BBox, Paragraph
from pdf_translator.output.summary_extractor import SummaryExtractor


class TestSummaryExtractor:
    """Tests for SummaryExtractor."""

    @pytest.fixture
    def sample_paragraphs(self) -> list[Paragraph]:
        """Create sample paragraphs for testing."""
        return [
            Paragraph(
                id="p0_0",
                page_number=0,
                block_bbox=BBox(x0=100, y0=700, x1=500, y1=750),
                text="Test Document Title",
                line_count=1,
                translated_text=None,
                category="doc_title",
            ),
            Paragraph(
                id="p0_1",
                page_number=0,
                block_bbox=BBox(x0=100, y0=600, x1=500, y1=680),
                text="This is the abstract of the document.",
                line_count=2,
                translated_text="これはドキュメントの要約です。",
                category="abstract",
            ),
            Paragraph(
                id="p0_2",
                page_number=0,
                block_bbox=BBox(x0=100, y0=400, x1=500, y1=580),
                text="This is the main text content.",
                line_count=3,
                translated_text="これは本文です。",
                category="text",
            ),
        ]

    @pytest.fixture
    def sample_pdf(self) -> Path:
        """Get sample PDF path."""
        pdf_path = Path("tests/fixtures/sample_llama.pdf")
        if not pdf_path.exists():
            pytest.skip("Sample PDF not found")
        return pdf_path

    async def test_extract_without_thumbnail_or_llm(
        self, sample_paragraphs: list[Paragraph]
    ) -> None:
        """Test extraction without thumbnail or LLM."""
        extractor = SummaryExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await extractor.extract(
                paragraphs=sample_paragraphs,
                pdf_path=Path("nonexistent.pdf"),
                output_dir=Path(tmpdir),
                output_stem="test",
                source_lang="en",
                target_lang="ja",
                page_count=1,
                generate_thumbnail=False,
            )

        assert summary.title == "Test Document Title"
        assert summary.title_translated is None  # No translator provided
        assert summary.abstract == "This is the abstract of the document."
        assert summary.abstract_translated == "これはドキュメントの要約です。"
        assert summary.title_source == "layout"
        assert summary.abstract_source == "layout"
        assert summary.thumbnail_path is None
        assert summary.page_count == 1
        assert summary.source_lang == "en"
        assert summary.target_lang == "ja"

    async def test_extract_with_thumbnail(
        self, sample_paragraphs: list[Paragraph], sample_pdf: Path
    ) -> None:
        """Test extraction with thumbnail generation."""
        extractor = SummaryExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await extractor.extract(
                paragraphs=sample_paragraphs,
                pdf_path=sample_pdf,
                output_dir=Path(tmpdir),
                output_stem="test",
                source_lang="en",
                target_lang="ja",
                page_count=1,
                generate_thumbnail=True,
            )

            assert summary.thumbnail_path == "test_thumbnail.png"
            assert summary.thumbnail_width == 400
            assert summary.thumbnail_height > 0
            assert summary._thumbnail_bytes is not None

            # Verify thumbnail file was created
            thumb_file = Path(tmpdir) / "test_thumbnail.png"
            assert thumb_file.exists()

    async def test_extract_with_translator(
        self, sample_paragraphs: list[Paragraph]
    ) -> None:
        """Test extraction with translator for title."""
        extractor = SummaryExtractor()

        # Create mock translator
        mock_translator = AsyncMock()
        mock_translator.translate = AsyncMock(return_value="テストドキュメントタイトル")

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await extractor.extract(
                paragraphs=sample_paragraphs,
                pdf_path=Path("nonexistent.pdf"),
                output_dir=Path(tmpdir),
                output_stem="test",
                source_lang="en",
                target_lang="ja",
                page_count=1,
                generate_thumbnail=False,
                translator=mock_translator,
            )

        assert summary.title == "Test Document Title"
        assert summary.title_translated == "テストドキュメントタイトル"
        mock_translator.translate.assert_called_once_with(
            "Test Document Title", "en", "ja"
        )

    async def test_extract_multiple_abstract_paragraphs(self) -> None:
        """Test extraction with multiple abstract paragraphs."""
        paragraphs = [
            Paragraph(
                id="p0_0",
                page_number=0,
                block_bbox=BBox(x0=100, y0=700, x1=500, y1=750),
                text="First abstract paragraph.",
                line_count=1,
                translated_text="最初の要約段落。",
                category="abstract",
            ),
            Paragraph(
                id="p0_1",
                page_number=0,
                block_bbox=BBox(x0=100, y0=650, x1=500, y1=700),
                text="Second abstract paragraph.",
                line_count=1,
                translated_text="2番目の要約段落。",
                category="abstract",
            ),
        ]

        extractor = SummaryExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await extractor.extract(
                paragraphs=paragraphs,
                pdf_path=Path("nonexistent.pdf"),
                output_dir=Path(tmpdir),
                output_stem="test",
                generate_thumbnail=False,
            )

        # Paragraphs should be merged with double newline
        assert "First abstract paragraph." in summary.abstract
        assert "Second abstract paragraph." in summary.abstract
        assert "\n\n" in summary.abstract

    async def test_extract_no_title_or_abstract(self) -> None:
        """Test extraction when no title or abstract is found."""
        paragraphs = [
            Paragraph(
                id="p0_0",
                page_number=0,
                block_bbox=BBox(x0=100, y0=400, x1=500, y1=580),
                text="Just regular text.",
                line_count=1,
                translated_text="通常のテキスト。",
                category="text",
            ),
        ]

        extractor = SummaryExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = await extractor.extract(
                paragraphs=paragraphs,
                pdf_path=Path("nonexistent.pdf"),
                output_dir=Path(tmpdir),
                output_stem="test",
                generate_thumbnail=False,
            )

        assert summary.title is None
        assert summary.abstract is None
        assert summary.title_source == "layout"
        assert summary.abstract_source == "layout"


class TestFindAndMergeByCategory:
    """Tests for _find_and_merge_by_category static method."""

    def test_single_paragraph(self) -> None:
        """Test with single matching paragraph."""
        paragraphs = [
            Paragraph(
                id="p0_0",
                page_number=0,
                block_bbox=BBox(x0=0, y0=700, x1=100, y1=750),
                text="Test Title",
                line_count=1,
                translated_text="テストタイトル",
                category="doc_title",
            ),
        ]

        original, translated = SummaryExtractor._find_and_merge_by_category(
            paragraphs, "doc_title"
        )

        assert original == "Test Title"
        assert translated == "テストタイトル"

    def test_multiple_paragraphs(self) -> None:
        """Test with multiple matching paragraphs."""
        paragraphs = [
            Paragraph(
                id="p0_0",
                page_number=0,
                block_bbox=BBox(x0=0, y0=700, x1=100, y1=750),
                text="First paragraph",
                line_count=1,
                translated_text="最初の段落",
                category="abstract",
            ),
            Paragraph(
                id="p0_1",
                page_number=0,
                block_bbox=BBox(x0=0, y0=600, x1=100, y1=650),
                text="Second paragraph",
                line_count=1,
                translated_text="2番目の段落",
                category="abstract",
            ),
        ]

        original, translated = SummaryExtractor._find_and_merge_by_category(
            paragraphs, "abstract"
        )

        assert "First paragraph" in original
        assert "Second paragraph" in original
        assert "\n\n" in original

    def test_no_matching_category(self) -> None:
        """Test with no matching category."""
        paragraphs = [
            Paragraph(
                id="p0_0",
                page_number=0,
                block_bbox=BBox(x0=0, y0=700, x1=100, y1=750),
                text="Text content",
                line_count=1,
                translated_text="テキストコンテンツ",
                category="text",
            ),
        ]

        original, translated = SummaryExtractor._find_and_merge_by_category(
            paragraphs, "doc_title"
        )

        assert original is None
        assert translated is None


class TestGetFirstPageText:
    """Tests for _get_first_page_text static method."""

    def test_first_page_only(self) -> None:
        """Test that only first page text is returned."""
        paragraphs = [
            Paragraph(
                id="p0_0",
                page_number=0,
                block_bbox=BBox(x0=0, y0=700, x1=100, y1=750),
                text="First page text",
                line_count=1,
                category="text",
            ),
            Paragraph(
                id="p1_0",
                page_number=1,
                block_bbox=BBox(x0=0, y0=700, x1=100, y1=750),
                text="Second page text",
                line_count=1,
                category="text",
            ),
        ]

        result = SummaryExtractor._get_first_page_text(paragraphs)

        assert "First page text" in result
        assert "Second page text" not in result

    def test_sorted_by_y_coordinate(self) -> None:
        """Test that paragraphs are sorted by y coordinate (top to bottom)."""
        paragraphs = [
            Paragraph(
                id="p0_0",
                page_number=0,
                block_bbox=BBox(x0=0, y0=500, x1=100, y1=550),  # Lower on page
                text="Lower text",
                line_count=1,
                category="text",
            ),
            Paragraph(
                id="p0_1",
                page_number=0,
                block_bbox=BBox(x0=0, y0=700, x1=100, y1=750),  # Higher on page
                text="Upper text",
                line_count=1,
                category="text",
            ),
        ]

        result = SummaryExtractor._get_first_page_text(paragraphs)

        # In PDF coordinates, higher y = higher on page
        # So "Upper text" should come first
        upper_idx = result.index("Upper text")
        lower_idx = result.index("Lower text")
        assert upper_idx < lower_idx
