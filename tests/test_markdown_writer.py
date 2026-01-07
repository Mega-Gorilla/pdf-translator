# SPDX-License-Identifier: Apache-2.0
"""Tests for Markdown writer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pdf_translator.core.models import BBox, Paragraph
from pdf_translator.output.markdown_writer import (
    DEFAULT_CATEGORY_MAPPING,
    DEFAULT_NONE_CATEGORY_MAPPING,
    MarkdownConfig,
    MarkdownOutputMode,
    MarkdownWriter,
)


class TestMarkdownWriter:
    """Test MarkdownWriter class."""

    def _create_test_paragraphs(self) -> list[Paragraph]:
        """Create test paragraphs."""
        return [
            Paragraph(
                id="p1",
                page_number=0,
                text="Document Title",
                block_bbox=BBox(x0=100, y0=700, x1=300, y1=730),
                line_count=1,
                category="doc_title",
            ),
            Paragraph(
                id="p2",
                page_number=0,
                text="This is the abstract.",
                block_bbox=BBox(x0=50, y0=600, x1=350, y1=650),
                line_count=2,
                category="abstract",
                translated_text="これは概要です。",
            ),
            Paragraph(
                id="p3",
                page_number=0,
                text="Body text here.",
                block_bbox=BBox(x0=50, y0=500, x1=350, y1=550),
                line_count=1,
                category="text",
                translated_text="本文はここです。",
            ),
        ]

    def test_write_single_paragraph(self) -> None:
        """Test writing single paragraph."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Hello world",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
            translated_text="こんにちは世界",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert "こんにちは世界" in md

    def test_write_multiple_paragraphs(self) -> None:
        """Test writing multiple paragraphs."""
        paras = self._create_test_paragraphs()
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write(paras)

        assert "# Document Title" in md
        assert "> これは概要です。" in md
        assert "本文はここです。" in md

    def test_write_with_page_breaks(self) -> None:
        """Test page break generation."""
        paras = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Page 1 content",
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
                category="text",
            ),
            Paragraph(
                id="p2",
                page_number=1,
                text="Page 2 content",
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
                category="text",
            ),
        ]
        writer = MarkdownWriter(
            MarkdownConfig(include_metadata=False, include_page_breaks=True)
        )
        md = writer.write(paras)

        assert "<!-- Page 2 -->" in md
        assert "---" in md

    def test_write_without_page_breaks(self) -> None:
        """Test output without page breaks."""
        paras = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Page 1",
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
            ),
            Paragraph(
                id="p2",
                page_number=1,
                text="Page 2",
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
            ),
        ]
        writer = MarkdownWriter(
            MarkdownConfig(include_metadata=False, include_page_breaks=False)
        )
        md = writer.write(paras)

        assert "<!-- Page" not in md

    def test_generate_metadata(self) -> None:
        """Test YAML frontmatter generation."""
        writer = MarkdownWriter(
            MarkdownConfig(
                include_metadata=True,
                source_filename="test.pdf",
                source_lang="en",
                target_lang="ja",
            )
        )
        md = writer.write([])

        assert "---" in md
        assert "title: test.pdf" in md
        assert "source_lang: en" in md
        assert "target_lang: ja" in md
        assert "translated_at:" in md

    def test_translated_only_mode(self) -> None:
        """Test TRANSLATED_ONLY mode."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Original text",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
            translated_text="翻訳されたテキスト",
        )
        writer = MarkdownWriter(
            MarkdownConfig(
                include_metadata=False,
                output_mode=MarkdownOutputMode.TRANSLATED_ONLY,
            )
        )
        md = writer.write([para])

        assert "翻訳されたテキスト" in md
        assert "Original text" not in md

    def test_translated_only_fallback_to_original(self) -> None:
        """Test TRANSLATED_ONLY falls back to original when no translation."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Original text",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
            # No translated_text
        )
        writer = MarkdownWriter(
            MarkdownConfig(
                include_metadata=False,
                output_mode=MarkdownOutputMode.TRANSLATED_ONLY,
            )
        )
        md = writer.write([para])

        assert "Original text" in md

    def test_original_only_mode(self) -> None:
        """Test ORIGINAL_ONLY mode."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Original text",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
            translated_text="翻訳されたテキスト",
        )
        writer = MarkdownWriter(
            MarkdownConfig(
                include_metadata=False,
                output_mode=MarkdownOutputMode.ORIGINAL_ONLY,
            )
        )
        md = writer.write([para])

        assert "Original text" in md
        assert "翻訳されたテキスト" not in md

    def test_parallel_mode(self) -> None:
        """Test PARALLEL mode shows both original and translation."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Original text",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
            translated_text="翻訳されたテキスト",
        )
        writer = MarkdownWriter(
            MarkdownConfig(
                include_metadata=False,
                output_mode=MarkdownOutputMode.PARALLEL,
            )
        )
        md = writer.write([para])

        assert "Original text" in md
        assert "翻訳されたテキスト" in md

    def test_paragraph_with_bold(self) -> None:
        """Test bold text formatting."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Bold text",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
            is_bold=True,
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert "**Bold text**" in md

    def test_paragraph_with_italic(self) -> None:
        """Test italic text formatting."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Italic text",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
            is_italic=True,
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert "*Italic text*" in md

    def test_paragraph_with_bold_and_italic(self) -> None:
        """Test bold and italic text formatting."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Bold italic",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
            is_bold=True,
            is_italic=True,
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert "***Bold italic***" in md

    def test_write_to_file(self) -> None:
        """Test writing to file."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Test content",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.md"
            writer.write_to_file([para], path)

            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "Test content" in content


class TestCategoryMapping:
    """Test category mapping functionality."""

    def test_default_mapping_doc_title(self) -> None:
        """Test doc_title -> h1 mapping."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Title",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="doc_title",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert md.strip() == "# Title"

    def test_default_mapping_paragraph_title(self) -> None:
        """Test paragraph_title -> h2 mapping."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Section",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="paragraph_title",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert md.strip() == "## Section"

    def test_default_mapping_text(self) -> None:
        """Test text -> p mapping."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Body text",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="text",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert md.strip() == "Body text"

    def test_default_mapping_abstract(self) -> None:
        """Test abstract -> blockquote mapping."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Abstract content",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="abstract",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert md.strip() == "> Abstract content"

    def test_default_mapping_skip_categories(self) -> None:
        """Test skip categories are not output."""
        paras = [
            Paragraph(
                id=f"p{i}",
                page_number=0,
                text=f"Skip {cat}",
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
                category=cat,
            )
            for i, cat in enumerate(["header", "footer", "number", "formula_number"])
        ]
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write(paras)

        assert md.strip() == ""

    def test_default_mapping_code(self) -> None:
        """Test inline_formula -> code mapping."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="E=mc^2",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="inline_formula",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert "`E=mc^2`" in md

    def test_default_mapping_code_block(self) -> None:
        """Test display_formula -> code_block mapping."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="E=mc^2",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="display_formula",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert "```" in md
        assert "E=mc^2" in md

    def test_category_override_heading_level(self) -> None:
        """Test category override for heading level."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Title",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="doc_title",
        )
        writer = MarkdownWriter(
            MarkdownConfig(
                include_metadata=False,
                category_mapping_overrides={"doc_title": "h2"},
            )
        )
        md = writer.write([para])

        assert md.strip() == "## Title"

    def test_category_override_element_type(self) -> None:
        """Test category override for element type."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Abstract",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="abstract",
        )
        writer = MarkdownWriter(
            MarkdownConfig(
                include_metadata=False,
                category_mapping_overrides={"abstract": "h3"},
            )
        )
        md = writer.write([para])

        assert md.strip() == "### Abstract"

    def test_heading_offset_applies_to_all(self) -> None:
        """Test heading_offset applies to all headings."""
        paras = [
            Paragraph(
                id="p1",
                page_number=0,
                text="Doc Title",
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
                category="doc_title",
            ),
            Paragraph(
                id="p2",
                page_number=0,
                text="Section",
                block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
                line_count=1,
                category="paragraph_title",
            ),
        ]
        writer = MarkdownWriter(
            MarkdownConfig(
                include_metadata=False,
                heading_offset=1,
            )
        )
        md = writer.write(paras)

        assert "## Doc Title" in md  # h1 + 1 = h2
        assert "### Section" in md  # h2 + 1 = h3

    def test_heading_offset_clamps_to_valid_range(self) -> None:
        """Test heading_offset clamps to 1-6 range."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Title",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="doc_title",  # h1
        )
        writer = MarkdownWriter(
            MarkdownConfig(
                include_metadata=False,
                heading_offset=10,  # h1 + 10 should clamp to h6
            )
        )
        md = writer.write([para])

        assert md.strip() == "###### Title"

    def test_none_category_uses_default(self) -> None:
        """Test None category uses default paragraph mapping."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="No category",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category=None,
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert md.strip() == "No category"

    def test_unknown_category_uses_default(self) -> None:
        """Test unknown category uses default paragraph mapping."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Unknown category",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="nonexistent_category",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert md.strip() == "Unknown category"

    def test_caption_formatting(self) -> None:
        """Test figure_title -> caption (italic) mapping."""
        para = Paragraph(
            id="p1",
            page_number=0,
            text="Figure 1: Description",
            block_bbox=BBox(x0=0, y0=0, x1=100, y1=20),
            line_count=1,
            category="figure_title",
        )
        writer = MarkdownWriter(MarkdownConfig(include_metadata=False))
        md = writer.write([para])

        assert "*Figure 1: Description*" in md


class TestCategoryMappingConstants:
    """Test category mapping constants."""

    def test_all_raw_categories_have_mapping(self) -> None:
        """Test all PP-DocLayoutV2 categories have mappings."""
        expected_categories = [
            "text",
            "vertical_text",
            "paragraph_title",
            "doc_title",
            "abstract",
            "aside_text",
            "inline_formula",
            "display_formula",
            "formula_number",
            "algorithm",
            "table",
            "image",
            "figure_title",
            "chart",
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "reference",
            "reference_content",
            "footnote",
            "vision_footnote",
            "seal",
            "content",
            "unknown",
        ]
        for cat in expected_categories:
            assert cat in DEFAULT_CATEGORY_MAPPING, f"Missing mapping for {cat}"

    def test_default_none_mapping(self) -> None:
        """Test default None category mapping is 'p'."""
        assert DEFAULT_NONE_CATEGORY_MAPPING == "p"
