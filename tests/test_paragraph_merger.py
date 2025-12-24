# SPDX-License-Identifier: Apache-2.0
"""Tests for paragraph_merger module."""

import pytest

from pdf_translator.core.models import BBox, Paragraph
from pdf_translator.core.paragraph_merger import (
    MergeConfig,
    SENTENCE_ENDING_PUNCTUATION,
    _calc_x_overlap,
    _can_merge,
    _ends_with_sentence_punctuation,
    _merge_two_paragraphs,
    merge_adjacent_paragraphs,
)


# =============================================================================
# Fixtures
# =============================================================================


def make_paragraph(
    id: str = "para_p0_b0",
    page_number: int = 0,
    text: str = "Hello world",
    x0: float = 0,
    y0: float = 0,
    x1: float = 100,
    y1: float = 50,
    line_count: int = 1,
    original_font_size: float = 12.0,
    category: str | None = "text",
    category_confidence: float | None = 0.9,
    alignment: str = "left",
) -> Paragraph:
    """Helper to create Paragraph instances for testing."""
    return Paragraph(
        id=id,
        page_number=page_number,
        text=text,
        block_bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        line_count=line_count,
        original_font_size=original_font_size,
        category=category,
        category_confidence=category_confidence,
        alignment=alignment,
    )


# =============================================================================
# Tests for BBox.union()
# =============================================================================


class TestBBoxUnion:
    """Tests for BBox.union() method."""

    def test_bbox_union_overlapping(self) -> None:
        """Union of overlapping boxes."""
        bbox1 = BBox(0, 0, 100, 50)
        bbox2 = BBox(50, 25, 150, 75)
        result = bbox1.union(bbox2)

        assert result.x0 == 0
        assert result.y0 == 0
        assert result.x1 == 150
        assert result.y1 == 75

    def test_bbox_union_non_overlapping(self) -> None:
        """Union of non-overlapping boxes."""
        bbox1 = BBox(0, 0, 50, 25)
        bbox2 = BBox(100, 50, 150, 75)
        result = bbox1.union(bbox2)

        assert result.x0 == 0
        assert result.y0 == 0
        assert result.x1 == 150
        assert result.y1 == 75

    def test_bbox_union_contained(self) -> None:
        """Union where one box contains the other."""
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(25, 25, 75, 75)
        result = bbox1.union(bbox2)

        assert result.x0 == 0
        assert result.y0 == 0
        assert result.x1 == 100
        assert result.y1 == 100


# =============================================================================
# Tests for _calc_x_overlap()
# =============================================================================


class TestCalcXOverlap:
    """Tests for _calc_x_overlap() function."""

    def test_full_overlap(self) -> None:
        """Two boxes with same X range."""
        bbox1 = BBox(0, 0, 100, 50)
        bbox2 = BBox(0, 50, 100, 100)

        assert _calc_x_overlap(bbox1, bbox2) == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        """Partial X overlap."""
        bbox1 = BBox(0, 0, 100, 50)
        bbox2 = BBox(50, 50, 150, 100)

        # Overlap width = 50, min width = 100
        assert _calc_x_overlap(bbox1, bbox2) == pytest.approx(0.5)

    def test_no_overlap(self) -> None:
        """No X overlap (different columns)."""
        bbox1 = BBox(0, 0, 50, 50)
        bbox2 = BBox(100, 50, 150, 100)

        assert _calc_x_overlap(bbox1, bbox2) == 0.0

    def test_narrow_contained_in_wide(self) -> None:
        """Narrow box fully contained in wide box."""
        bbox1 = BBox(0, 0, 100, 50)  # Wide
        bbox2 = BBox(25, 50, 75, 100)  # Narrow (width=50)

        # Overlap = 50 (full width of narrow), min width = 50
        assert _calc_x_overlap(bbox1, bbox2) == pytest.approx(1.0)

    def test_zero_width_box(self) -> None:
        """Box with zero width."""
        bbox1 = BBox(50, 0, 50, 50)  # Zero width
        bbox2 = BBox(0, 50, 100, 100)

        assert _calc_x_overlap(bbox1, bbox2) == 0.0


# =============================================================================
# Tests for _ends_with_sentence_punctuation()
# =============================================================================


class TestEndsWithSentencePunctuation:
    """Tests for _ends_with_sentence_punctuation() function."""

    def test_english_period(self) -> None:
        assert _ends_with_sentence_punctuation("This is a sentence.") is True

    def test_japanese_period(self) -> None:
        assert _ends_with_sentence_punctuation("これは文です。") is True

    def test_exclamation(self) -> None:
        assert _ends_with_sentence_punctuation("Hello!") is True
        assert _ends_with_sentence_punctuation("こんにちは！") is True

    def test_question(self) -> None:
        assert _ends_with_sentence_punctuation("What?") is True
        assert _ends_with_sentence_punctuation("なに？") is True

    def test_comma_ending(self) -> None:
        assert _ends_with_sentence_punctuation("First item,") is False

    def test_no_punctuation(self) -> None:
        assert _ends_with_sentence_punctuation("No punctuation") is False

    def test_whitespace_after_punctuation(self) -> None:
        """Trailing whitespace should be ignored."""
        assert _ends_with_sentence_punctuation("End.   ") is True

    def test_empty_string(self) -> None:
        assert _ends_with_sentence_punctuation("") is False

    def test_whitespace_only(self) -> None:
        assert _ends_with_sentence_punctuation("   ") is False

    def test_all_punctuation_types(self) -> None:
        """Verify all punctuation in SENTENCE_ENDING_PUNCTUATION."""
        for punct in SENTENCE_ENDING_PUNCTUATION:
            assert _ends_with_sentence_punctuation(f"Text{punct}") is True


# =============================================================================
# Tests for _can_merge()
# =============================================================================


class TestCanMerge:
    """Tests for _can_merge() function."""

    def test_can_merge_same_category_adjacent(self) -> None:
        """Two adjacent text paragraphs should merge."""
        # para1 is above para2 in PDF coords (y1 > y0)
        para1 = make_paragraph(
            text="First part",
            y0=50, y1=100,  # Upper paragraph
        )
        para2 = make_paragraph(
            text="second part",
            y0=0, y1=48,  # Lower paragraph (gap = 2)
        )
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is True

    def test_cannot_merge_different_pages(self) -> None:
        """Paragraphs on different pages should not merge."""
        para1 = make_paragraph(page_number=0, y0=50, y1=100)
        para2 = make_paragraph(page_number=1, y0=0, y1=48)
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is False

    def test_cannot_merge_different_categories(self) -> None:
        """Paragraphs with different categories should not merge."""
        para1 = make_paragraph(category="text", y0=50, y1=100)
        para2 = make_paragraph(category="paragraph_title", y0=0, y1=48)
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is False

    def test_cannot_merge_none_category(self) -> None:
        """Paragraphs with None category should not merge."""
        para1 = make_paragraph(category=None, y0=50, y1=100)
        para2 = make_paragraph(category=None, y0=0, y1=48)
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is False

    def test_cannot_merge_non_translatable_category(self) -> None:
        """Paragraphs with non-translatable categories should not merge."""
        para1 = make_paragraph(category="table", y0=50, y1=100)
        para2 = make_paragraph(category="table", y0=0, y1=48)
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is False

    def test_cannot_merge_large_gap(self) -> None:
        """Paragraphs with gap > tolerance should not merge."""
        para1 = make_paragraph(
            y0=100, y1=150,
            original_font_size=10.0,
        )
        para2 = make_paragraph(
            y0=0, y1=50,  # Gap = 50 > 10 * 1.5 = 15
        )
        config = MergeConfig(gap_tolerance=1.5)

        assert _can_merge(para1, para2, config) is False

    def test_cannot_merge_overlapping_vertically(self) -> None:
        """Overlapping paragraphs should not merge."""
        para1 = make_paragraph(y0=50, y1=100)
        para2 = make_paragraph(y0=60, y1=110)  # Overlaps with para1
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is False

    def test_cannot_merge_insufficient_x_overlap(self) -> None:
        """Paragraphs without enough X overlap should not merge."""
        para1 = make_paragraph(x0=0, x1=100, y0=50, y1=100)
        para2 = make_paragraph(x0=80, x1=180, y0=0, y1=48)  # Only 20% overlap
        config = MergeConfig(x_overlap_threshold=0.7)

        assert _can_merge(para1, para2, config) is False

    def test_cannot_merge_different_font_sizes(self) -> None:
        """Paragraphs with font size difference > tolerance should not merge."""
        para1 = make_paragraph(original_font_size=12.0, y0=50, y1=100)
        para2 = make_paragraph(original_font_size=16.0, y0=0, y1=48)
        config = MergeConfig(font_size_tolerance=1.0)

        assert _can_merge(para1, para2, config) is False

    def test_cannot_merge_different_alignments(self) -> None:
        """Paragraphs with different alignments should not merge."""
        para1 = make_paragraph(alignment="left", y0=50, y1=100)
        para2 = make_paragraph(alignment="center", y0=0, y1=48)
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is False

    def test_cannot_merge_sentence_ending(self) -> None:
        """First paragraph ending with sentence punctuation should not merge."""
        para1 = make_paragraph(text="Complete sentence.", y0=50, y1=100)
        para2 = make_paragraph(text="New sentence", y0=0, y1=48)
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is False

    def test_can_merge_comma_ending(self) -> None:
        """First paragraph ending with comma should merge."""
        para1 = make_paragraph(text="First part,", y0=50, y1=100)
        para2 = make_paragraph(text="second part", y0=0, y1=48)
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is True

    def test_can_merge_japanese_sentence(self) -> None:
        """Japanese sentence ending should not merge."""
        para1 = make_paragraph(text="日本語の文。", y0=50, y1=100)
        para2 = make_paragraph(text="次の文", y0=0, y1=48)
        config = MergeConfig()

        assert _can_merge(para1, para2, config) is False


# =============================================================================
# Tests for _merge_two_paragraphs()
# =============================================================================


class TestMergeTwoParagraphs:
    """Tests for _merge_two_paragraphs() function."""

    def test_merged_text_concatenation(self) -> None:
        """Merged text should be joined with space."""
        para1 = make_paragraph(text="First part")
        para2 = make_paragraph(text="second part")

        merged = _merge_two_paragraphs(para1, para2)

        assert merged.text == "First part second part"

    def test_merged_bbox_is_union(self) -> None:
        """Merged bbox should be union of original bboxes."""
        para1 = make_paragraph(x0=0, y0=50, x1=100, y1=100)
        para2 = make_paragraph(x0=10, y0=0, x1=110, y1=48)

        merged = _merge_two_paragraphs(para1, para2)

        assert merged.block_bbox.x0 == 0
        assert merged.block_bbox.y0 == 0
        assert merged.block_bbox.x1 == 110
        assert merged.block_bbox.y1 == 100

    def test_merged_line_count_sum(self) -> None:
        """Merged line_count should be sum of originals."""
        para1 = make_paragraph(line_count=2)
        para2 = make_paragraph(line_count=3)

        merged = _merge_two_paragraphs(para1, para2)

        assert merged.line_count == 5

    def test_merged_id_from_first(self) -> None:
        """Merged paragraph should use first paragraph's ID."""
        para1 = make_paragraph(id="para_p0_b0")
        para2 = make_paragraph(id="para_p0_b1")

        merged = _merge_two_paragraphs(para1, para2)

        assert merged.id == "para_p0_b0"

    def test_merged_category_confidence_minimum(self) -> None:
        """Merged category_confidence should be minimum of both."""
        para1 = make_paragraph(category_confidence=0.9)
        para2 = make_paragraph(category_confidence=0.7)

        merged = _merge_two_paragraphs(para1, para2)

        assert merged.category_confidence == 0.7

    def test_merged_category_confidence_one_none(self) -> None:
        """If one confidence is None, use the other."""
        para1 = make_paragraph(category_confidence=0.9)
        para2 = make_paragraph(category_confidence=None)

        merged = _merge_two_paragraphs(para1, para2)

        assert merged.category_confidence == 0.9

    def test_merged_category_confidence_both_none(self) -> None:
        """If both confidences are None, result is None."""
        para1 = make_paragraph(category_confidence=None)
        para2 = make_paragraph(category_confidence=None)

        merged = _merge_two_paragraphs(para1, para2)

        assert merged.category_confidence is None

    def test_merged_translated_text_reset(self) -> None:
        """Merged paragraph should have translated_text reset to None."""
        para1 = make_paragraph()
        para1.translated_text = "Translated"
        para2 = make_paragraph()

        merged = _merge_two_paragraphs(para1, para2)

        assert merged.translated_text is None

    def test_merged_preserves_font_size(self) -> None:
        """Merged paragraph should use first paragraph's font size."""
        para1 = make_paragraph(original_font_size=12.0)
        para2 = make_paragraph(original_font_size=11.5)

        merged = _merge_two_paragraphs(para1, para2)

        assert merged.original_font_size == 12.0


# =============================================================================
# Tests for merge_adjacent_paragraphs()
# =============================================================================


class TestMergeAdjacentParagraphs:
    """Tests for merge_adjacent_paragraphs() function."""

    def test_merge_empty_list(self) -> None:
        """Empty input should return empty output."""
        result = merge_adjacent_paragraphs([])

        assert result == []

    def test_merge_single_paragraph(self) -> None:
        """Single paragraph should return unchanged."""
        para = make_paragraph()
        result = merge_adjacent_paragraphs([para])

        assert len(result) == 1
        assert result[0].text == para.text

    def test_merge_two_adjacent(self) -> None:
        """Two adjacent paragraphs should merge into one."""
        para1 = make_paragraph(
            id="para_p0_b0",
            text="First part",
            y0=50, y1=100,
        )
        para2 = make_paragraph(
            id="para_p0_b1",
            text="second part",
            y0=0, y1=48,
        )

        result = merge_adjacent_paragraphs([para1, para2])

        assert len(result) == 1
        assert result[0].text == "First part second part"

    def test_merge_chain_of_three(self) -> None:
        """Three adjacent paragraphs should merge into one."""
        para1 = make_paragraph(
            id="para_p0_b0",
            text="First",
            y0=100, y1=150,
        )
        para2 = make_paragraph(
            id="para_p0_b1",
            text="second",
            y0=50, y1=98,
        )
        para3 = make_paragraph(
            id="para_p0_b2",
            text="third",
            y0=0, y1=48,
        )

        result = merge_adjacent_paragraphs([para1, para2, para3])

        assert len(result) == 1
        assert result[0].text == "First second third"

    def test_merge_partial_chain(self) -> None:
        """Chain broken by sentence ending."""
        para1 = make_paragraph(
            id="para_p0_b0",
            text="Complete sentence.",
            y0=100, y1=150,
        )
        para2 = make_paragraph(
            id="para_p0_b1",
            text="New paragraph",
            y0=50, y1=98,
        )
        para3 = make_paragraph(
            id="para_p0_b2",
            text="continues here",
            y0=0, y1=48,
        )

        result = merge_adjacent_paragraphs([para1, para2, para3])

        assert len(result) == 2
        assert result[0].text == "Complete sentence."
        assert result[1].text == "New paragraph continues here"

    def test_merge_preserves_non_translatable(self) -> None:
        """Non-translatable paragraphs should pass through unchanged."""
        para1 = make_paragraph(
            id="para_p0_b0",
            text="Formula 1",
            category="display_formula",
            y0=100, y1=150,
        )
        para2 = make_paragraph(
            id="para_p0_b1",
            text="Formula 2",
            category="display_formula",
            y0=50, y1=98,
        )

        result = merge_adjacent_paragraphs([para1, para2])

        assert len(result) == 2
        assert result[0].text == "Formula 1"
        assert result[1].text == "Formula 2"

    def test_merge_multiple_pages(self) -> None:
        """Paragraphs on different pages should not merge."""
        para1 = make_paragraph(
            id="para_p0_b0",
            page_number=0,
            text="Page 1 text",
            y0=50, y1=100,
        )
        para2 = make_paragraph(
            id="para_p1_b0",
            page_number=1,
            text="Page 2 text",
            y0=50, y1=100,
        )

        result = merge_adjacent_paragraphs([para1, para2])

        assert len(result) == 2

    def test_merge_with_config(self) -> None:
        """Custom config should be respected."""
        para1 = make_paragraph(
            text="First",
            y0=50, y1=100,
            original_font_size=12.0,
        )
        para2 = make_paragraph(
            text="second",
            y0=0, y1=30,  # Gap = 20 > 12 * 1.0 = 12
            original_font_size=12.0,
        )

        # With strict gap tolerance, should not merge
        config = MergeConfig(gap_tolerance=1.0)
        result = merge_adjacent_paragraphs([para1, para2], config)

        assert len(result) == 2

        # With relaxed gap tolerance, should merge
        config = MergeConfig(gap_tolerance=2.0)
        result = merge_adjacent_paragraphs([para1, para2], config)

        assert len(result) == 1

    def test_merge_preserves_page_order(self) -> None:
        """Result should preserve page order."""
        para_p1 = make_paragraph(id="para_p1_b0", page_number=1, y0=50, y1=100)
        para_p0 = make_paragraph(id="para_p0_b0", page_number=0, y0=50, y1=100)

        # Input in wrong order
        result = merge_adjacent_paragraphs([para_p1, para_p0])

        assert len(result) == 2
        assert result[0].page_number == 0
        assert result[1].page_number == 1

    def test_merge_unsorted_input(self) -> None:
        """Input paragraphs in random order should still merge correctly."""
        para1 = make_paragraph(
            id="para_p0_b0",
            text="Top paragraph",
            y0=100, y1=150,
        )
        para2 = make_paragraph(
            id="para_p0_b1",
            text="middle",
            y0=50, y1=98,
        )
        para3 = make_paragraph(
            id="para_p0_b2",
            text="bottom",
            y0=0, y1=48,
        )

        # Input in wrong order
        result = merge_adjacent_paragraphs([para3, para1, para2])

        assert len(result) == 1
        assert result[0].text == "Top paragraph middle bottom"
