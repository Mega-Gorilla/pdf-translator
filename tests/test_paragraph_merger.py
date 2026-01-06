# SPDX-License-Identifier: Apache-2.0
"""Tests for paragraph_merger module."""

import pytest

from pdf_translator.core.models import BBox, Paragraph
from pdf_translator.core.paragraph_merger import (
    MergeConfig,
    _calc_x_overlap,
    _can_merge,
    _detect_columns,
    _merge_column,
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

    def test_cannot_merge_different_widths(self) -> None:
        """Paragraphs with significantly different widths should not merge.

        This prevents short headings or list items from being absorbed
        into adjacent body text paragraphs.
        """
        # para1: full width (width=400)
        para1 = make_paragraph(x0=100, x1=500, y0=50, y1=100)
        # para2: narrow (width=150) - ratio = 150/400 = 0.375 < 0.8
        para2 = make_paragraph(x0=100, x1=250, y0=0, y1=48)
        config = MergeConfig(width_tolerance=0.8)

        assert _can_merge(para1, para2, config) is False

    def test_can_merge_similar_widths(self) -> None:
        """Paragraphs with similar widths should merge normally."""
        # para1: width=400
        para1 = make_paragraph(x0=100, x1=500, y0=50, y1=100)
        # para2: width=380 - ratio = 380/400 = 0.95 >= 0.8
        para2 = make_paragraph(x0=100, x1=480, y0=0, y1=48)
        config = MergeConfig(width_tolerance=0.8)

        assert _can_merge(para1, para2, config) is True

    def test_width_tolerance_configurable(self) -> None:
        """Width tolerance should be configurable."""
        # para1: width=400, para2: width=280
        # ratio = 280/400 = 0.70
        para1 = make_paragraph(x0=100, x1=500, y0=50, y1=100)
        para2 = make_paragraph(x0=100, x1=380, y0=0, y1=48)

        # With 80% tolerance: 0.70 < 0.80 → no merge
        config_strict = MergeConfig(width_tolerance=0.8)
        assert _can_merge(para1, para2, config_strict) is False

        # With 60% tolerance: 0.70 >= 0.60 → merge
        config_relaxed = MergeConfig(width_tolerance=0.6)
        assert _can_merge(para1, para2, config_relaxed) is True

    def test_can_merge_different_alignments(self) -> None:
        """Different alignments should NOT prevent merging.

        Alignment check was removed because _estimate_alignment() is unreliable
        for short paragraphs and causes false negatives (e.g., left vs justify).
        """
        para1 = make_paragraph(alignment="left", y0=50, y1=100)
        para2 = make_paragraph(alignment="justify", y0=0, y1=48)
        config = MergeConfig()

        # Should merge even with different alignments
        assert _can_merge(para1, para2, config) is True

    def test_can_merge_sentence_ending(self) -> None:
        """Sentence-ending punctuation should NOT prevent merging.

        Sentence-ending check was removed because the purpose of merging is
        layout optimization (larger bbox = better font size and line-breaking),
        not semantic analysis.
        """
        para1 = make_paragraph(text="Complete sentence.", y0=50, y1=100)
        para2 = make_paragraph(text="New sentence", y0=0, y1=48)
        config = MergeConfig()

        # Should merge even with sentence-ending punctuation
        assert _can_merge(para1, para2, config) is True

    def test_can_merge_japanese_sentence(self) -> None:
        """Japanese sentence ending should also merge."""
        para1 = make_paragraph(text="日本語の文。", y0=50, y1=100)
        para2 = make_paragraph(text="次の文", y0=0, y1=48)
        config = MergeConfig()

        # Should merge even with Japanese sentence-ending punctuation
        assert _can_merge(para1, para2, config) is True

    def test_cannot_merge_exceeds_max_length(self) -> None:
        """Paragraphs should not merge if result exceeds max_merged_length."""
        # Each paragraph has 50 chars, merged would be 101 chars (50 + 50 + 1 space)
        long_text = "A" * 50
        para1 = make_paragraph(text=long_text, y0=50, y1=100)
        para2 = make_paragraph(text=long_text, y0=0, y1=48)
        config = MergeConfig(max_merged_length=100)

        assert _can_merge(para1, para2, config) is False

    def test_can_merge_within_max_length(self) -> None:
        """Paragraphs can merge if result is within max_merged_length."""
        # Each paragraph has 50 chars, merged would be 101 chars
        long_text = "A" * 50
        para1 = make_paragraph(text=long_text, y0=50, y1=100)
        para2 = make_paragraph(text=long_text, y0=0, y1=48)
        config = MergeConfig(max_merged_length=101)  # Exactly fits

        assert _can_merge(para1, para2, config) is True

    def test_max_length_zero_disables_check(self) -> None:
        """max_merged_length=0 should disable the length check."""
        # Very long texts that would normally exceed any limit
        very_long_text = "A" * 10000
        para1 = make_paragraph(text=very_long_text, y0=50, y1=100)
        para2 = make_paragraph(text=very_long_text, y0=0, y1=48)
        config = MergeConfig(max_merged_length=0)  # Disabled

        assert _can_merge(para1, para2, config) is True

    def test_max_length_default_is_4000(self) -> None:
        """Default max_merged_length should be 4000."""
        config = MergeConfig()
        assert config.max_merged_length == 4000


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
        """Chain broken by category change."""
        para1 = make_paragraph(
            id="para_p0_b0",
            text="First paragraph",
            category="text",
            y0=100, y1=150,
        )
        para2 = make_paragraph(
            id="para_p0_b1",
            text="Title in between",
            category="paragraph_title",  # Different category breaks chain
            y0=50, y1=98,
        )
        para3 = make_paragraph(
            id="para_p0_b2",
            text="continues here",
            category="text",
            y0=0, y1=48,
        )

        result = merge_adjacent_paragraphs([para1, para2, para3])

        # Three separate paragraphs (title in the middle breaks both chains)
        assert len(result) == 3
        assert result[0].text == "First paragraph"
        assert result[1].text == "Title in between"
        assert result[2].text == "continues here"

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

    def test_merge_chain_stops_at_length_limit(self) -> None:
        """Chain merging should stop when max_merged_length is reached."""
        # Three paragraphs with 40 chars each
        # Merged: para1 + para2 = 81 chars (40 + 40 + 1)
        # Merged: para1+2 + para3 = 122 chars (81 + 40 + 1) > 100
        text_40 = "A" * 40
        para1 = make_paragraph(
            id="para_p0_b0",
            text=text_40,
            y0=100, y1=150,
        )
        para2 = make_paragraph(
            id="para_p0_b1",
            text=text_40,
            y0=50, y1=98,
        )
        para3 = make_paragraph(
            id="para_p0_b2",
            text=text_40,
            y0=0, y1=48,
        )

        config = MergeConfig(max_merged_length=100)
        result = merge_adjacent_paragraphs([para1, para2, para3], config)

        # para1 and para2 merge (81 chars), but para3 cannot join (would be 122)
        assert len(result) == 2
        assert len(result[0].text) == 81  # para1 + " " + para2
        assert len(result[1].text) == 40  # para3 alone


# =============================================================================
# Tests for _detect_columns()
# =============================================================================


class TestDetectColumns:
    """Tests for _detect_columns() function."""

    def test_single_column(self) -> None:
        """All paragraphs in a single column."""
        para1 = make_paragraph(id="p1", x0=50, x1=300, y0=100, y1=150)
        para2 = make_paragraph(id="p2", x0=50, x1=300, y0=50, y1=98)
        para3 = make_paragraph(id="p3", x0=50, x1=300, y0=0, y1=48)

        columns = _detect_columns([para1, para2, para3])

        assert len(columns) == 1
        assert len(columns[0]) == 3

    def test_two_columns(self) -> None:
        """Paragraphs in two separate columns."""
        # Left column (x: 50-300)
        left1 = make_paragraph(id="l1", x0=50, x1=300, y0=100, y1=150)
        left2 = make_paragraph(id="l2", x0=50, x1=300, y0=50, y1=98)

        # Right column (x: 350-600)
        right1 = make_paragraph(id="r1", x0=350, x1=600, y0=100, y1=150)
        right2 = make_paragraph(id="r2", x0=350, x1=600, y0=50, y1=98)

        columns = _detect_columns([left1, left2, right1, right2])

        assert len(columns) == 2
        assert len(columns[0]) == 2  # Left column
        assert len(columns[1]) == 2  # Right column

        # Check column contents
        left_ids = {p.id for p in columns[0]}
        right_ids = {p.id for p in columns[1]}
        assert left_ids == {"l1", "l2"}
        assert right_ids == {"r1", "r2"}

    def test_three_columns(self) -> None:
        """Paragraphs in three columns."""
        left = make_paragraph(id="left", x0=50, x1=200)
        middle = make_paragraph(id="middle", x0=250, x1=400)
        right = make_paragraph(id="right", x0=450, x1=600)

        columns = _detect_columns([left, middle, right])

        assert len(columns) == 3

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        columns = _detect_columns([])

        assert columns == []

    def test_overlapping_x_ranges(self) -> None:
        """Paragraphs with overlapping X ranges go to same column."""
        # Wide paragraph
        wide = make_paragraph(id="wide", x0=50, x1=300)
        # Narrower paragraph within the same X range
        narrow = make_paragraph(id="narrow", x0=100, x1=250)

        columns = _detect_columns([wide, narrow])

        assert len(columns) == 1
        assert len(columns[0]) == 2

    def test_x_overlap_threshold(self) -> None:
        """Paragraphs with significant X overlap go to same column."""
        # 80% overlap (para2 mostly within para1's X range)
        para1 = make_paragraph(id="p1", x0=50, x1=150)  # width=100
        para2 = make_paragraph(id="p2", x0=60, x1=140)  # width=80, overlap=80

        # High overlap -> same column
        columns = _detect_columns([para1, para2])
        assert len(columns) == 1
        assert len(columns[0]) == 2

    def test_no_x_overlap_separate_columns(self) -> None:
        """Paragraphs with no X overlap go to different columns."""
        para1 = make_paragraph(id="p1", x0=50, x1=100)
        para2 = make_paragraph(id="p2", x0=120, x1=170)  # No overlap

        columns = _detect_columns([para1, para2])
        assert len(columns) == 2


# =============================================================================
# Tests for _merge_column()
# =============================================================================


class TestMergeColumn:
    """Tests for _merge_column() function."""

    def test_merge_two_in_column(self) -> None:
        """Two adjacent paragraphs in same column should merge."""
        para1 = make_paragraph(id="p1", text="First", y0=50, y1=100)
        para2 = make_paragraph(id="p2", text="second", y0=0, y1=48)
        config = MergeConfig()

        result = _merge_column([para1, para2], config)

        assert len(result) == 1
        assert result[0].text == "First second"

    def test_merge_three_in_column(self) -> None:
        """Three adjacent paragraphs in same column should merge."""
        para1 = make_paragraph(id="p1", text="First", y0=100, y1=150)
        para2 = make_paragraph(id="p2", text="second", y0=50, y1=98)
        para3 = make_paragraph(id="p3", text="third", y0=0, y1=48)
        config = MergeConfig()

        result = _merge_column([para1, para2, para3], config)

        assert len(result) == 1
        assert result[0].text == "First second third"

    def test_no_merge_different_categories(self) -> None:
        """Different categories should not merge within column."""
        para1 = make_paragraph(id="p1", text="Text", category="text", y0=50, y1=100)
        para2 = make_paragraph(id="p2", text="Title", category="paragraph_title", y0=0, y1=48)
        config = MergeConfig()

        result = _merge_column([para1, para2], config)

        assert len(result) == 2

    def test_empty_column(self) -> None:
        """Empty column returns empty list."""
        config = MergeConfig()

        result = _merge_column([], config)

        assert result == []


# =============================================================================
# Tests for multi-column merge scenarios
# =============================================================================


class TestMultiColumnMerge:
    """Tests for merge_adjacent_paragraphs with multi-column layouts."""

    def test_two_column_independent_merge(self) -> None:
        """Each column should merge independently.

        This simulates a typical academic paper layout where left and right
        columns have vertically adjacent paragraphs that should merge within
        their respective columns but not across columns.
        """
        # Left column paragraphs (x: 50-300)
        left1 = make_paragraph(
            id="l1", text="Left top", x0=50, x1=300, y0=100, y1=150
        )
        left2 = make_paragraph(
            id="l2", text="left bottom", x0=50, x1=300, y0=50, y1=98
        )

        # Right column paragraphs (x: 350-600)
        right1 = make_paragraph(
            id="r1", text="Right top", x0=350, x1=600, y0=100, y1=150
        )
        right2 = make_paragraph(
            id="r2", text="right bottom", x0=350, x1=600, y0=50, y1=98
        )

        config = MergeConfig()
        result = merge_adjacent_paragraphs([left1, left2, right1, right2], config)

        # Should have 2 merged paragraphs (one per column)
        assert len(result) == 2

        # Check merged texts
        texts = {p.text for p in result}
        assert "Left top left bottom" in texts
        assert "Right top right bottom" in texts

    def test_two_column_no_cross_merge(self) -> None:
        """Paragraphs in different columns should NOT merge.

        Even if a paragraph from the right column appears "adjacent" by Y
        coordinate to a paragraph in the left column, they should not merge.
        """
        # Left column paragraph
        left = make_paragraph(
            id="left", text="Left text", x0=50, x1=300, y0=100, y1=150
        )

        # Right column paragraph - appears "adjacent" by Y but different column
        right = make_paragraph(
            id="right", text="Right text", x0=350, x1=600, y0=50, y1=98
        )

        config = MergeConfig()
        result = merge_adjacent_paragraphs([left, right], config)

        # Should remain as 2 separate paragraphs
        assert len(result) == 2
        assert result[0].text == "Left text"
        assert result[1].text == "Right text"

    def test_full_width_header_with_two_columns(self) -> None:
        """Full-width header should not merge with column content.

        A common layout pattern where a title spans the full page width
        above a two-column body.
        """
        # Full-width header (x: 50-600)
        header = make_paragraph(
            id="header",
            text="Paper Title",
            category="paragraph_title",
            x0=50, x1=600,
            y0=200, y1=250,
        )

        # Left column (x: 50-300)
        left = make_paragraph(
            id="left", text="Left column", x0=50, x1=300, y0=100, y1=150
        )

        # Right column (x: 350-600)
        right = make_paragraph(
            id="right", text="Right column", x0=350, x1=600, y0=100, y1=150
        )

        config = MergeConfig()
        result = merge_adjacent_paragraphs([header, left, right], config)

        # Should have 3 paragraphs (header separate, columns separate)
        assert len(result) == 3

    def test_realistic_paper_layout(self) -> None:
        """Realistic academic paper layout with interleaved Y positions.

        This tests the exact scenario that was failing before column detection:
        paragraphs from different columns have Y positions that interleave when
        sorted globally.
        """
        # Left column paragraphs (x: 70-290)
        left1 = make_paragraph(
            id="l1", text="Left para 1", x0=70, x1=290, y0=242, y1=431
        )
        left2 = make_paragraph(
            id="l2", text="left para 2", x0=70, x1=290, y0=106, y1=241
        )

        # Right column paragraphs (x: 320-540) - gap of 30 points
        # Their Y values interleave with left column
        right1 = make_paragraph(
            id="r1", text="Right para 1", x0=320, x1=540, y0=255, y1=336
        )
        right2 = make_paragraph(
            id="r2", text="right para 2", x0=320, x1=540, y0=200, y1=254
        )

        # When sorted by y1 descending: l1(431), r1(336), r2(254), l2(241)
        # Without column detection, l2 would try to merge with r2 (and fail on gap)
        # or worse, the merge chain would be broken

        # Left column x: 70-290, Right column x: 320-540
        # No X-axis overlap, so columns are detected as separate
        config = MergeConfig()
        result = merge_adjacent_paragraphs([left1, left2, right1, right2], config)

        # Should have 2 paragraphs (one merged per column)
        assert len(result) == 2

        texts = {p.text for p in result}
        assert "Left para 1 left para 2" in texts
        assert "Right para 1 right para 2" in texts

    def test_sidebar_layout(self) -> None:
        """Sidebar with main content should be treated as separate columns."""
        # Main content (wider, left side)
        main1 = make_paragraph(
            id="m1", text="Main top", x0=50, x1=400, y0=100, y1=150
        )
        main2 = make_paragraph(
            id="m2", text="main bottom", x0=50, x1=400, y0=50, y1=98
        )

        # Sidebar (narrower, right side)
        sidebar = make_paragraph(
            id="s1", text="Sidebar note", x0=450, x1=550, y0=100, y1=150
        )

        config = MergeConfig()
        result = merge_adjacent_paragraphs([main1, main2, sidebar], config)

        # Main content should merge, sidebar stays separate
        assert len(result) == 2

        merged_main = next(p for p in result if "Main" in p.text)
        assert merged_main.text == "Main top main bottom"
