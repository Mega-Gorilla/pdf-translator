# SPDX-License-Identifier: Apache-2.0
"""Tests for TextMerger class."""

import pytest

from pdf_translator.core.models import BBox, ProjectCategory, TextObject
from pdf_translator.core.text_merger import (
    ALL_TERMINALS,
    CJK_TERMINALS,
    SENTENCE_TERMINALS,
    TextMerger,
)


def make_text_object(
    id: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str = "test",
) -> TextObject:
    """Helper to create TextObject with minimal data."""
    return TextObject(
        id=id,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        text=text,
    )


class TestTextMergerInit:
    """Tests for TextMerger initialization."""

    def test_default_values(self) -> None:
        """Test default parameter values."""
        merger = TextMerger()
        assert merger._line_y_tolerance == 3.0
        assert merger._merge_threshold_x == 20.0
        assert merger._merge_threshold_y == 5.0
        assert merger._x_overlap_ratio == 0.5

    def test_custom_values(self) -> None:
        """Test custom parameter values."""
        merger = TextMerger(
            line_y_tolerance=5.0,
            merge_threshold_x=15.0,
            merge_threshold_y=8.0,
            x_overlap_ratio=0.3,
        )
        assert merger._line_y_tolerance == 5.0
        assert merger._merge_threshold_x == 15.0
        assert merger._merge_threshold_y == 8.0
        assert merger._x_overlap_ratio == 0.3


class TestReadingOrderSingleLine:
    """Tests for single-line reading order sorting."""

    def test_single_line_left_to_right(self) -> None:
        """Objects on same line should be sorted left-to-right."""
        merger = TextMerger()

        # Three objects on same y-level, in random x order
        obj_right = make_text_object("3", x0=200, y0=100, x1=250, y1=120, text="right")
        obj_left = make_text_object("1", x0=0, y0=100, x1=50, y1=120, text="left")
        obj_middle = make_text_object("2", x0=100, y0=100, x1=150, y1=120, text="middle")

        text_objects = [obj_right, obj_left, obj_middle]
        categories = {
            "1": ProjectCategory.TEXT,
            "2": ProjectCategory.TEXT,
            "3": ProjectCategory.TEXT,
        }

        result = merger.merge(text_objects, categories)

        assert len(result) == 3
        assert result[0].id == "1"  # left
        assert result[1].id == "2"  # middle
        assert result[2].id == "3"  # right

    def test_single_object(self) -> None:
        """Single object should be returned as-is."""
        merger = TextMerger()

        obj = make_text_object("1", x0=0, y0=100, x1=50, y1=120)
        categories = {"1": ProjectCategory.TEXT}

        result = merger.merge([obj], categories)

        assert len(result) == 1
        assert result[0].id == "1"


class TestReadingOrderMultipleLines:
    """Tests for multiple-line reading order sorting."""

    def test_multiple_lines_top_to_bottom(self) -> None:
        """Lines should be sorted top-to-bottom (higher y1 first in PDF)."""
        merger = TextMerger()

        # Two lines: top (y1=200) and bottom (y1=100)
        obj_bottom = make_text_object("2", x0=0, y0=80, x1=50, y1=100, text="bottom")
        obj_top = make_text_object("1", x0=0, y0=180, x1=50, y1=200, text="top")

        text_objects = [obj_bottom, obj_top]
        categories = {
            "1": ProjectCategory.TEXT,
            "2": ProjectCategory.TEXT,
        }

        result = merger.merge(text_objects, categories)

        assert len(result) == 2
        assert result[0].id == "1"  # top line first
        assert result[1].id == "2"  # bottom line second

    def test_complex_layout_two_lines(self) -> None:
        """Complex layout with 2 lines, 2 objects each."""
        merger = TextMerger()

        # Line 1 (top, y1=200): obj1 (left), obj2 (right)
        # Line 2 (bottom, y1=100): obj3 (left), obj4 (right)
        obj1 = make_text_object("1", x0=0, y0=180, x1=50, y1=200, text="line1-left")
        obj2 = make_text_object("2", x0=100, y0=180, x1=150, y1=200, text="line1-right")
        obj3 = make_text_object("3", x0=0, y0=80, x1=50, y1=100, text="line2-left")
        obj4 = make_text_object("4", x0=100, y0=80, x1=150, y1=100, text="line2-right")

        # Shuffle order
        text_objects = [obj4, obj1, obj3, obj2]
        categories = {
            "1": ProjectCategory.TEXT,
            "2": ProjectCategory.TEXT,
            "3": ProjectCategory.TEXT,
            "4": ProjectCategory.TEXT,
        }

        result = merger.merge(text_objects, categories)

        assert len(result) == 4
        # Line 1 (top)
        assert result[0].id == "1"
        assert result[1].id == "2"
        # Line 2 (bottom)
        assert result[2].id == "3"
        assert result[3].id == "4"


class TestReadingOrderWithYTolerance:
    """Tests for y-coordinate tolerance in line clustering."""

    def test_slight_y_variance_same_line(self) -> None:
        """Objects with slight y variance should be on same line."""
        merger = TextMerger(line_y_tolerance=3.0)

        # y1 values: 100, 101.5, 102 - within tolerance
        obj1 = make_text_object("1", x0=0, y0=80, x1=50, y1=100)
        obj2 = make_text_object("2", x0=100, y0=81.5, x1=150, y1=101.5)
        obj3 = make_text_object("3", x0=200, y0=82, x1=250, y1=102)

        text_objects = [obj3, obj1, obj2]
        categories = {
            "1": ProjectCategory.TEXT,
            "2": ProjectCategory.TEXT,
            "3": ProjectCategory.TEXT,
        }

        result = merger.merge(text_objects, categories)

        assert len(result) == 3
        # All on same line, sorted by x
        assert result[0].id == "1"
        assert result[1].id == "2"
        assert result[2].id == "3"

    def test_y_variance_exceeds_tolerance(self) -> None:
        """Objects with y variance exceeding tolerance should be on different lines."""
        merger = TextMerger(line_y_tolerance=3.0)

        # y1 values: 100, 110 - exceeds 3.0 tolerance
        obj1 = make_text_object("1", x0=0, y0=90, x1=50, y1=110)  # top
        obj2 = make_text_object("2", x0=0, y0=80, x1=50, y1=100)  # bottom

        text_objects = [obj2, obj1]
        categories = {
            "1": ProjectCategory.TEXT,
            "2": ProjectCategory.TEXT,
        }

        result = merger.merge(text_objects, categories)

        assert len(result) == 2
        assert result[0].id == "1"  # top line
        assert result[1].id == "2"  # bottom line


class TestCategoryFiltering:
    """Tests for category-based filtering."""

    def test_filters_translatable_categories(self) -> None:
        """Only TEXT, TITLE, CAPTION should be included."""
        merger = TextMerger()

        obj_text = make_text_object("1", x0=0, y0=0, x1=50, y1=20)
        obj_title = make_text_object("2", x0=0, y0=30, x1=50, y1=50)
        obj_caption = make_text_object("3", x0=0, y0=60, x1=50, y1=80)

        text_objects = [obj_text, obj_title, obj_caption]
        categories = {
            "1": ProjectCategory.TEXT,
            "2": ProjectCategory.TITLE,
            "3": ProjectCategory.CAPTION,
        }

        result = merger.merge(text_objects, categories)

        assert len(result) == 3

    def test_excludes_formula(self) -> None:
        """FORMULA category should be excluded."""
        merger = TextMerger()

        obj_text = make_text_object("1", x0=0, y0=0, x1=50, y1=20)
        obj_formula = make_text_object("2", x0=100, y0=0, x1=150, y1=20)

        text_objects = [obj_text, obj_formula]
        categories = {
            "1": ProjectCategory.TEXT,
            "2": ProjectCategory.FORMULA,
        }

        result = merger.merge(text_objects, categories)

        assert len(result) == 1
        assert result[0].id == "1"

    def test_excludes_table(self) -> None:
        """TABLE category should be excluded."""
        merger = TextMerger()

        obj_text = make_text_object("1", x0=0, y0=0, x1=50, y1=20)
        obj_table = make_text_object("2", x0=100, y0=0, x1=150, y1=20)

        text_objects = [obj_text, obj_table]
        categories = {
            "1": ProjectCategory.TEXT,
            "2": ProjectCategory.TABLE,
        }

        result = merger.merge(text_objects, categories)

        assert len(result) == 1
        assert result[0].id == "1"

    def test_excludes_multiple_non_translatable(self) -> None:
        """Multiple non-translatable categories should be excluded."""
        merger = TextMerger()

        obj_text = make_text_object("1", x0=0, y0=200, x1=50, y1=220)
        obj_formula = make_text_object("2", x0=0, y0=150, x1=50, y1=170)
        obj_table = make_text_object("3", x0=0, y0=100, x1=50, y1=120)
        obj_code = make_text_object("4", x0=0, y0=50, x1=50, y1=70)
        obj_header = make_text_object("5", x0=0, y0=0, x1=50, y1=20)

        text_objects = [obj_text, obj_formula, obj_table, obj_code, obj_header]
        categories = {
            "1": ProjectCategory.TEXT,
            "2": ProjectCategory.FORMULA,
            "3": ProjectCategory.TABLE,
            "4": ProjectCategory.CODE,
            "5": ProjectCategory.HEADER,
        }

        result = merger.merge(text_objects, categories)

        assert len(result) == 1
        assert result[0].id == "1"

    def test_empty_when_no_translatable(self) -> None:
        """Empty result when no translatable objects."""
        merger = TextMerger()

        obj = make_text_object("1", x0=0, y0=0, x1=50, y1=20)
        categories = {"1": ProjectCategory.FORMULA}

        result = merger.merge([obj], categories)

        assert len(result) == 0

    def test_missing_category_excluded(self) -> None:
        """Objects with missing category should be excluded."""
        merger = TextMerger()

        obj1 = make_text_object("1", x0=0, y0=0, x1=50, y1=20)
        obj2 = make_text_object("2", x0=100, y0=0, x1=150, y1=20)

        text_objects = [obj1, obj2]
        # Only obj1 has category
        categories = {"1": ProjectCategory.TEXT}

        result = merger.merge(text_objects, categories)

        assert len(result) == 1
        assert result[0].id == "1"


class TestSentenceEndDetection:
    """Tests for sentence end detection."""

    def test_sentence_end_detection_english(self) -> None:
        """English sentence-ending punctuation should be detected."""
        merger = TextMerger()

        assert merger.is_sentence_end("Hello world.")
        assert merger.is_sentence_end("What?")
        assert merger.is_sentence_end("Wow!")
        assert merger.is_sentence_end("Note:")
        assert merger.is_sentence_end("Item;")

    def test_sentence_end_detection_japanese(self) -> None:
        """Japanese sentence-ending punctuation should be detected."""
        merger = TextMerger()

        assert merger.is_sentence_end("これは文です。")
        assert merger.is_sentence_end("何ですか？")
        assert merger.is_sentence_end("すごい！")
        assert merger.is_sentence_end("注意：")
        assert merger.is_sentence_end("項目；")

    def test_not_sentence_end(self) -> None:
        """Non-sentence-ending text should return False."""
        merger = TextMerger()

        assert not merger.is_sentence_end("Hello world")
        assert not merger.is_sentence_end("This is")
        assert not merger.is_sentence_end("word,")
        assert not merger.is_sentence_end("")

    def test_whitespace_trimmed(self) -> None:
        """Trailing whitespace should be trimmed before checking."""
        merger = TextMerger()

        assert merger.is_sentence_end("Hello.  ")
        assert merger.is_sentence_end("Hello.\n")
        assert merger.is_sentence_end("Hello.\t")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_input(self) -> None:
        """Empty input should return empty result."""
        merger = TextMerger()
        result = merger.merge([], {})
        assert result == []

    def test_all_excluded(self) -> None:
        """All excluded objects should return empty result."""
        merger = TextMerger()

        objs = [make_text_object(str(i), 0, i * 20, 50, i * 20 + 20) for i in range(3)]
        categories = {str(i): ProjectCategory.FORMULA for i in range(3)}

        result = merger.merge(objs, categories)
        assert result == []


class TestConstants:
    """Tests for module constants."""

    def test_sentence_terminals(self) -> None:
        """Verify sentence terminal constants."""
        assert "." in SENTENCE_TERMINALS
        assert "!" in SENTENCE_TERMINALS
        assert "?" in SENTENCE_TERMINALS
        assert ":" in SENTENCE_TERMINALS
        assert ";" in SENTENCE_TERMINALS

    def test_cjk_terminals(self) -> None:
        """Verify CJK terminal constants."""
        assert "。" in CJK_TERMINALS
        assert "！" in CJK_TERMINALS
        assert "？" in CJK_TERMINALS
        assert "：" in CJK_TERMINALS
        assert "；" in CJK_TERMINALS

    def test_all_terminals_union(self) -> None:
        """ALL_TERMINALS should be union of both sets."""
        assert ALL_TERMINALS == SENTENCE_TERMINALS | CJK_TERMINALS


class TestV2Methods:
    """Tests for v2 merge methods (forward compatibility)."""

    def test_should_merge_same_line_basic(self) -> None:
        """Basic same-line merge check."""
        merger = TextMerger(merge_threshold_x=20.0)

        # Objects 10pt apart (within threshold)
        obj1 = make_text_object("1", x0=0, y0=0, x1=50, y1=20, text="hello")
        obj2 = make_text_object("2", x0=60, y0=0, x1=110, y1=20, text="world")

        assert merger._should_merge_same_line(obj1, obj2)

    def test_should_merge_same_line_sentence_end(self) -> None:
        """Don't merge if current ends with sentence punctuation."""
        merger = TextMerger(merge_threshold_x=20.0)

        obj1 = make_text_object("1", x0=0, y0=0, x1=50, y1=20, text="hello.")
        obj2 = make_text_object("2", x0=60, y0=0, x1=110, y1=20, text="world")

        assert not merger._should_merge_same_line(obj1, obj2)

    def test_should_merge_same_line_too_far(self) -> None:
        """Don't merge if gap exceeds threshold."""
        merger = TextMerger(merge_threshold_x=20.0)

        obj1 = make_text_object("1", x0=0, y0=0, x1=50, y1=20, text="hello")
        obj2 = make_text_object("2", x0=100, y0=0, x1=150, y1=20, text="world")

        assert not merger._should_merge_same_line(obj1, obj2)

    def test_should_merge_next_line_basic(self) -> None:
        """Basic next-line merge check."""
        merger = TextMerger(merge_threshold_y=5.0, x_overlap_ratio=0.5)

        # Vertically adjacent, same x range
        obj1 = make_text_object("1", x0=0, y0=20, x1=100, y1=40, text="line1")
        obj2 = make_text_object("2", x0=0, y0=10, x1=100, y1=18, text="line2")

        assert merger._should_merge_next_line(obj1, obj2)

    def test_should_merge_next_line_sentence_end(self) -> None:
        """Don't merge across lines if sentence ends."""
        merger = TextMerger(merge_threshold_y=5.0, x_overlap_ratio=0.5)

        obj1 = make_text_object("1", x0=0, y0=20, x1=100, y1=40, text="line1.")
        obj2 = make_text_object("2", x0=0, y0=10, x1=100, y1=18, text="line2")

        assert not merger._should_merge_next_line(obj1, obj2)

    def test_should_merge_next_line_no_overlap(self) -> None:
        """Don't merge if no x overlap (multi-column)."""
        merger = TextMerger(merge_threshold_y=5.0, x_overlap_ratio=0.5)

        # Different columns (no x overlap)
        obj1 = make_text_object("1", x0=0, y0=20, x1=100, y1=40, text="col1")
        obj2 = make_text_object("2", x0=200, y0=10, x1=300, y1=18, text="col2")

        assert not merger._should_merge_next_line(obj1, obj2)
