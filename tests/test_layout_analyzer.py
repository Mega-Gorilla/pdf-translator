# SPDX-License-Identifier: Apache-2.0
"""Tests for layout analysis module.

This module tests:
- RawLayoutCategory enum
- LayoutBlock data class
- BBox utility functions
- Coordinate conversion
- Text-layout matching
- Filtering logic
"""

from __future__ import annotations

import os

import pytest

from pdf_translator.core.layout_utils import (
    CATEGORY_PRIORITY,
    bbox_area,
    bbox_intersection,
    calc_containment,
    convert_image_to_pdf_coords,
    filter_translatable,
    match_text_with_layout,
)
from pdf_translator.core.models import (
    DEFAULT_TRANSLATABLE_RAW_CATEGORIES,
    BBox,
    LayoutBlock,
    RawLayoutCategory,
    TextObject,
)

# =============================================================================
# Test: Enum coverage
# =============================================================================


class TestRawLayoutCategory:
    """Tests for RawLayoutCategory enum."""

    def test_all_categories_defined(self) -> None:
        """Verify all 25 PP-DocLayoutV2 categories are defined."""
        expected_categories = {
            # テキスト系
            "text",
            "paragraph_title",
            "doc_title",
            "abstract",
            "aside_text",
            # 数式系
            "inline_formula",
            "display_formula",
            "formula_number",
            "algorithm",
            # 図表系
            "table",
            "image",
            "figure_title",
            "chart",
            # ナビゲーション系
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            # 参照系
            "reference",
            "reference_content",
            "footnote",
            "vision_footnote",
            # その他
            "seal",
            "content",
            "vertical_text",
            # 未知
            "unknown",
        }
        actual_values = {cat.value for cat in RawLayoutCategory}
        assert actual_values == expected_categories

    def test_str_enum_inheritance(self) -> None:
        """Verify str inheritance works for comparison and .value access."""
        # str(Enum) behavior changed in Python 3.11+
        # Use .value for string serialization
        assert RawLayoutCategory.TEXT.value == "text"
        # Direct comparison with str works via __eq__
        assert RawLayoutCategory.TEXT == "text"

    def test_matches_official_label_list(self) -> None:
        """Verify RawLayoutCategory matches PP-DocLayoutV2 official label_list.

        Official label_list from:
        https://huggingface.co/PaddlePaddle/PP-DocLayoutV2/raw/main/config.json
        """
        official_label_list = {
            "text",
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
            "vertical_text",
        }
        # UNKNOWN is our fallback, not in official list
        our_categories = {
            cat.value for cat in RawLayoutCategory if cat != RawLayoutCategory.UNKNOWN
        }
        assert our_categories == official_label_list


class TestTranslatableCategories:
    """Tests for DEFAULT_TRANSLATABLE_RAW_CATEGORIES."""

    def test_translatable_categories_defined(self) -> None:
        """Verify translatable categories are correct."""
        expected = {
            "text",
            "vertical_text",
            "abstract",
            "aside_text",
            "paragraph_title",
            "doc_title",
            "figure_title",
        }
        assert DEFAULT_TRANSLATABLE_RAW_CATEGORIES == expected

    def test_text_is_translatable(self) -> None:
        """Text categories should be translatable."""
        assert "text" in DEFAULT_TRANSLATABLE_RAW_CATEGORIES
        assert "paragraph_title" in DEFAULT_TRANSLATABLE_RAW_CATEGORIES

    def test_formula_is_not_translatable(self) -> None:
        """Formula categories should not be translatable."""
        assert "inline_formula" not in DEFAULT_TRANSLATABLE_RAW_CATEGORIES
        assert "display_formula" not in DEFAULT_TRANSLATABLE_RAW_CATEGORIES


# =============================================================================
# Test: LayoutBlock
# =============================================================================


class TestLayoutBlock:
    """Tests for LayoutBlock data class."""

    def test_creation_with_enums(self) -> None:
        """Test LayoutBlock creation with enum types."""
        block = LayoutBlock(
            id="test-1",
            bbox=BBox(x0=0, y0=0, x1=100, y1=50),
            raw_category=RawLayoutCategory.TEXT,
            confidence=0.95,
            page_num=0,
        )
        assert block.raw_category == RawLayoutCategory.TEXT
        assert block.confidence == 0.95
        assert block.page_num == 0

    def test_type_property_backward_compatibility(self) -> None:
        """Test backward-compatible type property."""
        block = LayoutBlock(
            id="test-1",
            bbox=BBox(x0=0, y0=0, x1=100, y1=50),
            raw_category=RawLayoutCategory.INLINE_FORMULA,
        )
        assert block.type == "inline_formula"

    def test_is_translatable_property(self) -> None:
        """Test is_translatable property."""
        text_block = LayoutBlock(
            id="text-1",
            bbox=BBox(x0=0, y0=0, x1=100, y1=50),
            raw_category=RawLayoutCategory.TEXT,
        )
        assert text_block.is_translatable is True

        formula_block = LayoutBlock(
            id="formula-1",
            bbox=BBox(x0=0, y0=0, x1=100, y1=50),
            raw_category=RawLayoutCategory.INLINE_FORMULA,
        )
        assert formula_block.is_translatable is False

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        block = LayoutBlock(
            id="test-1",
            bbox=BBox(x0=10, y0=20, x1=110, y1=70),
            raw_category=RawLayoutCategory.TEXT,
            confidence=0.9,
            page_num=2,
        )
        d = block.to_dict()
        assert d["raw_category"] == "text"
        assert "project_category" not in d  # Removed
        assert d["page_num"] == 2

    def test_from_dict_new_format(self) -> None:
        """Test deserialization from new format."""
        data = {
            "id": "test-1",
            "bbox": {"x0": 10, "y0": 20, "x1": 110, "y1": 70},
            "raw_category": "inline_formula",
            "confidence": 0.85,
            "page_num": 1,
        }
        block = LayoutBlock.from_dict(data)
        assert block.raw_category == RawLayoutCategory.INLINE_FORMULA
        assert block.page_num == 1

    def test_from_dict_legacy_format(self) -> None:
        """Test deserialization from legacy format with 'type' field."""
        data = {
            "id": "test-1",
            "bbox": {"x0": 10, "y0": 20, "x1": 110, "y1": 70},
            "type": "text",
            "confidence": 0.9,
        }
        block = LayoutBlock.from_dict(data)
        assert block.raw_category == RawLayoutCategory.TEXT

    def test_from_dict_unknown_category(self) -> None:
        """Test handling of unknown category in dict."""
        data = {
            "id": "test-1",
            "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 50},
            "raw_category": "nonexistent_category",
        }
        block = LayoutBlock.from_dict(data)
        assert block.raw_category == RawLayoutCategory.UNKNOWN


# =============================================================================
# Test: BBox utilities
# =============================================================================


class TestBBoxUtils:
    """Tests for BBox utility functions."""

    def test_bbox_area(self) -> None:
        """Test area calculation."""
        bbox = BBox(x0=0, y0=0, x1=10, y1=20)
        assert bbox_area(bbox) == 200.0

    def test_bbox_area_zero(self) -> None:
        """Test area of zero-size bbox."""
        bbox = BBox(x0=5, y0=5, x1=5, y1=5)
        assert bbox_area(bbox) == 0.0

    def test_bbox_intersection_full_overlap(self) -> None:
        """Test intersection of fully overlapping bboxes."""
        bbox1 = BBox(x0=0, y0=0, x1=100, y1=100)
        bbox2 = BBox(x0=25, y0=25, x1=75, y1=75)

        intersection = bbox_intersection(bbox1, bbox2)
        assert intersection is not None
        assert intersection.x0 == 25
        assert intersection.y0 == 25
        assert intersection.x1 == 75
        assert intersection.y1 == 75

    def test_bbox_intersection_partial_overlap(self) -> None:
        """Test intersection of partially overlapping bboxes."""
        bbox1 = BBox(x0=0, y0=0, x1=50, y1=50)
        bbox2 = BBox(x0=25, y0=25, x1=75, y1=75)

        intersection = bbox_intersection(bbox1, bbox2)
        assert intersection is not None
        assert intersection.x0 == 25
        assert intersection.y0 == 25
        assert intersection.x1 == 50
        assert intersection.y1 == 50

    def test_bbox_intersection_no_overlap(self) -> None:
        """Test intersection of non-overlapping bboxes."""
        bbox1 = BBox(x0=0, y0=0, x1=50, y1=50)
        bbox2 = BBox(x0=100, y0=100, x1=150, y1=150)

        intersection = bbox_intersection(bbox1, bbox2)
        assert intersection is None

    def test_bbox_intersection_edge_touch(self) -> None:
        """Test intersection when bboxes touch at edge (no overlap)."""
        bbox1 = BBox(x0=0, y0=0, x1=50, y1=50)
        bbox2 = BBox(x0=50, y0=0, x1=100, y1=50)

        intersection = bbox_intersection(bbox1, bbox2)
        assert intersection is None

    def test_calc_containment_full(self) -> None:
        """Test full containment (100%)."""
        text_bbox = BBox(x0=25, y0=25, x1=75, y1=75)
        block_bbox = BBox(x0=0, y0=0, x1=100, y1=100)

        containment = calc_containment(text_bbox, block_bbox)
        assert containment == 1.0

    def test_calc_containment_partial(self) -> None:
        """Test partial containment."""
        text_bbox = BBox(x0=0, y0=0, x1=100, y1=100)
        block_bbox = BBox(x0=50, y0=50, x1=150, y1=150)

        containment = calc_containment(text_bbox, block_bbox)
        # Intersection: 50x50 = 2500, Text area: 100x100 = 10000
        assert containment == pytest.approx(0.25)

    def test_calc_containment_no_overlap(self) -> None:
        """Test no overlap (0%)."""
        text_bbox = BBox(x0=0, y0=0, x1=50, y1=50)
        block_bbox = BBox(x0=100, y0=100, x1=200, y1=200)

        containment = calc_containment(text_bbox, block_bbox)
        assert containment == 0.0

    def test_calc_containment_zero_area_text(self) -> None:
        """Test with zero-area text bbox."""
        text_bbox = BBox(x0=25, y0=25, x1=25, y1=25)
        block_bbox = BBox(x0=0, y0=0, x1=100, y1=100)

        containment = calc_containment(text_bbox, block_bbox)
        assert containment == 0.0


# =============================================================================
# Test: Coordinate conversion
# =============================================================================


class TestCoordinateConversion:
    """Tests for image-to-PDF coordinate conversion."""

    def test_convert_image_to_pdf_coords_basic(self) -> None:
        """Test basic coordinate conversion."""
        # 600x800 pixel image -> 595x842 point PDF
        image_bbox = (100.0, 200.0, 300.0, 400.0)
        result = convert_image_to_pdf_coords(
            image_bbox,
            page_width=595.0,
            page_height=842.0,
            image_width=600,
            image_height=800,
        )

        # X scales: 595/600 ≈ 0.992
        # Y inverts and scales: 842/800 = 1.0525
        assert isinstance(result, BBox)

    def test_y_axis_inversion(self) -> None:
        """Test that Y-axis is properly inverted."""
        # Top of image (y=0) -> Top of PDF (y=page_height)
        # Bottom of image (y=800) -> Bottom of PDF (y=0)
        image_bbox = (0.0, 0.0, 100.0, 100.0)  # Top-left corner in image
        result = convert_image_to_pdf_coords(
            image_bbox,
            page_width=800.0,
            page_height=800.0,
            image_width=800,
            image_height=800,
        )

        # In PDF coords, top of page is y=800
        assert result.y1 == pytest.approx(800.0)
        assert result.y0 == pytest.approx(700.0)


# =============================================================================
# Test: Text-layout matching
# =============================================================================


class TestMatchTextWithLayout:
    """Tests for match_text_with_layout function."""

    @staticmethod
    def create_text_object(id: str, bbox: BBox) -> TextObject:
        """Helper to create TextObject."""
        return TextObject(id=id, bbox=bbox, text="dummy")

    @staticmethod
    def create_layout_block(
        id: str,
        bbox: BBox,
        raw_category: RawLayoutCategory,
    ) -> LayoutBlock:
        """Helper to create LayoutBlock."""
        return LayoutBlock(
            id=id,
            bbox=bbox,
            raw_category=raw_category,
        )

    def test_single_match(self) -> None:
        """Test simple single block match."""
        text_obj = self.create_text_object("t1", BBox(x0=10, y0=10, x1=20, y1=20))
        block = self.create_layout_block(
            "b1",
            BBox(x0=0, y0=0, x1=100, y1=100),
            RawLayoutCategory.TEXT,
        )

        result = match_text_with_layout([text_obj], [block])
        assert result["t1"] == "text"

    def test_priority_matching_formula_wins(self) -> None:
        """Test that formula (high priority) beats text (low priority)."""
        # Text inside both a text block and an inline_formula block
        text_obj = self.create_text_object("t1", BBox(x0=50, y0=50, x1=60, y1=60))

        text_block = self.create_layout_block(
            "b1",
            BBox(x0=0, y0=0, x1=200, y1=200),  # Large text block
            RawLayoutCategory.TEXT,
        )
        formula_block = self.create_layout_block(
            "b2",
            BBox(x0=40, y0=40, x1=70, y1=70),  # Smaller formula block
            RawLayoutCategory.INLINE_FORMULA,
        )

        result = match_text_with_layout([text_obj], [text_block, formula_block])
        # Formula should win due to higher priority (lower number)
        assert result["t1"] == "inline_formula"

    def test_no_match_returns_unknown(self) -> None:
        """Test unmatched text defaults to unknown."""
        text_obj = self.create_text_object("t1", BBox(x0=500, y0=500, x1=510, y1=510))
        block = self.create_layout_block(
            "b1",
            BBox(x0=0, y0=0, x1=100, y1=100),  # Non-overlapping
            RawLayoutCategory.TEXT,
        )

        result = match_text_with_layout([text_obj], [block])
        assert result["t1"] == "unknown"

    def test_containment_threshold(self) -> None:
        """Test containment threshold filtering."""
        # Text partially overlapping block
        text_obj = self.create_text_object("t1", BBox(x0=0, y0=0, x1=100, y1=100))
        block = self.create_layout_block(
            "b1",
            BBox(x0=50, y0=50, x1=150, y1=150),  # 25% overlap
            RawLayoutCategory.TEXT,
        )

        # Default threshold 0.5 - should not match
        result = match_text_with_layout([text_obj], [block], containment_threshold=0.5)
        assert result["t1"] == "unknown"

        # Lower threshold 0.2 - should match
        result = match_text_with_layout([text_obj], [block], containment_threshold=0.2)
        assert result["t1"] == "text"

    def test_table_cell_scenario(self) -> None:
        """Test that table text is correctly categorized."""
        text_obj = self.create_text_object("t1", BBox(x0=50, y0=50, x1=60, y1=60))

        text_block = self.create_layout_block(
            "b1",
            BBox(x0=0, y0=0, x1=200, y1=200),
            RawLayoutCategory.TEXT,
        )
        table_block = self.create_layout_block(
            "b2",
            BBox(x0=40, y0=40, x1=100, y1=100),
            RawLayoutCategory.TABLE,
        )

        result = match_text_with_layout([text_obj], [text_block, table_block])
        # Table has priority 3, Text has priority 6 → Table wins
        assert result["t1"] == "table"


class TestCategoryPriority:
    """Tests for category priority definitions."""

    def test_formula_highest_priority(self) -> None:
        """Formula categories should have highest priority (1)."""
        assert CATEGORY_PRIORITY[RawLayoutCategory.INLINE_FORMULA] == 1
        assert CATEGORY_PRIORITY[RawLayoutCategory.DISPLAY_FORMULA] == 1

    def test_text_lower_priority(self) -> None:
        """Text categories should have lower priority (6)."""
        assert CATEGORY_PRIORITY[RawLayoutCategory.TEXT] == 6
        assert CATEGORY_PRIORITY[RawLayoutCategory.PARAGRAPH_TITLE] == 6

    def test_all_important_categories_have_priority(self) -> None:
        """All categories used in matching should have defined priority."""
        important_categories = [
            RawLayoutCategory.TEXT,
            RawLayoutCategory.INLINE_FORMULA,
            RawLayoutCategory.DISPLAY_FORMULA,
            RawLayoutCategory.TABLE,
            RawLayoutCategory.IMAGE,
            RawLayoutCategory.ALGORITHM,
        ]
        for cat in important_categories:
            assert cat in CATEGORY_PRIORITY


# =============================================================================
# Test: Filter translatable
# =============================================================================


class TestFilterTranslatable:
    """Tests for filter_translatable function."""

    def test_filter_keeps_translatable(self) -> None:
        """Test that translatable categories are kept."""
        text_obj = TextObject(id="t1", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text="Hello")
        categories = {"t1": "text"}

        result = filter_translatable([text_obj], categories)
        assert len(result) == 1
        assert result[0].id == "t1"

    def test_filter_removes_non_translatable(self) -> None:
        """Test that non-translatable categories are removed."""
        text_obj = TextObject(id="t1", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text="x^2")
        categories = {"t1": "inline_formula"}

        result = filter_translatable([text_obj], categories)
        assert len(result) == 0

    def test_filter_mixed(self) -> None:
        """Test filtering with mixed categories."""
        objs = [
            TextObject(id="t1", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text="Text"),
            TextObject(id="t2", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text="Formula"),
            TextObject(id="t3", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text="Title"),
        ]
        categories = {
            "t1": "text",
            "t2": "inline_formula",
            "t3": "paragraph_title",
        }

        result = filter_translatable(objs, categories)
        assert len(result) == 2
        assert {obj.id for obj in result} == {"t1", "t3"}


# =============================================================================
# Integration tests (require PaddleOCR)
# =============================================================================


@pytest.mark.skipif(
    os.environ.get("RUN_LAYOUT_TESTS") != "1",
    reason="Layout tests require PaddleOCR (set RUN_LAYOUT_TESTS=1)",
)
class TestLayoutAnalyzerIntegration:
    """Integration tests for LayoutAnalyzer (requires PaddleOCR)."""

    def test_analyze_single_page(self) -> None:
        """Test analyzing a single page."""
        from pdf_translator.core.layout_analyzer import LayoutAnalyzer

        pdf_path = "tests/fixtures/sample_autogen_paper.pdf"
        analyzer = LayoutAnalyzer()
        blocks = analyzer.analyze(pdf_path, page_num=0)

        assert len(blocks) > 0
        for block in blocks:
            assert isinstance(block.raw_category, RawLayoutCategory)
            assert block.confidence > 0
