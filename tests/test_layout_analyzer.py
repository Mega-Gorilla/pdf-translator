# SPDX-License-Identifier: Apache-2.0
"""Tests for layout analysis module.

This module tests:
- RawLayoutCategory and ProjectCategory enums
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
    RAW_TO_PROJECT_MAPPING,
    TRANSLATABLE_CATEGORIES,
    BBox,
    LayoutBlock,
    ProjectCategory,
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

    def test_all_categories_have_mapping(self) -> None:
        """Verify all raw categories map to project categories."""
        for raw_cat in RawLayoutCategory:
            assert raw_cat in RAW_TO_PROJECT_MAPPING

    def test_matches_official_label_list(self) -> None:
        """Verify RawLayoutCategory matches PP-DocLayoutV2 official label_list.

        Source: https://huggingface.co/PaddlePaddle/PP-DocLayoutV2/raw/main/config.json
        """
        official_label_list = {
            "abstract",
            "algorithm",
            "aside_text",
            "chart",
            "content",
            "display_formula",
            "doc_title",
            "figure_title",
            "footer",
            "footer_image",
            "footnote",
            "formula_number",
            "header",
            "header_image",
            "image",
            "inline_formula",
            "number",
            "paragraph_title",
            "reference",
            "reference_content",
            "seal",
            "table",
            "text",
            "vertical_text",
            "vision_footnote",
        }
        # UNKNOWN is our fallback, not in official list
        our_categories = {
            cat.value for cat in RawLayoutCategory if cat != RawLayoutCategory.UNKNOWN
        }
        assert our_categories == official_label_list


class TestProjectCategory:
    """Tests for ProjectCategory enum."""

    def test_all_categories_defined(self) -> None:
        """Verify all 12 project categories are defined."""
        expected_categories = {
            "text",
            "title",
            "caption",
            "footnote",
            "formula",
            "code",
            "table",
            "image",
            "chart",
            "header",
            "reference",
            "other",
        }
        actual_values = {cat.value for cat in ProjectCategory}
        assert actual_values == expected_categories

    def test_translatable_categories(self) -> None:
        """Verify translatable categories are correct."""
        expected = {ProjectCategory.TEXT, ProjectCategory.TITLE, ProjectCategory.CAPTION}
        assert TRANSLATABLE_CATEGORIES == expected


class TestCategoryMapping:
    """Tests for RAW_TO_PROJECT_MAPPING."""

    def test_text_categories_map_correctly(self) -> None:
        """Verify text-like categories map to TEXT/TITLE/CAPTION."""
        assert RAW_TO_PROJECT_MAPPING[RawLayoutCategory.TEXT] == ProjectCategory.TEXT
        assert RAW_TO_PROJECT_MAPPING[RawLayoutCategory.ABSTRACT] == ProjectCategory.TEXT
        assert (
            RAW_TO_PROJECT_MAPPING[RawLayoutCategory.PARAGRAPH_TITLE]
            == ProjectCategory.TITLE
        )
        assert (
            RAW_TO_PROJECT_MAPPING[RawLayoutCategory.DOC_TITLE] == ProjectCategory.TITLE
        )
        assert (
            RAW_TO_PROJECT_MAPPING[RawLayoutCategory.FIGURE_TITLE]
            == ProjectCategory.CAPTION
        )

    def test_formula_categories_map_correctly(self) -> None:
        """Verify formula categories map to FORMULA."""
        assert (
            RAW_TO_PROJECT_MAPPING[RawLayoutCategory.INLINE_FORMULA]
            == ProjectCategory.FORMULA
        )
        assert (
            RAW_TO_PROJECT_MAPPING[RawLayoutCategory.DISPLAY_FORMULA]
            == ProjectCategory.FORMULA
        )
        assert (
            RAW_TO_PROJECT_MAPPING[RawLayoutCategory.FORMULA_NUMBER]
            == ProjectCategory.FORMULA
        )

    def test_footnote_maps_correctly(self) -> None:
        """Verify FOOTNOTE maps to FOOTNOTE (not TEXT)."""
        assert (
            RAW_TO_PROJECT_MAPPING[RawLayoutCategory.FOOTNOTE]
            == ProjectCategory.FOOTNOTE
        )


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
            project_category=ProjectCategory.TEXT,
            confidence=0.95,
            page_num=0,
        )
        assert block.raw_category == RawLayoutCategory.TEXT
        assert block.project_category == ProjectCategory.TEXT
        assert block.confidence == 0.95
        assert block.page_num == 0

    def test_type_property_backward_compatibility(self) -> None:
        """Test backward-compatible type property."""
        block = LayoutBlock(
            id="test-1",
            bbox=BBox(x0=0, y0=0, x1=100, y1=50),
            raw_category=RawLayoutCategory.INLINE_FORMULA,
            project_category=ProjectCategory.FORMULA,
        )
        assert block.type == "inline_formula"

    def test_is_translatable_property(self) -> None:
        """Test is_translatable property."""
        text_block = LayoutBlock(
            id="text-1",
            bbox=BBox(x0=0, y0=0, x1=100, y1=50),
            raw_category=RawLayoutCategory.TEXT,
            project_category=ProjectCategory.TEXT,
        )
        assert text_block.is_translatable is True

        formula_block = LayoutBlock(
            id="formula-1",
            bbox=BBox(x0=0, y0=0, x1=100, y1=50),
            raw_category=RawLayoutCategory.INLINE_FORMULA,
            project_category=ProjectCategory.FORMULA,
        )
        assert formula_block.is_translatable is False

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        block = LayoutBlock(
            id="test-1",
            bbox=BBox(x0=10, y0=20, x1=110, y1=70),
            raw_category=RawLayoutCategory.TEXT,
            project_category=ProjectCategory.TEXT,
            confidence=0.9,
            page_num=2,
        )
        d = block.to_dict()
        assert d["raw_category"] == "text"
        assert d["project_category"] == "text"
        assert d["page_num"] == 2

    def test_from_dict_new_format(self) -> None:
        """Test deserialization from new format."""
        data = {
            "id": "test-1",
            "bbox": {"x0": 10, "y0": 20, "x1": 110, "y1": 70},
            "raw_category": "inline_formula",
            "project_category": "formula",
            "confidence": 0.85,
            "page_num": 1,
        }
        block = LayoutBlock.from_dict(data)
        assert block.raw_category == RawLayoutCategory.INLINE_FORMULA
        assert block.project_category == ProjectCategory.FORMULA
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
        assert block.project_category == ProjectCategory.TEXT

    def test_from_dict_unknown_category(self) -> None:
        """Test handling of unknown category in dict."""
        data = {
            "id": "test-1",
            "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 50},
            "raw_category": "nonexistent_category",
        }
        block = LayoutBlock.from_dict(data)
        assert block.raw_category == RawLayoutCategory.UNKNOWN
        assert block.project_category == ProjectCategory.OTHER


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

    def test_bbox_intersection_overlap(self) -> None:
        """Test intersection of overlapping boxes."""
        bbox1 = BBox(x0=0, y0=0, x1=10, y1=10)
        bbox2 = BBox(x0=5, y0=5, x1=15, y1=15)
        result = bbox_intersection(bbox1, bbox2)
        assert result is not None
        assert result.x0 == 5
        assert result.y0 == 5
        assert result.x1 == 10
        assert result.y1 == 10

    def test_bbox_intersection_no_overlap(self) -> None:
        """Test intersection of non-overlapping boxes."""
        bbox1 = BBox(x0=0, y0=0, x1=10, y1=10)
        bbox2 = BBox(x0=20, y0=20, x1=30, y1=30)
        result = bbox_intersection(bbox1, bbox2)
        assert result is None

    def test_bbox_intersection_contained(self) -> None:
        """Test intersection when one box contains another."""
        outer = BBox(x0=0, y0=0, x1=100, y1=100)
        inner = BBox(x0=20, y0=20, x1=40, y1=40)
        result = bbox_intersection(outer, inner)
        assert result is not None
        assert result.x0 == inner.x0
        assert result.y0 == inner.y0
        assert result.x1 == inner.x1
        assert result.y1 == inner.y1

    def test_calc_containment_full(self) -> None:
        """Test containment when text is fully contained."""
        text_bbox = BBox(x0=20, y0=20, x1=40, y1=40)
        block_bbox = BBox(x0=0, y0=0, x1=100, y1=100)
        assert calc_containment(text_bbox, block_bbox) == 1.0

    def test_calc_containment_partial(self) -> None:
        """Test containment when text is partially contained."""
        text_bbox = BBox(x0=0, y0=0, x1=10, y1=10)  # area = 100
        block_bbox = BBox(x0=5, y0=5, x1=15, y1=15)  # overlaps (5,5)-(10,10) = 25
        containment = calc_containment(text_bbox, block_bbox)
        assert containment == pytest.approx(0.25)

    def test_calc_containment_none(self) -> None:
        """Test containment when no overlap."""
        text_bbox = BBox(x0=0, y0=0, x1=10, y1=10)
        block_bbox = BBox(x0=50, y0=50, x1=60, y1=60)
        assert calc_containment(text_bbox, block_bbox) == 0.0

    def test_calc_containment_zero_area_text(self) -> None:
        """Test containment when text has zero area."""
        text_bbox = BBox(x0=5, y0=5, x1=5, y1=5)
        block_bbox = BBox(x0=0, y0=0, x1=100, y1=100)
        assert calc_containment(text_bbox, block_bbox) == 0.0


# =============================================================================
# Test: Coordinate conversion
# =============================================================================


class TestCoordinateConversion:
    """Tests for image to PDF coordinate conversion."""

    def test_basic_conversion(self) -> None:
        """Test basic coordinate conversion with Y-axis inversion."""
        # Image: 1000x800 pixels, PDF: 500x400 points
        # Image bbox: (100, 100, 200, 200) - near top-left
        # Expected PDF: scaled and Y-inverted

        result = convert_image_to_pdf_coords(
            image_bbox=(100, 100, 200, 200),
            page_width=500,
            page_height=400,
            image_width=1000,
            image_height=800,
        )

        # X: 100 * 0.5 = 50, 200 * 0.5 = 100
        assert result.x0 == pytest.approx(50.0)
        assert result.x1 == pytest.approx(100.0)

        # Y: inverted
        # y0_pdf = 400 - (200 * 0.5) = 300
        # y1_pdf = 400 - (100 * 0.5) = 350
        assert result.y0 == pytest.approx(300.0)
        assert result.y1 == pytest.approx(350.0)

    def test_full_page_conversion(self) -> None:
        """Test converting full page bbox."""
        result = convert_image_to_pdf_coords(
            image_bbox=(0, 0, 1000, 800),
            page_width=500,
            page_height=400,
            image_width=1000,
            image_height=800,
        )
        assert result.x0 == pytest.approx(0.0)
        assert result.y0 == pytest.approx(0.0)
        assert result.x1 == pytest.approx(500.0)
        assert result.y1 == pytest.approx(400.0)

    def test_bottom_region_conversion(self) -> None:
        """Test converting region at bottom of image (top in PDF)."""
        # Image bbox near bottom: (0, 700, 100, 800)
        result = convert_image_to_pdf_coords(
            image_bbox=(0, 700, 100, 800),
            page_width=500,
            page_height=400,
            image_width=1000,
            image_height=800,
        )
        # Y near PDF origin (bottom)
        assert result.y0 == pytest.approx(0.0)  # 400 - 800*0.5 = 0
        assert result.y1 == pytest.approx(50.0)  # 400 - 700*0.5 = 50


# =============================================================================
# Test: Text-layout matching
# =============================================================================


class TestMatchTextWithLayout:
    """Tests for text-layout matching algorithm."""

    def create_text_object(self, id: str, bbox: BBox) -> TextObject:
        """Helper to create TextObject."""
        return TextObject(id=id, bbox=bbox, text="dummy")

    def create_layout_block(
        self,
        id: str,
        bbox: BBox,
        raw_category: RawLayoutCategory,
    ) -> LayoutBlock:
        """Helper to create LayoutBlock."""
        project_category = RAW_TO_PROJECT_MAPPING[raw_category]
        return LayoutBlock(
            id=id,
            bbox=bbox,
            raw_category=raw_category,
            project_category=project_category,
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
        assert result["t1"] == ProjectCategory.TEXT

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
        assert result["t1"] == ProjectCategory.FORMULA

    def test_no_match_returns_other(self) -> None:
        """Test unmatched text defaults to OTHER."""
        text_obj = self.create_text_object("t1", BBox(x0=500, y0=500, x1=510, y1=510))
        block = self.create_layout_block(
            "b1",
            BBox(x0=0, y0=0, x1=100, y1=100),  # Non-overlapping
            RawLayoutCategory.TEXT,
        )

        result = match_text_with_layout([text_obj], [block])
        assert result["t1"] == ProjectCategory.OTHER

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
        assert result["t1"] == ProjectCategory.OTHER

        # Lower threshold 0.2 - should match
        result = match_text_with_layout([text_obj], [block], containment_threshold=0.2)
        assert result["t1"] == ProjectCategory.TEXT

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
        assert result["t1"] == ProjectCategory.TABLE


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
        categories = {"t1": ProjectCategory.TEXT}

        result = filter_translatable([text_obj], categories)
        assert len(result) == 1
        assert result[0].id == "t1"

    def test_filter_removes_non_translatable(self) -> None:
        """Test that non-translatable categories are removed."""
        text_obj = TextObject(id="t1", bbox=BBox(x0=0, y0=0, x1=10, y1=10), text="x^2")
        categories = {"t1": ProjectCategory.FORMULA}

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
            "t1": ProjectCategory.TEXT,
            "t2": ProjectCategory.FORMULA,
            "t3": ProjectCategory.TITLE,
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
            assert isinstance(block.project_category, ProjectCategory)
            assert block.confidence > 0
