# SPDX-License-Identifier: Apache-2.0
"""Tests for assign_categories helper."""

from pdf_translator.core.layout_utils import assign_categories
from pdf_translator.core.models import (
    BBox,
    LayoutBlock,
    Paragraph,
    RawLayoutCategory,
)


def test_assign_categories_prefers_formula():
    paragraphs = [
        Paragraph(
            id="para_p0_b0",
            page_number=0,
            text="x = 1",
            block_bbox=BBox(0, 0, 100, 100),
            line_count=1,
        )
    ]

    layout_blocks = {
        0: [
            LayoutBlock(
                id="b1",
                bbox=BBox(0, 0, 100, 100),
                raw_category=RawLayoutCategory.TEXT,
            ),
            LayoutBlock(
                id="b2",
                bbox=BBox(0, 0, 100, 100),
                raw_category=RawLayoutCategory.INLINE_FORMULA,
            ),
        ]
    }

    assign_categories(paragraphs, layout_blocks, threshold=0.5)
    # Formula has higher priority, so category should be "inline_formula"
    assert paragraphs[0].category == "inline_formula"


def test_assign_categories_no_match_keeps_none():
    paragraphs = [
        Paragraph(
            id="para_p0_b0",
            page_number=0,
            text="hello",
            block_bbox=BBox(0, 0, 50, 50),
            line_count=1,
        )
    ]

    layout_blocks = {
        0: [
            LayoutBlock(
                id="b1",
                bbox=BBox(200, 200, 300, 300),
                raw_category=RawLayoutCategory.TEXT,
            )
        ]
    }

    assign_categories(paragraphs, layout_blocks, threshold=0.5)
    assert paragraphs[0].category is None


def test_assign_categories_threshold_filtering():
    paragraphs = [
        Paragraph(
            id="para_p0_b0",
            page_number=0,
            text="hello",
            block_bbox=BBox(0, 0, 100, 100),
            line_count=1,
        )
    ]

    layout_blocks = {
        0: [
            LayoutBlock(
                id="b1",
                bbox=BBox(90, 90, 110, 110),
                raw_category=RawLayoutCategory.TEXT,
            )
        ]
    }

    assign_categories(paragraphs, layout_blocks, threshold=0.5)
    assert paragraphs[0].category is None
