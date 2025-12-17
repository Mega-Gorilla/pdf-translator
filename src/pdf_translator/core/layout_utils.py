# SPDX-License-Identifier: Apache-2.0
"""Utility functions for layout analysis.

This module provides coordinate transformation, BBox operations, and
text-layout matching functions for PP-DocLayout integration.
"""

from __future__ import annotations

from .models import (
    TRANSLATABLE_CATEGORIES,
    BBox,
    LayoutBlock,
    ProjectCategory,
    RawLayoutCategory,
    TextObject,
)

# カテゴリ優先度 (数値が小さいほど優先)
# 翻訳除外すべきカテゴリを優先して「安全側に倒す」設計
CATEGORY_PRIORITY: dict[RawLayoutCategory, int] = {
    # 最優先: 絶対に翻訳してはいけないもの
    RawLayoutCategory.INLINE_FORMULA: 1,
    RawLayoutCategory.DISPLAY_FORMULA: 1,
    RawLayoutCategory.FORMULA_NUMBER: 1,
    RawLayoutCategory.ALGORITHM: 2,
    RawLayoutCategory.CODE_BLOCK: 2,
    # 次点: 通常は翻訳しないもの
    RawLayoutCategory.TABLE: 3,
    RawLayoutCategory.IMAGE: 4,
    RawLayoutCategory.CHART: 4,
    # キャプション系
    RawLayoutCategory.FIGURE_TITLE: 5,
    # テキスト系
    RawLayoutCategory.TEXT: 6,
    RawLayoutCategory.PARAGRAPH_TITLE: 6,
    RawLayoutCategory.DOC_TITLE: 6,
    RawLayoutCategory.ABSTRACT: 6,
    RawLayoutCategory.ASIDE_TEXT: 6,
    # その他
    RawLayoutCategory.FOOTNOTE: 7,
    RawLayoutCategory.HEADER: 8,
    RawLayoutCategory.FOOTER: 8,
    RawLayoutCategory.NUMBER: 8,
    RawLayoutCategory.REFERENCE: 9,
    RawLayoutCategory.REFERENCE_CONTENT: 9,
    # 未定義カテゴリは最低優先度
    RawLayoutCategory.SEAL: 99,
    RawLayoutCategory.CONTENT: 99,
    RawLayoutCategory.TABLE_OF_CONTENTS: 99,
    RawLayoutCategory.UNKNOWN: 99,
}

# デフォルト優先度（未定義カテゴリ用）
DEFAULT_PRIORITY = 99


def bbox_area(bbox: BBox) -> float:
    """Calculate area of a BBox.

    Args:
        bbox: Bounding box

    Returns:
        Area in square units (points²)
    """
    return bbox.width * bbox.height


def bbox_intersection(bbox1: BBox, bbox2: BBox) -> BBox | None:
    """Calculate intersection of two BBoxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        Intersection BBox, or None if no intersection
    """
    x0 = max(bbox1.x0, bbox2.x0)
    y0 = max(bbox1.y0, bbox2.y0)
    x1 = min(bbox1.x1, bbox2.x1)
    y1 = min(bbox1.y1, bbox2.y1)

    if x0 >= x1 or y0 >= y1:
        return None
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def calc_containment(text_bbox: BBox, block_bbox: BBox) -> float:
    """Calculate how much of text_bbox is contained within block_bbox.

    containment = intersection_area / text_area

    Args:
        text_bbox: TextObject bounding box
        block_bbox: LayoutBlock bounding box

    Returns:
        Containment ratio (0.0 - 1.0)
    """
    intersection = bbox_intersection(text_bbox, block_bbox)
    if intersection is None:
        return 0.0
    text_area = bbox_area(text_bbox)
    if text_area == 0:
        return 0.0
    return bbox_area(intersection) / text_area


def convert_image_to_pdf_coords(
    image_bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> BBox:
    """Convert image coordinates to PDF coordinates with Y-axis inversion.

    Coordinate system differences:
    - Image: origin=top-left, Y-axis=downward (y0 < y1 means top→bottom)
    - PDF: origin=bottom-left, Y-axis=upward (y0 < y1 means bottom→top)

    Args:
        image_bbox: Image coordinates (x0, y0, x1, y1) in pixels
        page_width: PDF page width in points
        page_height: PDF page height in points
        image_width: Rendered image width in pixels
        image_height: Rendered image height in pixels

    Returns:
        BBox in PDF coordinates
    """
    scale_x = page_width / image_width
    scale_y = page_height / image_height

    # X coordinate: simple scaling
    x0_pdf = image_bbox[0] * scale_x
    x1_pdf = image_bbox[2] * scale_x

    # Y coordinate: inversion + scaling
    # image y0 (top) → PDF y1 (top)
    # image y1 (bottom) → PDF y0 (bottom)
    y0_pdf = page_height - (image_bbox[3] * scale_y)
    y1_pdf = page_height - (image_bbox[1] * scale_y)

    return BBox(
        x0=x0_pdf,
        y0=y0_pdf,
        x1=x1_pdf,
        y1=y1_pdf,
    )


def match_text_with_layout(
    text_objects: list[TextObject],
    layout_blocks: list[LayoutBlock],
    containment_threshold: float = 0.5,
) -> dict[str, ProjectCategory]:
    """Match TextObjects with LayoutBlocks to determine categories.

    Algorithm:
    1. Calculate containment ratio between TextObject and each LayoutBlock
    2. Filter blocks with containment >= threshold as candidates
    3. Select the candidate with highest priority (lowest number)
    4. Tie-breaker: higher containment → smaller area

    This design prioritizes non-translatable categories (formula, code, table)
    to "fail safe" - we'd rather miss translating something than translate
    something that shouldn't be (like formulas).

    Args:
        text_objects: TextObjects extracted from PDF
        layout_blocks: LayoutBlocks from layout analysis
        containment_threshold: Minimum containment ratio (default: 0.5)

    Returns:
        Mapping of TextObject.id → ProjectCategory
    """
    result: dict[str, ProjectCategory] = {}

    for text_obj in text_objects:
        # Step 1: Extract candidate blocks
        candidates: list[dict] = []
        for block in layout_blocks:
            containment = calc_containment(text_obj.bbox, block.bbox)
            if containment >= containment_threshold:
                candidates.append(
                    {
                        "block": block,
                        "containment": containment,
                        "area": bbox_area(block.bbox),
                        "priority": CATEGORY_PRIORITY.get(
                            block.raw_category, DEFAULT_PRIORITY
                        ),
                    }
                )

        if not candidates:
            # Unmatched: default to OTHER (safe side)
            result[text_obj.id] = ProjectCategory.OTHER
            continue

        # Step 2: Sort by priority (asc) → containment (desc) → area (asc)
        candidates.sort(key=lambda x: (x["priority"], -x["containment"], x["area"]))

        # Select highest priority candidate
        best = candidates[0]["block"]
        result[text_obj.id] = best.project_category

    return result


def filter_translatable(
    text_objects: list[TextObject],
    categories: dict[str, ProjectCategory],
) -> list[TextObject]:
    """Filter TextObjects to only those that should be translated.

    Args:
        text_objects: All TextObjects
        categories: Mapping of TextObject.id → ProjectCategory

    Returns:
        TextObjects that should be translated
    """
    return [
        obj
        for obj in text_objects
        if categories.get(obj.id) in TRANSLATABLE_CATEGORIES
    ]
