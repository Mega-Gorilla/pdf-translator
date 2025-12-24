# SPDX-License-Identifier: Apache-2.0
"""Paragraph merging utilities for adjacent same-category paragraphs.

This module provides functionality to merge paragraphs that are:
- On the same page
- Have the same layout category
- Are vertically adjacent
- Meet various similarity criteria (font size, etc.)

This helps improve translation quality by maintaining context across
over-segmented paragraphs from pdftext.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pdf_translator.core.models import (
    DEFAULT_TRANSLATABLE_RAW_CATEGORIES,
    BBox,
    Paragraph,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class MergeConfig:
    """Configuration for paragraph merging.

    Attributes:
        gap_tolerance: Maximum gap as a multiple of font size.
            Gap <= font_size * gap_tolerance is allowed.
        x_overlap_threshold: Minimum X-axis overlap ratio (0.0 to 1.0).
            Overlap is calculated as overlap_width / min(width1, width2).
        font_size_tolerance: Maximum font size difference in points.
        translatable_categories: Set of category strings that are eligible
            for merging. If None, uses DEFAULT_TRANSLATABLE_RAW_CATEGORIES.
        column_gap_threshold_ratio: Minimum horizontal gap between columns
            as a ratio of page width (0.0 to 1.0). Paragraphs separated by
            more than this gap are considered to be in different columns.
    """

    gap_tolerance: float = 1.5
    x_overlap_threshold: float = 0.7
    font_size_tolerance: float = 1.0
    translatable_categories: frozenset[str] | None = None
    column_gap_threshold_ratio: float = 0.02


def _detect_columns(
    paragraphs: Sequence[Paragraph],
    gap_threshold: float,
    wide_element_ratio: float = 0.5,
    column_overlap_threshold: float = 0.5,
) -> list[list[Paragraph]]:
    """Detect columns based on X-axis overlap.

    This function clusters paragraphs into columns by analyzing X-axis
    overlap between them. Two paragraphs belong to the same column if
    they have significant X overlap (>= column_overlap_threshold).

    This handles multi-column layouts with:
    - Full-width elements (titles) that span multiple columns
    - Centered elements (like "Meta AI") that sit between columns
    - Narrow column paragraphs that should not merge across columns

    Args:
        paragraphs: List of paragraphs on a single page.
        gap_threshold: Minimum horizontal gap (in points) - used as fallback.
        wide_element_ratio: Elements wider than page_width * ratio are
            considered "wide" and handled separately.
        column_overlap_threshold: Minimum X overlap ratio (0.0 to 1.0)
            to consider paragraphs as belonging to the same column.

    Returns:
        List of column groups, where each group is a list of paragraphs
        belonging to the same column. Columns are ordered left to right.
    """
    if not paragraphs:
        return []

    # Estimate page width
    page_width = max(p.block_bbox.x1 for p in paragraphs)
    wide_threshold = page_width * wide_element_ratio

    # Separate wide elements from narrow elements
    narrow_paras: list[Paragraph] = []
    wide_paras: list[Paragraph] = []

    for para in paragraphs:
        if para.block_bbox.width > wide_threshold:
            wide_paras.append(para)
        else:
            narrow_paras.append(para)

    # If all elements are wide or no narrow elements, treat as single column
    if not narrow_paras:
        return [list(paragraphs)]

    # Detect columns using X overlap clustering
    sorted_by_x = sorted(
        narrow_paras,
        key=lambda p: (p.block_bbox.x0 + p.block_bbox.x1) / 2,
    )

    columns: list[list[Paragraph]] = [[sorted_by_x[0]]]

    for para in sorted_by_x[1:]:
        last_col = columns[-1]

        # Check if para has significant X overlap with any paragraph in column
        max_overlap = 0.0
        for col_para in last_col:
            overlap = _calc_x_overlap(para.block_bbox, col_para.block_bbox)
            max_overlap = max(max_overlap, overlap)

        if max_overlap >= column_overlap_threshold:
            # Significant overlap - same column
            last_col.append(para)
        else:
            # No significant overlap - new column
            columns.append([para])

    # Assign wide elements to the most overlapping column
    for wide_para in wide_paras:
        best_col_idx = 0
        best_overlap = 0.0

        for i, col in enumerate(columns):
            col_x0 = min(p.block_bbox.x0 for p in col)
            col_x1 = max(p.block_bbox.x1 for p in col)

            # Calculate X overlap
            overlap_x0 = max(wide_para.block_bbox.x0, col_x0)
            overlap_x1 = min(wide_para.block_bbox.x1, col_x1)
            overlap = max(0.0, overlap_x1 - overlap_x0)

            if overlap > best_overlap:
                best_overlap = overlap
                best_col_idx = i

        columns[best_col_idx].append(wide_para)

    return columns


def _merge_column(
    paragraphs: list[Paragraph],
    config: MergeConfig,
) -> list[Paragraph]:
    """Merge adjacent paragraphs within a single column.

    Paragraphs are processed top-to-bottom (by y1 descending in PDF
    coordinates) and merged if they meet the merge criteria.

    Args:
        paragraphs: Paragraphs in a single column.
        config: Merge configuration.

    Returns:
        List of merged paragraphs.
    """
    if not paragraphs:
        return []

    # Sort by y1 descending (top to bottom in PDF coordinates)
    sorted_paras = sorted(
        paragraphs, key=lambda p: p.block_bbox.y1, reverse=True
    )

    # Initialize with first paragraph
    merged: list[Paragraph] = [sorted_paras[0]]

    # Process remaining paragraphs
    for current in sorted_paras[1:]:
        last = merged[-1]

        if _can_merge(last, current, config):
            # Replace last with merged paragraph
            merged[-1] = _merge_two_paragraphs(last, current)
        else:
            merged.append(current)

    return merged


def merge_adjacent_paragraphs(
    paragraphs: Sequence[Paragraph],
    config: MergeConfig | None = None,
) -> list[Paragraph]:
    """Merge adjacent paragraphs with the same category.

    This function processes paragraphs page by page, with column detection
    for multi-column layouts. Within each column, paragraphs are merged
    top-to-bottom if they meet the merge criteria.

    The column detection ensures that paragraphs in different columns
    (e.g., left and right columns in academic papers) are not incorrectly
    merged even if they appear vertically adjacent.

    Args:
        paragraphs: List of paragraphs to process.
        config: Merge configuration. Uses defaults if None.

    Returns:
        New list with merged paragraphs. The original list is not modified.
    """
    if not paragraphs:
        return []

    if config is None:
        config = MergeConfig()

    # Group paragraphs by page
    pages: dict[int, list[Paragraph]] = {}
    for para in paragraphs:
        if para.page_number not in pages:
            pages[para.page_number] = []
        pages[para.page_number].append(para)

    result: list[Paragraph] = []

    # Process each page
    for page_num in sorted(pages.keys()):
        page_paragraphs = pages[page_num]

        if not page_paragraphs:
            continue

        # Estimate page width from paragraph positions
        page_width = max(p.block_bbox.x1 for p in page_paragraphs)
        gap_threshold = page_width * config.column_gap_threshold_ratio

        # Detect columns based on X positions
        columns = _detect_columns(page_paragraphs, gap_threshold)

        # Process each column independently
        merged_page: list[Paragraph] = []
        for column in columns:
            merged_column = _merge_column(column, config)
            merged_page.extend(merged_column)

        # Sort by y1 descending for consistent output order
        merged_page.sort(key=lambda p: p.block_bbox.y1, reverse=True)

        result.extend(merged_page)

    return result


def _can_merge(
    para1: Paragraph,
    para2: Paragraph,
    config: MergeConfig,
) -> bool:
    """Check if two paragraphs can be merged.

    Conditions (all must be true):
    1. Same page (already guaranteed by caller)
    2. Same category (non-None)
    3. Translatable category
    4. Vertically adjacent (para2 below para1 in PDF coords)
    5. Gap <= font_size * gap_tolerance
    6. X overlap >= threshold
    7. Same font size (within tolerance)

    Note: The following checks were intentionally removed:
    - Alignment check: _estimate_alignment() is unreliable for short paragraphs,
      and "left" vs "justify" distinction causes false negatives.
    - Sentence-ending punctuation check: The purpose of merging is layout
      optimization (larger bbox = better font size and line-breaking decisions),
      not semantic analysis. Complete sentences should still merge for
      consistent visual appearance.

    Args:
        para1: First paragraph (should be above para2).
        para2: Second paragraph (should be below para1).
        config: Merge configuration.

    Returns:
        True if paragraphs can be merged, False otherwise.
    """
    # 1. Different pages should not merge (caller should handle this)
    if para1.page_number != para2.page_number:
        return False

    # 2. Category must be the same and non-None
    if para1.category is None or para2.category is None:
        return False
    if para1.category != para2.category:
        return False

    # 3. Only translatable categories
    translatable = config.translatable_categories
    if translatable is None:
        translatable = DEFAULT_TRANSLATABLE_RAW_CATEGORIES
    if para1.category not in translatable:
        return False

    # 4 & 5. Vertical adjacency and gap check
    # para2 should be below para1 (para2.y1 <= para1.y0)
    # Gap = para1.y0 - para2.y1 (should be >= 0 and within tolerance)
    gap = para1.block_bbox.y0 - para2.block_bbox.y1
    max_gap = para1.original_font_size * config.gap_tolerance

    if gap < 0:
        # para2 overlaps or is above para1, not adjacent
        return False
    if gap > max_gap:
        # Too much gap
        return False

    # 6. X overlap check
    x_overlap = _calc_x_overlap(para1.block_bbox, para2.block_bbox)
    if x_overlap < config.x_overlap_threshold:
        return False

    # 7. Font size similarity
    font_diff = abs(para1.original_font_size - para2.original_font_size)
    if font_diff > config.font_size_tolerance:
        return False

    return True


def _calc_x_overlap(bbox1: BBox, bbox2: BBox) -> float:
    """Calculate X-axis overlap ratio.

    The overlap ratio is calculated as:
    overlap_width / min(width1, width2)

    This approach ensures that a narrow paragraph fully contained
    within a wider one will have 100% overlap.

    Args:
        bbox1: First bounding box.
        bbox2: Second bounding box.

    Returns:
        Overlap ratio between 0.0 and 1.0.
    """
    overlap_x0 = max(bbox1.x0, bbox2.x0)
    overlap_x1 = min(bbox1.x1, bbox2.x1)

    if overlap_x0 >= overlap_x1:
        return 0.0

    overlap_width = overlap_x1 - overlap_x0
    min_width = min(bbox1.width, bbox2.width)

    if min_width <= 0:
        return 0.0

    return overlap_width / min_width


def _merge_two_paragraphs(para1: Paragraph, para2: Paragraph) -> Paragraph:
    """Merge two paragraphs into one.

    The merged paragraph uses:
    - ID from para1 (first paragraph)
    - Union of bounding boxes
    - Concatenated text with space separator
    - Sum of line counts
    - Font size, style, and other attributes from para1
    - Minimum of category confidence (conservative)

    Args:
        para1: First paragraph (used as base for attributes).
        para2: Second paragraph to merge.

    Returns:
        New merged Paragraph instance.
    """
    # Combine text with space
    merged_text = para1.text + " " + para2.text

    # Union of bounding boxes
    merged_bbox = para1.block_bbox.union(para2.block_bbox)

    # Sum line counts
    merged_line_count = para1.line_count + para2.line_count

    # Conservative category confidence (minimum)
    merged_confidence: float | None = None
    if para1.category_confidence is not None and para2.category_confidence is not None:
        merged_confidence = min(para1.category_confidence, para2.category_confidence)
    elif para1.category_confidence is not None:
        merged_confidence = para1.category_confidence
    elif para2.category_confidence is not None:
        merged_confidence = para2.category_confidence

    # Create new paragraph using dataclass replace
    return replace(
        para1,
        text=merged_text,
        block_bbox=merged_bbox,
        line_count=merged_line_count,
        category_confidence=merged_confidence,
        # Reset translation fields
        translated_text=None,
        adjusted_font_size=None,
    )
