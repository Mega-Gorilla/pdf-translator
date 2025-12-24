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

# Sentence-ending punctuation for Japanese and English
SENTENCE_ENDING_PUNCTUATION: frozenset[str] = frozenset(
    {"。", ".", "!", "?", "！", "？"}
)


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
    """

    gap_tolerance: float = 1.5
    x_overlap_threshold: float = 0.7
    font_size_tolerance: float = 1.0
    translatable_categories: frozenset[str] | None = None


def merge_adjacent_paragraphs(
    paragraphs: Sequence[Paragraph],
    config: MergeConfig | None = None,
) -> list[Paragraph]:
    """Merge adjacent paragraphs with the same category.

    This function processes paragraphs page by page, merging those that
    meet the merge criteria. Paragraphs are processed top-to-bottom
    (by y1 descending in PDF coordinates).

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

        # Sort by y1 descending (top to bottom in PDF coordinates)
        sorted_paras = sorted(
            page_paragraphs, key=lambda p: p.block_bbox.y1, reverse=True
        )

        if not sorted_paras:
            continue

        # Initialize with first paragraph
        merged_page: list[Paragraph] = [sorted_paras[0]]

        # Process remaining paragraphs
        for current in sorted_paras[1:]:
            last = merged_page[-1]

            if _can_merge(last, current, config):
                # Replace last with merged paragraph
                merged_page[-1] = _merge_two_paragraphs(last, current)
            else:
                merged_page.append(current)

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
    8. para1 doesn't end with sentence-ending punctuation

    Note: Alignment check was intentionally removed because:
    - _estimate_alignment() is unreliable for short paragraphs
    - Category and font size checks already distinguish headers from body text
    - "left" vs "justify" distinction causes false negatives in continuous text

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

    # 8. First paragraph should not end with sentence-ending punctuation
    if _ends_with_sentence_punctuation(para1.text):
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


def _ends_with_sentence_punctuation(text: str) -> bool:
    """Check if text ends with sentence-ending punctuation.

    Args:
        text: Text to check.

    Returns:
        True if text ends with sentence-ending punctuation.
    """
    stripped = text.rstrip()
    if not stripped:
        return False
    return stripped[-1] in SENTENCE_ENDING_PUNCTUATION
