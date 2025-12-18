# SPDX-License-Identifier: Apache-2.0
"""Text merger for reading order sorting and filtering.

This module provides the TextMerger class for sorting TextObjects
in reading order (top-to-bottom, left-to-right) and filtering
translatable objects.
"""

from __future__ import annotations

from .models import TRANSLATABLE_CATEGORIES, ProjectCategory, TextObject

# Sentence-ending punctuation marks
SENTENCE_TERMINALS: frozenset[str] = frozenset({".", "!", "?", ":", ";"})
CJK_TERMINALS: frozenset[str] = frozenset({"。", "！", "？", "：", "；"})
ALL_TERMINALS: frozenset[str] = SENTENCE_TERMINALS | CJK_TERMINALS


class TextMerger:
    """Sorts TextObjects in reading order and filters translatable ones.

    This class handles:
    1. Filtering TextObjects by category (only TRANSLATABLE_CATEGORIES)
    2. Clustering objects into lines based on y-coordinate tolerance
    3. Sorting within lines by x-coordinate (left-to-right)
    4. Sorting lines by y-coordinate (top-to-bottom)

    Note:
        In v1, this class only performs sorting (no actual merging).
        The "merge" name is forward-looking for v2 cross-block translation.
    """

    def __init__(
        self,
        line_y_tolerance: float = 3.0,
        merge_threshold_x: float = 20.0,
        merge_threshold_y: float = 5.0,
        x_overlap_ratio: float = 0.5,
    ) -> None:
        """Initialize TextMerger.

        Args:
            line_y_tolerance: Y-coordinate tolerance for same-line detection (pt)
            merge_threshold_x: X gap threshold for same-line merge (pt) [v2]
            merge_threshold_y: Y gap threshold for next-line merge (pt) [v2]
            x_overlap_ratio: X overlap ratio required for next-line merge [v2]

        Note:
            In v1, only `line_y_tolerance` is used for line clustering.
            Other parameters are defined for v2 cross-block merging.
        """
        self._line_y_tolerance = line_y_tolerance
        self._merge_threshold_x = merge_threshold_x
        self._merge_threshold_y = merge_threshold_y
        self._x_overlap_ratio = x_overlap_ratio

    def merge(
        self,
        text_objects: list[TextObject],
        categories: dict[str, ProjectCategory],
    ) -> list[TextObject]:
        """Filter and sort TextObjects in reading order.

        Filters TextObjects to only those in TRANSLATABLE_CATEGORIES,
        then sorts them in reading order (top-to-bottom, left-to-right).

        Args:
            text_objects: All TextObjects from a page
            categories: Mapping of TextObject.id -> ProjectCategory

        Returns:
            Reading-order sorted list of translatable TextObjects
        """
        # Step 1: Filter to translatable categories
        translatable = [
            obj
            for obj in text_objects
            if categories.get(obj.id) in TRANSLATABLE_CATEGORIES
        ]

        if not translatable:
            return []

        # Step 2: Cluster by line and sort
        lines = self._cluster_by_line(translatable)

        # Step 3: Flatten to reading-order list
        result: list[TextObject] = []
        for line in lines:
            result.extend(line)

        return result

    def _cluster_by_line(
        self,
        text_objects: list[TextObject],
    ) -> list[list[TextObject]]:
        """Cluster TextObjects into lines based on y-coordinate.

        Groups objects that have similar y1 (top) coordinates within
        the tolerance threshold. Each line is sorted by x0 (left-to-right).
        Lines are sorted by y1 (top-to-bottom, descending in PDF coords).

        Args:
            text_objects: TextObjects to cluster

        Returns:
            List of lines, each line is a list of TextObjects sorted by x0
        """
        if not text_objects:
            return []

        # Sort by y1 (top coordinate) descending (top of page first)
        # In PDF coordinates, higher y1 = higher on page
        sorted_objs = sorted(text_objects, key=lambda o: -o.bbox.y1)

        lines: list[list[TextObject]] = []
        current_line: list[TextObject] = [sorted_objs[0]]
        current_y = sorted_objs[0].bbox.y1

        for obj in sorted_objs[1:]:
            if abs(obj.bbox.y1 - current_y) <= self._line_y_tolerance:
                # Same line
                current_line.append(obj)
            else:
                # New line - sort current line by x0 and save
                current_line.sort(key=lambda o: o.bbox.x0)
                lines.append(current_line)
                current_line = [obj]
                current_y = obj.bbox.y1

        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda o: o.bbox.x0)
            lines.append(current_line)

        return lines

    def is_sentence_end(self, text: str) -> bool:
        """Check if text ends with a sentence-ending punctuation.

        Supports both Western and CJK punctuation marks.

        Args:
            text: Text to check

        Returns:
            True if text ends with sentence-ending punctuation
        """
        text = text.rstrip()
        if not text:
            return False
        return text[-1] in ALL_TERMINALS

    # v2 methods (not used in v1, but defined for forward compatibility)

    def _should_merge_same_line(
        self,
        current: TextObject,
        next_obj: TextObject,
    ) -> bool:
        """Check if two objects on the same line should be merged.

        Note: Not used in v1 (1:1 translation). Defined for v2.

        Args:
            current: Current TextObject
            next_obj: Next TextObject (to the right)

        Returns:
            True if objects should be merged
        """
        # Don't merge if current ends with sentence punctuation
        if self.is_sentence_end(current.text):
            return False

        # Check x gap
        x_gap = next_obj.bbox.x0 - current.bbox.x1
        return x_gap <= self._merge_threshold_x

    def _should_merge_next_line(
        self,
        current: TextObject,
        next_obj: TextObject,
    ) -> bool:
        """Check if an object should merge with the next line.

        Note: Not used in v1 (1:1 translation). Defined for v2.

        Args:
            current: Current TextObject (end of current line)
            next_obj: Next TextObject (start of next line)

        Returns:
            True if objects should be merged across lines
        """
        # Don't merge if current ends with sentence punctuation
        if self.is_sentence_end(current.text):
            return False

        # Check y gap
        y_gap = current.bbox.y0 - next_obj.bbox.y1
        if y_gap > self._merge_threshold_y:
            return False

        # Check x overlap (prevents multi-column mis-merging)
        x_overlap = min(current.bbox.x1, next_obj.bbox.x1) - max(
            current.bbox.x0, next_obj.bbox.x0
        )
        min_width = min(current.bbox.width, next_obj.bbox.width)

        if min_width <= 0:
            return False

        return x_overlap >= min_width * self._x_overlap_ratio
