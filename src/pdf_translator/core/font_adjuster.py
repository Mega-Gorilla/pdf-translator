# SPDX-License-Identifier: Apache-2.0
"""Font size adjustment utilities."""

from __future__ import annotations

from pdf_translator.core.models import BBox


class FontSizeAdjuster:
    """Adjust font size to fit translated text into a bounding box."""

    def __init__(
        self,
        min_font_size: float = 6.0,
        font_size_decrement: float = 0.1,
    ) -> None:
        """Initialize FontSizeAdjuster.

        Args:
            min_font_size: Minimum font size in points.
            font_size_decrement: Decrement step in points.
        """
        self._min_font_size = float(min_font_size)
        self._font_size_decrement = float(font_size_decrement)

    def calculate_font_size(
        self,
        text: str,
        bbox: BBox,
        original_font_size: float,
        target_lang: str,
    ) -> float:
        """Calculate font size that fits text within bbox width.

        Args:
            text: Translated text.
            bbox: Bounding box for placement.
            original_font_size: Original font size.
            target_lang: Target language code (used for width estimate).

        Returns:
            Adjusted font size (>= min_font_size).
        """
        if not text:
            return max(float(original_font_size), self._min_font_size)

        font_size = max(float(original_font_size), self._min_font_size)
        target = target_lang.lower()
        while font_size >= self._min_font_size:
            char_width = self._estimate_char_width(font_size, target)
            text_width = len(text) * char_width
            if text_width <= bbox.width:
                return font_size
            font_size = round(font_size - self._font_size_decrement, 2)

        return self._min_font_size

    @staticmethod
    def _estimate_char_width(font_size: float, target_lang: str) -> float:
        if target_lang in {"ja", "zh", "ko"}:
            return font_size * 0.9
        return font_size * 0.55
