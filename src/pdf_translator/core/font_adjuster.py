# SPDX-License-Identifier: Apache-2.0
"""Font size adjuster for translated text fitting.

This module provides the FontSizeAdjuster class for calculating
optimal font sizes to fit translated text within original bounding boxes.
"""

from __future__ import annotations

from .models import BBox

# Default constants
DEFAULT_MIN_FONT_SIZE = 6.0  # Minimum readable font size (pt)
DEFAULT_FONT_SIZE_DECREMENT = 0.1  # Step size for reduction (pt)

# CJK languages that use wider characters
CJK_LANGUAGES = frozenset({"ja", "zh", "ko", "zh-cn", "zh-tw", "zh-hans", "zh-hant"})


class FontSizeAdjuster:
    """Adjusts font size to fit translated text within bounding boxes.

    This class estimates the optimal font size for translated text
    to fit within the original TextObject's bounding box. It uses
    character width estimation based on target language.

    Note:
        Current implementation is width-based only (single line).
        PDFProcessor.insert_text_object() does not auto-wrap text,
        so we only check if text fits horizontally.
    """

    def __init__(
        self,
        min_font_size: float = DEFAULT_MIN_FONT_SIZE,
        font_size_decrement: float = DEFAULT_FONT_SIZE_DECREMENT,
    ) -> None:
        """Initialize FontSizeAdjuster.

        Args:
            min_font_size: Minimum font size in points (default: 6.0)
            font_size_decrement: Step size for font reduction (default: 0.1)
        """
        self._min_font_size = min_font_size
        self._font_size_decrement = font_size_decrement

    def calculate_font_size(
        self,
        text: str,
        bbox: BBox,
        original_font_size: float,
        target_lang: str,
    ) -> float:
        """Calculate font size that fits text within bbox width.

        Iteratively reduces font size until the estimated text width
        fits within the bbox, or reaches the minimum font size.

        Args:
            text: Translated text to fit
            bbox: Original TextObject's bounding box
            original_font_size: Original font size in points
            target_lang: Target language code (e.g., "ja", "en")

        Returns:
            Adjusted font size in points (>= min_font_size)
        """
        if not text:
            return original_font_size

        font_size = original_font_size

        while font_size >= self._min_font_size:
            char_width = self._estimate_char_width(font_size, target_lang)
            text_width = len(text) * char_width

            if text_width <= bbox.width:
                return font_size

            font_size -= self._font_size_decrement

        # If still doesn't fit, return minimum size
        return self._min_font_size

    def _estimate_char_width(self, font_size: float, target_lang: str) -> float:
        """Estimate average character width for a given font size and language.

        CJK characters are approximately square, while Latin characters
        are narrower on average.

        Args:
            font_size: Font size in points
            target_lang: Target language code

        Returns:
            Estimated average character width in points
        """
        # Normalize language code
        lang = target_lang.lower().split("-")[0] if target_lang else "en"

        if lang in CJK_LANGUAGES or target_lang.lower() in CJK_LANGUAGES:
            # CJK: Characters are approximately square
            # Using 0.9 as CJK fonts are typically slightly narrower than height
            return font_size * 0.9
        else:
            # Latin/other: Average character width is narrower
            # Using 0.55 based on typical proportional font metrics
            return font_size * 0.55

    @property
    def min_font_size(self) -> float:
        """Get minimum font size."""
        return self._min_font_size

    @property
    def font_size_decrement(self) -> float:
        """Get font size decrement step."""
        return self._font_size_decrement
