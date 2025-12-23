# SPDX-License-Identifier: Apache-2.0
"""Text layout engine for fitting text within bounding boxes.

This module provides text layout capabilities including:
- Accurate text width calculation using FPDFFont_GetGlyphWidth
- Text wrapping with word/character boundary support
- Font size adjustment to fit text within bounding boxes
- Basic Japanese line-break rules (kinsoku)
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import pypdfium2 as pdfium  # type: ignore[import-untyped]

from .models import BBox

# Characters that should not appear at the start of a line (Japanese kinsoku)
KINSOKU_NOT_AT_LINE_START: set[str] = {
    # Punctuation
    "。", "、", "．", "，", "：", "；", "！", "？",
    # Closing brackets
    "）", "」", "』", "】", "〉", "》", "〕", "］", "｝", ")",
    # Small kana
    "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "っ", "ゃ", "ゅ", "ょ", "ゎ",
    "ァ", "ィ", "ゥ", "ェ", "ォ", "ッ", "ャ", "ュ", "ョ", "ヮ",
    # Long vowel mark
    "ー",
    # Repetition marks
    "々", "ゝ", "ゞ", "ヽ", "ヾ",
}

# Characters that should not appear at the end of a line (Japanese kinsoku)
KINSOKU_NOT_AT_LINE_END: set[str] = {
    # Opening brackets
    "（", "「", "『", "【", "〈", "《", "〔", "［", "｛", "(",
}


@dataclass
class LayoutLine:
    """A single line of laid out text."""

    text: str
    width: float
    y_position: float


@dataclass
class LayoutResult:
    """Result of text layout calculation."""

    lines: list[LayoutLine]
    font_size: float
    total_height: float
    fits_in_bbox: bool


class TextLayoutEngine:
    """Engine for laying out text within bounding boxes.

    This class provides accurate text layout using PDFium's font metrics,
    including proper text wrapping and font size adjustment.
    """

    def __init__(
        self,
        min_font_size: float = 6.0,
        font_size_step: float = 0.5,
        line_height_factor: float = 1.2,
    ) -> None:
        """Initialize TextLayoutEngine.

        Args:
            min_font_size: Minimum allowed font size in points.
            font_size_step: Step size for font size reduction.
            line_height_factor: Multiplier for line height (1.0 = tight, 1.5 = loose).
        """
        self._min_font_size = min_font_size
        self._font_size_step = font_size_step
        self._line_height_factor = line_height_factor

    def calculate_text_width(
        self,
        text: str,
        font_handle: ctypes.c_void_p,
        font_size: float,
    ) -> float:
        """Calculate the width of text using font metrics.

        Args:
            text: Text to measure.
            font_handle: PDFium font handle (FPDF_FONT).
            font_size: Font size in points.

        Returns:
            Total width in points.
        """
        if not text:
            return 0.0

        total_width = 0.0
        width_out = ctypes.c_float()

        for char in text:
            result = pdfium.raw.FPDFFont_GetGlyphWidth(
                font_handle,
                ord(char),
                ctypes.c_float(font_size),
                ctypes.byref(width_out),
            )
            if result:
                total_width += width_out.value

        return total_width

    def get_line_height(
        self,
        font_handle: ctypes.c_void_p,
        font_size: float,
    ) -> float:
        """Calculate line height using font metrics.

        Args:
            font_handle: PDFium font handle (FPDF_FONT).
            font_size: Font size in points.

        Returns:
            Line height in points.
        """
        ascent = ctypes.c_float()
        descent = ctypes.c_float()

        pdfium.raw.FPDFFont_GetAscent(
            font_handle,
            ctypes.c_float(font_size),
            ctypes.byref(ascent),
        )
        pdfium.raw.FPDFFont_GetDescent(
            font_handle,
            ctypes.c_float(font_size),
            ctypes.byref(descent),
        )

        # descent is negative, so we subtract it (add absolute value)
        base_height = ascent.value - descent.value
        return base_height * self._line_height_factor

    def get_ascent(
        self,
        font_handle: ctypes.c_void_p,
        font_size: float,
    ) -> float:
        """Get font ascent (distance from baseline to top).

        Args:
            font_handle: PDFium font handle (FPDF_FONT).
            font_size: Font size in points.

        Returns:
            Ascent in points.
        """
        ascent = ctypes.c_float()
        pdfium.raw.FPDFFont_GetAscent(
            font_handle,
            ctypes.c_float(font_size),
            ctypes.byref(ascent),
        )
        return ascent.value

    def _is_cjk_char(self, char: str) -> bool:
        """Check if a character is CJK (Chinese, Japanese, Korean).

        Args:
            char: Single character to check.

        Returns:
            True if the character is CJK.
        """
        code = ord(char)
        return (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3040 <= code <= 0x309F  # Hiragana
            or 0x30A0 <= code <= 0x30FF  # Katakana
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
            or 0x3000 <= code <= 0x303F  # CJK Punctuation
            or 0xFF00 <= code <= 0xFFEF  # Fullwidth Forms
        )

    def _find_break_point(
        self,
        text: str,
        max_chars: int,
        font_handle: ctypes.c_void_p,
        font_size: float,
        max_width: float,
    ) -> int:
        """Find the best break point for text wrapping.

        This method finds where to break text considering:
        - Maximum width constraint
        - Word boundaries for non-CJK text
        - Character boundaries for CJK text
        - Basic kinsoku (Japanese line-break rules)

        Args:
            text: Text to wrap.
            max_chars: Maximum characters to consider.
            font_handle: PDFium font handle.
            font_size: Font size in points.
            max_width: Maximum line width in points.

        Returns:
            Number of characters to include in this line.
        """
        if not text:
            return 0

        # Find the point where text exceeds max_width
        current_width = 0.0
        width_out = ctypes.c_float()
        break_point = 0

        for i, char in enumerate(text[:max_chars]):
            result = pdfium.raw.FPDFFont_GetGlyphWidth(
                font_handle,
                ord(char),
                ctypes.c_float(font_size),
                ctypes.byref(width_out),
            )
            char_width = width_out.value if result else 0.0

            if current_width + char_width > max_width:
                break_point = i
                break

            current_width += char_width
            break_point = i + 1

        if break_point == 0:
            # Even single character doesn't fit - force at least one char
            return 1

        if break_point >= len(text[:max_chars]):
            # All text fits
            return break_point

        # Apply kinsoku rules
        break_point = self._apply_kinsoku(text, break_point)

        # For non-CJK text, try to break at word boundary
        if break_point > 0 and not self._is_cjk_char(text[break_point - 1]):
            # Look backwards for a space
            for j in range(break_point - 1, max(0, break_point - 20), -1):
                if text[j] == " ":
                    return j + 1  # Include the space in the line
            # No space found, break at character boundary

        return break_point

    def _apply_kinsoku(self, text: str, break_point: int) -> int:
        """Apply Japanese line-break rules (kinsoku).

        Args:
            text: Full text being wrapped.
            break_point: Proposed break point.

        Returns:
            Adjusted break point.
        """
        if break_point <= 0 or break_point >= len(text):
            return break_point

        # Check if next character should not start a line
        next_char = text[break_point] if break_point < len(text) else ""
        if next_char in KINSOKU_NOT_AT_LINE_START:
            # Move break point forward (include this char in current line)
            # But we need to be careful not to exceed width too much
            return break_point + 1

        # Check if current last character should not end a line
        prev_char = text[break_point - 1] if break_point > 0 else ""
        if prev_char in KINSOKU_NOT_AT_LINE_END:
            # Move break point backward
            if break_point > 1:
                return break_point - 1

        return break_point

    def wrap_text(
        self,
        text: str,
        max_width: float,
        font_handle: ctypes.c_void_p,
        font_size: float,
    ) -> list[str]:
        """Wrap text to fit within a maximum width.

        Args:
            text: Text to wrap.
            max_width: Maximum line width in points.
            font_handle: PDFium font handle.
            font_size: Font size in points.

        Returns:
            List of lines.
        """
        if not text:
            return []

        lines: list[str] = []
        remaining = text.strip()

        while remaining:
            # Find break point
            break_point = self._find_break_point(
                remaining,
                len(remaining),
                font_handle,
                font_size,
                max_width,
            )

            if break_point <= 0:
                break_point = 1  # Force at least one character

            line = remaining[:break_point].rstrip()
            if line:
                lines.append(line)

            remaining = remaining[break_point:].lstrip()

        return lines

    def fit_text_in_bbox(
        self,
        text: str,
        bbox: BBox,
        font_handle: ctypes.c_void_p,
        initial_font_size: float,
    ) -> LayoutResult:
        """Fit text within a bounding box, adjusting font size if necessary.

        Args:
            text: Text to layout.
            bbox: Bounding box to fit text into.
            font_handle: PDFium font handle.
            initial_font_size: Starting font size in points.

        Returns:
            LayoutResult with lines, final font size, and fit status.
        """
        if not text:
            return LayoutResult(
                lines=[],
                font_size=initial_font_size,
                total_height=0.0,
                fits_in_bbox=True,
            )

        font_size = initial_font_size
        bbox_width = bbox.width
        bbox_height = bbox.height

        while font_size >= self._min_font_size:
            # Wrap text at current font size
            wrapped_lines = self.wrap_text(text, bbox_width, font_handle, font_size)

            if not wrapped_lines:
                break

            # Calculate total height
            line_height = self.get_line_height(font_handle, font_size)
            total_height = line_height * len(wrapped_lines)

            if total_height <= bbox_height:
                # Text fits! Calculate y positions (top to bottom)
                ascent = self.get_ascent(font_handle, font_size)
                lines: list[LayoutLine] = []

                # Start from top of bbox, offset by ascent for first line
                y = bbox.y1 - ascent

                for line_text in wrapped_lines:
                    line_width = self.calculate_text_width(
                        line_text, font_handle, font_size
                    )
                    lines.append(LayoutLine(
                        text=line_text,
                        width=line_width,
                        y_position=y,
                    ))
                    y -= line_height

                return LayoutResult(
                    lines=lines,
                    font_size=font_size,
                    total_height=total_height,
                    fits_in_bbox=True,
                )

            # Reduce font size and try again
            font_size = round(font_size - self._font_size_step, 2)

        # Could not fit even at minimum font size
        # Return best effort at minimum size
        wrapped_lines = self.wrap_text(text, bbox_width, font_handle, self._min_font_size)
        line_height = self.get_line_height(font_handle, self._min_font_size)
        ascent = self.get_ascent(font_handle, self._min_font_size)

        lines = []
        y = bbox.y1 - ascent

        for line_text in wrapped_lines:
            line_width = self.calculate_text_width(
                line_text, font_handle, self._min_font_size
            )
            lines.append(LayoutLine(
                text=line_text,
                width=line_width,
                y_position=y,
            ))
            y -= line_height

        return LayoutResult(
            lines=lines,
            font_size=self._min_font_size,
            total_height=line_height * len(wrapped_lines),
            fits_in_bbox=False,
        )
