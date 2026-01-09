# SPDX-License-Identifier: Apache-2.0
"""Tests for TextLayoutEngine."""

from __future__ import annotations

import ctypes

import pypdfium2 as pdfium
import pytest

from pdf_translator.core.models import BBox
from pdf_translator.core.text_layout import (
    KINSOKU_NOT_AT_LINE_END,
    KINSOKU_NOT_AT_LINE_START,
    PARAGRAPH_BREAK_MARKER,
    TextLayoutEngine,
)


@pytest.fixture
def layout_engine() -> TextLayoutEngine:
    """Create a TextLayoutEngine instance."""
    return TextLayoutEngine(
        min_font_size=6.0,
        font_size_step=0.5,
        line_height_factor=1.2,
    )


@pytest.fixture
def pdf_doc() -> pdfium.PdfDocument:
    """Create a new PDF document for testing."""
    return pdfium.PdfDocument.new()


@pytest.fixture
def helvetica_font(pdf_doc: pdfium.PdfDocument) -> ctypes.c_void_p:
    """Load Helvetica font."""
    return pdfium.raw.FPDFText_LoadStandardFont(pdf_doc, b"Helvetica")


class TestTextLayoutEngine:
    """Tests for TextLayoutEngine class."""

    def test_calculate_text_width_empty(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Empty text should have zero width."""
        width = layout_engine.calculate_text_width("", helvetica_font, 12.0)
        assert width == 0.0

    def test_calculate_text_width_single_char(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Single character should have non-zero width."""
        width = layout_engine.calculate_text_width("A", helvetica_font, 12.0)
        assert width > 0.0

    def test_calculate_text_width_proportional(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Different characters should have different widths."""
        width_i = layout_engine.calculate_text_width("i", helvetica_font, 12.0)
        width_w = layout_engine.calculate_text_width("W", helvetica_font, 12.0)
        # 'W' should be wider than 'i'
        assert width_w > width_i

    def test_calculate_text_width_additive(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Width of 'AB' should equal width of 'A' + width of 'B'."""
        width_a = layout_engine.calculate_text_width("A", helvetica_font, 12.0)
        width_b = layout_engine.calculate_text_width("B", helvetica_font, 12.0)
        width_ab = layout_engine.calculate_text_width("AB", helvetica_font, 12.0)
        assert abs(width_ab - (width_a + width_b)) < 0.01

    def test_get_line_height(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Line height should be positive and increase with font size."""
        height_12 = layout_engine.get_line_height(helvetica_font, 12.0)
        height_24 = layout_engine.get_line_height(helvetica_font, 24.0)
        assert height_12 > 0
        assert height_24 > height_12

    def test_get_ascent(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Ascent should be positive."""
        ascent = layout_engine.get_ascent(helvetica_font, 12.0)
        assert ascent > 0

    def test_is_cjk_char(self, layout_engine: TextLayoutEngine) -> None:
        """Test CJK character detection."""
        # Japanese hiragana
        assert layout_engine._is_cjk_char("あ")
        # Japanese katakana
        assert layout_engine._is_cjk_char("ア")
        # CJK ideograph
        assert layout_engine._is_cjk_char("漢")
        # Korean
        assert layout_engine._is_cjk_char("한")
        # ASCII
        assert not layout_engine._is_cjk_char("A")
        assert not layout_engine._is_cjk_char("1")

    def test_wrap_text_short(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Short text should not wrap."""
        lines = layout_engine.wrap_text("Hello", 200.0, helvetica_font, 12.0)
        assert len(lines) == 1
        assert lines[0] == "Hello"

    def test_wrap_text_long(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Long text should wrap into multiple lines."""
        text = "This is a long sentence that should wrap into multiple lines."
        lines = layout_engine.wrap_text(text, 100.0, helvetica_font, 12.0)
        assert len(lines) > 1
        # All words should be preserved
        joined = " ".join(lines)
        assert "long" in joined
        assert "multiple" in joined

    def test_wrap_text_word_boundary(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Text should wrap at word boundaries."""
        text = "Hello World Test"
        lines = layout_engine.wrap_text(text, 80.0, helvetica_font, 12.0)
        # Lines should not have trailing spaces
        for line in lines:
            assert not line.endswith(" ")

    def test_fit_text_in_bbox_fits(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Short text should fit without font size reduction."""
        bbox = BBox(x0=0, y0=0, x1=200, y1=100)
        result = layout_engine.fit_text_in_bbox("Hello", bbox, helvetica_font, 12.0)

        assert result.fits_in_bbox
        assert result.font_size == 12.0
        assert len(result.lines) == 1
        assert result.lines[0].text == "Hello"

    def test_fit_text_in_bbox_reduces_font(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Long text should trigger font size reduction."""
        bbox = BBox(x0=0, y0=0, x1=100, y1=30)  # Small bbox
        long_text = "This is a very long text that will not fit in the small box."
        result = layout_engine.fit_text_in_bbox(long_text, bbox, helvetica_font, 12.0)

        # Font size should be reduced
        assert result.font_size < 12.0

    def test_fit_text_in_bbox_y_positions(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Lines should be positioned from top to bottom."""
        bbox = BBox(x0=0, y0=0, x1=100, y1=200)
        text = "Line one Line two Line three"
        result = layout_engine.fit_text_in_bbox(text, bbox, helvetica_font, 12.0)

        if len(result.lines) > 1:
            # First line should have higher y than second line (PDF coords)
            assert result.lines[0].y_position > result.lines[1].y_position

    def test_fit_text_in_bbox_empty(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Empty text should return empty result."""
        bbox = BBox(x0=0, y0=0, x1=100, y1=100)
        result = layout_engine.fit_text_in_bbox("", bbox, helvetica_font, 12.0)

        assert result.fits_in_bbox
        assert len(result.lines) == 0


class TestKinsokuRules:
    """Tests for Japanese line-break rules."""

    def test_kinsoku_punctuation_not_at_start(self) -> None:
        """Punctuation should be in KINSOKU_NOT_AT_LINE_START."""
        assert "。" in KINSOKU_NOT_AT_LINE_START
        assert "、" in KINSOKU_NOT_AT_LINE_START
        assert "！" in KINSOKU_NOT_AT_LINE_START
        assert "？" in KINSOKU_NOT_AT_LINE_START

    def test_kinsoku_closing_brackets_not_at_start(self) -> None:
        """Closing brackets should be in KINSOKU_NOT_AT_LINE_START."""
        assert "）" in KINSOKU_NOT_AT_LINE_START
        assert "」" in KINSOKU_NOT_AT_LINE_START
        assert "』" in KINSOKU_NOT_AT_LINE_START

    def test_kinsoku_opening_brackets_not_at_end(self) -> None:
        """Opening brackets should be in KINSOKU_NOT_AT_LINE_END."""
        assert "（" in KINSOKU_NOT_AT_LINE_END
        assert "「" in KINSOKU_NOT_AT_LINE_END
        assert "『" in KINSOKU_NOT_AT_LINE_END

    def test_kinsoku_small_kana_not_at_start(self) -> None:
        """Small kana should be in KINSOKU_NOT_AT_LINE_START."""
        assert "っ" in KINSOKU_NOT_AT_LINE_START
        assert "ゃ" in KINSOKU_NOT_AT_LINE_START
        assert "ッ" in KINSOKU_NOT_AT_LINE_START


class TestParagraphBreaks:
    """Tests for paragraph break handling in TextLayoutEngine.

    Note: `\\n\\n` creates paragraph breaks (with spacing),
    `\\n` creates line breaks (no extra spacing).
    """

    def test_wrap_text_with_single_newline(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Single newline is a line break, not paragraph break."""
        text = "First line.\nSecond line."
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        # Should have 2 lines, no paragraph break marker
        assert len(lines) == 2
        assert lines[0] == "First line."
        assert lines[1] == "Second line."
        assert PARAGRAPH_BREAK_MARKER not in lines

    def test_wrap_text_with_double_newline(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Double newline creates paragraph break marker."""
        text = "First paragraph.\n\nSecond paragraph."
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        # Should have: text, marker, text
        assert len(lines) == 3
        assert lines[0] == "First paragraph."
        assert lines[1] == PARAGRAPH_BREAK_MARKER
        assert lines[2] == "Second paragraph."

    def test_wrap_text_multiple_paragraphs(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Multiple double newlines create multiple paragraph breaks."""
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        # Should have: text, marker, text, marker, text
        assert len(lines) == 5
        markers = [line for line in lines if line == PARAGRAPH_BREAK_MARKER]
        assert len(markers) == 2

    def test_wrap_text_no_trailing_marker(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Trailing newlines should not result in trailing markers."""
        text = "First paragraph.\n\n"
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        # Should have just the text, no trailing marker
        assert len(lines) == 1
        assert lines[0] == "First paragraph."

    def test_wrap_text_mixed_breaks(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Mixed line breaks and paragraph breaks."""
        text = "Line 1\nLine 2\n\nPara 2 Line 1\nPara 2 Line 2"
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        # Should have 5 items: 2 lines, 1 marker, 2 lines
        assert len(lines) == 5
        assert lines[0] == "Line 1"
        assert lines[1] == "Line 2"
        assert lines[2] == PARAGRAPH_BREAK_MARKER
        assert lines[3] == "Para 2 Line 1"
        assert lines[4] == "Para 2 Line 2"

    def test_wrap_text_long_segments_with_paragraphs(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Long segments should wrap and have markers between paragraphs."""
        text = (
            "This is a long first paragraph that should wrap.\n\n"
            "This is a long second paragraph that should also wrap."
        )
        lines = layout_engine.wrap_text(text, 100.0, helvetica_font, 12.0)

        # Should have multiple wrapped lines with markers between paragraphs
        assert PARAGRAPH_BREAK_MARKER in lines
        # Text from both paragraphs should be present
        full_text = " ".join([line for line in lines if line != PARAGRAPH_BREAK_MARKER])
        assert "first" in full_text
        assert "second" in full_text

    def test_fit_text_with_paragraph_breaks(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """fit_text_in_bbox should handle paragraph breaks correctly."""
        bbox = BBox(x0=0, y0=0, x1=200, y1=100)
        text = "First paragraph.\n\nSecond paragraph."
        result = layout_engine.fit_text_in_bbox(text, bbox, helvetica_font, 12.0)

        # Should have 2 text lines (markers are not included in result.lines)
        assert len(result.lines) == 2
        assert result.lines[0].text == "First paragraph."
        assert result.lines[1].text == "Second paragraph."

    def test_fit_text_with_line_breaks(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """fit_text_in_bbox should handle line breaks without extra spacing."""
        bbox = BBox(x0=0, y0=0, x1=200, y1=100)
        text = "First line.\nSecond line."
        result = layout_engine.fit_text_in_bbox(text, bbox, helvetica_font, 12.0)

        # Should have 2 text lines
        assert len(result.lines) == 2
        assert result.lines[0].text == "First line."
        assert result.lines[1].text == "Second line."

    def test_fit_text_paragraph_spacing(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Paragraph breaks should have additional vertical spacing."""
        bbox = BBox(x0=0, y0=0, x1=200, y1=200)

        # Text with line break (no extra spacing)
        text_line_break = "First line.\nSecond line."
        result_line_break = layout_engine.fit_text_in_bbox(
            text_line_break, bbox, helvetica_font, 12.0
        )

        # Text with paragraph break (extra spacing)
        text_para_break = "First line.\n\nSecond line."
        result_para_break = layout_engine.fit_text_in_bbox(
            text_para_break, bbox, helvetica_font, 12.0
        )

        # Both should have 2 lines of text
        assert len(result_line_break.lines) == 2
        assert len(result_para_break.lines) == 2

        # The version with paragraph break should have greater total height
        assert result_para_break.total_height > result_line_break.total_height

    def test_paragraph_spacing_factor(
        self,
        pdf_doc: pdfium.PdfDocument,
    ) -> None:
        """Paragraph spacing factor should affect total height."""
        font = pdfium.raw.FPDFText_LoadStandardFont(pdf_doc, b"Helvetica")
        bbox = BBox(x0=0, y0=0, x1=200, y1=200)
        text = "Para 1.\n\nPara 2."  # Use double newline for paragraph break

        # Engine with no paragraph spacing
        engine_no_space = TextLayoutEngine(paragraph_spacing_factor=0.0)
        result_no_space = engine_no_space.fit_text_in_bbox(text, bbox, font, 12.0)

        # Engine with paragraph spacing
        engine_with_space = TextLayoutEngine(paragraph_spacing_factor=0.5)
        result_with_space = engine_with_space.fit_text_in_bbox(text, bbox, font, 12.0)

        # With spacing should have greater total height
        assert result_with_space.total_height > result_no_space.total_height
