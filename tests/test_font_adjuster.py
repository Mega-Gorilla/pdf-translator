# SPDX-License-Identifier: Apache-2.0
"""Tests for FontSizeAdjuster."""

from pdf_translator.core.font_adjuster import FontSizeAdjuster
from pdf_translator.core.models import BBox


def test_text_fits_original_size():
    adjuster = FontSizeAdjuster(min_font_size=6.0, font_size_decrement=1.0)
    bbox = BBox(0, 0, 100, 20)
    size = adjuster.calculate_font_size("hello", bbox, 12.0, "en")
    assert size == 12.0


def test_text_requires_reduction():
    adjuster = FontSizeAdjuster(min_font_size=6.0, font_size_decrement=1.0)
    bbox = BBox(0, 0, 100, 20)
    size = adjuster.calculate_font_size("a" * 20, bbox, 12.0, "en")
    assert size == 9.0


def test_minimum_font_size_limit():
    adjuster = FontSizeAdjuster(min_font_size=6.0, font_size_decrement=1.0)
    bbox = BBox(0, 0, 40, 20)
    size = adjuster.calculate_font_size("a" * 100, bbox, 12.0, "en")
    assert size == 6.0


def test_cjk_character_width():
    adjuster = FontSizeAdjuster(min_font_size=6.0, font_size_decrement=1.0)
    bbox = BBox(0, 0, 90, 20)
    size = adjuster.calculate_font_size("あ" * 10, bbox, 12.0, "ja")
    assert size == 10.0
