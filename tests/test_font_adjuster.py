# SPDX-License-Identifier: Apache-2.0
"""Tests for FontSizeAdjuster class."""

import pytest

from pdf_translator.core.font_adjuster import (
    CJK_LANGUAGES,
    DEFAULT_FONT_SIZE_DECREMENT,
    DEFAULT_MIN_FONT_SIZE,
    FontSizeAdjuster,
)
from pdf_translator.core.models import BBox


class TestFontSizeAdjusterInit:
    """Tests for FontSizeAdjuster initialization."""

    def test_default_values(self) -> None:
        """Test default parameter values."""
        adjuster = FontSizeAdjuster()
        assert adjuster.min_font_size == DEFAULT_MIN_FONT_SIZE
        assert adjuster.font_size_decrement == DEFAULT_FONT_SIZE_DECREMENT

    def test_custom_values(self) -> None:
        """Test custom parameter values."""
        adjuster = FontSizeAdjuster(min_font_size=8.0, font_size_decrement=0.5)
        assert adjuster.min_font_size == 8.0
        assert adjuster.font_size_decrement == 0.5


class TestTextFitsOriginalSize:
    """Tests for text that fits at original size."""

    def test_short_text_fits(self) -> None:
        """Short text should use original font size."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=200, y1=20)  # 200pt wide
        original_size = 12.0
        text = "Hello"  # 5 chars * 12 * 0.55 = 33pt < 200pt

        result = adjuster.calculate_font_size(text, bbox, original_size, "en")

        assert result == original_size

    def test_exact_fit(self) -> None:
        """Text that exactly fits should use original size."""
        adjuster = FontSizeAdjuster()

        # 10 chars * 12 * 0.55 = 66pt, bbox = 66pt
        bbox = BBox(x0=0, y0=0, x1=66, y1=20)
        original_size = 12.0
        text = "0123456789"  # 10 chars

        result = adjuster.calculate_font_size(text, bbox, original_size, "en")

        assert result == original_size


class TestTextRequiresReduction:
    """Tests for text that requires font size reduction."""

    def test_long_text_reduced(self) -> None:
        """Long text should have reduced font size."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=50, y1=20)  # 50pt wide
        original_size = 12.0
        text = "This is a longer text"  # 21 chars

        result = adjuster.calculate_font_size(text, bbox, original_size, "en")

        # Should be reduced from 12.0
        assert result < original_size
        # But still >= min
        assert result >= DEFAULT_MIN_FONT_SIZE

    def test_reduction_amount_reasonable(self) -> None:
        """Reduction should be proportional to overflow."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=100, y1=20)
        original_size = 12.0

        # Text that's about 2x too wide needs ~50% reduction
        # 20 chars * 12 * 0.55 = 132pt > 100pt
        text = "12345678901234567890"

        result = adjuster.calculate_font_size(text, bbox, original_size, "en")

        # Should fit: result * 0.55 * 20 <= 100
        # result <= 100 / (0.55 * 20) = 9.09
        assert result <= 9.1


class TestMinimumFontSizeLimit:
    """Tests for minimum font size limit."""

    def test_very_long_text_hits_minimum(self) -> None:
        """Very long text should hit minimum font size."""
        adjuster = FontSizeAdjuster(min_font_size=6.0)

        bbox = BBox(x0=0, y0=0, x1=20, y1=10)  # Very narrow
        original_size = 12.0
        text = "This is a very very very long text that cannot fit"

        result = adjuster.calculate_font_size(text, bbox, original_size, "en")

        assert result == 6.0

    def test_custom_minimum_respected(self) -> None:
        """Custom minimum font size should be respected."""
        adjuster = FontSizeAdjuster(min_font_size=8.0)

        bbox = BBox(x0=0, y0=0, x1=10, y1=10)  # Tiny
        original_size = 12.0
        text = "Cannot fit this text"

        result = adjuster.calculate_font_size(text, bbox, original_size, "en")

        assert result == 8.0


class TestCJKCharacterWidth:
    """Tests for CJK character width estimation."""

    def test_japanese_wider_than_latin(self) -> None:
        """Japanese text should use wider character width."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=100, y1=20)
        original_size = 12.0

        # Same character count
        latin_text = "ABCDEFGHIJ"  # 10 chars
        japanese_text = "あいうえおかきくけこ"  # 10 chars

        latin_result = adjuster.calculate_font_size(
            latin_text, bbox, original_size, "en"
        )
        japanese_result = adjuster.calculate_font_size(
            japanese_text, bbox, original_size, "ja"
        )

        # Japanese needs smaller font for same char count due to wider chars
        assert japanese_result < latin_result

    def test_chinese_uses_cjk_width(self) -> None:
        """Chinese text should use CJK character width."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=100, y1=20)
        original_size = 12.0
        text = "1234567890"

        zh_result = adjuster.calculate_font_size(text, bbox, original_size, "zh")
        en_result = adjuster.calculate_font_size(text, bbox, original_size, "en")

        # Chinese uses wider chars
        assert zh_result < en_result

    def test_korean_uses_cjk_width(self) -> None:
        """Korean text should use CJK character width."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=100, y1=20)
        original_size = 12.0
        text = "1234567890"

        ko_result = adjuster.calculate_font_size(text, bbox, original_size, "ko")
        en_result = adjuster.calculate_font_size(text, bbox, original_size, "en")

        # Korean uses wider chars
        assert ko_result < en_result


class TestLatinCharacterWidth:
    """Tests for Latin character width estimation."""

    def test_english_uses_narrow_width(self) -> None:
        """English text should use narrow character width."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=66, y1=20)  # 10 * 12 * 0.55 = 66
        original_size = 12.0
        text = "1234567890"

        result = adjuster.calculate_font_size(text, bbox, original_size, "en")

        # Should fit at original size
        assert result == original_size

    def test_various_latin_languages(self) -> None:
        """Various Latin-script languages should use narrow width."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=66, y1=20)
        original_size = 12.0
        text = "1234567890"

        for lang in ["en", "de", "fr", "es", "it", "pt"]:
            result = adjuster.calculate_font_size(text, bbox, original_size, lang)
            assert result == original_size, f"Failed for {lang}"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self) -> None:
        """Empty text should return original size."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=100, y1=20)
        result = adjuster.calculate_font_size("", bbox, 12.0, "en")

        assert result == 12.0

    def test_single_character(self) -> None:
        """Single character should fit easily."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=10, y1=20)
        result = adjuster.calculate_font_size("A", bbox, 12.0, "en")

        # 1 * 12 * 0.55 = 6.6pt < 10pt
        assert result == 12.0

    def test_zero_width_bbox(self) -> None:
        """Zero-width bbox should return minimum size."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=0, y0=0, x1=0, y1=20)
        result = adjuster.calculate_font_size("text", bbox, 12.0, "en")

        assert result == DEFAULT_MIN_FONT_SIZE

    def test_negative_width_bbox(self) -> None:
        """Negative-width bbox should return minimum size."""
        adjuster = FontSizeAdjuster()

        bbox = BBox(x0=100, y0=0, x1=50, y1=20)  # x1 < x0
        result = adjuster.calculate_font_size("text", bbox, 12.0, "en")

        assert result == DEFAULT_MIN_FONT_SIZE


class TestLanguageCodeNormalization:
    """Tests for language code normalization."""

    def test_lowercase_language(self) -> None:
        """Lowercase language codes should work."""
        adjuster = FontSizeAdjuster()
        bbox = BBox(x0=0, y0=0, x1=100, y1=20)

        result = adjuster.calculate_font_size("test", bbox, 12.0, "ja")
        assert result > 0

    def test_uppercase_language(self) -> None:
        """Uppercase language codes should work."""
        adjuster = FontSizeAdjuster()
        bbox = BBox(x0=0, y0=0, x1=100, y1=20)

        # Should normalize to lowercase
        result_upper = adjuster.calculate_font_size("test", bbox, 12.0, "JA")
        result_lower = adjuster.calculate_font_size("test", bbox, 12.0, "ja")

        assert result_upper == result_lower

    def test_language_with_region(self) -> None:
        """Language codes with region should work."""
        adjuster = FontSizeAdjuster()
        bbox = BBox(x0=0, y0=0, x1=100, y1=20)
        text = "1234567890"

        # zh-CN, zh-TW should be treated as CJK
        result_zh_cn = adjuster.calculate_font_size(text, bbox, 12.0, "zh-CN")
        result_zh_tw = adjuster.calculate_font_size(text, bbox, 12.0, "zh-TW")
        result_en_us = adjuster.calculate_font_size(text, bbox, 12.0, "en-US")

        # CJK should have smaller result (wider chars)
        assert result_zh_cn < result_en_us
        assert result_zh_tw < result_en_us


class TestConstants:
    """Tests for module constants."""

    def test_default_min_font_size(self) -> None:
        """Default minimum font size should be 6.0."""
        assert DEFAULT_MIN_FONT_SIZE == 6.0

    def test_default_font_size_decrement(self) -> None:
        """Default font size decrement should be 0.1."""
        assert DEFAULT_FONT_SIZE_DECREMENT == 0.1

    def test_cjk_languages(self) -> None:
        """CJK languages set should contain expected languages."""
        assert "ja" in CJK_LANGUAGES
        assert "zh" in CJK_LANGUAGES
        assert "ko" in CJK_LANGUAGES
        assert "zh-cn" in CJK_LANGUAGES
        assert "zh-tw" in CJK_LANGUAGES
