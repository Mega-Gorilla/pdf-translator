# SPDX-License-Identifier: Apache-2.0
"""Tests for font_subsetter module."""

import tempfile
from pathlib import Path

import pytest

from pdf_translator.core.font_subsetter import (
    FontSubsetter,
    SubsetConfig,
    _find_font_variant,
    has_truetype_outlines,
)

# =============================================================================
# Constants
# =============================================================================

# Path to NotoSansCJK font (skip tests if not available)
NOTO_SANS_CJK_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
NOTO_SANS_CJK_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

# Skip condition for font-dependent tests
requires_noto_font = pytest.mark.skipif(
    not NOTO_SANS_CJK_REGULAR.exists(),
    reason="NotoSansCJK-Regular.ttc not found",
)

requires_noto_font_bold = pytest.mark.skipif(
    not NOTO_SANS_CJK_BOLD.exists(),
    reason="NotoSansCJK-Bold.ttc not found",
)


# Bundled Koruri font (TrueType)
BUNDLED_KORURI = Path(__file__).parent.parent / "src" / "pdf_translator" / "resources" / "fonts" / "Koruri-Regular.ttf"

requires_bundled_koruri = pytest.mark.skipif(
    not BUNDLED_KORURI.exists(),
    reason="Bundled Koruri font not found",
)


# =============================================================================
# Tests for has_truetype_outlines()
# =============================================================================


class TestHasTrueTypeOutlines:
    """Tests for has_truetype_outlines() function."""

    @requires_noto_font
    def test_noto_sans_cjk_is_cff(self) -> None:
        """NotoSansCJK uses CFF outlines (not TrueType)."""
        result = has_truetype_outlines(NOTO_SANS_CJK_REGULAR, font_number=0)
        assert result is False

    @requires_bundled_koruri
    def test_koruri_is_truetype(self) -> None:
        """Bundled Koruri uses TrueType outlines."""
        result = has_truetype_outlines(BUNDLED_KORURI)
        assert result is True

    def test_nonexistent_font_returns_false(self) -> None:
        """Non-existent font returns False."""
        result = has_truetype_outlines(Path("/nonexistent/font.ttf"))
        assert result is False


# =============================================================================
# Tests for _find_font_variant()
# =============================================================================


class TestFindFontVariant:
    """Tests for _find_font_variant() function."""

    @requires_noto_font
    @requires_noto_font_bold
    def test_find_bold_variant(self) -> None:
        """Find Bold variant from Regular path."""
        result = _find_font_variant(NOTO_SANS_CJK_REGULAR, is_bold=True, is_italic=False)
        assert result is not None
        assert result.name == "NotoSansCJK-Bold.ttc"

    @requires_noto_font
    def test_find_regular_variant(self) -> None:
        """Find Regular variant returns the Regular path."""
        result = _find_font_variant(NOTO_SANS_CJK_REGULAR, is_bold=False, is_italic=False)
        # Returns Regular path since weight pattern exists
        assert result is not None
        assert result.name == "NotoSansCJK-Regular.ttc"

    @requires_noto_font
    def test_italic_fallback(self) -> None:
        """Italic falls back to Regular (NotoSansCJK has no Italic)."""
        result = _find_font_variant(NOTO_SANS_CJK_REGULAR, is_bold=False, is_italic=True)
        # Falls back to Regular since no Italic variant exists
        assert result is not None
        assert result.name == "NotoSansCJK-Regular.ttc"

    def test_no_weight_pattern(self) -> None:
        """Returns None if no weight pattern found."""
        fake_path = Path("/fake/Font.ttf")
        result = _find_font_variant(fake_path, is_bold=True, is_italic=False)
        assert result is None


# =============================================================================
# Tests for SubsetConfig
# =============================================================================


class TestSubsetConfig:
    """Tests for SubsetConfig dataclass."""

    def test_default_values(self) -> None:
        """Default configuration values."""
        config = SubsetConfig()
        assert config.include_common_punctuation is True
        assert config.cache_dir is None

    def test_custom_cache_dir(self) -> None:
        """Custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SubsetConfig(cache_dir=Path(tmpdir))
            assert config.cache_dir == Path(tmpdir)


# =============================================================================
# Tests for FontSubsetter
# =============================================================================


class TestFontSubsetter:
    """Tests for FontSubsetter class."""

    @requires_noto_font
    def test_subset_for_texts_creates_file(self) -> None:
        """subset_for_texts creates a subset font file."""
        subsetter = FontSubsetter()
        texts = ["こんにちは", "世界"]

        result = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=texts,
            font_number=0,
        )

        assert result is not None
        assert result.exists()
        assert result.suffix == ".ttf"

        # Cleanup
        subsetter.cleanup()
        assert not result.exists()

    @requires_noto_font
    def test_subset_size_reduction(self) -> None:
        """Subset font is much smaller than original."""
        subsetter = FontSubsetter()
        texts = ["テスト"]

        original_size = NOTO_SANS_CJK_REGULAR.stat().st_size
        result = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=texts,
            font_number=0,
        )

        assert result is not None
        subset_size = result.stat().st_size

        # Subset should be < 1% of original (~18MB -> ~100KB or less)
        assert subset_size < original_size * 0.01

        subsetter.cleanup()

    @requires_noto_font
    @requires_noto_font_bold
    def test_subset_with_bold_variant(self) -> None:
        """subset_for_texts uses Bold variant when is_bold=True."""
        subsetter = FontSubsetter()
        texts = ["太字テスト"]

        result = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=texts,
            font_number=0,
            is_bold=True,
        )

        assert result is not None
        assert result.exists()

        subsetter.cleanup()

    @requires_noto_font
    def test_cache_returns_same_path(self) -> None:
        """Same inputs return cached path."""
        subsetter = FontSubsetter()
        texts = ["キャッシュ"]

        result1 = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=texts,
            font_number=0,
        )
        result2 = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=texts,
            font_number=0,
        )

        assert result1 == result2

        subsetter.cleanup()

    @requires_noto_font
    def test_different_texts_different_cache(self) -> None:
        """Different texts create different subsets."""
        subsetter = FontSubsetter()

        result1 = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=["テスト一"],
            font_number=0,
        )
        result2 = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=["テスト二"],
            font_number=0,
        )

        assert result1 != result2

        subsetter.cleanup()

    def test_empty_texts_returns_none(self) -> None:
        """Empty texts list returns None."""
        subsetter = FontSubsetter()

        result = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=[],
            font_number=0,
        )

        assert result is None

    def test_only_empty_strings_returns_none(self) -> None:
        """List of only empty strings returns None."""
        # With include_common_punctuation=True, safety chars are added
        # So result is NOT None. Test with include_common_punctuation=False
        config = SubsetConfig(include_common_punctuation=False)
        subsetter = FontSubsetter(config)

        result = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=["", ""],
            font_number=0,
        )

        assert result is None

    @requires_noto_font
    def test_custom_cache_dir(self) -> None:
        """Subset saved to custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            config = SubsetConfig(cache_dir=cache_dir)
            subsetter = FontSubsetter(config)

            result = subsetter.subset_for_texts(
                font_path=NOTO_SANS_CJK_REGULAR,
                texts=["カスタム"],
                font_number=0,
            )

            assert result is not None
            assert result.parent == cache_dir

            # No cleanup needed, tmpdir will be removed

    @requires_noto_font
    def test_cleanup_removes_files(self) -> None:
        """cleanup() removes all cached subset files."""
        subsetter = FontSubsetter()

        result1 = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=["テスト一"],
            font_number=0,
        )
        result2 = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=["テスト二"],
            font_number=0,
        )

        assert result1 is not None and result1.exists()
        assert result2 is not None and result2.exists()

        subsetter.cleanup()

        assert not result1.exists()
        assert not result2.exists()


# =============================================================================
# Tests for CID Font Compatibility
# =============================================================================


class TestCIDFontCompatibility:
    """Tests for CID font compatibility with pypdfium2."""

    @requires_noto_font
    def test_subset_loadable_as_cid_font(self) -> None:
        """Subset font can be loaded into pypdfium2 as CID font."""
        import ctypes

        import pypdfium2 as pdfium

        from pdf_translator.core.pdf_processor import to_byte_array

        subsetter = FontSubsetter()
        texts = ["CID互換性テスト"]

        subset_path = subsetter.subset_for_texts(
            font_path=NOTO_SANS_CJK_REGULAR,
            texts=texts,
            font_number=0,
        )
        assert subset_path is not None

        # Read subset font data
        font_data = subset_path.read_bytes()

        # Try to load into pypdfium2
        pdf = pdfium.PdfDocument.new()
        font_arr = to_byte_array(font_data)

        font_handle = pdfium.raw.FPDFText_LoadFont(
            pdf.raw,
            font_arr,
            ctypes.c_uint(len(font_data)),
            ctypes.c_int(pdfium.raw.FPDF_FONT_TRUETYPE),
            ctypes.c_int(1),  # is_cid = True
        )

        assert font_handle is not None

        pdf.close()
        subsetter.cleanup()


# =============================================================================
# Tests for Bundled Koruri Font
# =============================================================================


class TestBundledKoruriFont:
    """Tests for bundled Koruri font subsetting."""

    @requires_bundled_koruri
    def test_koruri_subset_creates_file(self) -> None:
        """Koruri subset creates a valid font file."""
        subsetter = FontSubsetter()
        texts = ["こんにちは世界"]

        result = subsetter.subset_for_texts(
            font_path=BUNDLED_KORURI,
            texts=texts,
        )

        assert result is not None
        assert result.exists()
        assert result.suffix == ".ttf"

        subsetter.cleanup()

    @requires_bundled_koruri
    def test_koruri_subset_size_reduction(self) -> None:
        """Koruri subset is much smaller than original."""
        subsetter = FontSubsetter()
        texts = ["テスト"]

        original_size = BUNDLED_KORURI.stat().st_size
        result = subsetter.subset_for_texts(
            font_path=BUNDLED_KORURI,
            texts=texts,
        )

        assert result is not None
        subset_size = result.stat().st_size

        # Subset should be < 5% of original (~1.8MB -> ~100KB or less)
        assert subset_size < original_size * 0.05

        subsetter.cleanup()

    @requires_bundled_koruri
    def test_koruri_subset_loadable_in_pypdfium2(self) -> None:
        """Koruri subset can be loaded into pypdfium2 as CID font."""
        import ctypes

        import pypdfium2 as pdfium

        from pdf_translator.core.pdf_processor import to_byte_array

        subsetter = FontSubsetter()
        texts = ["PDF互換性テスト"]

        subset_path = subsetter.subset_for_texts(
            font_path=BUNDLED_KORURI,
            texts=texts,
        )
        assert subset_path is not None

        font_data = subset_path.read_bytes()

        pdf = pdfium.PdfDocument.new()
        font_arr = to_byte_array(font_data)

        font_handle = pdfium.raw.FPDFText_LoadFont(
            pdf.raw,
            font_arr,
            ctypes.c_uint(len(font_data)),
            ctypes.c_int(pdfium.raw.FPDF_FONT_TRUETYPE),
            ctypes.c_int(1),  # is_cid = True
        )

        assert font_handle is not None

        pdf.close()
        subsetter.cleanup()
