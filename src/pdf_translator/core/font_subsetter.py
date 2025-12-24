# SPDX-License-Identifier: Apache-2.0
"""Font subsetting using fonttools.

This module provides font subsetting functionality to reduce PDF file sizes
by including only the glyphs used in the translated text.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Common punctuation and digits to include in subset
SAFETY_MARGIN_CHARS = "。、！？「」『』（）…―　0123456789"

# Font weight patterns for variant detection
WEIGHT_PATTERN = re.compile(r"-(Regular|Bold|Light|Medium|Thin|Black)")


@dataclass
class SubsetConfig:
    """Font subsetting configuration."""

    include_common_punctuation: bool = True
    cache_dir: Path | None = None  # None = use temp directory


def _find_font_variant(
    base_font_path: Path,
    is_bold: bool,
    is_italic: bool,
) -> Path | None:
    """Find font file for the given style variant.

    NotoSansCJK-Regular.ttc → NotoSansCJK-Bold.ttc (is_bold=True)

    Note: NotoSansCJK does not have Italic variants.
    When Italic is requested but not available, falls back to Regular/Bold.

    Args:
        base_font_path: Path to the base font file (typically Regular).
        is_bold: Whether Bold variant is requested.
        is_italic: Ignored for NotoSansCJK (no italic files).

    Returns:
        Path to variant font file, or None if not found.
    """
    stem = base_font_path.stem
    parent = base_font_path.parent

    # Determine target weight
    target_weight = "Bold" if is_bold else "Regular"

    # Replace weight in filename
    if WEIGHT_PATTERN.search(stem):
        new_stem = WEIGHT_PATTERN.sub(f"-{target_weight}", stem)
    else:
        # No weight pattern found, return None
        return None

    variant_path = parent / f"{new_stem}{base_font_path.suffix}"
    return variant_path if variant_path.exists() else None


class FontSubsetter:
    """Font subsetter using fonttools.

    Creates subset fonts containing only the glyphs needed for specific texts,
    dramatically reducing PDF file sizes when embedding CJK fonts.

    Example:
        subsetter = FontSubsetter()
        subset_path = subsetter.subset_for_texts(
            font_path=Path("/usr/share/fonts/noto/NotoSansCJK-Regular.ttc"),
            texts=["こんにちは", "世界"],
        )
        # Use subset_path for PDF text insertion
    """

    def __init__(self, config: SubsetConfig | None = None) -> None:
        """Initialize the font subsetter.

        Args:
            config: Subsetting configuration. Uses defaults if None.
        """
        self._config = config or SubsetConfig()
        self._cache: dict[str, Path] = {}

    def subset_for_texts(
        self,
        font_path: Path,
        texts: list[str],
        font_number: int = 0,
        is_bold: bool = False,
        is_italic: bool = False,
    ) -> Path | None:
        """Create subset font containing only characters used in texts.

        Args:
            font_path: Path to the base font file (TTF or TTC).
            texts: List of texts to extract characters from.
            font_number: Font index for TTC files (default: 0).
            is_bold: Use Bold variant if available.
            is_italic: Use Italic variant if available (fallback to base if not).

        Returns:
            Path to the subset font file, or None if subsetting failed.
        """
        from fontTools.subset import Options, Subsetter  # type: ignore[import-untyped]
        from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

        # Collect unique characters
        chars: set[str] = set()
        for text in texts:
            if text:
                chars.update(text)

        if not chars:
            return None

        # Add safety margin characters
        if self._config.include_common_punctuation:
            chars.update(SAFETY_MARGIN_CHARS)

        # Resolve font variant
        actual_font_path = font_path
        if is_bold or is_italic:
            variant_path = _find_font_variant(font_path, is_bold, is_italic)
            if variant_path:
                actual_font_path = variant_path
                logger.debug(
                    "Using font variant: %s (bold=%s, italic=%s)",
                    variant_path.name,
                    is_bold,
                    is_italic,
                )
            else:
                logger.warning(
                    "Font variant not found for bold=%s, italic=%s. Using %s",
                    is_bold,
                    is_italic,
                    font_path.name,
                )

        # Check cache
        cache_key = self._get_cache_key(actual_font_path, chars, font_number)
        if cache_key in self._cache:
            cached_path = self._cache[cache_key]
            if cached_path.exists():
                return cached_path

        try:
            # Load font
            if actual_font_path.suffix.lower() == ".ttc":
                font = TTFont(actual_font_path, fontNumber=font_number)
            else:
                font = TTFont(actual_font_path)

            # Create subset
            options = Options()
            options.layout_features = ["*"]
            options.name_IDs = ["*"]
            options.notdef_glyph = True
            options.notdef_outline = True

            subsetter = Subsetter(options=options)
            subsetter.populate(text="".join(chars))
            subsetter.subset(font)

            # Save to cache directory
            subset_path = self._get_subset_path(cache_key)
            font.save(subset_path)
            font.close()

            self._cache[cache_key] = subset_path
            logger.info(
                "Created subset font: %d chars, %s (bold=%s)",
                len(chars),
                subset_path.name,
                is_bold,
            )

            return subset_path

        except Exception as e:
            logger.warning("Font subsetting failed: %s", e)
            return None

    def _get_cache_key(
        self,
        font_path: Path,
        chars: set[str],
        font_number: int,
    ) -> str:
        """Generate cache key for subset.

        Uses full path, size, and mtime to avoid collisions
        with same-named fonts in different directories.
        """
        stat = font_path.stat()
        font_identity = f"{font_path.resolve()}:{stat.st_size}:{stat.st_mtime}"
        chars_str = "".join(sorted(chars))
        full_key = f"{font_identity}:{font_number}:{chars_str}"
        return hashlib.sha256(full_key.encode()).hexdigest()[:16]

    def _get_subset_path(self, cache_key: str) -> Path:
        """Get path for subset file.

        Uses mkstemp for safety instead of mktemp.
        """
        if self._config.cache_dir:
            self._config.cache_dir.mkdir(parents=True, exist_ok=True)
            return self._config.cache_dir / f"{cache_key}.ttf"
        else:
            import tempfile as tmp

            fd, path = tmp.mkstemp(suffix=f"_{cache_key}.ttf")
            os.close(fd)  # Close fd, we'll write via fontTools
            return Path(path)

    def cleanup(self) -> None:
        """Clean up cached subset files."""
        for path in self._cache.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._cache.clear()
