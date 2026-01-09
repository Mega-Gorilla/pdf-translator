# SPDX-License-Identifier: Apache-2.0
"""Tests for image extractor module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pdf_translator.core.models import BBox, LayoutBlock, RawLayoutCategory
from pdf_translator.output.image_extractor import (
    IMAGE_CATEGORIES,
    ExtractedImage,
    ImageExtractionConfig,
    ImageExtractor,
    extract_image_as_fallback,
)


class TestImageExtractionConfig:
    """Test ImageExtractionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ImageExtractionConfig()

        assert config.enabled is True
        assert config.output_dir is None
        assert config.relative_path == "images"
        assert config.format == "png"
        assert config.quality == 95
        assert config.min_size == (50, 50)
        assert config.dpi == 150
        assert config.naming == "sequential"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ImageExtractionConfig(
            enabled=False,
            output_dir=Path("/tmp/images"),
            relative_path="assets",
            format="jpeg",
            quality=80,
            min_size=(100, 100),
            dpi=300,
            naming="page_index",
        )

        assert config.enabled is False
        assert config.output_dir == Path("/tmp/images")
        assert config.relative_path == "assets"
        assert config.format == "jpeg"
        assert config.quality == 80
        assert config.min_size == (100, 100)
        assert config.dpi == 300
        assert config.naming == "page_index"


class TestExtractedImage:
    """Test ExtractedImage dataclass."""

    def test_create_extracted_image(self) -> None:
        """Test creating ExtractedImage."""
        img = ExtractedImage(
            id="figure_001",
            path=Path("/tmp/images/figure_001.png"),
            relative_path="images/figure_001.png",
            layout_block_id="block_1",
            page_number=0,
            bbox=BBox(x0=100, y0=200, x1=300, y1=400),
            category="image",
            caption="Figure 1: Test image",
        )

        assert img.id == "figure_001"
        assert img.path == Path("/tmp/images/figure_001.png")
        assert img.relative_path == "images/figure_001.png"
        assert img.layout_block_id == "block_1"
        assert img.page_number == 0
        assert img.bbox.x0 == 100
        assert img.category == "image"
        assert img.caption == "Figure 1: Test image"

    def test_create_without_caption(self) -> None:
        """Test creating ExtractedImage without caption."""
        img = ExtractedImage(
            id="figure_001",
            path=Path("/tmp/images/figure_001.png"),
            relative_path="images/figure_001.png",
            layout_block_id="block_1",
            page_number=0,
            bbox=BBox(x0=100, y0=200, x1=300, y1=400),
            category="image",
        )

        assert img.caption is None


class TestImageExtractor:
    """Test ImageExtractor class."""

    def test_init_with_default_config(self) -> None:
        """Test initialization with default config."""
        extractor = ImageExtractor()
        assert extractor._config is not None
        assert extractor._config.enabled is True

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ImageExtractionConfig(enabled=False, dpi=300)
        extractor = ImageExtractor(config)
        assert extractor._config.enabled is False
        assert extractor._config.dpi == 300

    def test_extract_disabled(self) -> None:
        """Test extraction when disabled."""
        config = ImageExtractionConfig(enabled=False)
        extractor = ImageExtractor(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = extractor.extract(
                Path("dummy.pdf"),
                {},
                Path(tmpdir),
            )

        assert result == []

    def test_generate_filename_sequential(self) -> None:
        """Test sequential filename generation."""
        config = ImageExtractionConfig(naming="sequential")
        extractor = ImageExtractor(config)

        counters: dict[str, int] = {}
        filename1 = extractor._generate_filename("image", 0, counters)
        filename2 = extractor._generate_filename("image", 0, counters)
        filename3 = extractor._generate_filename("chart", 0, counters)

        assert filename1 == "figure_001"
        assert filename2 == "figure_002"
        assert filename3 == "chart_001"

    def test_generate_filename_page_index(self) -> None:
        """Test page_index filename generation."""
        config = ImageExtractionConfig(naming="page_index")
        extractor = ImageExtractor(config)

        counters: dict[str, int] = {}
        filename1 = extractor._generate_filename("image", 0, counters)
        filename2 = extractor._generate_filename("image", 0, counters)
        filename3 = extractor._generate_filename("image", 1, counters)

        assert filename1 == "p1_figure_001"
        assert filename2 == "p1_figure_002"
        assert filename3 == "p2_figure_001"

    def test_find_caption_returns_none(self) -> None:
        """Test _find_caption returns None (placeholder)."""
        extractor = ImageExtractor()
        block = LayoutBlock(
            id="block_1",
            bbox=BBox(x0=100, y0=200, x1=300, y1=400),
            raw_category=RawLayoutCategory.IMAGE,
            page_num=0,
        )

        result = extractor._find_caption(block, [])
        assert result is None


class TestImageCategories:
    """Test image categories constant."""

    def test_image_categories(self) -> None:
        """Test IMAGE_CATEGORIES contains expected values."""
        assert "image" in IMAGE_CATEGORIES
        assert "chart" in IMAGE_CATEGORIES
        assert len(IMAGE_CATEGORIES) == 2


class TestExtractImageAsFallback:
    """Test extract_image_as_fallback function."""

    def test_function_exists(self) -> None:
        """Test function is importable."""
        assert callable(extract_image_as_fallback)
