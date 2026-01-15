# SPDX-License-Identifier: Apache-2.0
"""Tests for ThumbnailGenerator."""

import tempfile
from pathlib import Path

import pytest

from pdf_translator.output.thumbnail_generator import ThumbnailConfig, ThumbnailGenerator


class TestThumbnailConfig:
    """Tests for ThumbnailConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ThumbnailConfig()

        assert config.width == 400
        assert config.page_number == 0

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = ThumbnailConfig(width=600, page_number=1)

        assert config.width == 600
        assert config.page_number == 1


class TestThumbnailGenerator:
    """Tests for ThumbnailGenerator."""

    @pytest.fixture
    def sample_pdf(self) -> Path:
        """Get sample PDF path."""
        pdf_path = Path("tests/fixtures/sample_llama.pdf")
        if not pdf_path.exists():
            pytest.skip("Sample PDF not found")
        return pdf_path

    def test_generate_default_config(self, sample_pdf: Path) -> None:
        """Test thumbnail generation with default config."""
        generator = ThumbnailGenerator()

        image_bytes, width, height = generator.generate(sample_pdf)

        assert len(image_bytes) > 0
        assert width == 400
        assert height > 0
        # Verify PNG header
        assert image_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_custom_width(self, sample_pdf: Path) -> None:
        """Test thumbnail generation with custom width."""
        config = ThumbnailConfig(width=200)
        generator = ThumbnailGenerator(config)

        image_bytes, width, height = generator.generate(sample_pdf)

        assert width == 200
        assert height > 0

    def test_generate_to_file(self, sample_pdf: Path) -> None:
        """Test thumbnail generation to file."""
        generator = ThumbnailGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thumbnail.png"
            width, height = generator.generate_to_file(sample_pdf, output_path)

            assert output_path.exists()
            assert width == 400
            assert height > 0
            # Verify file content is PNG
            content = output_path.read_bytes()
            assert content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_file_not_found(self) -> None:
        """Test with non-existent PDF."""
        generator = ThumbnailGenerator()

        with pytest.raises(FileNotFoundError):
            generator.generate(Path("nonexistent.pdf"))

    def test_generate_creates_parent_directory(self, sample_pdf: Path) -> None:
        """Test that generate_to_file creates parent directories."""
        generator = ThumbnailGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "thumbnail.png"
            generator.generate_to_file(sample_pdf, output_path)

            assert output_path.exists()
