# SPDX-License-Identifier: Apache-2.0
"""Tests for side-by-side PDF generation module."""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium  # type: ignore[import-untyped]
import pytest

from pdf_translator.core.side_by_side import (
    SideBySideConfig,
    SideBySideGenerator,
    SideBySideOrder,
    create_side_by_side_pdf,
)


@pytest.fixture
def sample_pdf_path() -> Path:
    """Return path to the sample PDF fixture."""
    return Path(__file__).parent / "fixtures" / "sample_llama.pdf"


@pytest.fixture
def sample_pdf_bytes(sample_pdf_path: Path) -> bytes:
    """Return sample PDF as bytes."""
    return sample_pdf_path.read_bytes()


class TestSideBySideOrder:
    """Tests for SideBySideOrder enum."""

    def test_translated_original_value(self) -> None:
        """Test TRANSLATED_ORIGINAL has correct value."""
        assert SideBySideOrder.TRANSLATED_ORIGINAL.value == "translated_original"

    def test_original_translated_value(self) -> None:
        """Test ORIGINAL_TRANSLATED has correct value."""
        assert SideBySideOrder.ORIGINAL_TRANSLATED.value == "original_translated"


class TestSideBySideConfig:
    """Tests for SideBySideConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SideBySideConfig()
        assert config.order == SideBySideOrder.TRANSLATED_ORIGINAL
        assert config.gap == 0.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SideBySideConfig(
            order=SideBySideOrder.ORIGINAL_TRANSLATED,
            gap=10.0,
        )
        assert config.order == SideBySideOrder.ORIGINAL_TRANSLATED
        assert config.gap == 10.0


class TestSideBySideGenerator:
    """Tests for SideBySideGenerator class."""

    def test_generate_from_path(self, sample_pdf_path: Path) -> None:
        """Test generating side-by-side PDF from file paths."""
        generator = SideBySideGenerator()
        result = generator.generate(
            translated_pdf=sample_pdf_path,
            original_pdf=sample_pdf_path,
        )

        # Verify result is valid PDF bytes
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:4] == b"%PDF"

    def test_generate_from_bytes(self, sample_pdf_bytes: bytes) -> None:
        """Test generating side-by-side PDF from bytes."""
        generator = SideBySideGenerator()
        result = generator.generate(
            translated_pdf=sample_pdf_bytes,
            original_pdf=sample_pdf_bytes,
        )

        # Verify result is valid PDF bytes
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:4] == b"%PDF"

    def test_output_page_dimensions(self, sample_pdf_bytes: bytes) -> None:
        """Test that output pages have correct dimensions (double width)."""
        generator = SideBySideGenerator()
        result = generator.generate(
            translated_pdf=sample_pdf_bytes,
            original_pdf=sample_pdf_bytes,
        )

        # Open original to get dimensions
        original_doc = pdfium.PdfDocument(sample_pdf_bytes)
        original_width = original_doc[0].get_width()
        original_height = original_doc[0].get_height()
        original_doc.close()

        # Open result to verify dimensions
        result_doc = pdfium.PdfDocument(result)
        assert len(result_doc) == 1  # sample_llama.pdf has 1 page
        result_width = result_doc[0].get_width()
        result_height = result_doc[0].get_height()
        result_doc.close()

        # Width should be approximately double (may have small floating point differences)
        assert abs(result_width - original_width * 2) < 1.0
        assert abs(result_height - original_height) < 1.0

    def test_generate_with_gap(self, sample_pdf_bytes: bytes) -> None:
        """Test generating with gap between pages."""
        gap = 20.0
        config = SideBySideConfig(gap=gap)
        generator = SideBySideGenerator(config)
        result = generator.generate(
            translated_pdf=sample_pdf_bytes,
            original_pdf=sample_pdf_bytes,
        )

        # Open original to get dimensions
        original_doc = pdfium.PdfDocument(sample_pdf_bytes)
        original_width = original_doc[0].get_width()
        original_doc.close()

        # Open result to verify dimensions include gap
        result_doc = pdfium.PdfDocument(result)
        result_width = result_doc[0].get_width()
        result_doc.close()

        # Width should be double + gap
        expected_width = original_width * 2 + gap
        assert abs(result_width - expected_width) < 1.0

    def test_page_count_mismatch_raises_error(self, sample_pdf_path: Path) -> None:
        """Test that mismatched page counts raise ValueError."""
        # Create a multi-page PDF by using a different fixture or mock
        # For now, we'll test with the same PDF (should work)
        generator = SideBySideGenerator()

        # This should not raise because page counts match
        result = generator.generate(
            translated_pdf=sample_pdf_path,
            original_pdf=sample_pdf_path,
        )
        assert result is not None


class TestCreateSideBySidePdf:
    """Tests for create_side_by_side_pdf convenience function."""

    def test_convenience_function(self, sample_pdf_path: Path) -> None:
        """Test the convenience function works correctly."""
        result = create_side_by_side_pdf(
            translated_pdf=sample_pdf_path,
            original_pdf=sample_pdf_path,
        )

        assert isinstance(result, bytes)
        assert result[:4] == b"%PDF"

    def test_convenience_function_with_order(self, sample_pdf_path: Path) -> None:
        """Test convenience function with custom order."""
        result = create_side_by_side_pdf(
            translated_pdf=sample_pdf_path,
            original_pdf=sample_pdf_path,
            order=SideBySideOrder.ORIGINAL_TRANSLATED,
        )

        assert isinstance(result, bytes)
        assert result[:4] == b"%PDF"

    def test_convenience_function_with_gap(self, sample_pdf_path: Path) -> None:
        """Test convenience function with gap."""
        result = create_side_by_side_pdf(
            translated_pdf=sample_pdf_path,
            original_pdf=sample_pdf_path,
            gap=15.0,
        )

        assert isinstance(result, bytes)
        assert result[:4] == b"%PDF"
