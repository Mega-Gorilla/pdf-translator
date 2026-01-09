# SPDX-License-Identifier: Apache-2.0
"""Tests for table extractor module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_translator.core.models import BBox, LayoutBlock, RawLayoutCategory
from pdf_translator.output.table_extractor import (
    ExtractedTable,
    TableCell,
    TableExtractionConfig,
    TableExtractionError,
    TableExtractor,
)


class TestTableExtractionConfig:
    """Test TableExtractionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TableExtractionConfig()

        assert config.mode == "heuristic"
        assert config.y_tolerance == 5.0
        assert config.min_columns == 2
        assert config.min_rows == 2

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TableExtractionConfig(
            mode="pdfplumber",
            y_tolerance=10.0,
            min_columns=3,
            min_rows=4,
        )

        assert config.mode == "pdfplumber"
        assert config.y_tolerance == 10.0
        assert config.min_columns == 3
        assert config.min_rows == 4


class TestTableCell:
    """Test TableCell dataclass."""

    def test_create_simple_cell(self) -> None:
        """Test creating a simple table cell."""
        cell = TableCell(
            text="Hello",
            row=0,
            col=0,
        )

        assert cell.text == "Hello"
        assert cell.row == 0
        assert cell.col == 0
        assert cell.rowspan == 1
        assert cell.colspan == 1
        assert cell.is_header is False
        assert cell.alignment == "left"

    def test_create_cell_with_span(self) -> None:
        """Test creating cell with rowspan/colspan."""
        cell = TableCell(
            text="Merged",
            row=0,
            col=0,
            rowspan=2,
            colspan=3,
            is_header=True,
            alignment="center",
        )

        assert cell.text == "Merged"
        assert cell.rowspan == 2
        assert cell.colspan == 3
        assert cell.is_header is True
        assert cell.alignment == "center"


class TestExtractedTable:
    """Test ExtractedTable dataclass."""

    def _create_simple_table(self) -> ExtractedTable:
        """Create a simple 2x2 table for testing."""
        rows = [
            [
                TableCell(text="Header 1", row=0, col=0, is_header=True),
                TableCell(text="Header 2", row=0, col=1, is_header=True),
            ],
            [
                TableCell(text="Cell 1", row=1, col=0),
                TableCell(text="Cell 2", row=1, col=1),
            ],
        ]
        return ExtractedTable(
            id="table_001",
            layout_block_id="block_1",
            page_number=0,
            bbox=BBox(x0=100, y0=200, x1=400, y1=300),
            rows=rows,
            header_rows=1,
            extraction_method="heuristic",
        )

    def test_create_extracted_table(self) -> None:
        """Test creating ExtractedTable."""
        table = self._create_simple_table()

        assert table.id == "table_001"
        assert table.layout_block_id == "block_1"
        assert table.page_number == 0
        assert len(table.rows) == 2
        assert table.header_rows == 1
        assert table.extraction_method == "heuristic"

    def test_to_markdown_simple(self) -> None:
        """Test Markdown output for simple table."""
        table = self._create_simple_table()
        md = table.to_markdown()

        assert "| Header 1 | Header 2 |" in md
        assert "| --- | --- |" in md
        assert "| Cell 1 | Cell 2 |" in md

    def test_to_markdown_with_alignment(self) -> None:
        """Test Markdown output with alignment."""
        rows = [
            [
                TableCell(text="Left", row=0, col=0, is_header=True, alignment="left"),
                TableCell(
                    text="Center", row=0, col=1, is_header=True, alignment="center"
                ),
                TableCell(
                    text="Right", row=0, col=2, is_header=True, alignment="right"
                ),
            ],
            [
                TableCell(text="1", row=1, col=0, alignment="left"),
                TableCell(text="2", row=1, col=1, alignment="center"),
                TableCell(text="3", row=1, col=2, alignment="right"),
            ],
        ]
        table = ExtractedTable(
            id="table_001",
            layout_block_id="block_1",
            page_number=0,
            bbox=BBox(x0=100, y0=200, x1=400, y1=300),
            rows=rows,
            header_rows=1,
        )
        md = table.to_markdown()

        # Left alignment uses "---" (default)
        # Center alignment uses ":---:"
        # Right alignment uses "---:"
        assert ":---:" in md  # Center
        assert "---:" in md  # Right

    def test_to_markdown_no_header(self) -> None:
        """Test Markdown output without header."""
        rows = [
            [
                TableCell(text="Cell 1", row=0, col=0),
                TableCell(text="Cell 2", row=0, col=1),
            ],
            [
                TableCell(text="Cell 3", row=1, col=0),
                TableCell(text="Cell 4", row=1, col=1),
            ],
        ]
        table = ExtractedTable(
            id="table_001",
            layout_block_id="block_1",
            page_number=0,
            bbox=BBox(x0=100, y0=200, x1=400, y1=300),
            rows=rows,
            header_rows=0,
        )
        md = table.to_markdown()

        # Should still have separator after first row
        assert "|" in md
        assert "---" in md

    def test_to_markdown_escapes_pipe(self) -> None:
        """Test Markdown escapes pipe characters."""
        rows = [
            [
                TableCell(text="A | B", row=0, col=0, is_header=True),
            ],
            [
                TableCell(text="C | D", row=1, col=0),
            ],
        ]
        table = ExtractedTable(
            id="table_001",
            layout_block_id="block_1",
            page_number=0,
            bbox=BBox(x0=100, y0=200, x1=400, y1=300),
            rows=rows,
            header_rows=1,
        )
        md = table.to_markdown()

        assert r"A \| B" in md
        assert r"C \| D" in md

    def test_to_html_simple(self) -> None:
        """Test HTML output for simple table."""
        table = self._create_simple_table()
        html = table.to_html()

        assert "<table>" in html
        assert "</table>" in html
        assert "<tr>" in html
        assert "</tr>" in html
        assert "<th>" in html
        assert "<td>" in html

    def test_to_html_with_span(self) -> None:
        """Test HTML output with rowspan/colspan."""
        rows = [
            [
                TableCell(
                    text="Merged", row=0, col=0, is_header=True, rowspan=2, colspan=2
                ),
            ],
            [],  # Empty row due to span
            [
                TableCell(text="Normal", row=2, col=0),
                TableCell(text="Cell", row=2, col=1),
            ],
        ]
        table = ExtractedTable(
            id="table_001",
            layout_block_id="block_1",
            page_number=0,
            bbox=BBox(x0=100, y0=200, x1=400, y1=300),
            rows=rows,
            header_rows=1,
        )
        html = table.to_html()

        assert 'rowspan="2"' in html
        assert 'colspan="2"' in html

    def test_to_html_with_alignment(self) -> None:
        """Test HTML output with alignment styles."""
        rows = [
            [
                TableCell(text="Left", row=0, col=0, alignment="left"),
                TableCell(text="Center", row=0, col=1, alignment="center"),
                TableCell(text="Right", row=0, col=2, alignment="right"),
            ],
        ]
        table = ExtractedTable(
            id="table_001",
            layout_block_id="block_1",
            page_number=0,
            bbox=BBox(x0=100, y0=200, x1=400, y1=300),
            rows=rows,
            header_rows=0,
        )
        html = table.to_html()

        # Left is default, so no style attribute
        assert 'style="text-align: center"' in html
        assert 'style="text-align: right"' in html

    def test_has_complex_merge(self) -> None:
        """Test detection of complex cell merges."""
        # Simple table - no complex merge
        simple_table = self._create_simple_table()
        assert simple_table.has_complex_merge() is False

        # Table with rowspan/colspan
        rows = [
            [TableCell(text="Merged", row=0, col=0, rowspan=2, colspan=2)],
        ]
        complex_table = ExtractedTable(
            id="table_001",
            layout_block_id="block_1",
            page_number=0,
            bbox=BBox(x0=100, y0=200, x1=400, y1=300),
            rows=rows,
        )
        assert complex_table.has_complex_merge() is True


class TestTableExtractor:
    """Test TableExtractor class."""

    def test_init_with_default_config(self) -> None:
        """Test initialization with default config."""
        extractor = TableExtractor()
        assert extractor._config is not None
        assert extractor._config.mode == "heuristic"

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = TableExtractionConfig(mode="pdfplumber")
        extractor = TableExtractor(config)
        assert extractor._config.mode == "pdfplumber"

    def test_check_pdfplumber(self) -> None:
        """Test pdfplumber availability check."""
        extractor = TableExtractor()
        # Should return bool
        assert isinstance(extractor._pdfplumber_available, bool)


class TestTableExtractionError:
    """Test TableExtractionError exception."""

    def test_raise_error(self) -> None:
        """Test raising TableExtractionError."""
        with pytest.raises(TableExtractionError) as exc_info:
            raise TableExtractionError("Test error")

        assert "Test error" in str(exc_info.value)
