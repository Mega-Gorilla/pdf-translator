# SPDX-License-Identifier: Apache-2.0
"""Table extraction from PDF for Markdown output."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pdf_translator.core.models import BBox, LayoutBlock

if TYPE_CHECKING:
    from pdf_translator.output.image_extractor import ExtractedImage

logger = logging.getLogger(__name__)


class TableExtractionError(Exception):
    """Raised when table extraction fails."""

    pass


@dataclass
class TableExtractionConfig:
    """Table extraction configuration."""

    mode: str = "heuristic"  # heuristic, pdfplumber, image
    y_tolerance: float = 5.0  # Y-axis clustering tolerance
    min_columns: int = 2  # Minimum columns to be considered a table
    min_rows: int = 2  # Minimum rows to be considered a table


@dataclass
class TableCell:
    """Table cell data."""

    text: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    alignment: str = "left"  # left, center, right


@dataclass
class ExtractedTable:
    """Extracted table data."""

    id: str
    layout_block_id: str
    page_number: int
    bbox: BBox
    rows: list[list[TableCell]]
    header_rows: int = 1
    caption: str | None = None
    extraction_method: str = "heuristic"

    def to_markdown(self) -> str:
        """Convert table to Markdown format.

        Returns:
            Markdown table string.
        """
        if not self.rows:
            return ""

        lines: list[str] = []

        # Determine column count
        col_count = max(len(row) for row in self.rows) if self.rows else 0
        if col_count == 0:
            return ""

        # Build header row
        if self.rows:
            header = self.rows[0]
            header_cells = [self._escape_cell(cell.text) for cell in header]
            # Pad with empty cells if needed
            while len(header_cells) < col_count:
                header_cells.append("")
            lines.append("| " + " | ".join(header_cells) + " |")

            # Add separator row with alignment
            separators = []
            for i, cell in enumerate(header):
                if cell.alignment == "center":
                    separators.append(":---:")
                elif cell.alignment == "right":
                    separators.append("---:")
                else:
                    separators.append("---")
            # Pad with default separators
            while len(separators) < col_count:
                separators.append("---")
            lines.append("| " + " | ".join(separators) + " |")

        # Build data rows
        for row in self.rows[1:]:
            cells = [self._escape_cell(cell.text) for cell in row]
            # Pad with empty cells if needed
            while len(cells) < col_count:
                cells.append("")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Convert table to HTML format (for complex tables).

        Returns:
            HTML table string.
        """
        if not self.rows:
            return ""

        lines: list[str] = ["<table>"]

        for i, row in enumerate(self.rows):
            lines.append("  <tr>")
            for cell in row:
                tag = "th" if cell.is_header or i < self.header_rows else "td"
                attrs = []
                if cell.rowspan > 1:
                    attrs.append(f'rowspan="{cell.rowspan}"')
                if cell.colspan > 1:
                    attrs.append(f'colspan="{cell.colspan}"')
                if cell.alignment != "left":
                    attrs.append(f'style="text-align: {cell.alignment}"')

                attr_str = " " + " ".join(attrs) if attrs else ""
                text = self._escape_html(cell.text)
                lines.append(f"    <{tag}{attr_str}>{text}</{tag}>")
            lines.append("  </tr>")

        lines.append("</table>")
        return "\n".join(lines)

    def has_complex_merge(self) -> bool:
        """Check if table has complex cell merges (rowspan/colspan > 1).

        Returns:
            True if complex merges exist.
        """
        for row in self.rows:
            for cell in row:
                if cell.rowspan > 1 or cell.colspan > 1:
                    return True
        return False

    @staticmethod
    def _escape_cell(text: str) -> str:
        """Escape special characters for Markdown table cell.

        Args:
            text: Cell text.

        Returns:
            Escaped text.
        """
        # Replace pipe characters and newlines
        text = text.replace("|", "\\|")
        text = text.replace("\n", "<br>")
        return text.strip()

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape special characters for HTML.

        Args:
            text: Text to escape.

        Returns:
            Escaped text.
        """
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        return text


class TableExtractor:
    """Table extractor from PDF.

    Supports multiple extraction strategies:
    - heuristic: TextObject-based extraction for borderless tables
    - pdfplumber: pdfplumber-based extraction for lined tables (optional)
    - image: Image fallback for complex tables
    """

    def __init__(self, config: TableExtractionConfig | None = None) -> None:
        """Initialize TableExtractor.

        Args:
            config: Table extraction configuration.
        """
        self._config = config or TableExtractionConfig()
        self._pdfplumber_available = self._check_pdfplumber()
        self._counter = 0

    def _check_pdfplumber(self) -> bool:
        """Check if pdfplumber is available.

        Returns:
            True if pdfplumber is installed.
        """
        try:
            import pdfplumber  # noqa: F401

            return True
        except ImportError:
            return False

    def extract(
        self,
        pdf_path: Path,
        table_block: LayoutBlock,
        text_objects: list[Any],
        output_dir: Path | None = None,
    ) -> ExtractedTable | ExtractedImage:
        """Extract table from PDF.

        Tries extraction methods based on configuration mode.
        Falls back to image extraction if all methods fail.

        Args:
            pdf_path: Path to the PDF.
            table_block: LayoutBlock marking the table region.
            text_objects: List of TextObject from the page.
            output_dir: Output directory for image fallback.

        Returns:
            ExtractedTable or ExtractedImage (fallback).
        """
        self._counter += 1
        table_id = f"table_{self._counter:03d}"

        if self._config.mode == "image":
            # Direct image fallback
            return self._extract_as_image(pdf_path, table_block, table_id, output_dir)

        # Try heuristic extraction first (default)
        if self._config.mode == "heuristic":
            try:
                return self._extract_with_heuristics(
                    table_block, text_objects, table_id
                )
            except TableExtractionError:
                logger.debug(
                    "Heuristic extraction failed for %s, trying pdfplumber",
                    table_block.id,
                )

            # Try pdfplumber if available
            if self._pdfplumber_available:
                try:
                    return self._extract_with_pdfplumber(
                        pdf_path, table_block, table_id
                    )
                except TableExtractionError:
                    logger.debug(
                        "pdfplumber extraction failed for %s, using image fallback",
                        table_block.id,
                    )

        elif self._config.mode == "pdfplumber":
            # pdfplumber mode: try pdfplumber first
            if self._pdfplumber_available:
                try:
                    return self._extract_with_pdfplumber(
                        pdf_path, table_block, table_id
                    )
                except TableExtractionError:
                    logger.debug(
                        "pdfplumber extraction failed for %s, trying heuristics",
                        table_block.id,
                    )

            # Fallback to heuristics
            try:
                return self._extract_with_heuristics(
                    table_block, text_objects, table_id
                )
            except TableExtractionError:
                pass

        # Final fallback: image
        return self._extract_as_image(pdf_path, table_block, table_id, output_dir)

    def _extract_with_heuristics(
        self,
        table_block: LayoutBlock,
        text_objects: list[Any],
        table_id: str,
    ) -> ExtractedTable:
        """Extract table using TextObject heuristics.

        Args:
            table_block: LayoutBlock for the table.
            text_objects: TextObjects from the page.
            table_id: ID for the table.

        Returns:
            ExtractedTable.

        Raises:
            TableExtractionError: If extraction fails.
        """
        bbox = table_block.bbox

        # Filter text objects within table bbox
        table_texts = []
        for tobj in text_objects:
            tobj_bbox = tobj.bbox
            # Check if text object is within table bbox
            if (
                tobj_bbox.x0 >= bbox.x0 - 5
                and tobj_bbox.x1 <= bbox.x1 + 5
                and tobj_bbox.y0 >= bbox.y0 - 5
                and tobj_bbox.y1 <= bbox.y1 + 5
            ):
                table_texts.append(tobj)

        if not table_texts:
            raise TableExtractionError("No text objects found in table region")

        # Cluster by Y coordinate (rows)
        rows = self._cluster_by_y(table_texts)

        if len(rows) < self._config.min_rows:
            raise TableExtractionError(
                f"Too few rows: {len(rows)} < {self._config.min_rows}"
            )

        # Detect column boundaries
        col_boundaries = self._detect_column_boundaries(rows)

        if len(col_boundaries) - 1 < self._config.min_columns:
            raise TableExtractionError(
                f"Too few columns: {len(col_boundaries) - 1} < {self._config.min_columns}"
            )

        # Normalize rows to cells
        normalized_rows = self._normalize_rows(rows, col_boundaries)

        # Detect header rows
        header_rows = self._detect_header_rows(normalized_rows)

        return ExtractedTable(
            id=table_id,
            layout_block_id=table_block.id,
            page_number=table_block.page_num,
            bbox=bbox,
            rows=normalized_rows,
            header_rows=header_rows,
            extraction_method="heuristic",
        )

    def _cluster_by_y(self, text_objects: list[Any]) -> list[list[Any]]:
        """Cluster text objects by Y coordinate.

        Args:
            text_objects: List of text objects.

        Returns:
            List of rows (each row is a list of text objects).
        """
        if not text_objects:
            return []

        # Sort by Y coordinate (descending - PDF coordinates)
        sorted_texts = sorted(text_objects, key=lambda t: -t.bbox.y0)

        rows: list[list[Any]] = []
        current_row: list[Any] = []
        current_y: float | None = None

        tolerance = self._config.y_tolerance

        for tobj in sorted_texts:
            y = tobj.bbox.y0

            if current_y is None or abs(y - current_y) <= tolerance:
                current_row.append(tobj)
                if current_y is None:
                    current_y = y
            else:
                if current_row:
                    # Sort row by X coordinate
                    current_row.sort(key=lambda t: t.bbox.x0)
                    rows.append(current_row)
                current_row = [tobj]
                current_y = y

        if current_row:
            current_row.sort(key=lambda t: t.bbox.x0)
            rows.append(current_row)

        return rows

    def _detect_column_boundaries(
        self, rows: list[list[Any]]
    ) -> list[float]:
        """Detect column boundaries from all rows.

        Args:
            rows: List of rows.

        Returns:
            List of X coordinates marking column boundaries.
        """
        # Collect all X start positions
        x_positions: list[float] = []
        for row in rows:
            for tobj in row:
                x_positions.append(tobj.bbox.x0)

        if not x_positions:
            return []

        # Sort and find gaps
        x_positions.sort()

        # Start with minimum X
        boundaries = [x_positions[0] - 5]

        # Find significant gaps
        prev_x = x_positions[0]
        for x in x_positions[1:]:
            gap = x - prev_x
            # Use dynamic gap detection based on average text width
            if gap > 20:  # Significant gap threshold
                boundaries.append((prev_x + x) / 2)
            prev_x = x

        # Add end boundary
        boundaries.append(max(x_positions) + 100)

        return boundaries

    def _normalize_rows(
        self,
        rows: list[list[Any]],
        col_boundaries: list[float],
    ) -> list[list[TableCell]]:
        """Normalize rows to have consistent column count.

        Args:
            rows: Raw rows from clustering.
            col_boundaries: Column boundary X coordinates.

        Returns:
            Normalized rows as TableCell lists.
        """
        num_cols = len(col_boundaries) - 1
        normalized: list[list[TableCell]] = []

        for row_idx, row in enumerate(rows):
            cells: list[TableCell] = []

            for col_idx in range(num_cols):
                left = col_boundaries[col_idx]
                right = col_boundaries[col_idx + 1]

                # Find text objects in this column
                col_texts = []
                for tobj in row:
                    center_x = (tobj.bbox.x0 + tobj.bbox.x1) / 2
                    if left <= center_x < right:
                        col_texts.append(tobj.text)

                # Join texts in the cell
                cell_text = " ".join(col_texts) if col_texts else ""

                cells.append(
                    TableCell(
                        text=cell_text,
                        row=row_idx,
                        col=col_idx,
                        is_header=(row_idx == 0),
                    )
                )

            normalized.append(cells)

        return normalized

    def _detect_header_rows(
        self, rows: list[list[TableCell]]
    ) -> int:
        """Detect number of header rows.

        Args:
            rows: Normalized rows.

        Returns:
            Number of header rows.
        """
        # Simple heuristic: first row is header
        # Could be extended to detect bold/larger fonts
        return 1 if rows else 0

    def _extract_with_pdfplumber(
        self,
        pdf_path: Path,
        table_block: LayoutBlock,
        table_id: str,
    ) -> ExtractedTable:
        """Extract table using pdfplumber.

        Args:
            pdf_path: Path to the PDF.
            table_block: LayoutBlock for the table.
            table_id: ID for the table.

        Returns:
            ExtractedTable.

        Raises:
            TableExtractionError: If extraction fails.
        """
        try:
            import pdfplumber
        except ImportError:
            raise TableExtractionError("pdfplumber not installed")

        bbox = table_block.bbox
        page_num = table_block.page_num

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    raise TableExtractionError(f"Page {page_num} not found")

                page = pdf.pages[page_num]
                page_height = page.height

                # Convert PDF coordinates to pdfplumber coordinates
                # PDF: origin at bottom-left
                # pdfplumber: origin at top-left
                crop_bbox = (
                    bbox.x0,
                    page_height - bbox.y1,
                    bbox.x1,
                    page_height - bbox.y0,
                )

                cropped = page.within_bbox(crop_bbox)
                tables = cropped.extract_tables()

                if not tables:
                    raise TableExtractionError("No tables found by pdfplumber")

                # Use first table
                raw_table = tables[0]

                # Convert to TableCell format
                rows: list[list[TableCell]] = []
                for row_idx, row in enumerate(raw_table):
                    cells = []
                    for col_idx, cell_text in enumerate(row):
                        cells.append(
                            TableCell(
                                text=cell_text or "",
                                row=row_idx,
                                col=col_idx,
                                is_header=(row_idx == 0),
                            )
                        )
                    rows.append(cells)

                return ExtractedTable(
                    id=table_id,
                    layout_block_id=table_block.id,
                    page_number=page_num,
                    bbox=bbox,
                    rows=rows,
                    header_rows=1,
                    extraction_method="pdfplumber",
                )

        except Exception as e:
            raise TableExtractionError(f"pdfplumber extraction failed: {e}")

    def _extract_as_image(
        self,
        pdf_path: Path,
        table_block: LayoutBlock,
        table_id: str,
        output_dir: Path | None,
    ) -> ExtractedImage:
        """Extract table as image (fallback).

        Args:
            pdf_path: Path to the PDF.
            table_block: LayoutBlock for the table.
            table_id: ID for the table.
            output_dir: Output directory for images.

        Returns:
            ExtractedImage.
        """
        from pdf_translator.output.image_extractor import (
            ImageExtractionConfig,
            extract_image_as_fallback,
        )

        if output_dir is None:
            output_dir = pdf_path.parent / "images"

        config = ImageExtractionConfig()
        result = extract_image_as_fallback(pdf_path, table_block, output_dir, config)

        if result is None:
            # Create a placeholder ExtractedImage
            from pdf_translator.output.image_extractor import ExtractedImage

            return ExtractedImage(
                id=table_id,
                path=Path(""),
                relative_path="",
                layout_block_id=table_block.id,
                page_number=table_block.page_num,
                bbox=table_block.bbox,
                category="table",
                caption=None,
            )

        return result
