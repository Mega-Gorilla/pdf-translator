# SPDX-License-Identifier: Apache-2.0
"""Side-by-side PDF generation module.

This module provides functionality to create side-by-side PDF pages
with translated content on the left and original content on the right.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Union

import pypdfium2 as pdfium  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class SideBySideOrder(str, Enum):
    """Order of pages in side-by-side layout."""

    TRANSLATED_ORIGINAL = "translated_original"  # Left: translated, Right: original
    ORIGINAL_TRANSLATED = "original_translated"  # Left: original, Right: translated


@dataclass
class SideBySideConfig:
    """Configuration for side-by-side PDF generation."""

    order: SideBySideOrder = SideBySideOrder.TRANSLATED_ORIGINAL
    gap: float = 0.0  # Gap between pages in points (default: no gap)


class SideBySideGenerator:
    """Generator for side-by-side PDF layouts.

    Creates a new PDF where each page contains two source pages
    placed side by side, allowing comparison between translated
    and original content.
    """

    def __init__(self, config: SideBySideConfig | None = None) -> None:
        """Initialize SideBySideGenerator.

        Args:
            config: Configuration for side-by-side generation.
                   If None, uses default configuration.
        """
        self._config = config or SideBySideConfig()

    def generate(
        self,
        translated_pdf: Union[Path, bytes],
        original_pdf: Union[Path, bytes],
    ) -> bytes:
        """Generate a side-by-side PDF from translated and original PDFs.

        Args:
            translated_pdf: Path or bytes of the translated PDF.
            original_pdf: Path or bytes of the original PDF.

        Returns:
            bytes: The generated side-by-side PDF as bytes.

        Raises:
            ValueError: If the PDFs have different page counts.
        """
        # Open source documents
        translated_doc = self._open_document(translated_pdf)
        original_doc = self._open_document(original_pdf)

        try:
            # Validate page counts
            if len(translated_doc) != len(original_doc):
                raise ValueError(
                    f"Page count mismatch: translated={len(translated_doc)}, "
                    f"original={len(original_doc)}"
                )

            page_count = len(translated_doc)
            if page_count == 0:
                raise ValueError("PDFs have no pages")

            # Determine left/right based on order
            if self._config.order == SideBySideOrder.TRANSLATED_ORIGINAL:
                left_doc, right_doc = translated_doc, original_doc
            else:
                left_doc, right_doc = original_doc, translated_doc

            # Create output document
            output_doc = pdfium.PdfDocument.new()

            for i in range(page_count):
                self._create_side_by_side_page(
                    output_doc=output_doc,
                    left_doc=left_doc,
                    right_doc=right_doc,
                    page_index=i,
                )

            # Save to bytes
            output_bytes = self._save_to_bytes(output_doc)
            output_doc.close()

            logger.info(
                "Generated side-by-side PDF: %d pages, order=%s",
                page_count,
                self._config.order.value,
            )

            return output_bytes

        finally:
            translated_doc.close()
            original_doc.close()

    def _open_document(
        self, source: Union[Path, bytes]
    ) -> pdfium.PdfDocument:
        """Open a PDF document from path or bytes.

        Args:
            source: Path or bytes of the PDF.

        Returns:
            PdfDocument: The opened document.
        """
        if isinstance(source, bytes):
            return pdfium.PdfDocument(source)
        return pdfium.PdfDocument(source)

    def _create_side_by_side_page(
        self,
        output_doc: pdfium.PdfDocument,
        left_doc: pdfium.PdfDocument,
        right_doc: pdfium.PdfDocument,
        page_index: int,
    ) -> None:
        """Create a single side-by-side page.

        Args:
            output_doc: The output document to add the page to.
            left_doc: Source document for left side.
            right_doc: Source document for right side.
            page_index: Index of the page to process.
        """
        # Get source page dimensions
        left_page = left_doc[page_index]
        right_page = right_doc[page_index]

        left_width = left_page.get_width()
        left_height = left_page.get_height()
        right_width = right_page.get_width()
        right_height = right_page.get_height()

        # Calculate output page dimensions
        gap = self._config.gap
        new_width = left_width + gap + right_width
        new_height = max(left_height, right_height)

        # Create new page
        new_page = output_doc.new_page(new_width, new_height)

        # Capture and insert left page
        left_xobj = left_doc.page_as_xobject(page_index, output_doc)
        left_obj = left_xobj.as_pageobject()

        # Center vertically if heights differ
        left_y_offset = (new_height - left_height) / 2 if left_height < new_height else 0
        if left_y_offset > 0:
            left_obj.set_matrix(pdfium.PdfMatrix().translate(0, left_y_offset))

        new_page.insert_obj(left_obj)
        left_xobj.close()

        # Capture and insert right page
        right_xobj = right_doc.page_as_xobject(page_index, output_doc)
        right_obj = right_xobj.as_pageobject()

        # Position right page with translation
        right_x_offset = left_width + gap
        right_y_offset = (new_height - right_height) / 2 if right_height < new_height else 0
        right_obj.set_matrix(
            pdfium.PdfMatrix().translate(right_x_offset, right_y_offset)
        )

        new_page.insert_obj(right_obj)
        right_xobj.close()

        # Generate content stream
        new_page.gen_content()

    def _save_to_bytes(self, doc: pdfium.PdfDocument) -> bytes:
        """Save a PDF document to bytes.

        Args:
            doc: The document to save.

        Returns:
            bytes: The PDF as bytes.
        """
        buffer = BytesIO()
        doc.save(buffer)
        return buffer.getvalue()


def create_side_by_side_pdf(
    translated_pdf: Union[Path, bytes],
    original_pdf: Union[Path, bytes],
    order: SideBySideOrder = SideBySideOrder.TRANSLATED_ORIGINAL,
    gap: float = 0.0,
) -> bytes:
    """Convenience function to create a side-by-side PDF.

    Args:
        translated_pdf: Path or bytes of the translated PDF.
        original_pdf: Path or bytes of the original PDF.
        order: Order of pages (translated_original or original_translated).
        gap: Gap between pages in points.

    Returns:
        bytes: The generated side-by-side PDF as bytes.
    """
    config = SideBySideConfig(order=order, gap=gap)
    generator = SideBySideGenerator(config)
    return generator.generate(translated_pdf, original_pdf)
