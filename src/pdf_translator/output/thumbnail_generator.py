# SPDX-License-Identifier: Apache-2.0
"""Thumbnail generator for PDF documents."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ThumbnailConfig:
    """Configuration for thumbnail generation.

    Attributes:
        width: Target thumbnail width in pixels.
            Height is calculated to maintain aspect ratio.
        page_number: Page to render (0-indexed). Default: 0 (first page).

    Note:
        Output format is fixed to PNG for simplicity and transparency support.
    """

    width: int = 400
    page_number: int = 0


class ThumbnailGenerator:
    """Generate thumbnail images from PDF pages.

    Uses pypdfium2 for rendering, similar to ImageExtractor.
    """

    def __init__(self, config: ThumbnailConfig | None = None) -> None:
        """Initialize ThumbnailGenerator.

        Args:
            config: Thumbnail generation configuration.
        """
        self._config = config or ThumbnailConfig()

    def generate(self, pdf_path: Path) -> tuple[bytes, int, int]:
        """Generate thumbnail from PDF first page.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Tuple of (image_bytes, width, height).

        Raises:
            FileNotFoundError: If PDF file not found.
            ValueError: If PDF has no pages.
        """
        import pypdfium2 as pdfium

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = pdfium.PdfDocument(pdf_path)
        try:
            if len(doc) == 0:
                raise ValueError("PDF has no pages")

            page_num = min(self._config.page_number, len(doc) - 1)
            page = doc[page_num]

            # Calculate scale to achieve target width
            page_width = page.get_width()
            scale = self._config.width / page_width

            # Render page
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()

            # Get actual dimensions
            actual_width = pil_image.width
            actual_height = pil_image.height

            # Convert to PNG bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")

            return buffer.getvalue(), actual_width, actual_height

        finally:
            doc.close()

    def generate_to_file(self, pdf_path: Path, output_path: Path) -> tuple[int, int]:
        """Generate thumbnail and save to file.

        Args:
            pdf_path: Path to PDF file.
            output_path: Path to save thumbnail.

        Returns:
            Tuple of (width, height).
        """
        image_bytes, width, height = self.generate(pdf_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        return width, height
