# SPDX-License-Identifier: Apache-2.0
"""Image extraction from PDF for Markdown output."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pypdfium2 as pdfium

from pdf_translator.core.models import BBox, LayoutBlock

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Categories that should be extracted as images
IMAGE_CATEGORIES = frozenset({"image", "chart"})


@dataclass
class ImageExtractionConfig:
    """Image extraction configuration."""

    enabled: bool = True
    output_dir: Path | None = None
    relative_path: str = "images"
    format: str = "png"  # png or jpeg
    quality: int = 95  # JPEG quality (1-100)
    min_size: tuple[int, int] = (50, 50)  # Minimum size in pixels
    dpi: int = 150  # Render resolution
    naming: str = "sequential"  # sequential or page_index


@dataclass
class ExtractedImage:
    """Extracted image information."""

    id: str
    path: Path
    relative_path: str
    layout_block_id: str
    page_number: int
    bbox: BBox
    category: str
    caption: str | None = None


class ImageExtractor:
    """PDF image extractor.

    Extracts images from PDF based on LayoutBlock regions.
    """

    def __init__(self, config: ImageExtractionConfig | None = None) -> None:
        """Initialize ImageExtractor.

        Args:
            config: Image extraction configuration.
        """
        self._config = config or ImageExtractionConfig()

    def extract(
        self,
        pdf_path: Path,
        layout_blocks: dict[int, list[LayoutBlock]],
        output_dir: Path,
    ) -> list[ExtractedImage]:
        """Extract images from PDF based on layout blocks.

        Args:
            pdf_path: Path to the source PDF.
            layout_blocks: Dict of page_number -> list of LayoutBlocks.
            output_dir: Directory to save extracted images.

        Returns:
            List of ExtractedImage objects.
        """
        if not self._config.enabled:
            return []

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted: list[ExtractedImage] = []
        counters: dict[str, int] = {}

        # Open PDF
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            for page_num, blocks in layout_blocks.items():
                # Filter image/chart blocks
                image_blocks = [
                    b for b in blocks if b.raw_category.value in IMAGE_CATEGORIES
                ]

                for block in image_blocks:
                    result = self._extract_block_image(
                        pdf,
                        page_num,
                        block,
                        output_dir,
                        counters,
                        layout_blocks,
                    )
                    if result:
                        extracted.append(result)
        finally:
            pdf.close()

        logger.info("Extracted %d images from %s", len(extracted), pdf_path.name)
        return extracted

    def _extract_block_image(
        self,
        pdf: pdfium.PdfDocument,
        page_num: int,
        block: LayoutBlock,
        output_dir: Path,
        counters: dict[str, int],
        all_blocks: dict[int, list[LayoutBlock]],
    ) -> ExtractedImage | None:
        """Extract image from a single layout block.

        Args:
            pdf: Open PDF document.
            page_num: Page number (0-indexed).
            block: LayoutBlock to extract.
            output_dir: Directory to save images.
            counters: Counter dict for sequential naming.
            all_blocks: All layout blocks (for caption search).

        Returns:
            ExtractedImage or None if extraction failed.
        """
        try:
            page = pdf[page_num]
            page_width, page_height = page.get_size()

            # Convert bbox to render region
            # pypdfium2 uses PDF coordinate system (origin at bottom-left)
            bbox = block.bbox

            # Calculate scale factor for DPI
            scale = self._config.dpi / 72.0

            # Render the specific region
            bitmap = page.render(
                scale=scale,
                crop=(bbox.x0, bbox.y0, page_width - bbox.x1, page_height - bbox.y1),
            )

            # Check minimum size
            width, height = bitmap.width, bitmap.height
            if width < self._config.min_size[0] or height < self._config.min_size[1]:
                logger.debug(
                    "Skipping small image %s: %dx%d < %s",
                    block.id,
                    width,
                    height,
                    self._config.min_size,
                )
                return None

            # Generate filename
            category = block.raw_category.value
            filename = self._generate_filename(category, page_num, counters)

            # Convert to PIL Image and save
            pil_image = bitmap.to_pil()

            # Determine output format and path
            ext = self._config.format.lower()
            if ext not in ("png", "jpeg", "jpg"):
                ext = "png"
            if ext == "jpg":
                ext = "jpeg"

            image_path = output_dir / f"{filename}.{ext}"

            # Save image
            save_kwargs = {}
            if ext == "jpeg":
                save_kwargs["quality"] = self._config.quality
                # Convert RGBA to RGB for JPEG
                if pil_image.mode == "RGBA":
                    pil_image = pil_image.convert("RGB")

            pil_image.save(image_path, **save_kwargs)

            # Find caption
            caption = self._find_caption(block, all_blocks.get(page_num, []))

            # Calculate relative path
            rel_path = f"{self._config.relative_path}/{filename}.{ext}"

            return ExtractedImage(
                id=filename,
                path=image_path,
                relative_path=rel_path,
                layout_block_id=block.id,
                page_number=page_num,
                bbox=bbox,
                category=category,
                caption=caption,
            )

        except Exception as e:
            logger.warning(
                "Failed to extract image from block %s: %s",
                block.id,
                e,
            )
            return None

    def _generate_filename(
        self,
        category: str,
        page_num: int,
        counters: dict[str, int],
    ) -> str:
        """Generate filename for extracted image.

        Args:
            category: Image category (image, chart, etc.).
            page_num: Page number.
            counters: Counter dict for sequential naming.

        Returns:
            Filename without extension.
        """
        # Map category to prefix
        prefix_map = {
            "image": "figure",
            "chart": "chart",
            "table": "table",
        }
        prefix = prefix_map.get(category, "image")

        if self._config.naming == "page_index":
            # page_index naming: p1_figure_001
            key = f"p{page_num}_{prefix}"
            count = counters.get(key, 0) + 1
            counters[key] = count
            return f"p{page_num + 1}_{prefix}_{count:03d}"
        else:
            # sequential naming: figure_001
            count = counters.get(prefix, 0) + 1
            counters[prefix] = count
            return f"{prefix}_{count:03d}"

    def _find_caption(
        self,
        image_block: LayoutBlock,
        page_blocks: list[LayoutBlock],
    ) -> str | None:
        """Find caption for an image block.

        Searches for figure_title blocks near the image.

        Args:
            image_block: The image block to find caption for.
            page_blocks: All blocks on the same page.

        Returns:
            Caption text or None.
        """
        # Note: LayoutBlock does not contain text content directly.
        # Caption association is handled by MarkdownWriter using Paragraphs
        # with figure_title category that are spatially close to the image.
        # This method is a placeholder for future enhancement.
        return None


def extract_image_as_fallback(
    pdf_path: Path,
    block: LayoutBlock,
    output_dir: Path,
    config: ImageExtractionConfig | None = None,
) -> ExtractedImage | None:
    """Extract a single block as image (for table fallback).

    Args:
        pdf_path: Path to the PDF.
        block: LayoutBlock to extract.
        output_dir: Output directory for images.
        config: Extraction configuration.

    Returns:
        ExtractedImage or None if extraction failed.
    """
    cfg = config or ImageExtractionConfig()
    extractor = ImageExtractor(cfg)

    # Create a single-block dict
    blocks_dict = {block.page_num: [block]}

    # Use a counter dict
    counters: dict[str, int] = {"table": 0}

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        return extractor._extract_block_image(
            pdf,
            block.page_num,
            block,
            output_dir,
            counters,
            blocks_dict,
        )
    finally:
        pdf.close()
