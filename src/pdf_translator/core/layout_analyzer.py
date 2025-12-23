# SPDX-License-Identifier: Apache-2.0
"""Layout analysis using PP-DocLayoutV2.

This module provides the LayoutAnalyzer class for detecting document structure
elements (text, tables, figures, formulas, etc.) using PaddleOCR's PP-DocLayoutV2.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pypdfium2 as pdfium  # type: ignore[import-untyped]

from .layout_utils import convert_image_to_pdf_coords
from .models import (
    LayoutBlock,
    RawLayoutCategory,
)

if TYPE_CHECKING:
    from PIL import Image  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """Layout analyzer using PP-DocLayoutV2.

    This class detects document structure elements from PDF pages using
    PaddleOCR's PP-DocLayoutV2 model. It handles:
    - PDF rendering to images
    - Layout detection
    - Coordinate transformation (image → PDF coordinates)
    - Category mapping (raw → project)

    Attributes:
        DEFAULT_MODEL: Default model name (PP-DocLayoutV2)
        DEFAULT_RENDER_SCALE: Default render scale (2.0 for better detection)
    """

    DEFAULT_MODEL = "PP-DocLayoutV2"
    DEFAULT_RENDER_SCALE = 2.0  # 2x zoom for better detection

    def __init__(
        self,
        model_name: str | None = None,
        render_scale: float | None = None,
    ) -> None:
        """Initialize LayoutAnalyzer.

        Args:
            model_name: PP-DocLayout model name (default: PP-DocLayoutV2)
            render_scale: PDF rendering scale (default: 2.0)
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._render_scale = render_scale or self.DEFAULT_RENDER_SCALE
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        """Lazy initialization of the model.

        Returns:
            LayoutDetection model instance
        """
        if self._model is None:
            from paddleocr import LayoutDetection  # type: ignore[import-not-found]

            logger.info("Initializing %s model...", self._model_name)
            self._model = LayoutDetection(model_name=self._model_name)
            logger.info("Model initialized successfully")
        return self._model

    def _render_page_to_image(
        self,
        pdf_path: str | Path,
        page_num: int,
        pdf_doc: Any | None = None,
    ) -> tuple["Image.Image", float, float]:
        """Render a PDF page to an image.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            pdf_doc: Optional pre-opened PdfDocument for efficiency

        Returns:
            Tuple of (PIL Image, page_width, page_height)
        """
        from PIL import Image as PILImage

        if pdf_doc is None:
            pdf_doc = pdfium.PdfDocument(str(pdf_path))
        page = pdf_doc[page_num]

        # Get original page dimensions (in points)
        page_width = page.get_width()
        page_height = page.get_height()

        # Render with scale
        bitmap = page.render(scale=self._render_scale)
        image: PILImage.Image = bitmap.to_pil()

        return image, page_width, page_height

    def _parse_detections(
        self,
        output: list[dict[str, Any]],
        page_width: float,
        page_height: float,
        image_width: int,
        image_height: int,
        page_num: int,
    ) -> list[LayoutBlock]:
        """Parse model output into LayoutBlocks.

        Args:
            output: Raw model output
            page_width: PDF page width in points
            page_height: PDF page height in points
            image_width: Rendered image width in pixels
            image_height: Rendered image height in pixels
            page_num: Page number (0-indexed)

        Returns:
            List of LayoutBlock objects
        """
        blocks: list[LayoutBlock] = []

        for res in output:
            boxes = res.get("boxes", [])

            for det in boxes:
                label = det.get("label", "unknown")
                score = float(det.get("score", 0))
                coord = det.get("coordinate", [])

                if len(coord) != 4:
                    logger.warning("Invalid coordinate length: %s", coord)
                    continue

                # Convert numpy float32 to Python float
                image_bbox = tuple(float(c) for c in coord)

                # Convert to PDF coordinates
                pdf_bbox = convert_image_to_pdf_coords(
                    image_bbox,  # type: ignore[arg-type]
                    page_width,
                    page_height,
                    image_width,
                    image_height,
                )

                # Parse raw category
                try:
                    raw_category = RawLayoutCategory(label)
                except ValueError:
                    logger.warning("Unknown category: %s", label)
                    raw_category = RawLayoutCategory.UNKNOWN

                block = LayoutBlock(
                    id=str(uuid.uuid4()),
                    bbox=pdf_bbox,
                    raw_category=raw_category,
                    confidence=score,
                    page_num=page_num,
                )
                blocks.append(block)

        return blocks

    def analyze(
        self,
        pdf_path: str | Path,
        page_num: int,
    ) -> list[LayoutBlock]:
        """Analyze layout of a single page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            List of detected LayoutBlocks
        """
        model = self._ensure_model()

        # Render page to image
        image, page_width, page_height = self._render_page_to_image(pdf_path, page_num)
        image_width, image_height = image.size

        # Save to temp file for model input
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            image.save(temp_path)

        try:
            # Run detection
            output = model.predict(temp_path, batch_size=1, layout_nms=True)

            # Parse results
            blocks = self._parse_detections(
                output,
                page_width,
                page_height,
                image_width,
                image_height,
                page_num,
            )

            logger.info(
                "Page %d: detected %d blocks",
                page_num,
                len(blocks),
            )

            return blocks

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def analyze_all(
        self,
        pdf_path: str | Path,
    ) -> dict[int, list[LayoutBlock]]:
        """Analyze layout of all pages.

        Opens PDF once and reuses for all pages (more efficient than
        calling analyze() separately for each page).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict mapping page number → list of LayoutBlocks
        """
        model = self._ensure_model()
        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        page_count = len(pdf_doc)

        logger.info("Analyzing %d pages...", page_count)

        results: dict[int, list[LayoutBlock]] = {}
        for page_num in range(page_count):
            # Render page to image (reusing pdf_doc)
            image, page_width, page_height = self._render_page_to_image(
                pdf_path, page_num, pdf_doc=pdf_doc
            )
            image_width, image_height = image.size

            # Save to temp file for model input
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
                image.save(temp_path)

            try:
                # Run detection
                output = model.predict(temp_path, batch_size=1, layout_nms=True)

                # Parse results
                blocks = self._parse_detections(
                    output,
                    page_width,
                    page_height,
                    image_width,
                    image_height,
                    page_num,
                )

                logger.info(
                    "Page %d: detected %d blocks",
                    page_num,
                    len(blocks),
                )

                results[page_num] = blocks

            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

        return results
