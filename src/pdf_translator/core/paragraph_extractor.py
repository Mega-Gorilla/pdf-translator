# SPDX-License-Identifier: Apache-2.0
"""Paragraph extraction using pdftext block output."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pdf_translator.core.models import BBox, Paragraph


class ParagraphExtractor:
    """Extract Paragraph objects from pdftext dictionary output."""

    def __init__(self, page_heights: dict[int, float] | None = None) -> None:
        self._page_heights = page_heights or {}

    def extract(
        self,
        pdftext_result: list[dict[str, Any]],
        page_range: list[int] | None = None,
    ) -> list[Paragraph]:
        """Generate Paragraph list from pdftext output.

        Args:
            pdftext_result: Output of pdftext.dictionary_output().
            page_range: Optional page index list to include.

        Returns:
            List of Paragraphs.
        """
        paragraphs: list[Paragraph] = []
        page_filter = set(page_range) if page_range is not None else None
        page_numbers = (
            list(page_range)
            if page_range is not None and len(pdftext_result) == len(page_range)
            else None
        )
        if page_numbers is not None:
            page_filter = None

        for page_idx, page_data in enumerate(pdftext_result):
            if page_filter is not None and page_idx not in page_filter:
                continue
            page_number = page_numbers[page_idx] if page_numbers is not None else page_idx

            page_bbox = page_data.get("bbox")
            if page_bbox and len(page_bbox) >= 4:
                page_height = float(page_bbox[3])
            elif page_number in self._page_heights:
                page_height = float(self._page_heights[page_number])
            else:
                raise ValueError("pdftext result missing page bbox for extraction")
            blocks = page_data.get("blocks", [])

            for block_idx, block in enumerate(blocks):
                paragraph = self._process_block(
                    block,
                    page_number,
                    block_idx,
                    page_height,
                )
                if paragraph is not None:
                    paragraphs.append(paragraph)

        return paragraphs

    @staticmethod
    def extract_from_pdf(
        pdf_path: str | Path,
        page_range: list[int] | None = None,
    ) -> list[Paragraph]:
        """Extract Paragraphs directly from a PDF file.

        Args:
            pdf_path: PDF file path.
            page_range: Optional page index list.

        Returns:
            List of Paragraphs.
        """
        from pdftext.extraction import dictionary_output  # type: ignore[import-untyped]

        result = dictionary_output(str(pdf_path), page_range=page_range)
        page_heights: dict[int, float] = {}
        try:
            import pypdfium2 as pdfium  # type: ignore[import-untyped]

            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            try:
                page_heights = {
                    page_idx: pdf_doc[page_idx].get_height()
                    for page_idx in range(len(pdf_doc))
                }
            finally:
                pdf_doc.close()
        except Exception:
            page_heights = {}

        extractor = ParagraphExtractor(page_heights=page_heights)
        return extractor.extract(result, page_range)

    def _process_block(
        self,
        block: dict[str, Any],
        page_idx: int,
        block_idx: int,
        page_height: float,
    ) -> Paragraph | None:
        """Convert a single block to Paragraph."""
        lines: list[str] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            line_text = line_text.strip()
            if line_text:
                lines.append(line_text)

        if not lines:
            return None

        merged_text = " ".join(lines)
        merged_text = re.sub(r"\s+", " ", merged_text).strip()
        if not merged_text:
            return None

        block_bbox = block.get("bbox")
        if not block_bbox or len(block_bbox) < 4:
            return None

        x0, y0_top, x1, y1_bottom = block_bbox
        pdf_y0 = page_height - float(y1_bottom)
        pdf_y1 = page_height - float(y0_top)

        font_size = self._estimate_font_size(block)

        return Paragraph(
            id=f"para_p{page_idx}_b{block_idx}",
            page_number=page_idx,
            text=merged_text,
            block_bbox=BBox(x0=float(x0), y0=pdf_y0, x1=float(x1), y1=pdf_y1),
            line_count=len(lines),
            original_font_size=font_size,
        )

    @staticmethod
    def _estimate_font_size(block: dict[str, Any]) -> float:
        """Estimate font size using weighted mode across spans."""
        size_weights: dict[float, int] = defaultdict(int)
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "") or ""
                if not text:
                    continue
                font = span.get("font", {})
                size = font.get("size")
                if size is None:
                    continue
                normalized = round(float(size), 1)
                size_weights[normalized] += len(text)

        if not size_weights:
            return 12.0

        best_size, _ = max(
            size_weights.items(),
            key=lambda item: (item[1], -item[0]),
        )
        return float(best_size)
