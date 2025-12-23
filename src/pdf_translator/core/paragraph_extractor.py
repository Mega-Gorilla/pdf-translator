# SPDX-License-Identifier: Apache-2.0
"""Paragraph extraction using pdftext block output."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pdf_translator.core.models import BBox, Color, Paragraph


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
        is_bold = self._estimate_is_bold(block)
        is_italic = self._estimate_is_italic(block)
        font_name = self._estimate_font_name(block)
        text_color = self._estimate_text_color(block)
        rotation = self._estimate_rotation(block)
        alignment = self._estimate_alignment(block)

        return Paragraph(
            id=f"para_p{page_idx}_b{block_idx}",
            page_number=page_idx,
            text=merged_text,
            block_bbox=BBox(x0=float(x0), y0=pdf_y0, x1=float(x1), y1=pdf_y1),
            line_count=len(lines),
            original_font_size=font_size,
            is_bold=is_bold,
            is_italic=is_italic,
            font_name=font_name,
            text_color=text_color,
            rotation=rotation,
            alignment=alignment,
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

    @staticmethod
    def _estimate_is_bold(block: dict[str, Any]) -> bool:
        """Estimate if block is bold using font weight.

        Uses weighted voting based on text length to determine
        if the majority of the block is bold.

        Args:
            block: Block dictionary from pdftext.

        Returns:
            True if the block is predominantly bold.
        """
        bold_weight = 0
        normal_weight = 0

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "") or ""
                if not text.strip():
                    continue

                font = span.get("font", {})
                weight = font.get("weight", 400)
                font_name = font.get("name", "").lower()

                # Check weight (700+ is bold) or font name contains "bold"/"medi"
                is_bold = (
                    weight >= 700
                    or "bold" in font_name
                    or "medi" in font_name  # Medium weight fonts like NimbusRomNo9L-Medi
                )

                text_len = len(text)
                if is_bold:
                    bold_weight += text_len
                else:
                    normal_weight += text_len

        # Return True if majority of text is bold
        return bold_weight > normal_weight

    @staticmethod
    def _estimate_is_italic(block: dict[str, Any]) -> bool:
        """Estimate if block is italic using font flags or name.

        Args:
            block: Block dictionary from pdftext.

        Returns:
            True if the block is predominantly italic.
        """
        italic_weight = 0
        normal_weight = 0

        # PDF font flags bit 6 (0x40) indicates italic
        ITALIC_FLAG = 0x40

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "") or ""
                if not text.strip():
                    continue

                font = span.get("font", {})
                flags = font.get("flags", 0)
                font_name = font.get("name", "").lower()

                # Check italic flag or font name contains "italic"/"oblique"
                is_italic = (
                    bool(flags & ITALIC_FLAG)
                    or "italic" in font_name
                    or "oblique" in font_name
                )

                text_len = len(text)
                if is_italic:
                    italic_weight += text_len
                else:
                    normal_weight += text_len

        return italic_weight > normal_weight

    @staticmethod
    def _estimate_font_name(block: dict[str, Any]) -> str | None:
        """Estimate the predominant font name in the block.

        Args:
            block: Block dictionary from pdftext.

        Returns:
            Most common font name or None.
        """
        font_weights: dict[str, int] = defaultdict(int)

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "") or ""
                if not text.strip():
                    continue

                font = span.get("font", {})
                name = font.get("name", "")
                if name:
                    font_weights[name] += len(text)

        if not font_weights:
            return None

        best_font, _ = max(font_weights.items(), key=lambda x: x[1])
        return best_font

    @staticmethod
    def _estimate_text_color(block: dict[str, Any]) -> Color | None:
        """Estimate text color from pdftext data.

        Note: pdftext does not provide color information directly.
        This returns None; color extraction requires pypdfium2.

        Args:
            block: Block dictionary from pdftext.

        Returns:
            Color or None (pdftext doesn't provide color).
        """
        # pdftext doesn't provide color information
        # Color extraction would require pypdfium2 direct access
        return None

    @staticmethod
    def _estimate_rotation(block: dict[str, Any]) -> float:
        """Estimate text rotation from spans.

        Args:
            block: Block dictionary from pdftext.

        Returns:
            Predominant rotation angle in degrees.
        """
        rotation_weights: dict[float, int] = defaultdict(int)

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "") or ""
                if not text.strip():
                    continue

                rotation = span.get("rotation", 0.0)
                rotation_weights[rotation] += len(text)

        if not rotation_weights:
            return 0.0

        best_rotation, _ = max(rotation_weights.items(), key=lambda x: x[1])
        return float(best_rotation)

    @staticmethod
    def _estimate_alignment(block: dict[str, Any]) -> str:
        """Estimate text alignment from line positions.

        Args:
            block: Block dictionary from pdftext.

        Returns:
            Alignment string: "left", "center", "right", or "justify".
        """
        lines = block.get("lines", [])
        if not lines:
            return "left"

        block_bbox = block.get("bbox")
        if not block_bbox or len(block_bbox) < 4:
            return "left"

        block_x0, _, block_x1, _ = block_bbox

        if len(lines) == 1:
            # Single line: check position relative to block
            line = lines[0]
            line_bbox = line.get("bbox")
            if not line_bbox or len(line_bbox) < 4:
                return "left"

            line_x0, _, line_x1, _ = line_bbox
            line_center = (line_x0 + line_x1) / 2
            block_center = (block_x0 + block_x1) / 2

            # Check if centered
            if abs(line_center - block_center) < 5:
                return "center"

            return "left"

        # Multiple lines: analyze pattern
        left_positions = []
        right_positions = []

        for line in lines:
            line_bbox = line.get("bbox")
            if not line_bbox or len(line_bbox) < 4:
                continue
            line_x0, _, line_x1, _ = line_bbox
            left_positions.append(line_x0)
            right_positions.append(line_x1)

        if not left_positions:
            return "left"

        left_std = max(left_positions) - min(left_positions)
        right_std = max(right_positions) - min(right_positions)

        # Justified: both edges consistent
        if left_std < 5 and right_std < 5:
            return "justify"

        # Left aligned: consistent left edge
        if left_std < 5:
            return "left"

        # Right aligned: consistent right edge
        if right_std < 5:
            return "right"

        # Check center alignment
        centers = [(left + right) / 2 for left, right in zip(left_positions, right_positions)]
        center_std = max(centers) - min(centers)
        if center_std < 10:
            return "center"

        # Default to left
        return "left"
