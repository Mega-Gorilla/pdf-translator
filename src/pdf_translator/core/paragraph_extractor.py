# SPDX-License-Identifier: Apache-2.0
"""Paragraph extraction using pdftext block output."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pdf_translator.core.models import BBox, Color, ListMarker, Paragraph


# =============================================================================
# リスト検出用定数
# =============================================================================

# 箇条書きマーカー文字
BULLET_CHARS: frozenset[str] = frozenset("•◦○●◆◇▸▹‣⁃")

# 丸数字（①〜⑳）- 誤検知リスクが極めて低い
CIRCLED_NUMBERS: frozenset[str] = frozenset("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳")
CIRCLED_NUMBERS_STR = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

# 番号付きリストのパターン
NUMBERED_DOT_PATTERN = re.compile(r"^([1-9]\d{0,2})\.$")  # 1., 2., 3. (1-999)
NUMBERED_PAREN_PATTERN = re.compile(r"^([1-9]\d{0,2})\)$")  # 1), 2), 3) (1-999)

# マーカーspan幅の閾値
MAX_MARKER_SPAN_WIDTH_ABSOLUTE = 30.0  # 絶対上限 (pt)
MAX_MARKER_SPAN_WIDTH_RATIO = 2.5  # font_size に対する比率


@dataclass
class ExtractorConfig:
    """Configuration for paragraph extraction.

    Attributes:
        detect_lists: Enable list marker detection and line-level splitting.
            When True (default), list items are extracted as separate paragraphs.
            When False, all lines are merged with spaces (legacy behavior).
    """

    detect_lists: bool = True


class ParagraphExtractor:
    """Extract Paragraph objects from pdftext dictionary output."""

    def __init__(self, page_heights: dict[int, float] | None = None) -> None:
        self._page_heights = page_heights or {}

    def extract(
        self,
        pdftext_result: list[dict[str, Any]],
        page_range: list[int] | None = None,
        config: ExtractorConfig | None = None,
    ) -> list[Paragraph]:
        """Generate Paragraph list from pdftext output.

        Args:
            pdftext_result: Output of pdftext.dictionary_output().
            page_range: Optional page index list to include.
            config: Extraction configuration. Uses defaults if None.

        Returns:
            List of Paragraphs.
        """
        if config is None:
            config = ExtractorConfig()

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
                block_paragraphs = self._process_block(
                    block,
                    page_number,
                    block_idx,
                    page_height,
                    config,
                )
                paragraphs.extend(block_paragraphs)

        return paragraphs

    @staticmethod
    def extract_from_pdf(
        pdf_path: str | Path,
        page_range: list[int] | None = None,
        config: ExtractorConfig | None = None,
    ) -> list[Paragraph]:
        """Extract Paragraphs directly from a PDF file.

        Args:
            pdf_path: PDF file path.
            page_range: Optional page index list.
            config: Extraction configuration. Uses defaults if None.

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
        return extractor.extract(result, page_range, config)

    def _process_block(
        self,
        block: dict[str, Any],
        page_idx: int,
        block_idx: int,
        page_height: float,
        config: ExtractorConfig,
    ) -> list[Paragraph]:
        """Convert a single block to Paragraph(s) with list marker detection.

        Returns:
            List of Paragraphs. For list blocks with detect_lists=True, uses
            continuation line handling. For regular text, all lines are merged
            into one Paragraph.
        """
        lines = block.get("lines", [])
        if not lines:
            return []

        # Get block-level attributes
        font_size = self._estimate_font_size(block)
        is_bold = self._estimate_is_bold(block)
        is_italic = self._estimate_is_italic(block)
        font_name = self._estimate_font_name(block)
        text_color = self._estimate_text_color(block)
        rotation = self._estimate_rotation(block)
        alignment = self._estimate_alignment(block, rotation)

        # First pass: detect markers and extract content for each line
        line_data: list[tuple[ListMarker | None, str, dict[str, Any]]] = []

        for line in lines:
            spans = line.get("spans", [])
            line_font_size = self._estimate_line_font_size(spans) or font_size

            # Only detect markers if feature is enabled
            marker = None
            if config.detect_lists:
                marker = self._detect_list_marker(line, line_font_size)

            # Extract content text (skipping marker if present)
            content = self._extract_content_after_marker(spans, marker is not None)
            content = content.rstrip()  # 末尾空白のみ除去、先頭は保持

            if content or marker:
                line_data.append((marker, content, line))

        if not line_data:
            return []

        # Check if any line has a list marker
        has_list = any(m for m, _, _ in line_data)

        if has_list and config.detect_lists:
            # List block: use continuation line handling
            return self._process_block_with_continuation(
                line_data,
                page_idx,
                block_idx,
                page_height,
                font_size,
                is_bold,
                is_italic,
                font_name,
                text_color,
                rotation,
                alignment,
            )
        else:
            # Regular block: merge all lines into one Paragraph
            merged_text = " ".join(content for _, content, _ in line_data if content)
            merged_text = re.sub(r"\s+", " ", merged_text).strip()

            if not merged_text:
                return []

            block_bbox = block.get("bbox")
            if not block_bbox or len(block_bbox) < 4:
                return []

            x0, y0_top, x1, y1_bottom = block_bbox
            pdf_y0 = page_height - float(y1_bottom)
            pdf_y1 = page_height - float(y0_top)

            para = Paragraph(
                id=f"para_p{page_idx}_b{block_idx}_i0",
                page_number=page_idx,
                text=merged_text,
                block_bbox=BBox(x0=float(x0), y0=pdf_y0, x1=float(x1), y1=pdf_y1),
                line_count=len(line_data),
                original_font_size=font_size,
                is_bold=is_bold,
                is_italic=is_italic,
                font_name=font_name,
                text_color=text_color,
                rotation=rotation,
                alignment=alignment,
            )
            return [para]

    def _process_block_with_continuation(
        self,
        line_data: list[tuple[ListMarker | None, str, dict[str, Any]]],
        page_idx: int,
        block_idx: int,
        page_height: float,
        font_size: float,
        is_bold: bool,
        is_italic: bool,
        font_name: str | None,
        text_color: Color | None,
        rotation: float,
        alignment: str,
    ) -> list[Paragraph]:
        """Process lines with continuation line handling.

        Lines without markers that follow a marker line are treated as
        continuations of the previous list item.

        When continuation lines are merged:
        - bbox: Union of all merged lines' bboxes (with PDF coordinate conversion)
        - line_count: Sum of merged lines
        - id: Deterministic format para_p{page}_b{block}_i{item}
        """
        paragraphs: list[Paragraph] = []
        current_marker: ListMarker | None = None
        current_content: list[str] = []
        current_lines: list[dict[str, Any]] = []
        item_idx = 0

        for marker, content, line in line_data:
            if not content:
                continue

            if marker is not None:
                # New list item - flush previous if exists
                if current_content and current_marker is not None:
                    para = self._create_paragraph_from_lines(
                        current_lines,
                        " ".join(current_content),
                        page_idx,
                        block_idx,
                        item_idx,
                        page_height,
                        font_size,
                        is_bold,
                        is_italic,
                        font_name,
                        text_color,
                        rotation,
                        alignment,
                        current_marker,
                    )
                    paragraphs.append(para)
                    item_idx += 1

                # Start new item
                current_marker = marker
                current_content = [content]
                current_lines = [line]
            elif current_marker is not None:
                # Continuation line - append to current item
                current_content.append(content)
                current_lines.append(line)
            else:
                # Regular text line (no active list context)
                para = self._create_paragraph_from_lines(
                    [line],
                    content,
                    page_idx,
                    block_idx,
                    item_idx,
                    page_height,
                    font_size,
                    is_bold,
                    is_italic,
                    font_name,
                    text_color,
                    rotation,
                    alignment,
                    None,
                )
                paragraphs.append(para)
                item_idx += 1

        # Flush final item
        if current_content and current_marker is not None:
            para = self._create_paragraph_from_lines(
                current_lines,
                " ".join(current_content),
                page_idx,
                block_idx,
                item_idx,
                page_height,
                font_size,
                is_bold,
                is_italic,
                font_name,
                text_color,
                rotation,
                alignment,
                current_marker,
            )
            paragraphs.append(para)

        return paragraphs

    def _create_paragraph_from_lines(
        self,
        lines: list[dict[str, Any]],
        text: str,
        page_idx: int,
        block_idx: int,
        item_idx: int,
        page_height: float,
        font_size: float,
        is_bold: bool,
        is_italic: bool,
        font_name: str | None,
        text_color: Color | None,
        rotation: float,
        alignment: str,
        list_marker: ListMarker | None,
    ) -> Paragraph:
        """Create a Paragraph from one or more lines.

        Args:
            lines: List of line dictionaries (may contain multiple for continuations).
            text: Merged text content.
            page_idx: Page index.
            block_idx: Block index.
            item_idx: Item index within block (for deterministic ID).
            page_height: Page height for PDF coordinate conversion.
            font_size: Font size in points.
            is_bold: Whether text is bold.
            is_italic: Whether text is italic.
            font_name: Font name.
            text_color: Text color.
            rotation: Text rotation in degrees.
            alignment: Text alignment.
            list_marker: List marker if detected.

        Returns:
            Paragraph with bbox union and correct line_count.
        """
        # Calculate bbox as union of all lines (in pdftext top-left coordinates)
        raw_bboxes = [line["bbox"] for line in lines if "bbox" in line]
        if raw_bboxes:
            x0 = min(b[0] for b in raw_bboxes)
            y0_top = min(b[1] for b in raw_bboxes)
            x1 = max(b[2] for b in raw_bboxes)
            y1_bottom = max(b[3] for b in raw_bboxes)

            # Convert to PDF coordinates (origin at bottom-left)
            pdf_y0 = page_height - float(y1_bottom)
            pdf_y1 = page_height - float(y0_top)
            merged_bbox = BBox(x0=float(x0), y0=pdf_y0, x1=float(x1), y1=pdf_y1)
        else:
            merged_bbox = BBox(0, 0, 0, 0)

        return Paragraph(
            id=f"para_p{page_idx}_b{block_idx}_i{item_idx}",
            page_number=page_idx,
            text=text,
            block_bbox=merged_bbox,
            line_count=len(lines),
            original_font_size=font_size,
            is_bold=is_bold,
            is_italic=is_italic,
            font_name=font_name,
            text_color=text_color,
            rotation=rotation,
            alignment=alignment,
            list_marker=list_marker,
        )

    @staticmethod
    def _is_marker_span_width(span_width: float, font_size: float) -> bool:
        """Check if span width is within marker threshold.

        Uses both absolute and relative thresholds:
        - Absolute: span_width <= MAX_MARKER_SPAN_WIDTH_ABSOLUTE
        - Relative: span_width <= font_size * MAX_MARKER_SPAN_WIDTH_RATIO

        Args:
            span_width: Width of the span in points.
            font_size: Font size in points.

        Returns:
            True if span width is acceptable for a marker.
        """
        absolute_ok = span_width <= MAX_MARKER_SPAN_WIDTH_ABSOLUTE
        relative_ok = span_width <= font_size * MAX_MARKER_SPAN_WIDTH_RATIO
        return absolute_ok and relative_ok

    @staticmethod
    def _detect_list_marker(
        line: dict[str, Any],
        font_size: float,
    ) -> ListMarker | None:
        """Detect list marker from span structure.

        Analyzes the first span of a line to detect bullet or numbered markers.
        Markers must be in a separate, narrow span.

        Args:
            line: Line dictionary from pdftext.
            font_size: Font size for relative width check.

        Returns:
            ListMarker if detected, None otherwise.
        """
        spans = line.get("spans", [])
        if len(spans) < 2:
            # Need at least 2 spans (marker + content)
            return None

        first_span = spans[0]
        first_text = first_span.get("text", "").strip()
        if not first_text:
            return None

        # Check span width
        first_bbox = first_span.get("bbox", [])
        if len(first_bbox) >= 4:
            span_width = first_bbox[2] - first_bbox[0]
            if not ParagraphExtractor._is_marker_span_width(span_width, font_size):
                return None

        # Check for bullet markers
        if first_text in BULLET_CHARS:
            return ListMarker(
                marker_type="bullet",
                marker_text=first_text,
                number=None,
            )

        # Check for circled numbers (①, ②, など)
        if first_text in CIRCLED_NUMBERS:
            circled_index = CIRCLED_NUMBERS_STR.index(first_text) + 1
            return ListMarker(
                marker_type="numbered",
                marker_text=first_text,
                number=circled_index,
            )

        # Check for numbered list: 1., 2., etc.
        match = NUMBERED_DOT_PATTERN.match(first_text)
        if match:
            return ListMarker(
                marker_type="numbered",
                marker_text=first_text,
                number=int(match.group(1)),
            )

        # Check for numbered list: 1), 2), etc.
        match = NUMBERED_PAREN_PATTERN.match(first_text)
        if match:
            return ListMarker(
                marker_type="numbered",
                marker_text=first_text,
                number=int(match.group(1)),
            )

        return None

    @staticmethod
    def _extract_content_after_marker(
        spans: list[dict[str, Any]],
        has_marker: bool,
    ) -> str:
        """Extract text content, skipping marker span if present.

        Args:
            spans: List of span dictionaries.
            has_marker: Whether first span is a list marker.

        Returns:
            Concatenated text content.
        """
        if has_marker and len(spans) > 1:
            # Skip marker span and optional space span
            start_idx = 1
            if len(spans) > 2:
                second_text = spans[1].get("text", "").strip()
                if not second_text:  # Second span is just whitespace
                    start_idx = 2
            content_spans = spans[start_idx:]
        else:
            content_spans = spans

        return "".join(span.get("text", "") for span in content_spans)

    @staticmethod
    def _estimate_line_font_size(spans: list[dict[str, Any]]) -> float | None:
        """Estimate font size for a single line from its spans.

        Args:
            spans: List of span dictionaries.

        Returns:
            Estimated font size or None if not determinable.
        """
        size_weights: dict[float, int] = defaultdict(int)
        for span in spans:
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
            return None

        best_size, _ = max(
            size_weights.items(),
            key=lambda item: (item[1], -item[0]),
        )
        return float(best_size)

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

        Note:
            pdftext returns rotation in radians, so we convert to degrees here.
        """
        rotation_weights: dict[float, int] = defaultdict(int)

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "") or ""
                if not text.strip():
                    continue

                # pdftext returns rotation in radians
                rotation_radians = span.get("rotation", 0.0)
                rotation_weights[rotation_radians] += len(text)

        if not rotation_weights:
            return 0.0

        best_rotation_radians, _ = max(rotation_weights.items(), key=lambda x: x[1])
        # Convert radians to degrees
        return math.degrees(best_rotation_radians)

    @staticmethod
    def _estimate_alignment(block: dict[str, Any], rotation_degrees: float = 0.0) -> str:
        """Estimate text alignment from line positions.

        Args:
            block: Block dictionary from pdftext.
            rotation_degrees: Text rotation in degrees.

        Returns:
            Alignment string: "left", "center", "right", or "justify".

        Note:
            For 90°/270° rotated text, alignment detection is less reliable
            because the coordinate axes are swapped. In these cases, we default
            to "left" alignment to match the original text start position.
        """
        # For vertical text (90° or 270° rotation), default to left alignment
        # This ensures text starts at the edge of the bbox rather than being centered
        normalized_rotation = rotation_degrees % 360
        if 85 <= normalized_rotation <= 95 or 265 <= normalized_rotation <= 275:
            return "left"

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

        # Full variation
        left_std = max(left_positions) - min(left_positions)
        right_std = max(right_positions) - min(right_positions)

        # For 4+ lines, exclude first line (first-line indent) and last line
        # (often shorter) when calculating left edge variation
        if len(left_positions) >= 4:
            middle_left = left_positions[1:-1]
            left_std_middle = max(middle_left) - min(middle_left)
        else:
            left_std_middle = left_std

        # Justified: both edges consistent
        if left_std < 5 and right_std < 5:
            return "justify"

        # Left aligned: consistent left edge
        if left_std < 5:
            return "left"

        # For 4+ lines with first-line indent, check middle lines
        if len(left_positions) >= 4 and left_std_middle < 5:
            return "left"

        # Right aligned: consistent right edge
        # Only if left edge varies significantly (not just first-line indent)
        if right_std < 5 and left_std > 20:
            return "right"

        # Check center alignment
        centers = [(left + right) / 2 for left, right in zip(left_positions, right_positions)]
        center_std = max(centers) - min(centers)
        if center_std < 10:
            return "center"

        # Default to left
        return "left"
