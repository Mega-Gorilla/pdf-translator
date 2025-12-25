# SPDX-License-Identifier: Apache-2.0
"""PDF processing module using pypdfium2.

This module provides PDF text extraction, manipulation, and reinsertion
capabilities using the pypdfium2 library.
"""

from __future__ import annotations

import ctypes
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pikepdf  # type: ignore[import-untyped]
import pypdfium2 as pdfium  # type: ignore[import-untyped]

from .helpers import to_byte_array, to_widestring
from .models import (
    BBox,
    Color,
    Font,
    Metadata,
    Page,
    Paragraph,
    PDFDocument,
    TextObject,
    Transform,
)
from .text_layout import LayoutResult, TextLayoutEngine

# PDFium object type constant for text
FPDF_PAGEOBJ_TEXT = 1

# Debug bbox colors by raw_category (R, G, B)
# 各カテゴリーが視覚的に区別しやすいよう、異なる色相を使用
DEBUG_RAW_CATEGORY_COLORS: dict[str | None, tuple[int, int, int]] = {
    # テキスト系 (翻訳対象) - 各カテゴリーで異なる色
    "text": (34, 139, 34),            # Forest Green - 本文
    "vertical_text": (0, 206, 209),   # Dark Turquoise - 縦書き
    "abstract": (255, 165, 0),        # Orange - アブストラクト
    "aside_text": (147, 112, 219),    # Medium Purple - サイドテキスト
    # タイトル系 (翻訳対象)
    "paragraph_title": (255, 0, 0),   # Red - セクション見出し
    "doc_title": (220, 20, 60),       # Crimson - 文書タイトル
    # キャプション (翻訳対象)
    "figure_title": (255, 105, 180),  # Hot Pink - 図キャプション
    # 数式系 - 青系統で区別
    "inline_formula": (0, 0, 255),    # Blue - インライン数式
    "display_formula": (0, 191, 255), # Deep Sky Blue - ディスプレイ数式
    "formula_number": (100, 149, 237),# Cornflower Blue - 数式番号
    "algorithm": (138, 43, 226),      # Blue Violet - アルゴリズム
    # 図表系
    "table": (0, 128, 128),           # Teal - 表
    "image": (255, 0, 255),           # Magenta - 画像
    "chart": (255, 215, 0),           # Gold - チャート
    # ナビゲーション系 - 明確に区別できる色
    "header": (169, 169, 169),        # Dark Gray - ヘッダー
    "header_image": (112, 128, 144),  # Slate Gray - ヘッダー画像
    "footer": (105, 105, 105),        # Dim Gray - フッター
    "footer_image": (119, 136, 153),  # Light Slate Gray - フッター画像
    "number": (47, 79, 79),           # Dark Slate Gray - ページ番号
    # 参照系
    "reference": (139, 69, 19),       # Saddle Brown - 参考文献
    "reference_content": (210, 105, 30),# Chocolate - 参考文献内容
    "footnote": (188, 143, 143),      # Rosy Brown - 脚注
    "vision_footnote": (205, 92, 92), # Indian Red - 視覚的脚注
    # その他
    "seal": (128, 0, 0),              # Maroon - 印鑑
    "content": (70, 130, 180),        # Steel Blue - コンテンツ
    "unknown": (128, 128, 0),         # Olive - 不明
    # デバッグ用
    "MERGED": (255, 0, 128),          # Deep Pink - マージ結果
    "merged_result": (255, 0, 128),   # Deep Pink - マージ結果 (alias)
    None: (100, 100, 100),            # Default Gray
}

# Control characters to normalize (common in PDF hyphenation)
# \x02 (STX) is used by some PDFs for soft hyphens
CONTROL_CHAR_MAP: dict[int, str] = {
    0x02: "-",  # STX -> hyphen (soft hyphen in many PDFs)
    0x00: "",  # NUL -> remove
    0x01: "",  # SOH -> remove
    0x03: "",  # ETX -> remove
    0x04: "",  # EOT -> remove
    0x05: "",  # ENQ -> remove
    0x06: "",  # ACK -> remove
    0x07: "",  # BEL -> remove
    0x08: "",  # BS -> remove
    0x09: " ",  # TAB -> space
    0x0A: " ",  # LF (newline) -> space
    0x0B: "",  # VT -> remove
    0x0C: "",  # FF -> remove
    0x0D: "",  # CR (carriage return) -> remove
    0x0E: "",  # SO -> remove
    0x0F: "",  # SI -> remove
    0x10: "",  # DLE -> remove
    0x11: "",  # DC1 -> remove
    0x12: "",  # DC2 -> remove
    0x13: "",  # DC3 -> remove
    0x14: "",  # DC4 -> remove
    0x15: "",  # NAK -> remove
    0x16: "",  # SYN -> remove
    0x17: "",  # ETB -> remove
    0x18: "",  # CAN -> remove
    0x19: "",  # EM -> remove
    0x1A: "",  # SUB -> remove
    0x1B: "",  # ESC -> remove
    0x1C: "",  # FS -> remove
    0x1D: "",  # GS -> remove
    0x1E: "",  # RS -> remove
    0x1F: "",  # US -> remove
    0x7F: "",  # DEL -> remove
    0xAD: "-",  # Soft hyphen -> hyphen
    0xFF: "",  # ÿ (often encoding error) -> remove
}

# Unicode characters not supported by standard PDF fonts (WinAnsiEncoding)
# Map to ASCII equivalents for compatibility
UNICODE_TO_ASCII_MAP: dict[int, str] = {
    # Math operators
    0x2217: "*",  # ∗ ASTERISK OPERATOR -> *
    0x00D7: "x",  # × MULTIPLICATION SIGN -> x
    0x00F7: "/",  # ÷ DIVISION SIGN -> /
    0x2212: "-",  # − MINUS SIGN -> -
    0x2013: "-",  # – EN DASH -> -
    0x2014: "--",  # — EM DASH -> --
    0x2018: "'",  # ' LEFT SINGLE QUOTATION -> '
    0x2019: "'",  # ' RIGHT SINGLE QUOTATION -> '
    0x201C: '"',  # " LEFT DOUBLE QUOTATION -> "
    0x201D: '"',  # " RIGHT DOUBLE QUOTATION -> "
    0x2026: "...",  # … HORIZONTAL ELLIPSIS -> ...
    0x2022: "*",  # • BULLET -> *
    0x2032: "'",  # ′ PRIME -> '
    0x2033: '"',  # ″ DOUBLE PRIME -> "
    0x00B0: "o",  # ° DEGREE SIGN -> o
    0x00B7: ".",  # · MIDDLE DOT -> .
    0x2264: "<=",  # ≤ LESS-THAN OR EQUAL TO -> <=
    0x2265: ">=",  # ≥ GREATER-THAN OR EQUAL TO -> >=
    0x2260: "!=",  # ≠ NOT EQUAL TO -> !=
    0x00B1: "+/-",  # ± PLUS-MINUS SIGN -> +/-
    0x221E: "inf",  # ∞ INFINITY -> inf
    0x2248: "~=",  # ≈ ALMOST EQUAL TO -> ~=
    0x221A: "sqrt",  # √ SQUARE ROOT -> sqrt
    0x03B1: "alpha",  # α GREEK SMALL LETTER ALPHA
    0x03B2: "beta",  # β GREEK SMALL LETTER BETA
    0x03B3: "gamma",  # γ GREEK SMALL LETTER GAMMA
    0x03B4: "delta",  # δ GREEK SMALL LETTER DELTA
    0x03C0: "pi",  # π GREEK SMALL LETTER PI
    0x03C3: "sigma",  # σ GREEK SMALL LETTER SIGMA
    0x03BC: "mu",  # μ GREEK SMALL LETTER MU
}


def normalize_text(text: str, ascii_fallback: bool = False) -> str:
    """Normalize text by replacing control characters.

    This handles PDF-specific issues like soft hyphens represented
    as control characters (common in hyphenated text).

    Args:
        text: Raw text from PDF extraction
        ascii_fallback: If True, also convert unsupported Unicode characters
                       to ASCII equivalents for standard PDF font compatibility

    Returns:
        Normalized text with control characters replaced
    """
    result = []
    for char in text:
        code = ord(char)
        if code in CONTROL_CHAR_MAP:
            result.append(CONTROL_CHAR_MAP[code])
        elif ascii_fallback and code in UNICODE_TO_ASCII_MAP:
            result.append(UNICODE_TO_ASCII_MAP[code])
        else:
            result.append(char)
    return "".join(result)


class PDFProcessor:
    """PDF processor using pypdfium2.

    This class provides methods for extracting, removing, and inserting
    text objects in PDF documents. It uses the "template PDF" approach
    where the original PDF is preserved and only the text layer is modified.

    Example:
        >>> processor = PDFProcessor("input.pdf")
        >>> doc = processor.extract_text_objects()
        >>> print(doc.to_json())
        >>> processor.remove_text_objects(0, ["text_001"])
        >>> processor.insert_text_object(
        ...     page_num=0,
        ...     text="New text",
        ...     bbox=BBox(72, 750, 300, 770),
        ...     font=Font("Helvetica", 12.0)
        ... )
        >>> processor.save("output.pdf")
    """

    def __init__(self, pdf_source: Union[Path, str, bytes]) -> None:
        """Initialize the PDF processor.

        Args:
            pdf_source: Path to PDF file or PDF bytes

        Raises:
            TypeError: If pdf_source is not Path, str, or bytes
            FileNotFoundError: If the file path doesn't exist
            ValueError: If the PDF cannot be loaded
        """
        self._source_name: str
        self._pdf: Optional[pdfium.PdfDocument] = None
        # Store original PDF bytes for hybrid approach (pikepdf removal).
        # Note: This is updated by apply_paragraphs() after text removal.
        # If you modify the PDF via other methods before calling apply_paragraphs(),
        # those changes will be lost. For normal translation workflow, this is fine
        # since apply_paragraphs() is the final step.
        self._pdf_bytes: bytes
        self._loaded_fonts: dict[str, Any] = {}  # font_path -> font_handle
        self._loaded_font_buffers: dict[str, ctypes.Array[Any]] = {}  # keep buffers alive
        self._layout_engine = TextLayoutEngine()

        if isinstance(pdf_source, bytes):
            self._pdf_bytes = pdf_source
            self._pdf = pdfium.PdfDocument(pdf_source)
            self._source_name = "bytes"
        elif isinstance(pdf_source, (str, Path)):
            path = Path(pdf_source)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {path}")
            # Read and store PDF bytes for hybrid approach
            with open(path, "rb") as f:
                self._pdf_bytes = f.read()
            self._pdf = pdfium.PdfDocument(str(path))
            self._source_name = path.name
        else:
            raise TypeError(
                f"pdf_source must be Path, str, or bytes, got {type(pdf_source).__name__}"
            )

    def __enter__(self) -> PDFProcessor:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the PDF document and release resources."""
        # Clear font buffers first
        self._loaded_font_buffers.clear()
        self._loaded_fonts.clear()
        if hasattr(self, "_pdf") and self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    @property
    def page_count(self) -> int:
        """Get the number of pages in the document."""
        if self._pdf is None:
            raise RuntimeError("PDF document is not open")
        return len(self._pdf)

    def _ensure_open(self) -> pdfium.PdfDocument:
        """Ensure PDF document is open and return it."""
        if self._pdf is None:
            raise RuntimeError("PDF document is not open")
        return self._pdf

    def _generate_stable_id(
        self, page_num: int, obj_index: int, prefix: str = "text"
    ) -> str:
        """Generate a stable ID for objects based on page and index.

        This ensures that repeated extractions produce the same IDs,
        enabling the remove_text_objects(object_ids) workflow.

        Args:
            page_num: Page number
            obj_index: Object index within the page
            prefix: Prefix for the ID

        Returns:
            Stable identifier string
        """
        return f"{prefix}_p{page_num}_i{obj_index}"

    def _get_text_from_page(self, page: pdfium.PdfPage) -> str:
        """Extract all text from a page using textpage.

        Args:
            page: pypdfium2 page object

        Returns:
            All text content from the page
        """
        textpage = page.get_textpage()
        try:
            result: str = textpage.get_text_bounded()
            return result
        finally:
            textpage.close()

    def _get_text_object_text_direct(
        self, obj: pdfium.PdfObject, textpage: pdfium.PdfTextPage
    ) -> str:
        """Get text directly from a text object using FPDFTextObj_GetText.

        This method extracts text that belongs specifically to this text object,
        avoiding the bbox overlap issue present in get_text_bounded().

        Args:
            obj: Text page object
            textpage: The textpage for the page containing the object

        Returns:
            Text content of the object (empty string if extraction fails)
        """
        try:
            # First call: get required buffer size
            length = pdfium.raw.FPDFTextObj_GetText(
                obj.raw,
                textpage.raw,
                None,
                0
            )

            if length == 0:
                return ""

            # Allocate buffer (UTF-16LE, each char is 2 bytes)
            buffer = (ctypes.c_ushort * length)()

            # Second call: get the text
            pdfium.raw.FPDFTextObj_GetText(
                obj.raw,
                textpage.raw,
                buffer,
                length
            )

            # Convert UTF-16LE to Python string
            chars = []
            for i in range(length - 1):  # -1 to exclude null terminator
                if buffer[i] == 0:
                    break
                chars.append(chr(buffer[i]))

            return "".join(chars)
        except Exception:
            return ""

    def _get_text_object_content(
        self, page: pdfium.PdfPage, obj: pdfium.PdfObject
    ) -> str:
        """Get text content from a text object using textpage.

        Args:
            page: The page containing the object
            obj: Text page object

        Returns:
            Text content within the object's bounds

        Note:
            This method uses get_text_bounded() which may include text from
            overlapping objects. For more precise extraction, use
            _get_text_object_text_direct() instead.
        """
        bounds = obj.get_bounds()
        if bounds is None:
            return ""

        # bounds = (left, bottom, right, top)
        left, bottom, right, top = bounds

        textpage = page.get_textpage()
        try:
            # Get text bounded by the object's bbox
            text = textpage.get_text_bounded(
                left=left, bottom=bottom, right=right, top=top
            )
            return text.strip() if text else ""
        finally:
            textpage.close()

    def _extract_font_info(self, obj: pdfium.PdfObject) -> Optional[Font]:
        """Extract font information from a text object.

        Args:
            obj: Text page object

        Returns:
            Font information or None if not available
        """
        try:
            # Get font size using raw API (requires pointer argument)
            font_size_val = ctypes.c_float()
            ret = pdfium.raw.FPDFTextObj_GetFontSize(
                obj.raw, ctypes.byref(font_size_val)
            )
            if not ret:
                return None

            font_size = font_size_val.value

            # Get font handle
            font_handle = pdfium.raw.FPDFTextObj_GetFont(obj.raw)
            if not font_handle:
                return Font(name="Unknown", size=float(font_size))

            # Get font name from font handle
            buffer_len = 256
            buffer = ctypes.create_string_buffer(buffer_len)
            actual_len = pdfium.raw.FPDFFont_GetBaseFontName(
                font_handle, buffer, buffer_len
            )

            font_name = "Unknown"
            if actual_len > 0:
                font_name = buffer.value.decode("utf-8", errors="replace")

            # Determine bold/italic from font name heuristics
            name_lower = font_name.lower()
            is_bold = "bold" in name_lower or "medi" in name_lower
            is_italic = "italic" in name_lower or "oblique" in name_lower

            return Font(
                name=font_name,
                size=float(font_size),
                is_bold=is_bold,
                is_italic=is_italic,
            )
        except Exception:
            return None

    def _extract_transform(self, obj: pdfium.PdfObject) -> Optional[Transform]:
        """Extract transformation matrix from an object.

        Args:
            obj: Page object

        Returns:
            Transform object or None
        """
        try:
            # Use pypdfium2's high-level API to get the matrix
            matrix = obj.get_matrix()
            if matrix is None:
                return None

            transform = Transform(
                a=matrix.a,
                b=matrix.b,
                c=matrix.c,
                d=matrix.d,
                e=matrix.e,
                f=matrix.f,
            )
            # Only return if not identity transform
            if not transform.is_identity():
                return transform
            return None
        except Exception:
            return None

    def _extract_color(self, obj: pdfium.PdfObject) -> Optional[Color]:
        """Extract fill color from a text object.

        Args:
            obj: Page object

        Returns:
            Color object or None
        """
        try:
            r = ctypes.c_uint()
            g = ctypes.c_uint()
            b = ctypes.c_uint()
            a = ctypes.c_uint()

            success = pdfium.raw.FPDFPageObj_GetFillColor(
                obj.raw,
                ctypes.byref(r),
                ctypes.byref(g),
                ctypes.byref(b),
                ctypes.byref(a),
            )

            if success:
                return Color(r=r.value, g=g.value, b=b.value)
            return None
        except Exception:
            return None

    def extract_text_objects(self) -> PDFDocument:
        """Extract all text objects from the PDF.

        Returns:
            PDFDocument containing all extracted text objects

        Note:
            Object IDs are stable across repeated calls (based on page/index),
            enabling the remove_text_objects(object_ids) workflow.
        """
        pages = []
        pdf = self._ensure_open()

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            width = page.get_width()
            height = page.get_height()
            rotation = page.get_rotation()

            text_objects = []

            # Create textpage once for the entire page (performance optimization)
            textpage = page.get_textpage()
            try:
                obj_index = 0
                # Get all text objects from the page
                for obj in page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]):
                    bounds = obj.get_bounds()
                    if bounds is None:
                        continue

                    left, bottom, right, top = bounds
                    bbox = BBox(x0=left, y0=bottom, x1=right, y1=top)

                    # Get text content directly from the text object
                    # This avoids bbox overlap issues present in get_text_bounded()
                    text = self._get_text_object_text_direct(obj, textpage)
                    text = text.strip() if text else ""
                    if not text:
                        continue

                    # Normalize control characters (e.g., soft hyphens)
                    text = normalize_text(text)

                    # Generate stable ID based on page and object index
                    obj_id = self._generate_stable_id(page_num, obj_index)
                    obj_index += 1

                    # Extract metadata
                    font = self._extract_font_info(obj)
                    color = self._extract_color(obj)
                    transform = self._extract_transform(obj)

                    text_obj = TextObject(
                        id=obj_id,
                        bbox=bbox,
                        text=text,
                        font=font,
                        color=color,
                        transform=transform,
                    )
                    text_objects.append(text_obj)
            finally:
                textpage.close()

            page_data = Page(
                page_number=page_num,
                width=width,
                height=height,
                rotation=rotation,
                text_objects=text_objects,
            )
            pages.append(page_data)

        metadata = Metadata(
            source_file=self._source_name,
            created_at=datetime.now().isoformat(),
            page_count=len(pages),
        )

        return PDFDocument(pages=pages, metadata=metadata)

    def remove_text_objects(
        self, page_num: int, object_ids: Optional[list[str]] = None
    ) -> int:
        """Remove text objects from a page.

        Args:
            page_num: Page number (0-indexed)
            object_ids: List of object IDs to remove. If None, removes all text objects.
                       IDs should be from extract_text_objects() output (stable IDs).

        Returns:
            Number of objects removed

        Raises:
            IndexError: If page_num is out of range

        Note:
            IDs are stable across extract_text_objects() calls, based on
            page number and object index (e.g., "text_p0_i5").
        """
        pdf = self._ensure_open()
        if page_num < 0 or page_num >= len(pdf):
            raise IndexError(f"Page number {page_num} out of range")

        page = pdf[page_num]

        # If object_ids is None, remove all text objects
        if object_ids is None:
            objects_to_remove = list(page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]))
            for obj in objects_to_remove:
                page.remove_obj(obj)
            page.gen_content()
            return len(objects_to_remove)

        # Parse object IDs to extract indices
        # IDs are in format "text_p{page}_i{index}"
        target_indices: set[int] = set()
        for obj_id in object_ids:
            if obj_id.startswith(f"text_p{page_num}_i"):
                try:
                    idx = int(obj_id.split("_i")[1])
                    target_indices.add(idx)
                except (ValueError, IndexError):
                    pass

        if not target_indices:
            return 0

        # Build list of objects to remove by index
        removed = 0
        objects_to_remove = []
        obj_index = 0

        textpage = page.get_textpage()
        try:
            for obj in page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]):
                bounds = obj.get_bounds()
                if bounds is None:
                    continue

                # Check if this object has text (matching extract logic)
                # Use _get_text_object_text_direct for consistency with extract
                text = self._get_text_object_text_direct(obj, textpage)
                text = text.strip() if text else ""
                if not text:
                    continue

                if obj_index in target_indices:
                    objects_to_remove.append(obj)

                obj_index += 1
        finally:
            textpage.close()

        for obj in objects_to_remove:
            page.remove_obj(obj)
            removed += 1

        if removed > 0:
            page.gen_content()

        return removed

    def remove_text_with_pikepdf(
        self,
        bboxes_by_page: dict[int, list[BBox]],
        pdf_source: Union[Path, str, bytes],
    ) -> bytes:
        """Remove text in specified bboxes using pikepdf content stream editing.

        This method uses pikepdf to directly edit PDF content streams,
        filtering out TJ/Tj operators that fall within the specified bounding boxes.
        This approach preserves spacing information in adjacent text because it
        edits content streams directly without triggering gen_content() regeneration.

        Args:
            bboxes_by_page: Dictionary mapping page numbers to lists of bboxes to remove.
            pdf_source: Path to PDF file or PDF bytes.

        Returns:
            Modified PDF as bytes.

        Note:
            This is the standard method for text removal (Issue #47 fix).
            The returned bytes can be used to create a new PDFProcessor instance
            for further manipulation (e.g., inserting translated text).

        Known Limitations:
            - Text position after Tj/TJ is not tracked (would require font metrics
              for accurate character width calculation). This means if multiple
              Tj/TJ operations appear on the same line without intervening Tm/Td,
              only the first text block's position is used for bbox matching.
              Most PDFs use Tm to set absolute position before each text block,
              so this is rarely an issue in practice.
        """

        def point_in_any_bbox(x: float, y: float, bboxes: list[BBox]) -> bool:
            """Check if point (x, y) is inside any of the bboxes.

            Uses a small tolerance (0.5 pt) to handle floating-point precision
            differences between pdftext and pikepdf coordinate systems.
            """
            tolerance = 0.5  # Points - handles precision mismatch
            for bbox in bboxes:
                if (bbox.x0 - tolerance <= x <= bbox.x1 + tolerance and
                        bbox.y0 - tolerance <= y <= bbox.y1 + tolerance):
                    return True
            return False

        # Open PDF with pikepdf
        from io import BytesIO

        if isinstance(pdf_source, bytes):
            pdf = pikepdf.open(BytesIO(pdf_source))
        else:
            pdf = pikepdf.open(str(pdf_source))

        try:
            for page_num, bboxes in bboxes_by_page.items():
                if page_num >= len(pdf.pages):
                    continue

                page = pdf.pages[page_num]
                cs = list(pikepdf.parse_content_stream(page))

                new_cs = []
                in_text = False

                # Track current text position
                current_x: float = 0.0
                current_y: float = 0.0

                # Track text leading for T* operator
                text_leading: float = 0.0

                for cmd in cs:
                    operands, operator = cmd
                    op_str = str(operator)

                    if op_str == "BT":
                        # Begin Text block - reset text matrix to identity
                        in_text = True
                        current_x = 0.0
                        current_y = 0.0
                        new_cs.append(cmd)
                    elif op_str == "ET":
                        # End Text block
                        in_text = False
                        new_cs.append(cmd)
                    elif in_text:
                        # Inside text block - track position and filter text
                        if op_str == "Tm" and len(operands) >= 6:
                            # Text Matrix: a b c d e f (e=x, f=y)
                            current_x = float(operands[4])
                            current_y = float(operands[5])
                            new_cs.append(cmd)
                        elif op_str == "Td" and len(operands) >= 2:
                            # Text move: dx dy
                            current_x += float(operands[0])
                            current_y += float(operands[1])
                            new_cs.append(cmd)
                        elif op_str == "TD" and len(operands) >= 2:
                            # Text move with leading: dx dy (also sets TL = -dy)
                            current_x += float(operands[0])
                            current_y += float(operands[1])
                            text_leading = -float(operands[1])
                            new_cs.append(cmd)
                        elif op_str == "TL" and len(operands) >= 1:
                            # Set text leading
                            text_leading = float(operands[0])
                            new_cs.append(cmd)
                        elif op_str == "T*":
                            # Move to next line (equivalent to 0 -TL Td)
                            # Note: x position is maintained (relative move with dx=0)
                            current_y -= text_leading
                            new_cs.append(cmd)
                        elif op_str in ("TJ", "Tj"):
                            # Text showing operators - filter based on position
                            # Note: We don't update current_x after text rendering because
                            # accurate tracking requires font metrics (character widths).
                            # This is acceptable for bbox-based filtering since most PDFs
                            # use Tm to set absolute position before each text block.
                            if point_in_any_bbox(current_x, current_y, bboxes):
                                # Skip this text operation (remove it)
                                pass
                            else:
                                new_cs.append(cmd)
                        elif op_str == "'":
                            # Move to next line and show text (T* followed by Tj)
                            # Note: x position is maintained (relative move with dx=0)
                            current_y -= text_leading
                            if point_in_any_bbox(current_x, current_y, bboxes):
                                # Replace with just T* (move without text)
                                new_cs.append(([], pikepdf.Operator("T*")))
                            else:
                                new_cs.append(cmd)
                        elif op_str == '"':
                            # Set spacing, move to next line and show text
                            # Equivalent to: aw Tw ac Tc string '
                            # Operands: aw ac string (word spacing, char spacing, string)
                            # Note: x position is maintained (relative move with dx=0)
                            current_y -= text_leading
                            if point_in_any_bbox(current_x, current_y, bboxes):
                                # Keep Tw/Tc settings for subsequent text, add T* for line move
                                # This preserves word/char spacing for text that follows
                                if len(operands) >= 2:
                                    # aw Tw (word spacing)
                                    new_cs.append(([operands[0]], pikepdf.Operator("Tw")))
                                    # ac Tc (character spacing)
                                    new_cs.append(([operands[1]], pikepdf.Operator("Tc")))
                                # T* for line move
                                new_cs.append(([], pikepdf.Operator("T*")))
                            else:
                                new_cs.append(cmd)
                        else:
                            # Other text-related commands (Tf, Tc, Tw, etc.)
                            new_cs.append(cmd)
                    else:
                        # Outside text block - keep all commands
                        new_cs.append(cmd)

                # Unparse and update page content
                new_stream = pikepdf.unparse_content_stream(new_cs)
                page.Contents = pdf.make_stream(new_stream)

            # Save to bytes using BytesIO (cross-platform compatible)
            output = BytesIO()
            pdf.save(output)
            return output.getvalue()
        finally:
            pdf.close()

    def remove_all_text(self, page_num: int) -> int:
        """Remove all text objects from a page.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Number of objects removed
        """
        return self.remove_text_objects(page_num, None)

    def _find_font_variant(
        self,
        base_font_path: Path,
        is_bold: bool,
        is_italic: bool,
    ) -> Path:
        """Find appropriate font variant based on style flags.

        Attempts to find Bold/Italic variants of a font using common naming
        conventions. Falls back to the base font if no variant is found.

        Args:
            base_font_path: Path to the base font file.
            is_bold: Whether to find a bold variant.
            is_italic: Whether to find an italic variant.

        Returns:
            Path to the appropriate font variant, or base path if not found.
        """
        if not is_bold and not is_italic:
            return base_font_path

        base_name = base_font_path.stem
        parent_dir = base_font_path.parent
        extension = base_font_path.suffix

        # Remove common style suffixes to get the font family base name
        for style in ["-Regular", "-Text", "-Normal", "-Book", "Regular", "Text"]:
            if base_name.endswith(style):
                base_name = base_name[: -len(style)]
                break

        # Determine target suffixes based on style
        if is_bold and is_italic:
            target_suffixes = [
                "-BoldItalic",
                "-Bold Italic",
                "BoldItalic",
                "-SemiBoldItalic",
            ]
        elif is_bold:
            target_suffixes = ["-Bold", "Bold", "-SemiBold", "-Medium"]
        else:  # italic only
            target_suffixes = ["-Italic", "Italic", "-Oblique", "-TextItalic"]

        # Try to find variant
        for suffix in target_suffixes:
            variant_path = parent_dir / f"{base_name}{suffix}{extension}"
            if variant_path.exists():
                return variant_path

        # Return original if no variant found
        return base_font_path

    def load_font(self, font_path: Union[Path, str]) -> Optional[Any]:
        """Load a TrueType font for text insertion.

        Always loads as CID font to support both ASCII and CJK characters.
        CID fonts handle ASCII correctly, so this is safe for all text.

        Args:
            font_path: Path to TTF font file

        Returns:
            Font handle or None if loading failed
        """
        path = Path(font_path)
        path_str = str(path)

        # Return cached font if already loaded
        if path_str in self._loaded_fonts:
            return self._loaded_fonts[path_str]

        if not path.exists():
            return None

        with open(path, "rb") as f:
            font_data = f.read()

        font_arr = to_byte_array(font_data)
        # Keep the buffer alive for the lifetime of this processor
        self._loaded_font_buffers[path_str] = font_arr

        pdf = self._ensure_open()
        # Always use CID mode (is_cid=1) for full Unicode support
        font_handle = pdfium.raw.FPDFText_LoadFont(
            pdf.raw,
            font_arr,
            ctypes.c_uint(len(font_data)),
            ctypes.c_int(pdfium.raw.FPDF_FONT_TRUETYPE),
            ctypes.c_int(1),  # CID mode for CJK support
        )

        if font_handle:
            self._loaded_fonts[path_str] = font_handle
            return font_handle
        return None

    def load_standard_font(self, font_name: str) -> Optional[Any]:
        """Load a standard PDF font.

        Args:
            font_name: Standard font name (e.g., "Helvetica", "Times-Roman")

        Returns:
            Font handle or None if loading failed
        """
        if font_name in self._loaded_fonts:
            return self._loaded_fonts[font_name]

        pdf = self._ensure_open()
        font_handle = pdfium.raw.FPDFText_LoadStandardFont(
            pdf.raw, font_name.encode("utf-8")
        )

        if font_handle:
            self._loaded_fonts[font_name] = font_handle
            return font_handle
        return None

    def insert_text_object(
        self,
        page_num: int,
        text: str,
        bbox: BBox,
        font: Font,
        font_path: Optional[Union[Path, str]] = None,
        color: Optional[Color] = None,
        transform: Optional[Transform] = None,
    ) -> Optional[str]:
        """Insert a text object into a page.

        Args:
            page_num: Page number (0-indexed)
            text: Text content to insert
            bbox: Bounding box for positioning
            font: Font information
            font_path: Path to TTF font file (for custom fonts)
            color: Text color (default: black)
            transform: Transformation matrix (optional)

        Returns:
            Generated object ID or None if insertion failed

        Raises:
            IndexError: If page_num is out of range
        """
        pdf = self._ensure_open()
        if page_num < 0 or page_num >= len(pdf):
            raise IndexError(f"Page number {page_num} out of range")

        page = pdf[page_num]
        doc_handle = pdf.raw
        page_handle = page.raw

        # Load font
        font_handle = None
        if font_path:
            font_handle = self.load_font(font_path)
        else:
            font_handle = self.load_standard_font(font.name)

        if not font_handle:
            return None

        # Create text object
        text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
            doc_handle, font_handle, ctypes.c_float(font.size)
        )

        if not text_obj:
            return None

        # Set text content
        text_ws = to_widestring(text)
        success = pdfium.raw.FPDFText_SetText(text_obj, text_ws)
        if not success:
            return None

        # Set color if specified
        if color:
            pdfium.raw.FPDFPageObj_SetFillColor(
                text_obj, color.r, color.g, color.b, 255
            )

        # Apply transform for positioning
        if transform:
            pdfium.raw.FPDFPageObj_Transform(
                text_obj,
                ctypes.c_double(transform.a),
                ctypes.c_double(transform.b),
                ctypes.c_double(transform.c),
                ctypes.c_double(transform.d),
                ctypes.c_double(transform.e),
                ctypes.c_double(transform.f),
            )
        else:
            # Default: position at bbox origin (x0, y0)
            pdfium.raw.FPDFPageObj_Transform(
                text_obj,
                ctypes.c_double(1.0),
                ctypes.c_double(0.0),
                ctypes.c_double(0.0),
                ctypes.c_double(1.0),
                ctypes.c_double(bbox.x0),
                ctypes.c_double(bbox.y0),
            )

        # Insert into page
        pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)
        page.gen_content()

        # Return a simple ID (not stable, as inserted objects have no index)
        return "inserted"

    def insert_laid_out_text(
        self,
        page_num: int,
        layout_result: LayoutResult,
        bbox: BBox,
        font_handle: Any,
        font_size: float,
        color: Optional[Color] = None,
        alignment: str = "left",
        rotation: float = 0.0,
        pdftext_rotation_degrees: float = 0.0,
    ) -> bool:
        """Insert laid out text (multiple lines) into a page.

        This method inserts each line as a separate text object, positioned
        according to the layout result.

        Args:
            page_num: Page number (0-indexed).
            layout_result: Layout result from TextLayoutEngine.
                Note: For rotated text (90°/270°), the layout should have been
                calculated with swapped width/height by fit_text_in_bbox().
            bbox: Bounding box for positioning.
            font_handle: PDFium font handle.
            font_size: Font size to use.
            color: Text color (default: black).
            alignment: Text alignment ("left", "center", "right", "justify").
            rotation: Rotation angle in radians for the PDF transform matrix.
                This is the negated pdftext rotation (e.g., math.radians(-270)
                for pdftext 270° visual rotation).
            pdftext_rotation_degrees: Original pdftext visual rotation in degrees.
                Used for coordinate calculation to determine text flow direction.
                pdftext 270° = text reads bottom-to-top (sidebar text).
                pdftext 90° = text reads top-to-bottom.

        Returns:
            True if all lines were inserted successfully.
        """
        pdf = self._ensure_open()
        if page_num < 0 or page_num >= len(pdf):
            raise IndexError(f"Page number {page_num} out of range")

        page = pdf[page_num]
        doc_handle = pdf.raw
        page_handle = page.raw

        # Pre-calculate rotation matrix components
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        # Determine rotation type for coordinate calculation using pdftext visual rotation
        # (not the PDF transform rotation which is negated)
        visual_rotation = pdftext_rotation_degrees % 360
        is_270_rotation = 265 <= visual_rotation <= 275
        is_90_rotation = 85 <= visual_rotation <= 95

        success = True
        for line in layout_result.lines:
            # Create text object for this line
            text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                doc_handle, font_handle, ctypes.c_float(font_size)
            )

            if not text_obj:
                success = False
                continue

            # Set text content
            text_ws = to_widestring(line.text)
            if not pdfium.raw.FPDFText_SetText(text_obj, text_ws):
                success = False
                continue

            # Set color if specified
            if color:
                pdfium.raw.FPDFPageObj_SetFillColor(
                    text_obj, color.r, color.g, color.b, 255
                )

            # Calculate position based on rotation
            if is_270_rotation:
                # 270° rotation: text flows bottom-to-top, lines stack right-to-left
                # Origin is at (x1, y0) - bottom-right of bbox
                # For 270° rotated text with swapped width/height in fit_text_in_bbox:
                # - effective_width = bbox.height (text wrapping direction)
                # - effective_height = bbox.width (line stacking direction)
                # local_x: alignment offset along text direction (becomes y in PDF)
                # local_y: line position offset (becomes -x in PDF, from right edge)
                if alignment == "center":
                    local_x = (bbox.height - line.width) / 2
                elif alignment == "right":
                    local_x = bbox.height - line.width
                else:
                    local_x = 0.0

                # local_y is the offset from the first line position
                # line.y_position was calculated as bbox.y1 - ascent - (line_index * line_height)
                # We need to convert this to offset from first line
                local_y = line.y_position - bbox.y1  # negative value

                # For 270° rotation:
                # - PDF x = origin_x + local_y (moves left as local_y is negative)
                # - PDF y = origin_y + local_x (moves up for text direction)
                x_pos = bbox.x1 + local_y
                y_pos = bbox.y0 + local_x

            elif is_90_rotation:
                # 90° rotation: text flows top-to-bottom, lines stack left-to-right
                # Origin is at (x0, y1) - top-left of bbox
                if alignment == "center":
                    local_x = (bbox.height - line.width) / 2
                elif alignment == "right":
                    local_x = bbox.height - line.width
                else:
                    local_x = 0.0

                local_y = line.y_position - bbox.y1  # negative value

                # For 90° rotation:
                # - PDF x = origin_x - local_y (moves right as local_y is negative)
                # - PDF y = origin_y - local_x (moves down for text direction)
                x_pos = bbox.x0 - local_y
                y_pos = bbox.y1 - local_x

            else:
                # Normal horizontal text (0° or 180°)
                if alignment == "center":
                    local_x = (bbox.width - line.width) / 2
                elif alignment == "right":
                    local_x = bbox.width - line.width
                else:
                    local_x = 0.0

                local_y = line.y_position - bbox.y1

                if abs(rotation) > 0.001:
                    # Apply rotation transform for non-90/270 angles
                    rotated_x = local_x * cos_r - local_y * sin_r
                    rotated_y = local_x * sin_r + local_y * cos_r
                    x_pos = bbox.x0 + rotated_x
                    y_pos = bbox.y1 + rotated_y
                else:
                    x_pos = bbox.x0 + local_x
                    y_pos = line.y_position

            # Position and rotate the text using affine transform
            # [a b 0]   [cos  sin 0]
            # [c d 0] = [-sin cos 0]
            # [e f 1]   [x    y   1]
            pdfium.raw.FPDFPageObj_Transform(
                text_obj,
                ctypes.c_double(cos_r),
                ctypes.c_double(sin_r),
                ctypes.c_double(-sin_r),
                ctypes.c_double(cos_r),
                ctypes.c_double(x_pos),
                ctypes.c_double(y_pos),
            )

            # Insert into page
            pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)

        # Generate content for all inserted objects
        page.gen_content()
        return success

    def _insert_laid_out_text_no_gen(
        self,
        page_num: int,
        layout_result: LayoutResult,
        bbox: BBox,
        font_handle: Any,
        font_size: float,
        color: Optional[Color] = None,
        alignment: str = "left",
        rotation: float = 0.0,
        pdftext_rotation_degrees: float = 0.0,
        *,
        _page: Optional[pdfium.PdfPage] = None,
    ) -> bool:
        """Insert laid out text without calling gen_content().

        This is an internal method used by apply_paragraphs() to batch
        gen_content() calls for better performance and correct z-ordering.
        See insert_laid_out_text() for parameter documentation.

        Args:
            _page: Pre-fetched page object to ensure consistent object insertion.
        """
        pdf = self._ensure_open()
        if page_num < 0 or page_num >= len(pdf):
            raise IndexError(f"Page number {page_num} out of range")

        # Use provided page object or fetch new one
        page = _page if _page is not None else pdf[page_num]
        doc_handle = pdf.raw
        page_handle = page.raw

        # Pre-calculate rotation matrix components
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        # Determine rotation type for coordinate calculation
        visual_rotation = pdftext_rotation_degrees % 360
        is_270_rotation = 265 <= visual_rotation <= 275
        is_90_rotation = 85 <= visual_rotation <= 95

        success = True
        for line in layout_result.lines:
            text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                doc_handle, font_handle, ctypes.c_float(font_size)
            )

            if not text_obj:
                success = False
                continue

            text_ws = to_widestring(line.text)
            if not pdfium.raw.FPDFText_SetText(text_obj, text_ws):
                success = False
                continue

            if color:
                pdfium.raw.FPDFPageObj_SetFillColor(
                    text_obj, color.r, color.g, color.b, 255
                )

            # Calculate position based on rotation (same logic as insert_laid_out_text)
            if is_270_rotation:
                if alignment == "center":
                    local_x = (bbox.height - line.width) / 2
                elif alignment == "right":
                    local_x = bbox.height - line.width
                else:
                    local_x = 0.0
                local_y = line.y_position - bbox.y1
                x_pos = bbox.x1 + local_y
                y_pos = bbox.y0 + local_x
            elif is_90_rotation:
                if alignment == "center":
                    local_x = (bbox.height - line.width) / 2
                elif alignment == "right":
                    local_x = bbox.height - line.width
                else:
                    local_x = 0.0
                local_y = line.y_position - bbox.y1
                x_pos = bbox.x0 - local_y
                y_pos = bbox.y1 - local_x
            else:
                if alignment == "center":
                    local_x = (bbox.width - line.width) / 2
                elif alignment == "right":
                    local_x = bbox.width - line.width
                else:
                    local_x = 0.0
                local_y = line.y_position - bbox.y1
                if abs(rotation) > 0.001:
                    rotated_x = local_x * cos_r - local_y * sin_r
                    rotated_y = local_x * sin_r + local_y * cos_r
                    x_pos = bbox.x0 + rotated_x
                    y_pos = bbox.y1 + rotated_y
                else:
                    x_pos = bbox.x0 + local_x
                    y_pos = line.y_position

            pdfium.raw.FPDFPageObj_Transform(
                text_obj,
                ctypes.c_double(cos_r),
                ctypes.c_double(sin_r),
                ctypes.c_double(-sin_r),
                ctypes.c_double(cos_r),
                ctypes.c_double(x_pos),
                ctypes.c_double(y_pos),
            )

            pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)

        # Note: gen_content() is NOT called here - caller is responsible
        return success

    def apply_paragraphs(
        self,
        paragraphs: list[Paragraph],
        font_path: Optional[Union[Path, str]] = None,
        font_subsets: Optional[dict[tuple[bool, bool], Path]] = None,
        min_font_size: Optional[float] = None,
    ) -> None:
        """Apply translated paragraphs to the PDF.

        Uses the hybrid approach (pikepdf + pypdfium2) to:
        1. Remove original text using pikepdf (preserves spacing in adjacent text)
        2. Insert translated text using pypdfium2

        This approach solves Issue #47 (spacing corruption after text removal).

        Args:
            paragraphs: List of paragraphs to apply.
            font_path: Path to TTF font file for custom fonts.
            font_subsets: Pre-generated subset fonts keyed by (is_bold, is_italic).
                If provided, uses subsets instead of finding font variants.
            min_font_size: Minimum font size for text fitting. If None, uses engine default.
        """
        # Update TextLayoutEngine min_font_size if specified
        if min_font_size is not None:
            self._layout_engine._min_font_size = min_font_size

        # Collect translated paragraphs
        translated_paragraphs = [p for p in paragraphs if p.translated_text]
        if not translated_paragraphs:
            return

        # Collect bboxes by page for text removal
        bboxes_by_page: dict[int, list[BBox]] = {}
        for para in translated_paragraphs:
            if para.page_number not in bboxes_by_page:
                bboxes_by_page[para.page_number] = []
            bboxes_by_page[para.page_number].append(para.block_bbox)

        # Remove text using pikepdf (preserves spacing in adjacent text)
        modified_pdf_bytes = self.remove_text_with_pikepdf(
            bboxes_by_page, self._pdf_bytes
        )

        # Close current PDF and reopen with modified bytes
        self.close()
        self._pdf = pdfium.PdfDocument(modified_pdf_bytes)
        self._pdf_bytes = modified_pdf_bytes  # Update stored bytes
        self._source_name = "modified"

        pdf = self._ensure_open()

        # Track page objects that need gen_content() call
        pages_modified: dict[int, pdfium.PdfPage] = {}

        for para in translated_paragraphs:
            # Get the page object once and reuse it
            if para.page_number not in pages_modified:
                pages_modified[para.page_number] = pdf[para.page_number]

            page = pages_modified[para.page_number]

            # Determine initial font size
            initial_font_size = (
                para.adjusted_font_size
                if para.adjusted_font_size is not None
                else para.original_font_size
            ) or 12.0

            # Load font (use bold/italic variants if paragraph has those styles)
            text = para.translated_text
            font_handle = None
            if font_path:
                base_font_path = Path(font_path) if isinstance(font_path, str) else font_path
                style_key = (para.is_bold, para.is_italic)

                # Use subset if available, otherwise fall back to variant finding
                if font_subsets and style_key in font_subsets:
                    actual_font_path = font_subsets[style_key]
                else:
                    actual_font_path = self._find_font_variant(
                        base_font_path, para.is_bold, para.is_italic
                    )

                font_handle = self.load_font(actual_font_path)
            else:
                # Select standard font variant based on bold/italic flags
                if para.is_bold and para.is_italic:
                    font_name = "Helvetica-BoldOblique"
                elif para.is_bold:
                    font_name = "Helvetica-Bold"
                elif para.is_italic:
                    font_name = "Helvetica-Oblique"
                else:
                    font_name = "Helvetica"
                font_handle = self.load_standard_font(font_name)

            if not font_handle:
                # Fallback: create simple text object without gen_content()
                text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                    pdf.raw,
                    self.load_standard_font("Helvetica"),
                    ctypes.c_float(initial_font_size),
                )
                if text_obj:
                    text_ws = to_widestring(text)
                    pdfium.raw.FPDFText_SetText(text_obj, text_ws)
                    pdfium.raw.FPDFPageObj_Transform(
                        text_obj,
                        ctypes.c_double(1.0),
                        ctypes.c_double(0.0),
                        ctypes.c_double(0.0),
                        ctypes.c_double(1.0),
                        ctypes.c_double(para.block_bbox.x0),
                        ctypes.c_double(para.block_bbox.y0),
                    )
                    pdfium.raw.FPDFPage_InsertObject(page.raw, text_obj)
                continue

            # Use TextLayoutEngine to calculate layout
            layout_result = self._layout_engine.fit_text_in_bbox(
                text=text,
                bbox=para.block_bbox,
                font_handle=font_handle,
                initial_font_size=initial_font_size,
                rotation_degrees=para.rotation,
            )

            # Insert laid out text WITHOUT calling gen_content()
            rotation_radians = math.radians(-para.rotation)
            self._insert_laid_out_text_no_gen(
                page_num=para.page_number,
                layout_result=layout_result,
                bbox=para.block_bbox,
                font_handle=font_handle,
                font_size=layout_result.font_size,
                color=para.text_color,
                alignment=para.alignment,
                rotation=rotation_radians,
                pdftext_rotation_degrees=para.rotation,
                _page=page,
            )

        # Call gen_content() once per modified page at the end
        # This ensures proper z-ordering: original text -> rectangles -> translated text
        for page_num, page in pages_modified.items():
            page.gen_content()

    def _get_fallback_font(self, original_font: Font) -> Font:
        """Get appropriate fallback font for non-standard fonts.

        This tries to match the font family (serif/sans-serif/monospace)
        and style (bold/italic) of the original font.

        Args:
            original_font: Original font information

        Returns:
            Font with standard PDF font name
        """
        name_lower = original_font.name.lower()

        # Detect font family from name patterns
        is_serif = any(
            pattern in name_lower
            for pattern in [
                "times",
                "roman",
                "serif",
                "nimbus",  # NimbusRomNo9L is Times-like
                "palatino",
                "georgia",
                "garamond",
                "cambria",
                "book",
            ]
        )
        is_mono = any(
            pattern in name_lower
            for pattern in [
                "courier",
                "mono",
                "consola",
                "inconsolata",
                "menlo",
                "source code",
                "fira code",
            ]
        )

        # Determine fallback based on family and style
        if is_mono:
            if original_font.is_bold and original_font.is_italic:
                fallback_name = "Courier-BoldOblique"
            elif original_font.is_bold:
                fallback_name = "Courier-Bold"
            elif original_font.is_italic:
                fallback_name = "Courier-Oblique"
            else:
                fallback_name = "Courier"
        elif is_serif:
            if original_font.is_bold and original_font.is_italic:
                fallback_name = "Times-BoldItalic"
            elif original_font.is_bold:
                fallback_name = "Times-Bold"
            elif original_font.is_italic:
                fallback_name = "Times-Italic"
            else:
                fallback_name = "Times-Roman"
        else:
            # Default to sans-serif (Helvetica)
            if original_font.is_bold and original_font.is_italic:
                fallback_name = "Helvetica-BoldOblique"
            elif original_font.is_bold:
                fallback_name = "Helvetica-Bold"
            elif original_font.is_italic:
                fallback_name = "Helvetica-Oblique"
            else:
                fallback_name = "Helvetica"

        return Font(
            name=fallback_name,
            size=original_font.size,
            is_bold=original_font.is_bold,
            is_italic=original_font.is_italic,
        )

    def apply(self, doc: PDFDocument) -> None:
        """Apply intermediate data to the PDF (template PDF approach).

        This method removes all existing text objects and reinserts
        the text objects from the provided document. Non-text elements
        (images, figures) are preserved from the original PDF.

        Args:
            doc: PDFDocument containing text objects to apply
        """
        # Standard fonts that can be used directly
        STANDARD_FONTS = {
            "Helvetica",
            "Helvetica-Bold",
            "Helvetica-Oblique",
            "Helvetica-BoldOblique",
            "Times-Roman",
            "Times-Bold",
            "Times-Italic",
            "Times-BoldItalic",
            "Courier",
            "Courier-Bold",
            "Courier-Oblique",
            "Courier-BoldOblique",
        }

        pdf = self._ensure_open()
        for page_data in doc.pages:
            page_num = page_data.page_number
            if page_num >= len(pdf):
                continue

            # Remove all existing text
            self.remove_all_text(page_num)

            # Insert text objects from the document
            for text_obj in page_data.text_objects:
                original_font = text_obj.font or Font(name="Times-Roman", size=12.0)

                # Determine which font to use
                if original_font.name in STANDARD_FONTS:
                    font = original_font
                else:
                    # Use intelligent fallback based on font family
                    font = self._get_fallback_font(original_font)

                # Normalize text for standard font compatibility
                # (convert unsupported Unicode to ASCII equivalents)
                text_to_insert = normalize_text(text_obj.text, ascii_fallback=True)

                self.insert_text_object(
                    page_num=page_num,
                    text=text_to_insert,
                    bbox=text_obj.bbox,
                    font=font,
                    color=text_obj.color,
                    transform=text_obj.transform,
                )

    def to_json(self, indent: int = 2) -> str:
        """Export current PDF text objects as JSON.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        doc = self.extract_text_objects()
        return doc.to_json(indent=indent)

    def save(self, output_path: Union[Path, str]) -> None:
        """Save the PDF to a file.

        Args:
            output_path: Output file path
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf = self._ensure_open()
        with open(path, "wb") as f:
            pdf.save(f)

    def to_bytes(self) -> bytes:
        """Export the PDF as bytes."""
        from io import BytesIO

        buffer = BytesIO()
        pdf = self._ensure_open()
        pdf.save(buffer)
        return buffer.getvalue()

    def _draw_debug_bbox(
        self,
        page_num: int,
        bbox: BBox,
        raw_category: Optional[str] = None,
        confidence: Optional[float] = None,
        line_width: float = 1.5,
        label_font_size: float = 10.0,
        alpha: int = 200,
        label_position: str = "left",
        label_alpha: int = 220,
    ) -> None:
        """Draw a colored bbox outline with category label for debugging.

        Args:
            page_num: Page number (0-indexed).
            bbox: Bounding box to draw.
            raw_category: Raw category string for color selection and label.
            confidence: Detection confidence (0.0-1.0) to display in label.
            line_width: Stroke width in points.
            label_font_size: Font size for category label.
            alpha: Stroke alpha value (0-255, default 200).
            label_position: Label position - "left" (top-left) or "right" (top-right).
            label_alpha: Label background alpha value (0-255, default 220).
        """
        pdf = self._ensure_open()
        if page_num < 0 or page_num >= len(pdf):
            return

        page = pdf[page_num]
        page_handle = page.raw

        # Get color for raw_category
        color = DEBUG_RAW_CATEGORY_COLORS.get(
            raw_category, DEBUG_RAW_CATEGORY_COLORS[None]
        )
        r, g, b = color

        # Draw rectangle outline
        rect = pdfium.raw.FPDFPageObj_CreateNewRect(
            ctypes.c_float(bbox.x0),
            ctypes.c_float(bbox.y0),
            ctypes.c_float(bbox.width),
            ctypes.c_float(bbox.height),
        )

        # Set stroke color and width
        pdfium.raw.FPDFPageObj_SetStrokeColor(rect, r, g, b, alpha)
        pdfium.raw.FPDFPageObj_SetStrokeWidth(rect, ctypes.c_float(line_width))

        # Draw mode: stroke only (fill_mode=0, stroke=1)
        pdfium.raw.FPDFPath_SetDrawMode(rect, 0, ctypes.c_int(1))

        # Insert rectangle into page
        pdfium.raw.FPDFPage_InsertObject(page_handle, rect)

        # Draw category label with confidence above bbox
        if raw_category:
            if confidence is not None:
                label_text = f"{raw_category} ({confidence:.2f})"
            else:
                label_text = raw_category
        else:
            label_text = "none"
        self._draw_debug_label(
            doc_handle=pdf.raw,
            page_handle=page_handle,
            text=label_text,
            bbox=bbox,
            font_size=label_font_size,
            color=Color(r=r, g=g, b=b),
            position=label_position,
            alpha=label_alpha,
        )

        page.gen_content()

    def _draw_debug_label(
        self,
        doc_handle: Any,
        page_handle: Any,
        text: str,
        bbox: BBox,
        font_size: float = 10.0,
        color: Optional[Color] = None,
        position: str = "left",
        alpha: int = 220,
    ) -> None:
        """Draw a debug label with background at top corner of bbox.

        Args:
            doc_handle: PDFium document handle.
            page_handle: PDFium page handle.
            text: Label text.
            bbox: Bounding box to label.
            font_size: Font size in points.
            color: Text color (also used for background).
            position: Label position - "left" (top-left) or "right" (top-right).
            alpha: Background alpha value (0-255, default 220).
        """

        # Load Helvetica-Bold font for label
        font_handle = self.load_standard_font("Helvetica-Bold")
        if not font_handle:
            return

        # Estimate label dimensions
        char_width = font_size * 0.48
        label_width = len(text) * char_width + 4
        label_height = font_size + 4

        # Position: ABOVE bbox at top corner (outside the box)
        if position == "right":
            # Top-right corner
            text_x = bbox.x1 - label_width + 2
            bg_x = bbox.x1 - label_width
        else:
            # Top-left corner (default)
            text_x = bbox.x0 + 2
            bg_x = bbox.x0

        text_y = bbox.y1 + 2  # Above the box
        bg_y = bbox.y1 + 1

        # Draw background rectangle with category color
        if color:
            bg_rect = pdfium.raw.FPDFPageObj_CreateNewRect(
                ctypes.c_float(bg_x),
                ctypes.c_float(bg_y),
                ctypes.c_float(label_width),
                ctypes.c_float(label_height),
            )
            # Solid background using the category color
            pdfium.raw.FPDFPageObj_SetFillColor(
                bg_rect, color.r, color.g, color.b, alpha
            )
            # Fill mode: FPDF_FILLMODE_WINDING = 2
            pdfium.raw.FPDFPath_SetDrawMode(bg_rect, 2, ctypes.c_int(0))
            pdfium.raw.FPDFPage_InsertObject(page_handle, bg_rect)

        # Create text object
        text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
            doc_handle, font_handle, ctypes.c_float(font_size)
        )
        if not text_obj:
            return

        # Set text content
        text_ws = to_widestring(text)
        success = pdfium.raw.FPDFText_SetText(text_obj, text_ws)
        if not success:
            return

        # Set text color to white for contrast on colored background
        pdfium.raw.FPDFPageObj_SetFillColor(text_obj, 255, 255, 255, 255)

        # Position text at baseline
        pdfium.raw.FPDFPageObj_Transform(
            text_obj,
            ctypes.c_double(1.0),
            ctypes.c_double(0.0),
            ctypes.c_double(0.0),
            ctypes.c_double(1.0),
            ctypes.c_double(text_x),
            ctypes.c_double(text_y),
        )

        # Insert text into page
        pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)

    def draw_merge_debug_overlay(
        self,
        original_paragraphs: list[Paragraph],
        merged_paragraphs: list[Paragraph],
        line_width: float = 1.5,
        label_font_size: float = 10.0,
        merged_alpha: int = 128,
    ) -> None:
        """Draw debug bbox overlays showing both original and merged paragraphs.

        This method first draws original paragraph bboxes with labels at top-left,
        then overlays merged paragraph bboxes with 50% transparency and labels
        at top-right to avoid overlap.

        Args:
            original_paragraphs: List of original paragraphs (before merge).
            merged_paragraphs: List of merged paragraphs (after merge).
            line_width: Stroke width for bbox outlines.
            label_font_size: Font size for category labels.
            merged_alpha: Alpha value for merged bbox overlays (0-255, default 128 = 50%).
        """
        # First, draw original paragraphs with normal styling
        for para in original_paragraphs:
            self._draw_debug_bbox(
                page_num=para.page_number,
                bbox=para.block_bbox,
                raw_category=para.category,
                confidence=para.category_confidence,
                line_width=line_width,
                label_font_size=label_font_size,
                alpha=200,
                label_position="left",
                label_alpha=220,
            )

        # Then, overlay merged paragraphs with semi-transparent styling
        # Only draw merged paragraphs that actually grew (were merged with others)
        original_texts = {p.id: len(p.text) for p in original_paragraphs}
        for para in merged_paragraphs:
            original_len = original_texts.get(para.id, 0)
            # Only draw if this paragraph was merged (text length increased by 10%+)
            if len(para.text) > original_len * 1.1:
                self._draw_debug_bbox(
                    page_num=para.page_number,
                    bbox=para.block_bbox,
                    raw_category="MERGED",
                    confidence=None,
                    line_width=line_width * 1.5,  # Slightly thicker line
                    label_font_size=label_font_size,
                    alpha=merged_alpha,
                    label_position="right",
                    label_alpha=merged_alpha,
                )
