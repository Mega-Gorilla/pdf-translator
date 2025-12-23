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
        self._loaded_fonts: dict[str, Any] = {}  # font_path -> font_handle
        self._loaded_font_buffers: dict[str, ctypes.Array[Any]] = {}  # keep buffers alive
        self._layout_engine = TextLayoutEngine()

        if isinstance(pdf_source, bytes):
            self._pdf = pdfium.PdfDocument(pdf_source)
            self._source_name = "bytes"
        elif isinstance(pdf_source, (str, Path)):
            path = Path(pdf_source)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {path}")
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

    def remove_text_in_bbox(
        self,
        page_num: int,
        bbox: BBox,
        containment_threshold: float = 0.5,
    ) -> int:
        """Remove text objects contained in a bounding box.

        Args:
            page_num: Page number (0-indexed).
            bbox: Target bounding box (PDF coordinates).
            containment_threshold: Ratio of text bbox area contained in bbox.

        Returns:
            Number of objects removed.
        """
        pdf = self._ensure_open()
        if page_num < 0 or page_num >= len(pdf):
            raise IndexError(f"Page number {page_num} out of range")

        page = pdf[page_num]
        removed = 0
        objects_to_remove = []

        def containment_ratio(text_bbox: BBox, target_bbox: BBox) -> float:
            x0 = max(text_bbox.x0, target_bbox.x0)
            y0 = max(text_bbox.y0, target_bbox.y0)
            x1 = min(text_bbox.x1, target_bbox.x1)
            y1 = min(text_bbox.y1, target_bbox.y1)
            if x0 >= x1 or y0 >= y1:
                return 0.0
            inter_area = (x1 - x0) * (y1 - y0)
            text_area = text_bbox.width * text_bbox.height
            if text_area <= 0:
                return 0.0
            return inter_area / text_area

        textpage = page.get_textpage()
        try:
            for obj in page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]):
                bounds = obj.get_bounds()
                if bounds is None:
                    continue

                text = self._get_text_object_text_direct(obj, textpage)
                text = text.strip() if text else ""
                if not text:
                    continue

                left, bottom, right, top = bounds
                text_bbox = BBox(x0=left, y0=bottom, x1=right, y1=top)
                containment = containment_ratio(text_bbox, bbox)
                if containment >= containment_threshold:
                    objects_to_remove.append(obj)
        finally:
            textpage.close()

        for obj in objects_to_remove:
            page.remove_obj(obj)
            removed += 1

        if removed > 0:
            page.gen_content()

        return removed

    def remove_all_text(self, page_num: int) -> int:
        """Remove all text objects from a page.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Number of objects removed
        """
        return self.remove_text_objects(page_num, None)

    def load_font(
        self, font_path: Union[Path, str], is_cid: bool = False
    ) -> Optional[Any]:
        """Load a TrueType font for text insertion.

        Args:
            font_path: Path to TTF font file
            is_cid: Whether this is a CID font (for CJK characters)

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
        font_handle = pdfium.raw.FPDFText_LoadFont(
            pdf.raw,
            font_arr,
            ctypes.c_uint(len(font_data)),
            ctypes.c_int(pdfium.raw.FPDF_FONT_TRUETYPE),
            ctypes.c_int(1 if is_cid else 0),
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
            # Check if CID font needed for CJK
            is_cid = self._needs_cid_font(text)
            font_handle = self.load_font(font_path, is_cid=is_cid)
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
    ) -> bool:
        """Insert laid out text (multiple lines) into a page.

        This method inserts each line as a separate text object, positioned
        according to the layout result.

        Args:
            page_num: Page number (0-indexed).
            layout_result: Layout result from TextLayoutEngine.
            bbox: Bounding box for positioning.
            font_handle: PDFium font handle.
            font_size: Font size to use.
            color: Text color (default: black).
            alignment: Text alignment ("left", "center", "right", "justify").
            rotation: Rotation angle in radians.

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

            # Calculate x position based on alignment (in local coordinates)
            if alignment == "center":
                local_x = (bbox.width - line.width) / 2
            elif alignment == "right":
                local_x = bbox.width - line.width
            else:  # left or justify (justify treated as left for now)
                local_x = 0.0

            # Calculate local y offset from bbox top
            local_y = line.y_position - bbox.y1

            # Apply rotation transform around bbox origin (x0, y1)
            # Transform: rotate around origin, then translate to bbox position
            # For rotated text, we rotate the local offset and add bbox origin
            if abs(rotation) > 0.001:  # Only apply rotation if significant
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

    def apply_paragraphs(
        self,
        paragraphs: list[Paragraph],
        font_path: Optional[Union[Path, str]] = None,
        debug_draw_bbox: bool = False,
    ) -> None:
        """Apply translated paragraphs to the PDF.

        Uses TextLayoutEngine to properly fit text within bounding boxes,
        with automatic line wrapping and font size adjustment.

        Args:
            paragraphs: List of paragraphs to apply.
            font_path: Path to TTF font file for custom fonts.
            debug_draw_bbox: If True, draw colored bbox outlines with category labels.
        """
        self._ensure_open()

        for para in paragraphs:
            if not para.translated_text:
                # Even if no translation, draw debug bbox if enabled
                if debug_draw_bbox:
                    self._draw_debug_bbox(
                        page_num=para.page_number,
                        bbox=para.block_bbox,
                        raw_category=para.category,
                    )
                continue

            self.remove_text_in_bbox(
                page_num=para.page_number,
                bbox=para.block_bbox,
            )

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
                is_cid = self._needs_cid_font(text)
                font_handle = self.load_font(font_path, is_cid=is_cid)
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
                # Fallback to old method if font loading fails
                font = Font(name="Helvetica", size=initial_font_size)
                self.insert_text_object(
                    page_num=para.page_number,
                    text=text,
                    bbox=para.block_bbox,
                    font=font,
                    font_path=font_path,
                )
                continue

            # Use TextLayoutEngine to calculate layout
            layout_result = self._layout_engine.fit_text_in_bbox(
                text=text,
                bbox=para.block_bbox,
                font_handle=font_handle,
                initial_font_size=initial_font_size,
            )

            # Insert laid out text
            self.insert_laid_out_text(
                page_num=para.page_number,
                layout_result=layout_result,
                bbox=para.block_bbox,
                font_handle=font_handle,
                font_size=layout_result.font_size,
                color=para.text_color,
                alignment=para.alignment,
                rotation=para.rotation,
            )

            if debug_draw_bbox:
                self._draw_debug_bbox(
                    page_num=para.page_number,
                    bbox=para.block_bbox,
                    raw_category=para.category,
                )

    def _needs_cid_font(self, text: str) -> bool:
        """Check if text contains CJK characters requiring CID font.

        Args:
            text: Text to check

        Returns:
            True if CID font is needed
        """
        for char in text:
            code = ord(char)
            # CJK Unified Ideographs and common CJK ranges
            if (
                0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
                or 0x3040 <= code <= 0x309F  # Hiragana
                or 0x30A0 <= code <= 0x30FF  # Katakana
                or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
            ):
                return True
        return False

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
    ) -> None:
        """Draw a colored bbox outline with category label for debugging.

        Args:
            page_num: Page number (0-indexed).
            bbox: Bounding box to draw.
            raw_category: Raw category string for color selection and label.
            confidence: Detection confidence (0.0-1.0) to display in label.
            line_width: Stroke width in points.
            label_font_size: Font size for category label.
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
        pdfium.raw.FPDFPageObj_SetStrokeColor(rect, r, g, b, 200)
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
    ) -> None:
        """Draw a debug label with background at top-left of bbox.

        Args:
            doc_handle: PDFium document handle.
            page_handle: PDFium page handle.
            text: Label text.
            bbox: Bounding box to label.
            font_size: Font size in points.
            color: Text color (also used for background).
        """

        # Load Helvetica-Bold font for label
        font_handle = self.load_standard_font("Helvetica-Bold")
        if not font_handle:
            return

        # Estimate label dimensions
        char_width = font_size * 0.48
        label_width = len(text) * char_width + 4
        label_height = font_size + 4

        # Position: ABOVE bbox at top-left corner (outside the box)
        # Text baseline position
        text_x = bbox.x0 + 2
        text_y = bbox.y1 + 2  # Above the box

        # Background rectangle position (above the box)
        bg_x = bbox.x0
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
                bg_rect, color.r, color.g, color.b, 220
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

    def draw_debug_overlay(
        self,
        paragraphs: list[Paragraph],
        line_width: float = 1.5,
        label_font_size: float = 10.0,
    ) -> None:
        """Draw debug bbox overlays for all paragraphs without modifying text.

        This method draws colored bbox outlines and category labels on top of
        the existing PDF content, useful for verifying layout analysis results.

        Args:
            paragraphs: List of paragraphs to visualize.
            line_width: Stroke width for bbox outlines.
            label_font_size: Font size for category labels.
        """
        for para in paragraphs:
            self._draw_debug_bbox(
                page_num=para.page_number,
                bbox=para.block_bbox,
                raw_category=para.category,
                confidence=para.category_confidence,
                line_width=line_width,
                label_font_size=label_font_size,
            )
