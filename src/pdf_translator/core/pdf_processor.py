# SPDX-License-Identifier: Apache-2.0
"""PDF processing module using pypdfium2.

This module provides PDF text extraction, manipulation, and reinsertion
capabilities using the pypdfium2 library.
"""

from __future__ import annotations

import ctypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pypdfium2 as pdfium

from .helpers import to_byte_array, to_widestring
from .models import (
    BBox,
    Color,
    Font,
    Metadata,
    Page,
    PDFDocument,
    TextObject,
    Transform,
)

# PDFium object type constant for text
FPDF_PAGEOBJ_TEXT = 1


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
            FileNotFoundError: If the file path doesn't exist
            ValueError: If the PDF cannot be loaded
        """
        self._source_name: str
        self._pdf: pdfium.PdfDocument
        self._loaded_fonts: dict[str, int] = {}  # font_path -> font_handle

        if isinstance(pdf_source, bytes):
            self._pdf = pdfium.PdfDocument(pdf_source)
            self._source_name = "bytes"
        else:
            path = Path(pdf_source)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {path}")
            self._pdf = pdfium.PdfDocument(str(path))
            self._source_name = path.name

    def __enter__(self) -> PDFProcessor:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the PDF document and release resources."""
        if hasattr(self, "_pdf") and self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    @property
    def page_count(self) -> int:
        """Get the number of pages in the document."""
        return len(self._pdf)

    def _generate_id(self, prefix: str = "text") -> str:
        """Generate a unique ID for objects.

        Args:
            prefix: Prefix for the ID

        Returns:
            Unique identifier string
        """
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _get_text_from_page(self, page: pdfium.PdfPage) -> str:
        """Extract all text from a page using textpage.

        Args:
            page: pypdfium2 page object

        Returns:
            All text content from the page
        """
        textpage = page.get_textpage()
        try:
            return textpage.get_text_bounded()
        finally:
            textpage.close()

    def _get_text_object_content(
        self, page: pdfium.PdfPage, obj: pdfium.PdfObject
    ) -> str:
        """Get text content from a text object using textpage.

        Args:
            page: The page containing the object
            obj: Text page object

        Returns:
            Text content within the object's bounds
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
            # Create ctypes variables for the matrix components
            a = ctypes.c_double()
            b = ctypes.c_double()
            c = ctypes.c_double()
            d = ctypes.c_double()
            e = ctypes.c_double()
            f = ctypes.c_double()

            success = pdfium.raw.FPDFPageObj_GetMatrix(
                obj.raw,
                ctypes.byref(a),
                ctypes.byref(b),
                ctypes.byref(c),
                ctypes.byref(d),
                ctypes.byref(e),
                ctypes.byref(f),
            )

            if success:
                transform = Transform(
                    a=a.value,
                    b=b.value,
                    c=c.value,
                    d=d.value,
                    e=e.value,
                    f=f.value,
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
        """
        pages = []

        for page_num in range(len(self._pdf)):
            page = self._pdf[page_num]
            width = page.get_width()
            height = page.get_height()
            rotation = page.get_rotation()

            text_objects = []

            # Get all text objects from the page
            for obj in page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]):
                bounds = obj.get_bounds()
                if bounds is None:
                    continue

                left, bottom, right, top = bounds
                bbox = BBox(x0=left, y0=bottom, x1=right, y1=top)

                # Get text content
                text = self._get_text_object_content(page, obj)
                if not text:
                    continue

                # Generate unique ID
                obj_id = self._generate_id("text")

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

        Returns:
            Number of objects removed

        Raises:
            IndexError: If page_num is out of range
        """
        if page_num < 0 or page_num >= len(self._pdf):
            raise IndexError(f"Page number {page_num} out of range")

        page = self._pdf[page_num]

        # If object_ids is None, remove all text objects
        if object_ids is None:
            objects_to_remove = list(page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]))
            for obj in objects_to_remove:
                page.remove_obj(obj)
            page.gen_content()
            return len(objects_to_remove)

        # For specific IDs, we need to match by position since pypdfium2
        # doesn't track our custom IDs
        # First, extract current objects with their bounds
        doc = self.extract_text_objects()
        id_to_bbox = {}
        for text_obj in doc.pages[page_num].text_objects:
            if text_obj.id in object_ids:
                id_to_bbox[text_obj.id] = text_obj.bbox

        # Now remove objects that match the bboxes
        removed = 0
        objects_to_remove = []

        for obj in page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]):
            bounds = obj.get_bounds()
            if bounds is None:
                continue

            left, bottom, right, top = bounds
            for obj_id, bbox in id_to_bbox.items():
                # Match by approximate bbox
                if (
                    abs(bbox.x0 - left) < 1.0
                    and abs(bbox.y0 - bottom) < 1.0
                    and abs(bbox.x1 - right) < 1.0
                    and abs(bbox.y1 - top) < 1.0
                ):
                    objects_to_remove.append(obj)
                    break

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
    ) -> Optional[int]:
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
        font_handle = pdfium.raw.FPDFText_LoadFont(
            self._pdf.raw,
            font_arr,
            ctypes.c_uint(len(font_data)),
            ctypes.c_int(pdfium.raw.FPDF_FONT_TRUETYPE),
            ctypes.c_int(1 if is_cid else 0),
        )

        if font_handle:
            self._loaded_fonts[path_str] = font_handle
            return font_handle
        return None

    def load_standard_font(self, font_name: str) -> Optional[int]:
        """Load a standard PDF font.

        Args:
            font_name: Standard font name (e.g., "Helvetica", "Times-Roman")

        Returns:
            Font handle or None if loading failed
        """
        if font_name in self._loaded_fonts:
            return self._loaded_fonts[font_name]

        font_handle = pdfium.raw.FPDFText_LoadStandardFont(
            self._pdf.raw, font_name.encode("utf-8")
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
        if page_num < 0 or page_num >= len(self._pdf):
            raise IndexError(f"Page number {page_num} out of range")

        page = self._pdf[page_num]
        doc_handle = self._pdf.raw
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

        return self._generate_id("text")

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

    def apply(self, doc: PDFDocument) -> None:
        """Apply intermediate data to the PDF (template PDF approach).

        This method removes all existing text objects and reinserts
        the text objects from the provided document. Non-text elements
        (images, figures) are preserved from the original PDF.

        Args:
            doc: PDFDocument containing text objects to apply
        """
        # Standard fonts that can be used as fallback
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

        for page_data in doc.pages:
            page_num = page_data.page_number
            if page_num >= len(self._pdf):
                continue

            # Remove all existing text
            self.remove_all_text(page_num)

            # Insert text objects from the document
            for text_obj in page_data.text_objects:
                original_font = text_obj.font or Font(name="Helvetica", size=12.0)

                # Determine which font to use
                if original_font.name in STANDARD_FONTS:
                    font = original_font
                else:
                    # Use fallback standard font based on style
                    if original_font.is_bold and original_font.is_italic:
                        fallback_name = "Helvetica-BoldOblique"
                    elif original_font.is_bold:
                        fallback_name = "Helvetica-Bold"
                    elif original_font.is_italic:
                        fallback_name = "Helvetica-Oblique"
                    else:
                        fallback_name = "Helvetica"

                    font = Font(
                        name=fallback_name,
                        size=original_font.size,
                        is_bold=original_font.is_bold,
                        is_italic=original_font.is_italic,
                    )

                self.insert_text_object(
                    page_num=page_num,
                    text=text_obj.text,
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
        with open(path, "wb") as f:
            self._pdf.save(f)
