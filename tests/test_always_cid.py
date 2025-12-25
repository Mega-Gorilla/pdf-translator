#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test that CID fonts work correctly for both ASCII and CJK text."""

import ctypes
from pathlib import Path

import pypdfium2 as pdfium

from pdf_translator.core.helpers import to_widestring
from pdf_translator.core.pdf_processor import PDFProcessor


def test_cid_font_ascii_and_cjk():
    """Test ASCII and CJK text with CID font (always used now)."""
    # Create blank PDF
    doc = pdfium.PdfDocument.new()
    doc.new_page(595, 842)  # Create page (return value not used directly)
    temp_path = Path("tests/outputs/test_cid_font.pdf")
    with open(temp_path, "wb") as f:
        doc.save(f)
    doc.close()

    koruri_font = Path("src/pdf_translator/resources/fonts/Koruri-Regular.ttf")

    with PDFProcessor(temp_path) as processor:
        # Load font (always CID now)
        font_handle = processor.load_font(koruri_font)
        print(f"Font handle loaded: {font_handle is not None}")

        # Insert ASCII text
        page = processor._ensure_open()[0]
        doc_handle = processor._ensure_open().raw
        page_handle = page.raw

        # Create text object with ASCII
        text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
            doc_handle, font_handle, ctypes.c_float(12.0)
        )
        text_ws = to_widestring("Hello World - ASCII only text")
        success = pdfium.raw.FPDFText_SetText(text_obj, text_ws)
        print(f"ASCII text set: {success}")

        pdfium.raw.FPDFPageObj_Transform(
            text_obj, 1.0, 0.0, 0.0, 1.0, 50.0, 800.0
        )
        pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)

        # Create text object with CJK
        text_obj2 = pdfium.raw.FPDFPageObj_CreateTextObj(
            doc_handle, font_handle, ctypes.c_float(12.0)
        )
        text_ws2 = to_widestring("日本語テキスト Japanese text")
        success2 = pdfium.raw.FPDFText_SetText(text_obj2, text_ws2)
        print(f"CJK text set: {success2}")

        pdfium.raw.FPDFPageObj_Transform(
            text_obj2, 1.0, 0.0, 0.0, 1.0, 50.0, 750.0
        )
        pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj2)

        page.gen_content()
        output_path = Path("tests/outputs/test_cid_font_result.pdf")
        processor.save(output_path)

    # Extract and verify
    print("\n=== Extraction ===")
    result_doc = pdfium.PdfDocument(str(output_path))
    page = result_doc[0]
    textpage = page.get_textpage()
    extracted = textpage.get_text_bounded()
    textpage.close()
    result_doc.close()

    print(f"Extracted: {extracted}")
    y_count = extracted.count('\xff')
    print(f"ÿ count: {y_count}")

    # Cleanup
    temp_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)

    # Assert no corruption and text is extracted correctly
    assert y_count == 0, f"Found {y_count} corrupted characters (ÿ)"
    assert "Hello" in extracted, "ASCII text not found"
    assert "日本語" in extracted, "CJK text not found"


if __name__ == "__main__":
    test_cid_font_ascii_and_cjk()
