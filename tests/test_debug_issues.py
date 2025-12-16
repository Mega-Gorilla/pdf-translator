# SPDX-License-Identifier: Apache-2.0
"""Debug tests to identify root causes of PDF processing issues."""

import json
import tempfile
from pathlib import Path

import pytest

from pdf_translator.core import BBox, Font, PDFProcessor

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample_llama.pdf"
DEBUG_OUTPUT_DIR = Path(__file__).parent / "evaluation" / "debug"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Test fixture not available")
class TestDebugIssues:
    """Debug tests to identify specific issues."""

    @pytest.fixture(autouse=True)
    def setup_output_dir(self) -> None:
        """Ensure output directory exists."""
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_issue1_text_bbox_overlap(self) -> None:
        """Issue 1: Check if get_text_bounded returns overlapping text.

        Hypothesis: get_text_bounded() returns ALL text in a region,
        not just the text from the specific text object.
        """
        with PDFProcessor(SAMPLE_PDF) as processor:
            doc = processor.extract_text_objects()

            # Check for suspicious patterns
            overlaps = []
            page = doc.pages[0]

            for i, obj1 in enumerate(page.text_objects):
                for j, obj2 in enumerate(page.text_objects):
                    if i >= j:
                        continue

                    # Check if bboxes overlap
                    if (
                        obj1.bbox.x0 < obj2.bbox.x1
                        and obj1.bbox.x1 > obj2.bbox.x0
                        and obj1.bbox.y0 < obj2.bbox.y1
                        and obj1.bbox.y1 > obj2.bbox.y0
                    ):
                        overlaps.append(
                            {
                                "obj1_id": obj1.id,
                                "obj1_text": obj1.text[:50],
                                "obj1_bbox": obj1.bbox.to_dict(),
                                "obj2_id": obj2.id,
                                "obj2_text": obj2.text[:50],
                                "obj2_bbox": obj2.bbox.to_dict(),
                            }
                        )

            report = {
                "total_objects": len(page.text_objects),
                "overlapping_pairs": len(overlaps),
                "overlaps": overlaps[:20],  # Limit output
            }

            report_path = DEBUG_OUTPUT_DIR / "issue1_bbox_overlap.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

            print("\n=== Issue 1: BBox Overlap Analysis ===")
            print(f"Total objects: {len(page.text_objects)}")
            print(f"Overlapping pairs: {len(overlaps)}")
            if overlaps:
                print("Sample overlap:")
                print(f"  Obj1: {overlaps[0]['obj1_text']}")
                print(f"  Obj2: {overlaps[0]['obj2_text']}")

    def test_issue2_transform_behavior(self) -> None:
        """Issue 2: Check if transform is applied correctly.

        Hypothesis: The positioning via Transform may not work as expected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.pdf"

            with PDFProcessor(SAMPLE_PDF) as processor:
                # Insert a test text at known position
                processor.remove_all_text(0)

                # Insert at specific position
                test_bbox = BBox(x0=100.0, y0=700.0, x1=200.0, y1=720.0)
                obj_id = processor.insert_text_object(
                    page_num=0,
                    text="TEST_MARKER",
                    bbox=test_bbox,
                    font=Font(name="Helvetica", size=12.0),
                )
                processor.save(output_path)

            # Re-extract and check position
            with PDFProcessor(output_path) as verify:
                doc = verify.extract_text_objects()

                test_objs = [
                    o for o in doc.pages[0].text_objects if "TEST_MARKER" in o.text
                ]

                if test_objs:
                    actual_bbox = test_objs[0].bbox
                    position_error = {
                        "x0_diff": actual_bbox.x0 - test_bbox.x0,
                        "y0_diff": actual_bbox.y0 - test_bbox.y0,
                        "x1_diff": actual_bbox.x1 - test_bbox.x1,
                        "y1_diff": actual_bbox.y1 - test_bbox.y1,
                    }

                    report = {
                        "insertion_successful": obj_id is not None,
                        "original_bbox": test_bbox.to_dict(),
                        "actual_bbox": actual_bbox.to_dict(),
                        "position_error": position_error,
                    }

                    print("\n=== Issue 2: Transform Behavior ===")
                    print(f"Inserted at: ({test_bbox.x0}, {test_bbox.y0})")
                    print(f"Found at: ({actual_bbox.x0}, {actual_bbox.y0})")
                    x_diff = position_error['x0_diff']
                    y_diff = position_error['y0_diff']
                    print(f"Error: ({x_diff:.2f}, {y_diff:.2f})")
                else:
                    report = {
                        "insertion_successful": obj_id is not None,
                        "error": "TEST_MARKER not found in output",
                    }
                    print("\n=== Issue 2: Transform Behavior ===")
                    print("ERROR: TEST_MARKER not found!")

                report_path = DEBUG_OUTPUT_DIR / "issue2_transform.json"
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def test_issue3_font_fallback_effect(self) -> None:
        """Issue 3: Check the effect of font fallback on text rendering.

        Hypothesis: Using Helvetica instead of original fonts causes
        different text widths and bbox sizes.
        """
        with PDFProcessor(SAMPLE_PDF) as processor:
            doc = processor.extract_text_objects()

            # Analyze font distribution
            font_counts: dict[str, int] = {}
            non_standard_fonts: list[dict[str, str]] = []

            standard_fonts = {
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

            for page in doc.pages:
                for obj in page.text_objects:
                    if obj.font:
                        font_name = obj.font.name
                        font_counts[font_name] = font_counts.get(font_name, 0) + 1
                        if font_name not in standard_fonts:
                            non_standard_fonts.append(
                                {
                                    "id": obj.id,
                                    "font": font_name,
                                    "text": obj.text[:30],
                                }
                            )

            report = {
                "font_distribution": font_counts,
                "standard_font_count": sum(
                    c for f, c in font_counts.items() if f in standard_fonts
                ),
                "non_standard_font_count": sum(
                    c for f, c in font_counts.items() if f not in standard_fonts
                ),
                "non_standard_samples": non_standard_fonts[:20],
            }

            report_path = DEBUG_OUTPUT_DIR / "issue3_fonts.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

            print("\n=== Issue 3: Font Fallback Analysis ===")
            print(f"Fonts found: {list(font_counts.keys())}")
            print(f"Non-standard fonts: {len(non_standard_fonts)} objects")

    def test_issue4_encoding_problems(self) -> None:
        """Issue 4: Check for encoding issues with special characters.

        Hypothesis: The ÿ character appears due to encoding issues.
        """
        with PDFProcessor(SAMPLE_PDF) as processor:
            doc = processor.extract_text_objects()

            problematic_chars: list[dict[str, str | int]] = []

            for page in doc.pages:
                for obj in page.text_objects:
                    text = obj.text
                    for i, char in enumerate(text):
                        code = ord(char)
                        # Check for suspicious characters
                        if code == 0xFF:  # ÿ
                            problematic_chars.append(
                                {
                                    "id": obj.id,
                                    "char": repr(char),
                                    "code": code,
                                    "context": text[max(0, i - 10) : i + 10],
                                }
                            )
                        elif code < 0x20 and code not in (0x0A, 0x0D, 0x09):  # Control chars
                            problematic_chars.append(
                                {
                                    "id": obj.id,
                                    "char": repr(char),
                                    "code": code,
                                    "context": text[max(0, i - 10) : i + 10],
                                }
                            )

            report = {
                "problematic_char_count": len(problematic_chars),
                "samples": problematic_chars[:30],
            }

            report_path = DEBUG_OUTPUT_DIR / "issue4_encoding.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

            print("\n=== Issue 4: Encoding Analysis ===")
            print(f"Problematic characters found: {len(problematic_chars)}")
            if problematic_chars:
                print(f"Sample: {problematic_chars[0]}")

    def test_issue5_single_object_roundtrip(self) -> None:
        """Issue 5: Test roundtrip with a single simple text object.

        This isolates the basic insert/extract cycle.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "single.pdf"

            # Create minimal test
            with PDFProcessor(SAMPLE_PDF) as processor:
                processor.remove_all_text(0)

                original_text = "Simple ASCII text 123"
                original_bbox = BBox(x0=72.0, y0=750.0, x1=300.0, y1=770.0)
                original_font = Font(name="Helvetica", size=12.0)

                processor.insert_text_object(
                    page_num=0,
                    text=original_text,
                    bbox=original_bbox,
                    font=original_font,
                )
                processor.save(output_path)

            # Extract and compare
            with PDFProcessor(output_path) as verify:
                doc = verify.extract_text_objects()

                if doc.pages[0].text_objects:
                    obj = doc.pages[0].text_objects[0]
                    result = {
                        "original_text": original_text,
                        "extracted_text": obj.text,
                        "text_match": obj.text.strip() == original_text,
                        "original_font": original_font.name,
                        "extracted_font": obj.font.name if obj.font else None,
                        "original_bbox": original_bbox.to_dict(),
                        "extracted_bbox": obj.bbox.to_dict(),
                        "position_error": {
                            "x0": abs(obj.bbox.x0 - original_bbox.x0),
                            "y0": abs(obj.bbox.y0 - original_bbox.y0),
                        },
                    }

                    print("\n=== Issue 5: Single Object Roundtrip ===")
                    print(f"Original: '{original_text}'")
                    print(f"Extracted: '{obj.text}'")
                    print(f"Match: {result['text_match']}")
                    x_err = result['position_error']['x0']
                    y_err = result['position_error']['y0']
                    print(f"Position error: x0={x_err:.2f}, y0={y_err:.2f}")
                else:
                    result = {"error": "No text objects extracted"}
                    print("\n=== Issue 5: Single Object Roundtrip ===")
                    print("ERROR: No text objects found!")

                report_path = DEBUG_OUTPUT_DIR / "issue5_single_roundtrip.json"
                report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    def test_issue6_text_accumulation(self) -> None:
        """Issue 6: Check if text accumulates in the same bbox region.

        Hypothesis: Multiple conversions add text instead of replacing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf1_path = Path(tmpdir) / "pdf1.pdf"
            pdf2_path = Path(tmpdir) / "pdf2.pdf"

            # First conversion
            with PDFProcessor(SAMPLE_PDF) as p1:
                doc1 = p1.extract_text_objects()
                obj_count_original = len(doc1.pages[0].text_objects)
                p1.apply(doc1)
                p1.save(pdf1_path)

            # Check after first conversion
            with PDFProcessor(pdf1_path) as p2:
                doc2 = p2.extract_text_objects()
                obj_count_pdf1 = len(doc2.pages[0].text_objects)
                p2.apply(doc2)
                p2.save(pdf2_path)

            # Check after second conversion
            with PDFProcessor(pdf2_path) as p3:
                doc3 = p3.extract_text_objects()
                obj_count_pdf2 = len(doc3.pages[0].text_objects)

            result = {
                "original_count": obj_count_original,
                "after_pdf1_count": obj_count_pdf1,
                "after_pdf2_count": obj_count_pdf2,
                "count_stable": obj_count_pdf1 == obj_count_pdf2,
            }

            report_path = DEBUG_OUTPUT_DIR / "issue6_accumulation.json"
            report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

            print("\n=== Issue 6: Text Accumulation Check ===")
            print(f"Original: {obj_count_original} objects")
            print(f"After PDF1: {obj_count_pdf1} objects")
            print(f"After PDF2: {obj_count_pdf2} objects")
            print(f"Count stable: {result['count_stable']}")

    def test_issue7_text_extraction_method(self) -> None:
        """Issue 7: Compare text extraction methods.

        Check if get_text_bounded() is the problem by comparing
        with character-level extraction.
        """

        with PDFProcessor(SAMPLE_PDF) as processor:
            pdf = processor._ensure_open()
            page = pdf[0]

            # Get first few text objects
            text_objects = list(page.get_objects(filter=[1]))[:5]

            comparisons = []
            textpage = page.get_textpage()

            try:
                for i, obj in enumerate(text_objects):
                    bounds = obj.get_bounds()
                    if not bounds:
                        continue

                    left, bottom, right, top = bounds

                    # Method 1: get_text_bounded
                    text_bounded = textpage.get_text_bounded(
                        left=left, bottom=bottom, right=right, top=top
                    )

                    # Method 2: Character-level iteration
                    char_count = textpage.count_chars()
                    chars_in_bounds = []
                    for ci in range(char_count):
                        char_box = textpage.get_charbox(ci, loose=True)
                        if char_box:
                            cx0, cy0, cx1, cy1 = char_box
                            # Check if character center is in bounds
                            cx = (cx0 + cx1) / 2
                            cy = (cy0 + cy1) / 2
                            if left <= cx <= right and bottom <= cy <= top:
                                char = textpage.get_text_range(ci, 1)
                                chars_in_bounds.append(char)
                    text_chars = "".join(chars_in_bounds)

                    comparisons.append(
                        {
                            "obj_index": i,
                            "bbox": {"left": left, "bottom": bottom, "right": right, "top": top},
                            "text_bounded": text_bounded[:100] if text_bounded else "",
                            "text_chars": text_chars[:100] if text_chars else "",
                            "match": text_bounded == text_chars,
                        }
                    )
            finally:
                textpage.close()

            report_path = DEBUG_OUTPUT_DIR / "issue7_extraction_method.json"
            report_path.write_text(json.dumps(comparisons, indent=2, ensure_ascii=False))

            print("\n=== Issue 7: Text Extraction Method Comparison ===")
            for comp in comparisons:
                print(f"Obj {comp['obj_index']}: match={comp['match']}")
                if not comp["match"]:
                    print(f"  bounded: {comp['text_bounded'][:50]}")
                    print(f"  chars: {comp['text_chars'][:50]}")
