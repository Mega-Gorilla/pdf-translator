#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Quick evaluation script."""

import json
import re
import tempfile
from pathlib import Path

from pdf_translator.core import PDFProcessor

SAMPLE_PDF = Path("tests/fixtures/sample_llama.pdf")
OUTPUT_PATH = Path("tests/evaluation/outputs/quick_eval.json")


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison."""
    text = re.sub(r"[\x00-\x1f\x7f\xad\xff]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    """Run quick evaluation."""
    print("=== Quick Evaluation ===")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf1_path = Path(tmpdir) / "pdf1.pdf"
        pdf2_path = Path(tmpdir) / "pdf2.pdf"

        # First conversion
        print("Step 1: Original -> PDF1")
        with PDFProcessor(SAMPLE_PDF) as p1:
            doc = p1.extract_text_objects()
            p1.apply(doc)
            p1.save(pdf1_path)

        # Second conversion
        print("Step 2: PDF1 -> PDF2")
        with PDFProcessor(pdf1_path) as p2:
            doc2 = p2.extract_text_objects()
            p2.apply(doc2)
            p2.save(pdf2_path)

        # Extract from both
        print("Step 3: Compare PDF1 vs PDF2")
        with PDFProcessor(pdf1_path) as proc1:
            doc1_extracted = proc1.extract_text_objects()

        with PDFProcessor(pdf2_path) as proc2:
            doc2_extracted = proc2.extract_text_objects()

        # Compare
        obj_map1 = {
            obj.id: obj
            for page in doc1_extracted.pages
            for obj in page.text_objects
        }
        obj_map2 = {
            obj.id: obj
            for page in doc2_extracted.pages
            for obj in page.text_objects
        }

        total = len(obj_map1)
        matched = 0
        text_mismatches = 0
        position_errors = 0

        for obj_id in obj_map1:
            if obj_id not in obj_map2:
                continue

            obj1 = obj_map1[obj_id]
            obj2 = obj_map2[obj_id]

            text1 = normalize_for_comparison(obj1.text)
            text2 = normalize_for_comparison(obj2.text)

            is_match = True
            if text1 != text2:
                text_mismatches += 1
                is_match = False

            pos_error = max(
                abs(obj1.bbox.x0 - obj2.bbox.x0),
                abs(obj1.bbox.y0 - obj2.bbox.y0),
                abs(obj1.bbox.x1 - obj2.bbox.x1),
                abs(obj1.bbox.y1 - obj2.bbox.y1),
            )
            if pos_error > 5.0:
                position_errors += 1
                is_match = False

            if is_match:
                matched += 1

        match_rate = (matched / total * 100) if total > 0 else 0

        results = {
            "status": "completed",
            "total_objects": total,
            "matched": matched,
            "match_rate": match_rate,
            "text_mismatches": text_mismatches,
            "position_errors": position_errors,
        }

        OUTPUT_PATH.write_text(json.dumps(results, indent=2))

        print("")
        print("=== Results ===")
        print(f"Total objects: {total}")
        print(f"Matched: {matched}")
        print(f"Match rate: {match_rate:.2f}%")
        print(f"Text mismatches: {text_mismatches}")
        print(f"Position errors (>5pt): {position_errors}")


if __name__ == "__main__":
    main()
