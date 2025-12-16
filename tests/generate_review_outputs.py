#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Generate PDF outputs for human review."""

import json
from pathlib import Path

from pdf_translator.core import PDFProcessor

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample_llama.pdf"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def main() -> None:
    """Generate review outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Generating Review Outputs ===")
    print(f"Source: {SAMPLE_PDF}")
    print(f"Output: {OUTPUT_DIR}")

    # 1. Extract and save JSON
    print("\n1. Extracting text objects...")
    with PDFProcessor(SAMPLE_PDF) as processor:
        doc = processor.extract_text_objects()
        json_str = doc.to_json(indent=2)

        json_path = OUTPUT_DIR / "extracted_text.json"
        json_path.write_text(json_str, encoding="utf-8")
        print(f"   Saved: {json_path}")
        print(f"   Objects: {sum(len(p.text_objects) for p in doc.pages)}")

    # 2. Roundtrip PDF (extract -> apply -> save)
    print("\n2. Creating roundtrip PDF...")
    with PDFProcessor(SAMPLE_PDF) as processor:
        doc = processor.extract_text_objects()
        processor.apply(doc)
        roundtrip_path = OUTPUT_DIR / "roundtrip.pdf"
        processor.save(roundtrip_path)
        print(f"   Saved: {roundtrip_path}")

    # 3. Extract JSON from roundtrip PDF
    print("\n3. Extracting from roundtrip PDF...")
    with PDFProcessor(roundtrip_path) as processor:
        doc2 = processor.extract_text_objects()
        json_str2 = doc2.to_json(indent=2)

        json_path2 = OUTPUT_DIR / "roundtrip_extracted.json"
        json_path2.write_text(json_str2, encoding="utf-8")
        print(f"   Saved: {json_path2}")
        print(f"   Objects: {sum(len(p.text_objects) for p in doc2.pages)}")

    # 4. Double roundtrip (idempotency check)
    print("\n4. Creating double roundtrip PDF (idempotency)...")
    with PDFProcessor(roundtrip_path) as processor:
        doc3 = processor.extract_text_objects()
        processor.apply(doc3)
        double_path = OUTPUT_DIR / "double_roundtrip.pdf"
        processor.save(double_path)
        print(f"   Saved: {double_path}")

    # 5. Summary comparison
    print("\n5. Comparing text content...")
    with PDFProcessor(SAMPLE_PDF) as p1:
        original = p1.extract_text_objects()

    with PDFProcessor(roundtrip_path) as p2:
        after_roundtrip = p2.extract_text_objects()

    orig_texts = {obj.text for page in original.pages for obj in page.text_objects}
    rt_texts = {obj.text for page in after_roundtrip.pages for obj in page.text_objects}

    common = orig_texts & rt_texts
    missing = orig_texts - rt_texts
    extra = rt_texts - orig_texts

    summary = {
        "original_count": len(orig_texts),
        "roundtrip_count": len(rt_texts),
        "common_count": len(common),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "preservation_rate": len(common) / len(orig_texts) * 100 if orig_texts else 0,
    }

    summary_path = OUTPUT_DIR / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"   Saved: {summary_path}")
    print(f"   Preservation rate: {summary['preservation_rate']:.1f}%")

    print("\n=== Done ===")
    print("Review files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
