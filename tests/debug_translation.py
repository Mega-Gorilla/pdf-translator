#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Debug script to investigate partial translation issues."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

INPUT_PDF = PROJECT_ROOT / "tests" / "fixtures" / "sample_autogen_paper.pdf"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs"


def main() -> None:
    """Debug translation pipeline."""
    from pdftext.extraction import dictionary_output
    from pdf_translator.core.paragraph_extractor import ParagraphExtractor
    from pdf_translator.core.layout_analyzer import LayoutAnalyzer
    from pdf_translator.core.layout_utils import assign_categories
    from pdf_translator.core.models import DEFAULT_TRANSLATABLE_RAW_CATEGORIES

    print("=" * 70)
    print("Debug: Translation Pipeline Analysis")
    print("=" * 70)
    print(f"Input: {INPUT_PDF}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Step 1: Extract paragraphs using pdftext
    print("Step 1: Extracting paragraphs with pdftext...")
    pdftext_result = dictionary_output(str(INPUT_PDF))

    extractor = ParagraphExtractor()
    paragraphs = extractor.extract(pdftext_result)
    print(f"  Total paragraphs extracted: {len(paragraphs)}")

    # Save raw paragraphs
    paragraphs_data = []
    for p in paragraphs:
        paragraphs_data.append({
            "id": p.id,
            "page": p.page_number,
            "text": p.text[:200] + "..." if len(p.text) > 200 else p.text,
            "text_length": len(p.text),
            "bbox": {"x0": p.block_bbox.x0, "y0": p.block_bbox.y0,
                     "x1": p.block_bbox.x1, "y1": p.block_bbox.y1},
            "category": None,
        })

    # Step 2: Layout analysis
    print("\nStep 2: Running layout analysis...")
    analyzer = LayoutAnalyzer()
    layout_blocks = analyzer.analyze_all(INPUT_PDF)

    total_blocks = sum(len(blocks) for blocks in layout_blocks.values())
    print(f"  Total layout blocks detected: {total_blocks}")

    # Save layout blocks
    layout_data = {}
    for page_num, blocks in layout_blocks.items():
        layout_data[page_num] = []
        for b in blocks:
            layout_data[page_num].append({
                "raw_category": b.raw_category.value,
                "confidence": b.confidence,
                "bbox": {"x0": b.bbox.x0, "y0": b.bbox.y0,
                         "x1": b.bbox.x1, "y1": b.bbox.y1},
                "is_translatable": b.is_translatable,
            })

    with open(OUTPUT_DIR / "debug_layout_blocks.json", "w") as f:
        json.dump(layout_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved layout blocks to: debug_layout_blocks.json")

    # Step 3: Assign categories
    print("\nStep 3: Assigning categories...")
    assign_categories(paragraphs, layout_blocks)

    # Update paragraphs data with categories
    for i, p in enumerate(paragraphs):
        paragraphs_data[i]["category"] = p.category
        paragraphs_data[i]["is_translatable"] = p.is_translatable()

    with open(OUTPUT_DIR / "debug_paragraphs.json", "w") as f:
        json.dump(paragraphs_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved paragraphs to: debug_paragraphs.json")

    # Step 4: Analyze translation status
    print("\nStep 4: Analyzing translation status...")

    translatable = [p for p in paragraphs if p.is_translatable()]
    skipped = [p for p in paragraphs if not p.is_translatable()]

    print(f"  Translatable paragraphs: {len(translatable)}")
    print(f"  Skipped paragraphs: {len(skipped)}")
    print()

    # Count by category
    category_counts = {}
    for p in paragraphs:
        cat = p.category or "None"
        if cat not in category_counts:
            category_counts[cat] = {"total": 0, "translatable": 0, "skipped": 0}
        category_counts[cat]["total"] += 1
        if p.is_translatable():
            category_counts[cat]["translatable"] += 1
        else:
            category_counts[cat]["skipped"] += 1

    print("  Category breakdown:")
    print(f"  {'Category':<25} {'Total':>8} {'Translate':>10} {'Skip':>8}")
    print("  " + "-" * 55)
    for cat, counts in sorted(category_counts.items()):
        print(f"  {cat:<25} {counts['total']:>8} {counts['translatable']:>10} {counts['skipped']:>8}")

    print()
    print(f"  DEFAULT_TRANSLATABLE_RAW_CATEGORIES: {DEFAULT_TRANSLATABLE_RAW_CATEGORIES}")

    # Step 5: Show first few pages details
    print("\n" + "=" * 70)
    print("Page-by-page breakdown (first 5 pages)")
    print("=" * 70)

    for page_num in range(min(5, max(p.page_number for p in paragraphs) + 1)):
        page_paragraphs = [p for p in paragraphs if p.page_number == page_num]
        page_translatable = [p for p in page_paragraphs if p.is_translatable()]
        page_skipped = [p for p in page_paragraphs if not p.is_translatable()]

        print(f"\nPage {page_num}:")
        print(f"  Total: {len(page_paragraphs)}, Translatable: {len(page_translatable)}, Skipped: {len(page_skipped)}")

        # Show translatable paragraphs on this page
        if page_translatable:
            print(f"  Translatable paragraphs:")
            for p in page_translatable[:3]:
                text_preview = p.text[:80].replace('\n', ' ')
                print(f"    - [{p.category or 'None'}] {text_preview}...")

        # Show skipped paragraphs on this page
        if page_skipped:
            print(f"  Skipped paragraphs:")
            for p in page_skipped[:3]:
                text_preview = p.text[:80].replace('\n', ' ')
                print(f"    - [{p.category or 'None'}] {text_preview}...")

    # Step 6: Specifically look at introduction content
    print("\n" + "=" * 70)
    print("Looking for 'Introduction' content")
    print("=" * 70)

    intro_paragraphs = [p for p in paragraphs if "introduction" in p.text.lower() or "1." in p.text[:10]]
    print(f"\nFound {len(intro_paragraphs)} paragraphs with 'introduction' or starting with '1.':")
    for p in intro_paragraphs:
        text_preview = p.text[:150].replace('\n', ' ')
        print(f"\n  Page {p.page_number}, Category: {p.category}, Translatable: {p.is_translatable()}")
        print(f"  Text: {text_preview}...")


if __name__ == "__main__":
    main()
