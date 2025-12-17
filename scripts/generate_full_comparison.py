#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PP-DocLayout V1 vs V2 全ページ比較スクリプト

43ページ全件の比較結果を JSON に出力します。

Usage:
    .venv-layout-test/bin/python scripts/generate_full_comparison.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def render_pdf_page_to_image(pdf_path: str, page_num: int, scale: float = 2.0) -> Image.Image:
    """PDF ページを画像にレンダリング"""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_num]
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


def get_pdf_page_count(pdf_path: str) -> int:
    """PDF のページ数を取得"""
    pdf = pdfium.PdfDocument(pdf_path)
    return len(pdf)


def analyze_all_pages(model_name: str, pdf_path: str) -> dict:
    """全ページをモデルで解析"""
    from paddleocr import LayoutDetection

    print(f"\n{'='*60}")
    print(f"モデル: {model_name}")
    print('='*60)

    model = LayoutDetection(model_name=model_name)
    page_count = get_pdf_page_count(pdf_path)

    results = {
        "model_name": model_name,
        "pdf_path": pdf_path,
        "pages_analyzed": page_count,
        "pages": [],
        "summary": {
            "total_detections": 0,
            "categories": Counter(),
        }
    }

    for page_num in range(page_count):
        print(f"  Page {page_num + 1}/{page_count}...", end=" ", flush=True)

        # 画像に変換
        image = render_pdf_page_to_image(pdf_path, page_num, scale=2.0)
        temp_path = f"/tmp/layout_full_{model_name.replace('-', '_')}_p{page_num}.png"
        image.save(temp_path)

        # 解析
        output = model.predict(temp_path, batch_size=1, layout_nms=True)

        page_detections = []
        for res in output:
            boxes = res.get("boxes", [])
            for det in boxes:
                label = det.get("label", "unknown")
                score = float(det.get("score", 0))
                coord = [float(c) for c in det.get("coordinate", [])]

                page_detections.append({
                    "label": label,
                    "score": score,
                    "coordinate": coord,
                })
                results["summary"]["categories"][label] += 1

        results["pages"].append({
            "page_number": page_num,
            "detections": page_detections,
            "detection_count": len(page_detections),
        })
        results["summary"]["total_detections"] += len(page_detections)

        print(f"{len(page_detections)} 検出")

    return results


def main():
    pdf_path = "tests/fixtures/sample_autogen_paper.pdf"
    output_dir = Path("docs/research/pp-doclayout-v1-v2-comparison")

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    page_count = get_pdf_page_count(pdf_path)
    print(f"PDF: {pdf_path}")
    print(f"ページ数: {page_count}")

    # V1 解析
    v1_results = analyze_all_pages("PP-DocLayout-L", pdf_path)

    # V2 解析
    v2_results = analyze_all_pages("PP-DocLayoutV2", pdf_path)

    # 結果を保存
    v1_output = output_dir / "layout_v1_results.json"
    v2_output = output_dir / "layout_v2_results.json"
    summary_output = output_dir / "comparison_summary.json"

    # V1 結果
    v1_save = {
        "model_name": v1_results["model_name"],
        "pdf_path": v1_results["pdf_path"],
        "pages_analyzed": v1_results["pages_analyzed"],
        "pages": v1_results["pages"],
        "summary": {
            "total_detections": v1_results["summary"]["total_detections"],
            "categories": dict(v1_results["summary"]["categories"]),
        }
    }
    with open(v1_output, "w", encoding="utf-8") as f:
        json.dump(v1_save, f, indent=2, ensure_ascii=False)
    print(f"\nV1 結果を保存: {v1_output}")

    # V2 結果
    v2_save = {
        "model_name": v2_results["model_name"],
        "pdf_path": v2_results["pdf_path"],
        "pages_analyzed": v2_results["pages_analyzed"],
        "pages": v2_results["pages"],
        "summary": {
            "total_detections": v2_results["summary"]["total_detections"],
            "categories": dict(v2_results["summary"]["categories"]),
        }
    }
    with open(v2_output, "w", encoding="utf-8") as f:
        json.dump(v2_save, f, indent=2, ensure_ascii=False)
    print(f"V2 結果を保存: {v2_output}")

    # サマリー
    v1_cats = set(v1_results["summary"]["categories"].keys())
    v2_cats = set(v2_results["summary"]["categories"].keys())

    summary = {
        "v1": {
            "model": v1_results["model_name"],
            "total_detections": v1_results["summary"]["total_detections"],
            "categories": dict(v1_results["summary"]["categories"]),
        },
        "v2": {
            "model": v2_results["model_name"],
            "total_detections": v2_results["summary"]["total_detections"],
            "categories": dict(v2_results["summary"]["categories"]),
        },
        "comparison": {
            "v1_only_categories": sorted(list(v1_cats - v2_cats)),
            "v2_only_categories": sorted(list(v2_cats - v1_cats)),
            "common_categories": sorted(list(v1_cats & v2_cats)),
        }
    }
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"サマリーを保存: {summary_output}")

    # 結果表示
    print("\n" + "="*60)
    print("比較結果サマリー")
    print("="*60)
    print(f"\n総検出数:")
    print(f"  V1: {v1_results['summary']['total_detections']}")
    print(f"  V2: {v2_results['summary']['total_detections']}")
    print(f"\nV1のみのカテゴリ: {sorted(v1_cats - v2_cats)}")
    print(f"V2のみのカテゴリ: {sorted(v2_cats - v1_cats)}")


if __name__ == "__main__":
    main()
