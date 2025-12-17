#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PP-DocLayout V1 vs V2 比較テストスクリプト

このスクリプトは PP-DocLayout-L (V1) と PP-DocLayoutV2 (V2) の
カテゴリ体系と検出結果を比較します。

Usage:
    .venv-layout-test/bin/python scripts/compare_layout_models.py <pdf_path>
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def render_pdf_page_to_image(pdf_path: str, page_num: int = 0, scale: float = 2.0) -> Image.Image:
    """PDF ページを画像にレンダリング"""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_num]

    # スケールを適用してレンダリング
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil()

    return image


def analyze_with_model(model_name: str, image: Image.Image) -> dict:
    """指定モデルでレイアウト解析を実行"""
    from paddleocr import LayoutDetection

    print(f"\n{'='*60}")
    print(f"モデル: {model_name}")
    print('='*60)

    # モデル初期化
    model = LayoutDetection(model_name=model_name)

    # 一時ファイルに保存して解析
    temp_path = f"/tmp/layout_test_{model_name.replace('-', '_')}.png"
    image.save(temp_path)

    # 解析実行
    output = model.predict(temp_path, batch_size=1, layout_nms=True)

    results = {
        "model_name": model_name,
        "detections": [],
        "categories": Counter(),
    }

    for res in output:
        # boxes リストから検出結果を取得
        boxes = res.get("boxes", [])

        for det in boxes:
            label = det.get("label", "unknown")
            score = float(det.get("score", 0))
            coord = det.get("coordinate", [])
            # numpy float32 を Python float に変換
            coord = [float(c) for c in coord]

            results["detections"].append({
                "label": label,
                "score": score,
                "coordinate": coord,
            })
            results["categories"][label] += 1

    return results


def print_comparison(v1_results: dict, v2_results: dict) -> None:
    """比較結果を表示"""
    print("\n" + "="*80)
    print("比較結果サマリー")
    print("="*80)

    # カテゴリ一覧
    v1_cats = set(v1_results["categories"].keys())
    v2_cats = set(v2_results["categories"].keys())

    print("\n[検出数]")
    print(f"  V1 (PP-DocLayout-L): {len(v1_results['detections'])} 検出")
    print(f"  V2 (PP-DocLayoutV2): {len(v2_results['detections'])} 検出")

    print("\n[カテゴリ数]")
    print(f"  V1: {len(v1_cats)} カテゴリ")
    print(f"  V2: {len(v2_cats)} カテゴリ")

    print("\n[V1 のみに存在するカテゴリ]")
    v1_only = v1_cats - v2_cats
    if v1_only:
        for cat in sorted(v1_only):
            print(f"  - {cat}: {v1_results['categories'][cat]} 件")
    else:
        print("  (なし)")

    print("\n[V2 のみに存在するカテゴリ]")
    v2_only = v2_cats - v1_cats
    if v2_only:
        for cat in sorted(v2_only):
            print(f"  - {cat}: {v2_results['categories'][cat]} 件")
    else:
        print("  (なし)")

    print("\n[共通カテゴリの検出数比較]")
    common = v1_cats & v2_cats
    print(f"  {'カテゴリ':<25} {'V1':>6} {'V2':>6} {'差分':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8}")
    for cat in sorted(common):
        v1_count = v1_results["categories"][cat]
        v2_count = v2_results["categories"][cat]
        diff = v2_count - v1_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"  {cat:<25} {v1_count:>6} {v2_count:>6} {diff_str:>8}")

    # 詳細なカテゴリ一覧
    print("\n[V1 全カテゴリ詳細]")
    for cat, count in sorted(v1_results["categories"].items()):
        print(f"  {cat}: {count}")

    print("\n[V2 全カテゴリ詳細]")
    for cat, count in sorted(v2_results["categories"].items()):
        print(f"  {cat}: {count}")

    # キャプション関連の検証
    print("\n[キャプション検出検証]")
    caption_v1 = {k: v for k, v in v1_results["categories"].items()
                  if "title" in k.lower() or "caption" in k.lower()}
    caption_v2 = {k: v for k, v in v2_results["categories"].items()
                  if "title" in k.lower() or "caption" in k.lower()}

    print("  V1 キャプション関連:")
    for cat, count in sorted(caption_v1.items()):
        print(f"    - {cat}: {count}")

    print("  V2 キャプション関連:")
    for cat, count in sorted(caption_v2.items()):
        print(f"    - {cat}: {count}")

    # 数式関連の検証
    print("\n[数式検出検証]")
    formula_v1 = {k: v for k, v in v1_results["categories"].items()
                  if "formula" in k.lower()}
    formula_v2 = {k: v for k, v in v2_results["categories"].items()
                  if "formula" in k.lower()}

    print("  V1 数式関連:")
    for cat, count in sorted(formula_v1.items()):
        print(f"    - {cat}: {count}")
    if not formula_v1:
        print("    (なし)")

    print("  V2 数式関連:")
    for cat, count in sorted(formula_v2.items()):
        print(f"    - {cat}: {count}")
    if not formula_v2:
        print("    (なし)")


def save_results_json(v1_results: dict, v2_results: dict, output_path: str) -> None:
    """結果を JSON ファイルに保存"""
    combined = {
        "v1": {
            "model": v1_results["model_name"],
            "total_detections": len(v1_results["detections"]),
            "categories": dict(v1_results["categories"]),
            "detections": v1_results["detections"],
        },
        "v2": {
            "model": v2_results["model_name"],
            "total_detections": len(v2_results["detections"]),
            "categories": dict(v2_results["categories"]),
            "detections": v2_results["detections"],
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_layout_models.py <pdf_path> [page_num]")
        print("Example: python compare_layout_models.py tests/fixtures/sample_llama.pdf 0")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print(f"PDF: {pdf_path}")
    print(f"ページ: {page_num}")

    # PDF を画像に変換
    print("\nPDF を画像にレンダリング中...")
    image = render_pdf_page_to_image(pdf_path, page_num, scale=2.0)
    print(f"画像サイズ: {image.size}")

    # V1 (PP-DocLayout-L) で解析
    v1_results = analyze_with_model("PP-DocLayout-L", image)

    # V2 (PP-DocLayoutV2) で解析
    v2_results = analyze_with_model("PP-DocLayoutV2", image)

    # 比較結果を表示
    print_comparison(v1_results, v2_results)

    # 結果を JSON に保存
    output_dir = Path("scripts/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"layout_comparison_{Path(pdf_path).stem}_p{page_num}.json"
    save_results_json(v1_results, v2_results, str(output_path))


if __name__ == "__main__":
    main()
