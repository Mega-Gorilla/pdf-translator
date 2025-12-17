#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PP-DocLayoutV2 の読み順予測機能を確認するスクリプト

V2の predict() 返り値に reading_order 情報が含まれるかを検証します。

Usage:
    .venv-layout-test/bin/python scripts/check_v2_reading_order.py
"""

import json
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def render_pdf_page_to_image(pdf_path: str, page_num: int = 0, scale: float = 2.0) -> Image.Image:
    """PDF ページを画像にレンダリング"""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_num]
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


def check_reading_order():
    """V2 の読み順予測機能を確認"""
    from paddleocr import LayoutDetection

    pdf_path = "tests/fixtures/sample_autogen_paper.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print("="*60)
    print("PP-DocLayoutV2 読み順予測機能の確認")
    print("="*60)

    # 画像をレンダリング
    print(f"\nPDF: {pdf_path}")
    image = render_pdf_page_to_image(pdf_path, page_num=0, scale=2.0)
    temp_path = "/tmp/reading_order_test.png"
    image.save(temp_path)

    # V2 モデルで解析
    print("\nPP-DocLayoutV2 で解析中...")
    model = LayoutDetection(model_name="PP-DocLayoutV2")
    output = model.predict(temp_path, batch_size=1, layout_nms=True)

    print("\n" + "="*60)
    print("predict() 返り値の構造")
    print("="*60)

    for i, res in enumerate(output):
        print(f"\n--- Result {i} ---")
        print(f"Type: {type(res)}")

        if isinstance(res, dict):
            print(f"Keys: {list(res.keys())}")

            for key, value in res.items():
                if key == "boxes":
                    print(f"\n[boxes] ({len(value)} 件)")
                    if value:
                        first_keys = (
                            list(value[0].keys()) if isinstance(value[0], dict)
                            else "N/A"
                        )
                        print(f"  First box keys: {first_keys}")
                        sample = json.dumps(value[0], indent=4, default=str)[:500]
                        print(f"  First box sample: {sample}")
                elif key == "reading_order":
                    print("\n[reading_order] ✅ 存在!")
                    print(f"  Type: {type(value)}")
                    print(f"  Value: {value}")
                else:
                    val_str = str(value)[:200]
                    print(f"\n[{key}]")
                    print(f"  Type: {type(value)}")
                    print(f"  Value: {val_str}")
        else:
            print(f"Value: {str(res)[:500]}")

    # reading_order の有無を確認
    print("\n" + "="*60)
    print("結論")
    print("="*60)

    has_reading_order = False
    for res in output:
        if isinstance(res, dict) and "reading_order" in res:
            has_reading_order = True
            break

    if has_reading_order:
        print("✅ reading_order フィールドが存在します")
        print("   → V2 の読み順予測機能は API から取得可能")
    else:
        print("❌ reading_order フィールドは存在しません")
        print("   → 初期実装では「分類のみ」にスコープを限定すべき")

    # boxes の詳細構造も確認
    print("\n" + "="*60)
    print("boxes の詳細構造")
    print("="*60)

    for res in output:
        if isinstance(res, dict) and "boxes" in res:
            boxes = res["boxes"]
            if boxes:
                print(f"\n検出数: {len(boxes)}")
                box_keys = (
                    list(boxes[0].keys()) if isinstance(boxes[0], dict) else "N/A"
                )
                print(f"各 box のキー: {box_keys}")

                # 最初の3件を詳細表示
                print("\n最初の3件:")
                for j, box in enumerate(boxes[:3]):
                    print(f"\n  [{j}] {json.dumps(box, indent=6, default=str)}")


if __name__ == "__main__":
    check_reading_order()
