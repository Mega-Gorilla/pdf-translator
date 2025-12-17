#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PP-DocLayoutV2 検出結果を可視化するスクリプト

検出されたレイアウトブロックをカテゴリ別に色分けして
枠を描画した PDF を生成します。

Usage:
    uv run python scripts/visualize_layout.py <pdf_path> [output_path]

Examples:
    uv run python scripts/visualize_layout.py tests/fixtures/sample_autogen_paper.pdf
    uv run python scripts/visualize_layout.py paper.pdf output_visualized.pdf
    uv run python scripts/visualize_layout.py paper.pdf --pages 0,1,2

Requirements:
    - paddlepaddle >= 3.2.0
    - paddleocr >= 3.3.0
    - reportlab (for PDF generation)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# カテゴリ別の色定義 (RGB)
# PP-DocLayoutV2 公式 label_list に対応
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    # 翻訳対象 (緑系)
    "text": (0, 200, 0),
    "vertical_text": (0, 180, 0),
    "paragraph_title": (0, 150, 0),
    "doc_title": (0, 100, 0),
    "abstract": (50, 200, 50),
    "aside_text": (100, 200, 100),
    "figure_title": (0, 200, 100),
    # 数式 (赤系) - 翻訳除外
    "inline_formula": (255, 0, 0),
    "display_formula": (200, 0, 0),
    "formula_number": (150, 0, 0),
    # アルゴリズム (紫系) - 翻訳除外
    "algorithm": (150, 0, 150),
    # 表 (青系) - 翻訳除外
    "table": (0, 0, 255),
    # 画像/チャート (オレンジ系) - 翻訳除外
    "image": (255, 165, 0),
    "chart": (255, 140, 0),
    # ナビゲーション (グレー系) - 翻訳除外
    "header": (128, 128, 128),
    "header_image": (110, 110, 110),
    "footer": (100, 100, 100),
    "footer_image": (90, 90, 90),
    "number": (150, 150, 150),
    # 参照 (茶系) - 翻訳除外
    "reference": (139, 69, 19),
    "reference_content": (160, 82, 45),
    "footnote": (205, 133, 63),
    "vision_footnote": (180, 110, 50),
    # その他 (シアン系)
    "seal": (0, 200, 200),
    "content": (0, 150, 150),
    "unknown": (128, 128, 128),
}

# ProjectCategory 別の枠スタイル
PROJECT_CATEGORY_STYLE: dict[str, dict] = {
    "text": {"width": 2, "dash": None, "translatable": True},
    "title": {"width": 3, "dash": None, "translatable": True},
    "caption": {"width": 2, "dash": None, "translatable": True},
    "footnote": {"width": 2, "dash": (5, 5), "translatable": False},
    "formula": {"width": 3, "dash": None, "translatable": False},
    "code": {"width": 2, "dash": None, "translatable": False},
    "table": {"width": 2, "dash": None, "translatable": False},
    "image": {"width": 2, "dash": (10, 5), "translatable": False},
    "chart": {"width": 2, "dash": (10, 5), "translatable": False},
    "header": {"width": 1, "dash": (3, 3), "translatable": False},
    "reference": {"width": 2, "dash": (5, 5), "translatable": False},
    "other": {"width": 1, "dash": (2, 2), "translatable": False},
}


def get_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """フォントを取得（日本語対応）"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in font_paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_page_to_image(
    pdf_path: str | Path,
    page_num: int,
    scale: float = 2.0,
) -> tuple[Image.Image, float, float]:
    """PDF ページを画像にレンダリング"""
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_num]

    page_width = page.get_width()
    page_height = page.get_height()

    bitmap = page.render(scale=scale)
    image = bitmap.to_pil().convert("RGB")

    return image, page_width, page_height


def analyze_page(
    image: Image.Image,
    model_name: str = "PP-DocLayoutV2",
) -> list[dict]:
    """ページのレイアウト解析を実行"""
    from paddleocr import LayoutDetection

    model = LayoutDetection(model_name=model_name)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        image.save(temp_path)

    try:
        output = model.predict(temp_path, batch_size=1, layout_nms=True)

        detections = []
        for res in output:
            boxes = res.get("boxes", [])
            for det in boxes:
                detections.append({
                    "label": det.get("label", "unknown"),
                    "score": float(det.get("score", 0)),
                    "coordinate": [float(c) for c in det.get("coordinate", [])],
                })

        return detections
    finally:
        Path(temp_path).unlink(missing_ok=True)


def get_project_category(raw_label: str) -> str:
    """RawLayoutCategory → ProjectCategory マッピング"""
    mapping = {
        "text": "text",
        "vertical_text": "text",
        "abstract": "text",
        "aside_text": "text",
        "paragraph_title": "title",
        "doc_title": "title",
        "figure_title": "caption",
        "footnote": "footnote",
        "vision_footnote": "footnote",
        "inline_formula": "formula",
        "display_formula": "formula",
        "formula_number": "formula",
        "algorithm": "code",
        "table": "table",
        "image": "image",
        "chart": "chart",
        "header": "header",
        "header_image": "header",
        "footer": "header",
        "footer_image": "header",
        "number": "header",
        "reference": "reference",
        "reference_content": "reference",
        "seal": "other",
        "content": "other",
    }
    return mapping.get(raw_label, "other")


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    show_confidence: bool = True,
    show_labels: bool = True,
) -> Image.Image:
    """検出結果を画像に描画"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = get_font(14)
    small_font = get_font(10)

    for det in detections:
        label = det["label"]
        score = det["score"]
        coord = det["coordinate"]

        if len(coord) != 4:
            continue

        x0, y0, x1, y1 = coord
        color = CATEGORY_COLORS.get(label, (128, 128, 128))
        project_cat = get_project_category(label)
        style = PROJECT_CATEGORY_STYLE.get(project_cat, PROJECT_CATEGORY_STYLE["other"])

        # 枠を描画
        width = style["width"]
        for i in range(width):
            draw.rectangle(
                [x0 - i, y0 - i, x1 + i, y1 + i],
                outline=color,
            )

        # ラベル背景
        if show_labels:
            label_text = label
            if show_confidence:
                label_text = f"{label} ({score:.2f})"

            # テキストサイズを取得
            bbox = draw.textbbox((0, 0), label_text, font=small_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # ラベル背景を描画
            label_y = max(0, y0 - text_height - 4)
            draw.rectangle(
                [x0, label_y, x0 + text_width + 4, label_y + text_height + 4],
                fill=color,
            )

            # テキストを描画
            draw.text(
                (x0 + 2, label_y + 2),
                label_text,
                fill=(255, 255, 255),
                font=small_font,
            )

        # 翻訳対象/除外のマーク
        if style["translatable"]:
            # 緑のチェックマーク
            mark_x = x1 - 15
            mark_y = y0 + 5
            draw.text((mark_x, mark_y), "✓", fill=(0, 255, 0), font=font)
        else:
            # 赤のXマーク
            mark_x = x1 - 15
            mark_y = y0 + 5
            draw.text((mark_x, mark_y), "✗", fill=(255, 0, 0), font=font)

    return img


def create_legend_image(
    width: int = 300,
    detections: list[dict] | None = None,
) -> Image.Image:
    """凡例画像を生成"""
    font = get_font(14)
    small_font = get_font(11)

    # 使用されているカテゴリのみ表示
    if detections:
        used_labels = set(det["label"] for det in detections)
    else:
        used_labels = set(CATEGORY_COLORS.keys())

    # カテゴリをグループ化 (PP-DocLayoutV2 公式 label_list 対応)
    groups = {
        "翻訳対象 (Translatable)": ["text", "vertical_text", "paragraph_title",
                                     "doc_title", "abstract", "aside_text",
                                     "figure_title"],
        "数式 (Formula)": ["inline_formula", "display_formula", "formula_number"],
        "コード (Code)": ["algorithm"],
        "表/図 (Table/Figure)": ["table", "image", "chart"],
        "その他 (Other)": ["header", "header_image", "footer", "footer_image",
                          "number", "reference", "reference_content",
                          "footnote", "vision_footnote", "seal", "content",
                          "unknown"],
    }

    # 高さを計算
    line_height = 22
    group_spacing = 30
    total_height = 50  # タイトル

    for group_name, labels in groups.items():
        visible_labels = [lbl for lbl in labels if lbl in used_labels]
        if visible_labels:
            total_height += group_spacing + len(visible_labels) * line_height

    total_height += 50  # マージン

    # 画像作成
    img = Image.new("RGB", (width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = 15
    draw.text((10, y), "Legend / 凡例", fill=(0, 0, 0), font=font)
    y += 35

    for group_name, labels in groups.items():
        visible_labels = [lbl for lbl in labels if lbl in used_labels]
        if not visible_labels:
            continue

        # グループヘッダー
        draw.text((10, y), group_name, fill=(50, 50, 50), font=small_font)
        y += 20

        for label in visible_labels:
            color = CATEGORY_COLORS.get(label, (128, 128, 128))

            # 色のサンプル
            draw.rectangle([15, y + 2, 35, y + 18], fill=color, outline=(0, 0, 0))

            # ラベル名
            project_cat = get_project_category(label)
            style = PROJECT_CATEGORY_STYLE.get(project_cat, {})
            trans_mark = "✓" if style.get("translatable", False) else "✗"

            draw.text((45, y), f"{label} {trans_mark}", fill=(0, 0, 0), font=small_font)
            y += line_height

        y += 10

    return img


def combine_images_to_pdf(
    images: list[Image.Image],
    output_path: str | Path,
) -> None:
    """複数の画像を1つのPDFに結合"""
    if not images:
        return

    c = canvas.Canvas(str(output_path))

    for img in images:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            img.save(temp_path, "PNG")

        try:
            # ページサイズを画像に合わせる
            c.setPageSize((img.width, img.height))

            # 画像を描画
            c.drawImage(
                ImageReader(temp_path),
                0, 0,
                width=img.width,
                height=img.height,
            )
            c.showPage()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    c.save()


def process_pdf(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    pages: list[int] | None = None,
    scale: float = 2.0,
    show_confidence: bool = True,
    show_labels: bool = True,
) -> Path:
    """PDFを処理して可視化結果を生成"""
    pdf_path = Path(pdf_path)

    if output_path is None:
        output_path = pdf_path.parent / f"{pdf_path.stem}_layout_visualized.pdf"
    else:
        output_path = Path(output_path)

    # ページ数を取得
    pdf = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf)

    if pages is None:
        pages = list(range(total_pages))
    else:
        pages = [p for p in pages if 0 <= p < total_pages]

    print(f"PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Processing pages: {pages}")
    print()

    all_images: list[Image.Image] = []
    all_detections: list[dict] = []
    category_counts: Counter = Counter()

    for page_num in pages:
        print(f"Processing page {page_num + 1}/{total_pages}...")

        # レンダリング
        image, page_width, page_height = render_page_to_image(pdf_path, page_num, scale)
        print(f"  Image size: {image.size}")

        # レイアウト解析
        print("  Running layout detection...")
        detections = analyze_page(image)
        print(f"  Detected {len(detections)} blocks")

        # カウント
        for det in detections:
            category_counts[det["label"]] += 1
        all_detections.extend(detections)

        # 描画
        annotated = draw_detections(image, detections, show_confidence, show_labels)

        # 凡例を追加（最初のページのみ、または各ページ）
        legend = create_legend_image(300, detections)

        # 凡例を右側に配置
        combined_width = annotated.width + legend.width + 20
        combined_height = max(annotated.height, legend.height)
        combined = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
        combined.paste(annotated, (0, 0))
        combined.paste(legend, (annotated.width + 10, 10))

        all_images.append(combined)

    # PDF 出力
    print()
    print("Generating output PDF...")
    combine_images_to_pdf(all_images, output_path)
    print(f"Output: {output_path}")

    # サマリー
    print()
    print("=" * 60)
    print("Detection Summary")
    print("=" * 60)
    print(f"{'Category':<25} {'Count':>8} {'Translatable':>12}")
    print("-" * 50)
    for label, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        project_cat = get_project_category(label)
        style = PROJECT_CATEGORY_STYLE.get(project_cat, {})
        trans = "Yes" if style.get("translatable", False) else "No"
        print(f"{label:<25} {count:>8} {trans:>12}")
    print("-" * 50)
    print(f"{'Total':<25} {sum(category_counts.values()):>8}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="PP-DocLayoutV2 検出結果を可視化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.pdf
  %(prog)s paper.pdf output.pdf
  %(prog)s paper.pdf --pages 0,1,2
  %(prog)s paper.pdf --no-confidence
        """,
    )
    parser.add_argument("pdf_path", help="Input PDF file")
    parser.add_argument("output_path", nargs="?", help="Output PDF file (optional)")
    parser.add_argument(
        "--pages",
        type=str,
        help="Comma-separated page numbers (0-indexed), e.g., '0,1,2'",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Render scale (default: 2.0)",
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Hide confidence scores",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide category labels",
    )

    args = parser.parse_args()

    if not Path(args.pdf_path).exists():
        print(f"Error: File not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    pages = None
    if args.pages:
        try:
            pages = [int(p.strip()) for p in args.pages.split(",")]
        except ValueError:
            print(f"Error: Invalid page numbers: {args.pages}", file=sys.stderr)
            sys.exit(1)

    process_pdf(
        args.pdf_path,
        args.output_path,
        pages=pages,
        scale=args.scale,
        show_confidence=not args.no_confidence,
        show_labels=not args.no_labels,
    )


if __name__ == "__main__":
    main()
