# PP-DocLayout V1 vs V2 比較調査

## 概要

PP-DocLayout-L (V1) と PP-DocLayoutV2 (V2) のカテゴリ体系と検出精度を比較した調査結果。

**調査日**: 2024-12-17
**関連 Issue**: #2 (PP-DocLayout によるドキュメントレイアウト解析の統合)

## テスト環境

- PaddlePaddle 3.2.0 (CPU)
- PaddleOCR 3.3.2
- テスト PDF: `tests/fixtures/sample_autogen_paper.pdf` (43ページ)

## 比較結果サマリー

### 検出数

| モデル | 総検出数 |
|--------|----------|
| V1 (PP-DocLayout-L) | 401 |
| V2 (PP-DocLayoutV2) | 448 |

### カテゴリ比較

| カテゴリ | V1 | V2 | 備考 |
|---------|---:|---:|------|
| text | 165 | 142 | |
| formula | 55 | 0 | V2 で削除 |
| inline_formula | 0 | 80 | V2 で新規 |
| display_formula | 0 | 1 | V2 で新規 |
| table_title | 21 | 0 | V2 で削除 (figure_title に統合) |
| figure_title | 20 | 47 | V2 で統合 |
| chart_title | 6 | 0 | V2 で削除 (figure_title に統合) |
| reference_content | 0 | 47 | V2 で新規 |
| paragraph_title | 32 | 28 | |
| table | 20 | 20 | |
| image | 16 | 19 | |
| number | 40 | 40 | |
| footnote | 8 | 8 | |
| chart | 6 | 5 | |
| header | 5 | 4 | |
| reference | 3 | 3 | |

### キャプション統合の検証

```
V1: figure_title(20) + table_title(21) + chart_title(6) = 47
V2: figure_title(47)
```

V2 では `table_title`, `chart_title` が `figure_title` に統合されている。

### V1 のみのカテゴリ

- `formula` - 数式（inline/display 区別なし）
- `table_title` - 表タイトル
- `chart_title` - チャートタイトル

### V2 のみのカテゴリ

- `inline_formula` - インライン数式
- `display_formula` - ディスプレイ数式
- `reference_content` - 参照コンテンツ

## 結論

### 翻訳用途には V1 を推奨

| 要件 | V1 | V2 |
|------|:--:|:--:|
| キャプション種別の区別 (table/figure/chart) | ✅ | ❌ |
| inline/display 数式の区別 | ❌ | ✅ |
| 読み順予測 | ❌ | ✅ |

「キャプションだけ翻訳したい」要件に対応するため、V1 (PP-DocLayout-L) を採用する。

## ファイル構成

```
pp-doclayout-v1-v2-comparison/
├── README.md                    # 本ドキュメント
├── layout_v1_results.json       # V1 全ページ検出結果
├── layout_v2_results.json       # V2 全ページ検出結果
├── comparison_summary.json      # 比較サマリー (JSON)
└── v1_v2_comparison.pdf         # 見開き比較 PDF (gitignore)
```

## PDF 再生成

比較 PDF は以下のスクリプトで再生成可能:

```bash
.venv-layout-test/bin/python scripts/compare_layout_models.py
```

## 参照

- [PP-DocLayout-L (Hugging Face)](https://huggingface.co/PaddlePaddle/PP-DocLayout-L)
- [PP-DocLayoutV2 (Hugging Face)](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
