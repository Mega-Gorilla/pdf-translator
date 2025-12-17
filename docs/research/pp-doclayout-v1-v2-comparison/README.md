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
| formula_number | 1 | 1 | |
| abstract | 1 | 1 | |
| aside_text | 1 | 1 | |
| doc_title | 1 | 1 | |

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

## 読み順予測機能の調査

### 調査結果

**❌ reading_order は PaddleOCR API から取得できない**

`LayoutDetection.predict()` の返り値を調査した結果：

```python
# 返り値のキー
['input_path', 'page_index', 'input_img', 'boxes']

# boxes の各要素のキー
['cls_id', 'label', 'score', 'coordinate']
```

`reading_order` フィールドは存在しない。

### 結論

V2 の「読み順予測」機能はモデル内部で使用されている可能性があるが、
現在の PaddleOCR API (`LayoutDetection.predict()`) からは直接取得できない。

**→ 初期実装では「分類のみ」にスコープを限定すべき**

## 結論

### 翻訳用途には V2 を推奨

| 要件 | V1 | V2 |
|------|:--:|:--:|
| テキスト vs 数式の誤検知 | 多い | **少ない** |
| inline/display 数式の区別 | ❌ | ✅ |
| キャプション種別の区別 (table/figure/chart) | ✅ | ❌ |
| 読み順予測 (API取得) | ❌ | ❌ |

翻訳用途では：
- テキスト vs 数式の誤検知減少が品質向上に直結
- キャプション種別の区別は翻訳処理で不要
- 読み順予測は API から取得不可のため初期スコープ外

**V2 (PP-DocLayoutV2) を採用する。**

## ファイル構成

```
pp-doclayout-v1-v2-comparison/
├── README.md                    # 本ドキュメント
├── layout_v1_results.json       # V1 全43ページ検出結果
├── layout_v2_results.json       # V2 全43ページ検出結果
├── comparison_summary.json      # 比較サマリー (JSON)
└── v1_v2_comparison.pdf         # 見開き比較 PDF (gitignore)
```

## 再生成手順

### JSON 結果の再生成

```bash
# 43ページ全件の比較結果を生成
.venv-layout-test/bin/python scripts/generate_full_comparison.py
```

### 読み順予測の確認

```bash
# V2 の reading_order 取得可否を確認
.venv-layout-test/bin/python scripts/check_v2_reading_order.py
```

### 比較 PDF の再生成

```bash
# 見開き比較 PDF を生成（オプション）
.venv-layout-test/bin/python scripts/compare_layout_models.py tests/fixtures/sample_autogen_paper.pdf
```

## 参照

- [PP-DocLayout-L (Hugging Face)](https://huggingface.co/PaddlePaddle/PP-DocLayout-L)
- [PP-DocLayoutV2 (Hugging Face)](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
