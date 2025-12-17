# PP-DocLayout レイアウト解析モジュール設計書

## 1. 概要

本ドキュメントは Issue #2「PP-DocLayout によるドキュメントレイアウト解析の統合」の実装計画を定義する。

### 1.1 目的

PP-DocLayoutV2 を使用した ML ベースのドキュメントレイアウト解析を実装し、PDF 翻訳パイプラインにおいて翻訳対象（本文・キャプション）と除外対象（数式・図・ヘッダー等）を高精度に分類する。

### 1.2 スコープ

- PP-DocLayoutV2 モデルのラッパー実装
- PDF → 画像変換（pypdfium2 使用）
- レイアウト検出結果のデータモデル
- 画像座標 → PDF 座標の変換
- 翻訳対象フィルタリングロジック
- ユニットテスト

### 1.3 スコープ外

- モデルの学習・ファインチューニング
- 他のレイアウト検出モデル（DocLayout-YOLO 等）のサポート
- OCR 機能

---

## 2. 背景

### 2.1 現在の課題

従来のヒストグラムベースのブロック分類には以下の課題がある：

| 課題 | 説明 |
|------|------|
| 見出し検出不可 | トークン数・フォントサイズだけでは見出しと本文を区別困難 |
| 位置情報未使用 | ヘッダー/フッターの位置パターンを活用していない |
| 数式の誤翻訳 | 数式テキストが本文として翻訳される |
| 複雑なレイアウト | マルチカラム・図表混在ページで精度低下 |

### 2.2 PP-DocLayoutV2 の選定理由

| 観点 | PP-DocLayoutV2 | DocLayout-YOLO |
|------|----------------|----------------|
| ライセンス | **Apache-2.0** | AGPL-3.0 |
| カテゴリ数 | **25** | 10 |
| 数式検出 | **inline/display 区別** | なし |
| 読み順予測 | **あり** | なし |
| 最新リリース | 2025年3月 | 2024年10月 |

#### V1 vs V2 比較調査の結果

実機テスト（AutoGen論文 43ページ）により、V2 の採用を決定した。

| 観点 | V2 採用理由 |
|------|------------|
| テキスト vs 数式の誤検知 | V2 は誤検知が少ない（例: P21 `(-5,-1,-5)` を V2 は正しく text と判定） |
| セマンティック分離 | V2 は "Abstract" を paragraph_title + abstract に正しく分離 |
| 数式検出精度 | V2 は P36 の数式をより正確に検出 |
| キャプション種別区別 | 統合されるが翻訳用途では区別不要のため問題なし |

詳細: `docs/research/pp-doclayout-v1-v2-comparison/`

#### 対応言語

**英語・中国語のみ** を公式サポート。日本語文書への適用は実機テストが必要。

### 2.3 参照資料

- `_archive/Index_PDF_Translation/docs/research/layout-analysis/document-layout-analysis-survey.md`
- `_archive/Index_PDF_Translation/docs/research/layout-analysis/evaluations/pp-doclayout-evaluation.md`
- `_archive/Index_PDF_Translation/tests/evaluation/evaluate_pp_doclayout.py`

---

## 3. アーキテクチャ

### 3.1 ファイル構成

```
src/pdf_translator/
├── core/
│   ├── pdf_processor.py   # 既存: pypdfium2 PDF処理
│   ├── models.py          # 既存: データモデル
│   └── layout_analyzer.py # 新規: PP-DocLayoutV2 ラッパー
└── ...

tests/
└── test_layout_analyzer.py  # 新規: テスト
```

### 3.2 処理フロー

```
[PDFファイル]
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ PDFProcessor.extract_text_objects()                     │
│  - pypdfium2 でテキスト + bbox 座標を抽出              │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ LayoutAnalyzer.analyze()                                │
│  1. pypdfium2 で PDF ページを画像にレンダリング         │
│  2. PP-DocLayoutV2 でレイアウト検出                     │
│  3. 画像座標 → PDF座標に変換                           │
│  4. 検出結果を LayoutBlock リストとして返却             │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ match_text_with_layout()                                │
│  - TextObject と LayoutBlock の bbox を IoU でマッチング │
│  - 各 TextObject にカテゴリを付与                       │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ filter_translatable()                                   │
│  - カテゴリに基づいて翻訳対象をフィルタリング           │
│  - 本文 (text)、キャプション等を抽出                    │
│  - 数式、図、ヘッダー等を除外                           │
└─────────────────────────────────────────────────────────┘
```

### 3.3 クラス図

```
┌──────────────────────────────────────────────────────────┐
│                    LayoutAnalyzer                        │
├──────────────────────────────────────────────────────────┤
│ - _model: LayoutDetection                                │
│ - _render_scale: float                                   │
├──────────────────────────────────────────────────────────┤
│ + __init__(model_name: str, render_scale: float)         │
│ + analyze(pdf_path: str, page_num: int) -> list[Layout]  │
│ + analyze_all(pdf_path: str) -> dict[int, list[Layout]]  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                     LayoutBlock                          │
├──────────────────────────────────────────────────────────┤
│ + bbox: BBox                                             │
│ + category: LayoutCategory                               │
│ + confidence: float                                      │
│ + page_num: int                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│               LayoutCategory (Enum)                      │
├──────────────────────────────────────────────────────────┤
│ TEXT, PARAGRAPH_TITLE, DOC_TITLE, ABSTRACT,              │
│ INLINE_FORMULA, DISPLAY_FORMULA, ALGORITHM,              │
│ TABLE, IMAGE, FIGURE_TITLE, CHART, ...                   │
└──────────────────────────────────────────────────────────┘
```

---

## 4. データモデル

### 4.1 LayoutCategory Enum

PP-DocLayoutV2 の 25 カテゴリを Enum として定義：

```python
class LayoutCategory(str, Enum):
    """PP-DocLayoutV2 の検出カテゴリ"""

    # テキスト系
    TEXT = "text"
    PARAGRAPH_TITLE = "paragraph_title"
    DOC_TITLE = "doc_title"
    ABSTRACT = "abstract"
    ASIDE_TEXT = "aside_text"

    # 数式系 (V2: inline/display 区別)
    INLINE_FORMULA = "inline_formula"
    DISPLAY_FORMULA = "display_formula"
    ALGORITHM = "algorithm"

    # 図表系
    TABLE = "table"
    IMAGE = "image"
    FIGURE_TITLE = "figure_title"  # table/chart キャプション含む
    CHART = "chart"

    # コード
    CODE_BLOCK = "code_block"

    # ナビゲーション系
    HEADER = "header"
    FOOTER = "footer"
    NUMBER = "number"  # ページ番号等

    # 参照系
    REFERENCE = "reference"
    REFERENCE_CONTENT = "reference_content"
    FOOTNOTE = "footnote"

    # その他
    SEAL = "seal"
    CONTENT = "content"
    TABLE_OF_CONTENTS = "table_of_contents"

    # 未知のカテゴリ
    UNKNOWN = "unknown"
```

### 4.2 LayoutBlock データクラス

```python
@dataclass
class LayoutBlock:
    """レイアウト検出結果"""

    bbox: BBox
    category: LayoutCategory
    confidence: float
    page_num: int
```

### 4.3 翻訳対象分類

| 分類 | カテゴリ | 翻訳対象 | 備考 |
|------|---------|:--------:|------|
| 本文 | `TEXT`, `ABSTRACT` | ✅ ON | メインコンテンツ |
| 見出し | `PARAGRAPH_TITLE`, `DOC_TITLE` | ✅ ON | タイトル・セクション |
| キャプション | `FIGURE_TITLE` | ✅ ON | 図表・チャートキャプション統合 |
| 脚注 | `FOOTNOTE` | ✅ ON | 注釈 |
| 参照見出し | `REFERENCE` | ⚠️ 要検討 | "References" セクション見出し |
| 補足 | `ASIDE_TEXT` | ⚠️ 要検討 | サイドテキスト |
| 参照内容 | `REFERENCE_CONTENT` | ❌ OFF | 著者名・論文タイトル等 |
| 数式 | `INLINE_FORMULA`, `DISPLAY_FORMULA` | ❌ OFF | 数式は翻訳しない |
| アルゴリズム | `ALGORITHM` | ❌ OFF | 疑似コード等 |
| コード | `CODE_BLOCK` | ❌ OFF | ソースコード |
| 図表 | `TABLE`, `IMAGE`, `CHART` | ❌ OFF | 視覚要素 |
| ナビゲーション | `HEADER`, `FOOTER`, `NUMBER` | ❌ OFF | ヘッダー・フッター・ページ番号 |
| その他 | `SEAL`, `TABLE_OF_CONTENTS` | ❌ OFF | 印鑑・目次 |

---

## 5. 実装詳細

### 5.1 依存関係

```toml
# pyproject.toml
[project.optional-dependencies]
layout = [
    "paddlepaddle>=3.2.0",  # V2 モデルには 3.2.0 以上が必要
    "paddleocr>=3.3.0",
]
```

**注意**: PaddlePaddle は GPU 版と CPU 版でインストールコマンドが異なる：

```bash
# GPU (CUDA 12.6)
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# CPU
pip install paddlepaddle==3.2.0
```

**重要**: PP-DocLayoutV2 は PaddlePaddle 3.2.0 以上が必須。3.0.0 では動作しない。

### 5.2 LayoutAnalyzer クラス

```python
class LayoutAnalyzer:
    """PP-DocLayoutV2 を使用したレイアウト解析"""

    DEFAULT_MODEL = "PP-DocLayoutV2"
    DEFAULT_RENDER_SCALE = 2.0  # 2x zoom for better detection

    def __init__(
        self,
        model_name: str | None = None,
        render_scale: float | None = None,
    ) -> None:
        """
        Args:
            model_name: PP-DocLayout モデル名 (PP-DocLayoutV2 等)
            render_scale: PDF レンダリングスケール (default: 2.0)
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._render_scale = render_scale or self.DEFAULT_RENDER_SCALE
        self._model: LayoutDetection | None = None

    def _ensure_model(self) -> LayoutDetection:
        """モデルの遅延初期化"""
        if self._model is None:
            from paddleocr import LayoutDetection
            self._model = LayoutDetection(model_name=self._model_name)
        return self._model

    def analyze(
        self,
        pdf_path: str | Path,
        page_num: int,
    ) -> list[LayoutBlock]:
        """
        単一ページのレイアウト解析

        Args:
            pdf_path: PDF ファイルパス
            page_num: ページ番号 (0-indexed)

        Returns:
            検出された LayoutBlock のリスト
        """
        ...

    def analyze_all(
        self,
        pdf_path: str | Path,
    ) -> dict[int, list[LayoutBlock]]:
        """
        全ページのレイアウト解析

        Args:
            pdf_path: PDF ファイルパス

        Returns:
            ページ番号 → LayoutBlock リストの辞書
        """
        ...
```

### 5.3 座標変換

PP-DocLayoutV2 は画像座標（ピクセル）を返すため、PDF 座標に変換が必要：

```python
def _convert_image_to_pdf_coords(
    image_bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> BBox:
    """
    画像座標を PDF 座標に変換

    Args:
        image_bbox: 画像座標 (x0, y0, x1, y1) in pixels
        page_width: PDF ページ幅 (points)
        page_height: PDF ページ高さ (points)
        image_width: レンダリング画像幅 (pixels)
        image_height: レンダリング画像高さ (pixels)

    Returns:
        PDF 座標の BBox
    """
    scale_x = page_width / image_width
    scale_y = page_height / image_height

    return BBox(
        x0=image_bbox[0] * scale_x,
        y0=image_bbox[1] * scale_y,
        x1=image_bbox[2] * scale_x,
        y1=image_bbox[3] * scale_y,
    )
```

### 5.4 マッチング関数

TextObject と LayoutBlock を IoU (Intersection over Union) でマッチング：

```python
def match_text_with_layout(
    text_objects: list[TextObject],
    layout_blocks: list[LayoutBlock],
    iou_threshold: float = 0.3,
) -> dict[str, LayoutCategory]:
    """
    TextObject と LayoutBlock をマッチング

    Args:
        text_objects: PDFProcessor から抽出した TextObject リスト
        layout_blocks: LayoutAnalyzer から取得した LayoutBlock リスト
        iou_threshold: マッチング閾値 (default: 0.3)

    Returns:
        TextObject.id → LayoutCategory のマッピング
    """
    ...
```

### 5.5 フィルタリング関数

```python
# 翻訳対象カテゴリ (V2)
TRANSLATABLE_CATEGORIES = {
    LayoutCategory.TEXT,
    LayoutCategory.ABSTRACT,
    LayoutCategory.PARAGRAPH_TITLE,
    LayoutCategory.DOC_TITLE,
    LayoutCategory.FIGURE_TITLE,  # table/chart キャプション含む
    LayoutCategory.FOOTNOTE,
}

# オプション（設定で切り替え）
OPTIONAL_TRANSLATABLE = {
    LayoutCategory.REFERENCE,
    LayoutCategory.ASIDE_TEXT,
}

# 翻訳除外カテゴリ (V2)
NON_TRANSLATABLE_CATEGORIES = {
    LayoutCategory.INLINE_FORMULA,
    LayoutCategory.DISPLAY_FORMULA,
    LayoutCategory.ALGORITHM,
    LayoutCategory.CODE_BLOCK,
    LayoutCategory.TABLE,
    LayoutCategory.IMAGE,
    LayoutCategory.CHART,
    LayoutCategory.HEADER,
    LayoutCategory.FOOTER,
    LayoutCategory.NUMBER,
    LayoutCategory.REFERENCE_CONTENT,
    LayoutCategory.SEAL,
    LayoutCategory.TABLE_OF_CONTENTS,
}

def filter_translatable(
    text_objects: list[TextObject],
    categories: dict[str, LayoutCategory],
    include_optional: bool = False,
) -> list[TextObject]:
    """
    翻訳対象の TextObject をフィルタリング

    Args:
        text_objects: 全 TextObject リスト
        categories: TextObject.id → LayoutCategory のマッピング
        include_optional: REFERENCE, ASIDE_TEXT を含めるか

    Returns:
        翻訳対象の TextObject リスト
    """
    ...
```

---

## 6. テスト計画

### 6.1 ユニットテスト

| テストケース | 説明 |
|-------------|------|
| `test_layout_category_enum` | カテゴリ Enum の網羅性 |
| `test_layout_block_creation` | LayoutBlock データクラスの生成 |
| `test_coordinate_conversion` | 画像→PDF 座標変換の正確性 |
| `test_iou_calculation` | IoU 計算の正確性 |
| `test_filter_translatable` | フィルタリングロジック |

### 6.2 統合テスト（オプション）

```bash
# PP-DocLayoutV2 が利用可能な環境でのみ実行
RUN_LAYOUT_TESTS=1 pytest tests/test_layout_analyzer.py
```

| テストケース | 説明 |
|-------------|------|
| `test_analyze_single_page` | 単一ページの解析 |
| `test_analyze_all_pages` | 全ページの解析 |
| `test_match_with_real_pdf` | 実際の PDF でのマッチング |

### 6.3 評価用スクリプト

```
scripts/evaluation/
└── evaluate_layout_analyzer.py  # 精度評価用
```

---

## 7. 実装フェーズ

### Phase 1: データモデル定義

**出力**: `src/pdf_translator/core/models.py` (追加)

- [ ] `LayoutCategory` Enum の追加
- [ ] `LayoutBlock` データクラスの追加

### Phase 2: 座標変換・マッチング関数

**出力**: `src/pdf_translator/core/layout_utils.py`

- [ ] `calculate_iou()` 関数
- [ ] `_convert_image_to_pdf_coords()` 関数
- [ ] `match_text_with_layout()` 関数
- [ ] `filter_translatable()` 関数

### Phase 3: LayoutAnalyzer 実装

**出力**: `src/pdf_translator/core/layout_analyzer.py`

- [ ] `LayoutAnalyzer.__init__()` - モデル初期化
- [ ] `LayoutAnalyzer._render_page_to_image()` - pypdfium2 でレンダリング
- [ ] `LayoutAnalyzer.analyze()` - 単一ページ解析
- [ ] `LayoutAnalyzer.analyze_all()` - 全ページ解析

### Phase 4: テスト

**出力**: `tests/test_layout_analyzer.py`

- [ ] ユニットテスト
- [ ] 統合テスト（条件付き）

### Phase 5: pyproject.toml 更新

- [ ] `[layout]` オプション依存の追加
- [ ] インストール手順の文書化

---

## 8. 依存関係と注意事項

### 8.1 ライセンス

| コンポーネント | ライセンス | 互換性 |
|---------------|-----------|--------|
| PP-DocLayoutV2 モデル | Apache-2.0 | ✅ |
| PaddlePaddle | Apache-2.0 | ✅ |
| PaddleOCR | Apache-2.0 | ✅ |

### 8.2 GPU/CPU サポート

| 環境 | モデル | 推奨設定 |
|------|--------|---------|
| GPU (CUDA) | PP-DocLayoutV2 | `render_scale=2.0` |
| CPU | PP-DocLayoutV2 | `render_scale=1.5` |

**注意**: PP-DocLayoutV2 は単一モデルのみ提供。

### 8.3 _archive からのコード参照

`_archive/` 内のコードは AGPL-3.0 でライセンスされているため、コードのコピーは行わず、
「挙動確認」や「設計の参考」として参照するに留めること。

参照可能なファイル：
- `_archive/Index_PDF_Translation/tests/evaluation/evaluate_pp_doclayout.py` - 評価スクリプトの挙動確認
- `_archive/Index_PDF_Translation/docs/research/layout-analysis/` - 調査資料

---

## 9. 成功基準

| 項目 | 検証方法 | 許容基準 |
|------|----------|----------|
| 座標変換精度 | bbox 比較 | ±2.0pt |
| マッチング精度 | IoU 計算 | > 0.3 で正しくマッチ |
| カテゴリ分類 | 手動確認 | 95%+ 正確 |
| 処理速度 | 計測 | > 3 ページ/秒 (GPU) |

---

## 10. 参照

### 10.1 外部ドキュメント

- [PP-DocLayoutV2 (Hugging Face)](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-DocLayout Paper (arXiv)](https://arxiv.org/abs/2503.17213)

### 10.2 V1/V2 比較調査

- `docs/research/pp-doclayout-v1-v2-comparison/` - 比較調査結果

### 10.3 プロジェクト内ドキュメント

- `docs/archive/design/pdf-processor-design.md` - PDF 処理モジュール設計
- `docs/archive/design/translators-design.md` - 翻訳バックエンド設計
