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

#### 読み順予測機能について

V2 の「読み順予測」機能は PaddleOCR API からは**取得できない**ことが判明。

```python
# LayoutDetection.predict() の返り値
['input_path', 'page_index', 'input_img', 'boxes']

# boxes の各要素
['cls_id', 'label', 'score', 'coordinate']

# reading_order フィールドは存在しない
```

**結論**: 初期実装では「分類のみ」にスコープを限定。読み順は将来対応とする。

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
│  - TextObject と LayoutBlock を包含率+優先度でマッチング │
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
│ + raw_category: RawLayoutCategory                        │
│ + project_category: ProjectCategory                      │
│ + confidence: float                                      │
│ + page_num: int                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│            RawLayoutCategory (Enum)                      │
├──────────────────────────────────────────────────────────┤
│ TEXT, PARAGRAPH_TITLE, DOC_TITLE, ABSTRACT,              │
│ INLINE_FORMULA, DISPLAY_FORMULA, ALGORITHM, ...          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│            ProjectCategory (Enum)                        │
├──────────────────────────────────────────────────────────┤
│ TEXT, TITLE, CAPTION, FORMULA, TABLE, IMAGE, ...         │
└──────────────────────────────────────────────────────────┘
```

---

## 4. データモデル

### 4.1 2層カテゴリ構造

モデル出力と翻訳ロジックを分離し、将来のモデル変更に強い設計とする。

```
┌─────────────────────────────────────────────────────────┐
│                    Raw Layer                            │
│            (PP-DocLayoutV2 モデル出力)                   │
├─────────────────────────────────────────────────────────┤
│ text, paragraph_title, doc_title, abstract,             │
│ inline_formula, display_formula, figure_title, ...      │
└───────────────────────┬─────────────────────────────────┘
                        │ マッピング
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   Project Layer                         │
│              (プロジェクト内部カテゴリ)                   │
├─────────────────────────────────────────────────────────┤
│ TEXT, TITLE, CAPTION, FORMULA, TABLE, IMAGE, HEADER, ...│
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
              翻訳ロジック（安定 API）
```

**利点:**
- モデルバージョン変更時も翻訳ロジックは不変
- 将来的に他モデル（DocLayout-YOLO 等）への切り替えも容易
- テストが安定

### 4.2 RawLayoutCategory Enum (Raw Layer)

PP-DocLayoutV2 の生のラベルを Enum として定義：

```python
class RawLayoutCategory(str, Enum):
    """PP-DocLayoutV2 の生カテゴリ（モデル出力そのまま）"""

    # テキスト系
    TEXT = "text"
    PARAGRAPH_TITLE = "paragraph_title"
    DOC_TITLE = "doc_title"
    ABSTRACT = "abstract"
    ASIDE_TEXT = "aside_text"

    # 数式系
    INLINE_FORMULA = "inline_formula"
    DISPLAY_FORMULA = "display_formula"
    FORMULA_NUMBER = "formula_number"
    ALGORITHM = "algorithm"

    # 図表系
    TABLE = "table"
    IMAGE = "image"
    FIGURE_TITLE = "figure_title"
    CHART = "chart"

    # コード
    CODE_BLOCK = "code_block"

    # ナビゲーション系
    HEADER = "header"
    FOOTER = "footer"
    NUMBER = "number"

    # 参照系
    REFERENCE = "reference"
    REFERENCE_CONTENT = "reference_content"
    FOOTNOTE = "footnote"

    # その他
    SEAL = "seal"
    CONTENT = "content"
    TABLE_OF_CONTENTS = "table_of_contents"

    # 未知
    UNKNOWN = "unknown"
```

### 4.3 ProjectCategory Enum (Project Layer)

翻訳ロジックが依存する安定カテゴリ：

```python
class ProjectCategory(str, Enum):
    """プロジェクト内部の安定カテゴリ"""

    # 翻訳対象
    TEXT = "text"           # 本文テキスト
    TITLE = "title"         # 見出し・タイトル
    CAPTION = "caption"     # キャプション（図表共通）

    # 翻訳除外
    FOOTNOTE = "footnote"   # 脚注
    FORMULA = "formula"     # 数式（inline/display 統合）
    CODE = "code"           # コード・アルゴリズム
    TABLE = "table"         # 表
    IMAGE = "image"         # 画像
    CHART = "chart"         # チャート
    HEADER = "header"       # ヘッダー・フッター
    REFERENCE = "reference" # 参考文献

    # その他
    OTHER = "other"         # 上記以外
```

### 4.4 カテゴリマッピング

```python
RAW_TO_PROJECT_MAPPING: dict[RawLayoutCategory, ProjectCategory] = {
    # 翻訳対象
    RawLayoutCategory.TEXT: ProjectCategory.TEXT,
    RawLayoutCategory.ABSTRACT: ProjectCategory.TEXT,
    RawLayoutCategory.ASIDE_TEXT: ProjectCategory.TEXT,
    RawLayoutCategory.PARAGRAPH_TITLE: ProjectCategory.TITLE,
    RawLayoutCategory.DOC_TITLE: ProjectCategory.TITLE,
    RawLayoutCategory.FIGURE_TITLE: ProjectCategory.CAPTION,
    RawLayoutCategory.FOOTNOTE: ProjectCategory.FOOTNOTE,

    # 翻訳除外
    RawLayoutCategory.INLINE_FORMULA: ProjectCategory.FORMULA,
    RawLayoutCategory.DISPLAY_FORMULA: ProjectCategory.FORMULA,
    RawLayoutCategory.FORMULA_NUMBER: ProjectCategory.FORMULA,
    RawLayoutCategory.ALGORITHM: ProjectCategory.CODE,
    RawLayoutCategory.CODE_BLOCK: ProjectCategory.CODE,
    RawLayoutCategory.TABLE: ProjectCategory.TABLE,
    RawLayoutCategory.IMAGE: ProjectCategory.IMAGE,
    RawLayoutCategory.CHART: ProjectCategory.CHART,
    RawLayoutCategory.HEADER: ProjectCategory.HEADER,
    RawLayoutCategory.FOOTER: ProjectCategory.HEADER,
    RawLayoutCategory.NUMBER: ProjectCategory.HEADER,
    RawLayoutCategory.REFERENCE: ProjectCategory.REFERENCE,
    RawLayoutCategory.REFERENCE_CONTENT: ProjectCategory.REFERENCE,

    # その他
    RawLayoutCategory.SEAL: ProjectCategory.OTHER,
    RawLayoutCategory.CONTENT: ProjectCategory.OTHER,
    RawLayoutCategory.TABLE_OF_CONTENTS: ProjectCategory.OTHER,
    RawLayoutCategory.UNKNOWN: ProjectCategory.OTHER,
}
```

### 4.5 LayoutBlock データクラス

```python
@dataclass
class LayoutBlock:
    """レイアウト検出結果"""

    bbox: BBox
    raw_category: RawLayoutCategory   # モデル出力（Raw Layer）
    project_category: ProjectCategory  # 正規化済み（Project Layer）
    confidence: float
    page_num: int
```

### 4.6 翻訳対象分類 (Project Layer ベース)

| ProjectCategory | 翻訳対象 | 対応する RawLayoutCategory |
|-----------------|:--------:|----------------------------|
| `TEXT` | ✅ ON | text, abstract, aside_text |
| `TITLE` | ✅ ON | paragraph_title, doc_title |
| `CAPTION` | ✅ ON | figure_title |
| `FOOTNOTE` | ❌ OFF | footnote |
| `FORMULA` | ❌ OFF | inline_formula, display_formula, formula_number |
| `CODE` | ❌ OFF | algorithm, code_block |
| `TABLE` | ❌ OFF | table |
| `IMAGE` | ❌ OFF | image |
| `CHART` | ❌ OFF | chart |
| `HEADER` | ❌ OFF | header, footer, number |
| `REFERENCE` | ❌ OFF | reference, reference_content |
| `OTHER` | ❌ OFF | seal, content, table_of_contents, unknown |

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

PP-DocLayoutV2 は画像座標（ピクセル）を返すため、PDF 座標に変換が必要。

**重要**: 画像座標系は「原点が左上、Y軸は下向き」だが、PDF座標系は「原点が左下、Y軸は上向き」のため、**Y軸の反転**が必須。

```python
def _convert_image_to_pdf_coords(
    image_bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
) -> BBox:
    """
    画像座標を PDF 座標に変換（Y軸反転あり）

    座標系の違い:
    - 画像座標: 原点=左上, Y軸=下向き (y0 < y1 で上→下)
    - PDF座標: 原点=左下, Y軸=上向き (y0 < y1 で下→上)

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

    # X座標: 単純スケーリング
    x0_pdf = image_bbox[0] * scale_x
    x1_pdf = image_bbox[2] * scale_x

    # Y座標: 反転 + スケーリング
    # 画像の y0 (上) → PDF の y1 (上)
    # 画像の y1 (下) → PDF の y0 (下)
    y0_pdf = page_height - (image_bbox[3] * scale_y)
    y1_pdf = page_height - (image_bbox[1] * scale_y)

    return BBox(
        x0=x0_pdf,
        y0=y0_pdf,
        x1=x1_pdf,
        y1=y1_pdf,
    )
```

**注意事項**:
- ページ回転（`/Rotate`）がある場合は追加の変換が必要（初期実装では未対応）
- CropBox/MediaBox が異なる場合も考慮が必要（初期実装では MediaBox 前提）

### 5.4 マッチング関数

#### 5.4.1 ネストしたブロックの課題

PP-DocLayout は text ブロック内に inline_formula を検出することがある：

```
┌─────────────────────────────────────────────────────────┐
│ text ブロック                                           │
│                                                         │
│   "The equation ┌──────────────┐ shows that..."        │
│                 │inline_formula│                        │
│                 │  ∑_i x_i     │                        │
│                 └──────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

この場合、数式部分の TextObject は text と inline_formula 両方と重複する。
**単純な IoU では text ブロック（大）が勝ってしまい、数式が翻訳されてしまう。**

#### 5.4.2 カテゴリ優先度

翻訳除外すべきカテゴリを優先して「安全側に倒す」設計：

```python
# カテゴリ優先度 (数値が小さいほど優先)
CATEGORY_PRIORITY: dict[RawLayoutCategory, int] = {
    # 最優先: 絶対に翻訳してはいけないもの
    RawLayoutCategory.INLINE_FORMULA: 1,
    RawLayoutCategory.DISPLAY_FORMULA: 1,
    RawLayoutCategory.FORMULA_NUMBER: 1,
    RawLayoutCategory.ALGORITHM: 2,
    RawLayoutCategory.CODE_BLOCK: 2,

    # 次点: 通常は翻訳しないもの
    RawLayoutCategory.TABLE: 3,
    RawLayoutCategory.IMAGE: 4,
    RawLayoutCategory.CHART: 4,

    # キャプション系
    RawLayoutCategory.FIGURE_TITLE: 5,

    # テキスト系
    RawLayoutCategory.TEXT: 6,
    RawLayoutCategory.PARAGRAPH_TITLE: 6,
    RawLayoutCategory.DOC_TITLE: 6,
    RawLayoutCategory.ABSTRACT: 6,
    RawLayoutCategory.ASIDE_TEXT: 6,

    # その他
    RawLayoutCategory.FOOTNOTE: 7,
    RawLayoutCategory.HEADER: 8,
    RawLayoutCategory.FOOTER: 8,
    RawLayoutCategory.NUMBER: 8,
    RawLayoutCategory.REFERENCE: 9,
    RawLayoutCategory.REFERENCE_CONTENT: 9,
}
```

#### 5.4.3 BBox ユーティリティ関数

`layout_utils.py` に以下のユーティリティ関数を定義する（`BBox` クラスは変更しない）：

```python
def bbox_area(bbox: BBox) -> float:
    """BBox の面積を計算"""
    return bbox.width * bbox.height


def bbox_intersection(bbox1: BBox, bbox2: BBox) -> BBox | None:
    """2つの BBox の交差領域を返す（交差なしの場合 None）"""
    x0 = max(bbox1.x0, bbox2.x0)
    y0 = max(bbox1.y0, bbox2.y0)
    x1 = min(bbox1.x1, bbox2.x1)
    y1 = min(bbox1.y1, bbox2.y1)

    if x0 >= x1 or y0 >= y1:
        return None
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def calc_containment(text_bbox: BBox, block_bbox: BBox) -> float:
    """
    TextObject がブロック内にどれだけ含まれているか (0.0 - 1.0)

    containment = intersection_area / text_area
    """
    intersection = bbox_intersection(text_bbox, block_bbox)
    if intersection is None:
        return 0.0
    text_area = bbox_area(text_bbox)
    if text_area == 0:
        return 0.0
    return bbox_area(intersection) / text_area
```

#### 5.4.4 マッチングアルゴリズム

```python
def match_text_with_layout(
    text_objects: list[TextObject],
    layout_blocks: list[LayoutBlock],
    containment_threshold: float = 0.5,
) -> dict[str, ProjectCategory]:
    """
    TextObject と LayoutBlock をマッチング

    アルゴリズム:
    1. TextObject と各 LayoutBlock の包含率を計算
    2. 包含率が閾値以上のブロックを候補とする
    3. 候補の中からカテゴリ優先度が最高のものを選択
    4. 同一優先度なら包含率が高い → 面積が小さいものを選択

    Args:
        text_objects: PDFProcessor から抽出した TextObject リスト
        layout_blocks: LayoutAnalyzer から取得した LayoutBlock リスト
        containment_threshold: 包含率閾値 (default: 0.5)

    Returns:
        TextObject.id → ProjectCategory のマッピング
    """
    result = {}

    for text_obj in text_objects:
        # Step 1: 候補ブロックを抽出
        candidates = []
        for block in layout_blocks:
            containment = calc_containment(text_obj.bbox, block.bbox)
            if containment >= containment_threshold:
                candidates.append({
                    "block": block,
                    "containment": containment,
                    "area": bbox_area(block.bbox),
                    "priority": CATEGORY_PRIORITY.get(
                        block.raw_category, 99
                    ),
                })

        if not candidates:
            # 未マッチ時のデフォルト動作（安全側）
            result[text_obj.id] = ProjectCategory.OTHER
            continue

        # Step 2: ソート
        # 優先度昇順 → 包含率降順 → 面積昇順
        candidates.sort(
            key=lambda x: (x["priority"], -x["containment"], x["area"])
        )

        # 最優先の候補を採用
        best = candidates[0]["block"]
        result[text_obj.id] = best.project_category

    return result
```

#### 5.4.5 マッチング例

| ケース | TextObject | 候補ブロック | 結果 |
|--------|-----------|-------------|------|
| 数式テキスト | `∑_i x_i` | text(優先度6), inline_formula(優先度1) | **inline_formula** ✅ |
| 通常テキスト | `Hello` | text(優先度6) のみ | **text** ✅ |
| 表のセル | `123` | text(優先度6), table(優先度3) | **table** ✅ |
| 未マッチ | (孤立) | なし | **OTHER** (安全側) |

### 5.5 フィルタリング関数 (Project Layer ベース)

```python
# 翻訳対象カテゴリ (Project Layer)
TRANSLATABLE_CATEGORIES = {
    ProjectCategory.TEXT,
    ProjectCategory.TITLE,
    ProjectCategory.CAPTION,
}

# 翻訳除外カテゴリ (Project Layer)
NON_TRANSLATABLE_CATEGORIES = {
    ProjectCategory.FOOTNOTE,
    ProjectCategory.FORMULA,
    ProjectCategory.CODE,
    ProjectCategory.TABLE,
    ProjectCategory.IMAGE,
    ProjectCategory.CHART,
    ProjectCategory.HEADER,
    ProjectCategory.REFERENCE,
    ProjectCategory.OTHER,
}

def filter_translatable(
    text_objects: list[TextObject],
    categories: dict[str, ProjectCategory],
) -> list[TextObject]:
    """
    翻訳対象の TextObject をフィルタリング

    Args:
        text_objects: 全 TextObject リスト
        categories: TextObject.id → ProjectCategory のマッピング

    Returns:
        翻訳対象の TextObject リスト
    """
    return [
        obj for obj in text_objects
        if categories.get(obj.id) in TRANSLATABLE_CATEGORIES
    ]
```

**ポイント**: フィルタリングは `ProjectCategory` (Project Layer) を使用。
モデル出力の `RawLayoutCategory` は内部で `ProjectCategory` に変換される。

---

## 6. テスト計画

### 6.1 ユニットテスト

| テストケース | 説明 |
|-------------|------|
| `test_layout_category_enum` | カテゴリ Enum の網羅性 |
| `test_layout_block_creation` | LayoutBlock データクラスの生成 |
| `test_coordinate_conversion` | 画像→PDF 座標変換の正確性 |
| `test_bbox_utils` | bbox_area, bbox_intersection, calc_containment |
| `test_category_priority_matching` | 優先度ベースマッチングの正確性 |
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

- [ ] `RawLayoutCategory` Enum の追加（モデル出力用）
- [ ] `ProjectCategory` Enum の追加（翻訳ロジック用）
- [ ] `RAW_TO_PROJECT_MAPPING` の定義
- [ ] `LayoutBlock` データクラスの追加

### Phase 2: 座標変換・マッチング関数

**出力**: `src/pdf_translator/core/layout_utils.py`

- [ ] `bbox_area()` 関数
- [ ] `bbox_intersection()` 関数
- [ ] `calc_containment()` 関数
- [ ] `convert_image_to_pdf_coords()` 関数（Y軸反転あり）
- [ ] `CATEGORY_PRIORITY` 定義
- [ ] `match_text_with_layout()` 関数（包含率+優先度ベース）
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
| マッチング精度 | 包含率計算 | containment ≥ 0.5 で正しくマッチ |
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
