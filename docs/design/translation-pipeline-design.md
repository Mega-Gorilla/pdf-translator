# PDF翻訳パイプライン設計書

Issue #4: feat: PDF翻訳パイプラインの実装

## 1. 概要

### 1.1 目的

PDF翻訳の全体フローを統合するパイプラインを実装する。レイアウト解析、テキスト抽出、翻訳、再挿入を統合し、レイアウトを保持したまま翻訳されたPDFを出力する。

### 1.2 スコープ

- テキスト抽出とレイアウト解析の統合
- 翻訳対象テキストのフィルタリングと結合
- 翻訳処理（バッチ翻訳、リトライ）
- フォントサイズ自動調整
- 翻訳テキストの再挿入

### 1.3 スコープ外

- Markdown出力（Issue #5）
- 見開きPDF出力（Issue #6）
- CLIインターフェース（Issue #7）

### 1.4 制約事項

本設計における既知の制約事項を記載する。

#### 1.4.1 CJK フォント未対応

**制約**: 翻訳先言語が CJK（日本語・中国語・韓国語）の場合、正しく表示されない。

**理由**:
- `PDFProcessor.insert_text_object()` は標準フォント（Helvetica, Times-Roman 等）を使用
- CJK 文字は標準フォントに含まれないため、フォントファイル（TTF/TTC）の指定が必要
- CJK フォントの同梱はライセンス確認・サイズ（数十MB）の検討が必要

**現在の動作**:
- 翻訳先が CJK 言語の場合、警告ログを出力
- 翻訳処理は実行するが、PDF 出力時に文字化けする可能性あり

**対応予定** (Issue #18):
- `PipelineConfig.cjk_font_path: Optional[Path]` を追加
- システムフォント検索、または Noto Sans JP 同梱を検討
- ライセンス: SIL OFL 1.1（Apache-2.0 とバンドル互換）

##### デバッグ用: システムフォントの利用

本格実装前のデバッグでは、システムにインストールされた Noto Sans CJK を直接指定して日本語表示をテストできる。

**検証済みフォント**:
```
/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

**使用方法**:
```python
from pdf_translator.core.pdf_processor import PDFProcessor
from pdf_translator.core.models import Font, BBox

processor = PDFProcessor("input.pdf")

# CJK フォントを指定して日本語テキストを挿入
processor.insert_text_object(
    page_num=0,
    text="こんにちは日本語テスト",
    bbox=BBox(x0=50, y0=700, x1=300, y1=750),
    font=Font(name="NotoSansCJK", size=12.0),
    font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
```

**注意事項**:
- **⚠️ TTC フォントは非推奨**: TTC（TrueType Collection）フォーマットは pypdfium2 で読み込み可能だが、CJK 文字のグリフが正しく描画されない問題が確認されている。**TTF フォントの使用を推奨**。
- CJK 文字を含むテキストは自動的に CID フォントとして処理される（`_needs_cid_font()`）
- システムフォントのパスは環境依存のため、本番実装では Issue #18 で対応

#### 1.4.2 PDFテキスト構造の制約と対応

##### 問題: PDFテキストは行単位で分離される

PDF からテキストを抽出すると、**物理的な行単位**で TextObject が分離される。これは PDF フォーマットの構造上の制約であり、pypdfium2 に限らずすべての PDF 処理ライブラリで共通の挙動。

**抽出例（LLaMA 論文 Abstract）**:
```
[16] "We introduce LLaMA, a collection of founda-" (Y=595.5)
[17] "tion language models ranging from 7B to 65B"  (Y=582.8)
```

**行単位翻訳の問題**:
- ハイフネーション分断: 「founda-」と「tion」が別々に翻訳される
- 文脈喪失: 文の途中で分割されるため、翻訳品質が著しく低下
- 出力例:
  ```
  [翻訳] "We introduce LLaMA, a collection of founda-を紹介します"
  [翻訳] "tion言語モデルは7Bから65Bの範囲です"
  ```

##### 対応方針: pdftext によるブロック単位抽出

**pdftext ライブラリを使用してブロック（段落）単位で抽出する**。

```
PDF → pdftext (ブロック抽出) → 翻訳 → 配置 → 挿入
```

> **NOTE**: ハイフネーション前処理は不要。翻訳サービス（Google/DeepL/OpenAI）が
> ハイフネーションされたテキスト（例: `founda- tion`）を正しく認識し翻訳する（検証済み）。

**pdftext の利点**:

| 機能 | 従来（自前実装） | pdftext |
|------|-----------------|---------|
| 段落検出 | 複雑なアルゴリズム必要 | **自動（scikit-learn ベース）** |
| 行クラスタリング | Y-tolerance 計算必要 | **不要** |
| 多段組対応 | X-overlap 計算必要 | **自動** |
| ハイフネーション | カスタム処理 | **不要（翻訳サービスが処理）** |

**pdftext 検証結果（LLaMA 論文）**:
- Page 0: 15 ブロック検出
- Abstract（13行）が1ブロックに正しくグループ化
- 2段組レイアウトが自動で分離（左列 x~88-274, 右列 x~306-526）

**実装する機能**:

| 機能 | 説明 | 実装方法 |
|------|------|----------|
| ブロック抽出 | pdftext でブロック単位取得 | `dictionary_output()` API |
| 座標変換 | pdftext → PDF 座標系 | `pdf_y = page_height - pdftext_y` |
| テキスト結合 | ブロック内の行をスペースで結合 | シンプルな結合処理 |

> **NOTE**: ハイフネーション結合処理は不要。翻訳サービスが自動で処理する。

**現在の制約**:

| 制約 | 理由 | 将来拡張案 |
|------|------|-----------|
| 翻訳後テキストはブロック BBox に配置 | 行ごとの分配は複雑で破綻しやすい | LLM で段落レイアウト再構成 |
| 段落が元の BBox を超過する場合はフォント縮小 | シンプルな対応 | 動的レイアウト調整 |

##### 翻訳後テキスト配置戦略

pdftext で検出されたブロックの BBox を基準として配置。

```
元の PDF:
┌──────────────────────────────────┐
│ We introduce LLaMA, a collection │ ← pdftext block
│ of foundation language models    │   bbox で
│ ranging from 7B to 65B.         │   1ブロックとして検出
└──────────────────────────────────┘

翻訳後:
┌──────────────────────────────────┐
│ 我々はLLaMAを紹介します。これは  │ ← block_bbox を基準に配置
│ 7Bから65Bまでの基盤言語モデル   │   （元のテキストは削除）
│ のコレクションです。             │
└──────────────────────────────────┘
```

**注意点**:
- pdftext block bbox を翻訳テキスト配置の基準とする
- 元のテキストオブジェクトは全削除
- フォントサイズは block_bbox に収まるよう調整

##### 座標系の違い（重要）

pdftext と PDF/pypdfium2 では Y 座標系が異なる:

| ライブラリ | Y 座標原点 | Y 増加方向 |
|-----------|-----------|-----------|
| pdftext | ページ左上 | 上→下 |
| PDF/pypdfium2 | ページ左下 | 下→上 |

**変換式**:
```python
# pdftext bbox: [x0, y0_top, x1, y1_bottom]
# PDF bbox: (x0, y0_bottom, x1, y1_top)
pdf_y0 = page_height - y1_bottom  # PDF の下端
pdf_y1 = page_height - y0_top     # PDF の上端
```

##### 将来拡張

- LLM バックエンドでの structured output を活用したクロスブロック翻訳
- 文脈を考慮した翻訳品質向上
- 段落レイアウトの動的再構成

---

## 2. 現状分析

### 2.1 実装済みコンポーネント

| コンポーネント | ファイル | 機能 |
|---------------|---------|------|
| PDFProcessor | `core/pdf_processor.py` | テキスト抽出・削除・挿入・apply |
| LayoutAnalyzer | `core/layout_analyzer.py` | PP-DocLayoutV2によるレイアウト解析 |
| layout_utils | `core/layout_utils.py` | TextObject↔LayoutBlockマッチング |
| TranslatorBackend | `translators/*.py` | Google/DeepL/OpenAI翻訳バックエンド |
| pdftext | 外部ライブラリ | ブロック（段落）単位テキスト抽出 |

#### pdftext ライブラリ

**概要**: pypdfium2 ベースの PDF テキスト抽出ライブラリ。scikit-learn の決定木を使用してブロック（段落）を自動検出。

**ライセンス**: Apache-2.0（プロジェクト互換）

**主要 API**:
```python
from pdftext.extraction import dictionary_output

result = dictionary_output(
    pdf_path,
    page_range=[0, 1, 2],  # 対象ページ（省略で全ページ）
    sort=False,            # ソート無効（デフォルト）
    keep_chars=False,      # 文字単位情報は不要
)
# 戻り値: list[dict] - ページごとの辞書
# result[0]['blocks']: ブロックリスト
# result[0]['blocks'][0]['lines']: 行リスト
# result[0]['blocks'][0]['bbox']: [x0, y0_top, x1, y1_bottom]
```

**依存関係設定** (`pyproject.toml`):
```toml
dependencies = [
    "pypdfium2>=4.30.0",
    "pdftext>=0.6.0",
    ...
]

[tool.uv]
override-dependencies = ["pypdfium2>=5.2.0"]
```

> **NOTE**: pdftext は `pypdfium2==4.30.0` を要求するが、実際には pypdfium2 5.2.0 でも動作する。
> `[tool.uv]` の `override-dependencies` で最新版を強制。

#### 2.1.1 依存関係の運用方針

**採用方針: uv override + ドキュメント明記**

| 環境 | pypdfium2 バージョン | 備考 |
|------|---------------------|------|
| uv (推奨) | 5.2.0 | override-dependencies により強制 |
| pip | 4.30.0 | pdftext の要求バージョン |

**理由**:
- pypdfium2 5.2.0 は PDFProcessor の機能（テキスト挿入等）に必要
- pdftext のブロック抽出は 4.30.0/5.2.0 どちらでも動作確認済み
- 本プロジェクトは uv を公式ビルドツールとして採用

**代替案（不採用）**:
- (A) dependencies で `pypdfium2>=5.2.0` を直接指定 → pdftext との整合性エラー
- (B) 4.30.0 を前提に設計 → PDFProcessor の一部機能に制限
- (C) pdftext を optional extra に → セットアップが複雑化

**pip 利用者への対応**:
README に「`uv sync` 推奨、pip 使用時は `pip install pypdfium2>=5.2.0` を追加実行」と明記。

### 2.2 依存Issue

- ✅ #1 pypdfium2 ラッパー（実装済み）
- ✅ #2 PP-DocLayout 統合（実装済み・PR #15 マージ済み）
- ✅ #3 翻訳バックエンド（実装済み）

### 2.3 未実装機能

| 機能 | 説明 |
|------|------|
| テキスト結合 | 複数TextObjectを文の連続性を保って結合 |
| クロスブロック翻訳 | 終端句読点がない場合に次ブロックと結合 |
| フォントサイズ調整 | 翻訳後テキストがbboxに収まるよう調整 |
| パイプライン統合 | 全ステージの統合、進捗、エラーハンドリング |
| PDFProcessor.to_bytes() | PDFをbytes出力するメソッド（パイプライン出力に必要） |

### 2.4 PDFProcessor.to_bytes() 追加

パイプラインの出力が `TranslationResult.pdf_bytes` であるため、`PDFProcessor` に bytes 出力メソッドが必要。

**追加理由**:
- 後方互換性: 既存の `save(path)` の動作を変えない
- 意図の明確さ: 「ファイル保存」と「バイト取得」は別の操作

**実装イメージ**:
```python
def to_bytes(self) -> bytes:
    """Export PDF as bytes."""
    from io import BytesIO
    buffer = BytesIO()
    pdf = self._ensure_open()
    pdf.save(buffer)
    return buffer.getvalue()
```

> **NOTE**: Issue #4 のスコープ外だが、パイプラインが必要とするので同時に実装する。

---

## 3. アーキテクチャ

### 3.1 データフロー

```
PDF Input
    │
    ▼
┌─────────────────────────────────────────┐
│ pdftext.dictionary_output()            [NEW] │
│ → ブロック（段落）単位のテキスト             │
│ ※ 多段組も自動で分離                        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ ParagraphExtractor.extract()           [NEW] │
│ → list[Paragraph]                            │
│ ※ pdftext blocks → Paragraph 変換           │
│ ※ 座標変換（pdftext → PDF 座標系）          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ LayoutAnalyzer.analyze_all()      (Optional) │
│ → dict[int, list[LayoutBlock]]               │
│ ※ 数式・表・図の検出                         │
│ ※ asyncio.to_thread() 経由で呼び出し         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ layout_utils.assign_categories()       [NEW] │
│ → list[Paragraph] (カテゴリ付与済み)         │
│ ※ pdftext bbox と LayoutBlock bbox を比較   │
│ ※ 重複する Paragraph に category を設定     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ TranslatorBackend.translate_batch()          │
│ → 翻訳済みテキスト（段落単位）               │
│ ※ para.is_translatable == True のみ翻訳     │
│ ※ 文脈を保持した翻訳が可能                   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ TextLayoutEngine.fit_text_in_bbox()    [NEW] │
│ → レイアウト結果（行分割、フォントサイズ）     │
│ ※ 段落の block_bbox に収まるよう自動調整      │
│ ※ 自動改行、禁則処理対応                      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ PDFProcessor.apply_paragraphs()        [NEW] │
│ → 翻訳済みPDF                                │
│ ※ 元テキストを削除し、翻訳テキストを挿入     │
└─────────────────────────────────────────┘
    │
    ▼
Output PDF
```

**データフロー変更点（pdftext 統合前との比較）**:

| 従来 | 新（pdftext 統合） |
|------|-------------------|
| PDFProcessor で行単位抽出 | pdftext でブロック単位抽出 |
| TextMerger で段落マージ（複雑） | ParagraphExtractor（シンプル） |
| 行クラスタリング必要 | **不要** |
| 列分離ロジック必要 | **不要** |
| 段落境界検出必要 | **不要** |

### 3.2 ファイル構成

```
src/pdf_translator/
├── core/
│   ├── models.py              # 既存 + Paragraph 追加
│   ├── pdf_processor.py       # 既存 + to_bytes(), apply_paragraphs() 追加
│   ├── layout_analyzer.py     # 既存
│   ├── layout_utils.py        # 既存 + filter_paragraphs() 追加
│   ├── paragraph_extractor.py # 新規: pdftext ブロック → Paragraph 変換
│   └── font_adjuster.py       # 新規: フォント調整
├── pipeline/                   # 新規ディレクトリ
│   ├── __init__.py            # 公開 API エクスポート
│   ├── translation_pipeline.py  # TranslationPipeline, PipelineConfig, TranslationResult
│   ├── progress.py            # ProgressCallback
│   └── errors.py              # PipelineError 等
└── translators/               # 既存
```

**変更点（pdftext 統合）**:
- `text_merger.py` → `paragraph_extractor.py` に変更（役割が大幅に簡略化）
- `layout_utils.py` に `assign_categories()` 追加（カテゴリ付与）

---

## 4. データモデル

### 4.1 Paragraph データ構造（新規）

#### 4.1.1 設計方針: pdftext ブロックベース

pdftext が自動でブロック（段落）を検出するため、`Paragraph` dataclass はシンプルな構造になる。
pdftext block の情報を保持し、翻訳・配置に必要な情報を提供する。

**配置**: `src/pdf_translator/core/models.py`

```python
@dataclass
class Paragraph:
    """段落（翻訳単位）.

    pdftext で検出されたブロックを翻訳単位として表す。
    翻訳後テキストは block_bbox の位置に配置される。

    Attributes:
        id: 段落ID（"para_p{page}_b{block_index}" 形式）
        page_number: ページ番号
        text: マージされたテキスト
        block_bbox: pdftext ブロックの BBox（PDF 座標系に変換済み）
        line_count: 元の行数（デバッグ/統計用）
        original_font_size: 元のフォントサイズ（推定値）
        category: PP-DocLayout によるカテゴリ（"text", "formula", "table" 等、raw_category 文字列）
        category_confidence: レイアウト検出の信頼度 (0.0-1.0)
    """
    id: str
    page_number: int
    text: str
    block_bbox: BBox
    line_count: int
    original_font_size: float = 12.0  # デフォルト値

    # PP-DocLayout によるカテゴリ分類（raw_category 文字列を直接使用）
    category: Optional[str] = None  # None = 未分類
    category_confidence: Optional[float] = None

    # 翻訳後に設定されるフィールド
    translated_text: Optional[str] = None
    adjusted_font_size: Optional[float] = None

    def is_translatable(
        self,
        translatable_categories: frozenset[str] | None = None,
    ) -> bool:
        """翻訳対象かどうかを判定.

        Args:
            translatable_categories: 翻訳対象カテゴリのセット。
                None の場合は DEFAULT_TRANSLATABLE_RAW_CATEGORIES を使用。
        """
        # category が None（レイアウト解析無効）の場合は翻訳対象
        if self.category is None:
            return True
        # DEFAULT_TRANSLATABLE_RAW_CATEGORIES と照合
        if translatable_categories is None:
            translatable_categories = DEFAULT_TRANSLATABLE_RAW_CATEGORIES
        return self.category in translatable_categories
```

**変更点（従来設計との比較）**:

| 従来 | pdftext 統合後 |
|------|---------------|
| `text_object_ids: list[str]` | 削除（pdftext はブロック単位で管理） |
| `anchor_bbox` (最初の行) | `block_bbox` (ブロック全体) |
| `anchor_font`, `anchor_transform` | 削除（シンプル化） |
| なし | `category: str` 追加（raw_category 文字列を直接使用） |
| なし | `category_confidence: float` 追加（検出信頼度） |

#### 4.1.2 Paragraph 生成フロー

```python
from pdftext.extraction import dictionary_output

# pdftext でブロック抽出
pdftext_result = dictionary_output(pdf_path, page_range=page_range)

# ParagraphExtractor が Paragraph リストを生成
paragraphs = extractor.extract(pdftext_result)

# 翻訳（段落単位）
texts_to_translate = [p.text for p in paragraphs]
translated_texts = await translator.translate_batch(
    texts_to_translate, source_lang, target_lang
)

# 翻訳結果を Paragraph に設定
for para, translated in zip(paragraphs, translated_texts):
    para.translated_text = translated

# フォントサイズ調整
for para in paragraphs:
    para.adjusted_font_size = adjuster.calculate_font_size(
        para.translated_text,
        para.block_bbox,
        para.original_font_size,
        target_lang,
    )

# PDF 適用
processor.apply_paragraphs(paragraphs)
```

#### 4.1.3 PDFProcessor.apply_paragraphs() の動作

```python
def apply_paragraphs(
    self,
    paragraphs: list[Paragraph],
    font_path: Optional[Path] = None,
) -> None:
    """段落単位で翻訳テキストを適用.

    各 Paragraph について:
    1. block_bbox 内の既存テキストを削除
    2. block_bbox の位置に translated_text を挿入

    Args:
        paragraphs: 翻訳済み Paragraph リスト
        font_path: カスタムフォントパス（CJK用, Optional）
    """
    for para in paragraphs:
        if not para.translated_text:
            continue

        # block_bbox 内のテキストを削除
        self.remove_text_in_bbox(
            page_num=para.page_number,
            bbox=para.block_bbox,
        )

        # 翻訳テキストを挿入（既存 API に合わせる）
        font = Font(name="Helvetica", size=para.adjusted_font_size or 12.0)
        self.insert_text_object(
            page_num=para.page_number,
            text=para.translated_text,
            bbox=para.block_bbox,
            font=font,
            font_path=font_path,  # CJK 対応時に使用
        )
```

> **NOTE**: `remove_text_in_bbox()` は新規メソッド。block_bbox 内のテキストオブジェクトを
> 一括削除する。pdftext のブロック境界は正確なので、bbox ベースの削除が可能。
> 既存 `insert_text_object(page_num, text, bbox, font, font_path, ...)` API を使用。

#### 4.1.3.1 remove_text_in_bbox() の安全性ルール

**部分重複（partial overlap）の扱い**:

```python
def remove_text_in_bbox(
    self,
    page_num: int,
    bbox: BBox,
    containment_threshold: float = 0.5,  # 削除基準
) -> int:
    """BBox 内のテキストオブジェクトを削除.

    Args:
        page_num: ページ番号
        bbox: 削除対象範囲
        containment_threshold: 削除判定の閾値（TextObject の何%が bbox 内にあれば削除）

    Returns:
        削除したオブジェクト数

    削除判定:
        containment = (intersection_area / text_object_area)
        if containment >= containment_threshold:
            削除する
    """
```

**設計方針（保守的削除）**:

| 状況 | containment | 判定 |
|------|-------------|------|
| 完全包含 | 1.0 | ✅ 削除 |
| 80% 重複 | 0.8 | ✅ 削除（threshold=0.5） |
| 50% 重複 | 0.5 | ✅ 削除（境界） |
| 30% 重複 | 0.3 | ❌ 保持 |
| 重複なし | 0.0 | ❌ 保持 |

**リスク軽減策**:
- `containment_threshold` をデフォルト 0.5 に設定（50% 以上重複で削除）
- 隣接ブロックの誤削除を防ぐため、pdftext bbox をそのまま使用（マージンなし）
- デバッグ用にログ出力（削除対象オブジェクト ID と containment 値）

#### 4.1.4 将来拡張

LLM バックエンドを活用した高度な翻訳を将来的に検討:

```python
# 将来: LLM での structured output を活用
@dataclass
class Paragraph:
    ...
    # 将来追加予定
    # layout_hint: Optional[str] = None  # "single_column", "two_column", etc.
    # semantic_type: Optional[str] = None  # "abstract", "heading", "body", etc.
```

### 4.2 TranslationResult（新規）

**配置**: `pipeline/translation_pipeline.py`

```python
@dataclass
class TranslationResult:
    """翻訳パイプラインの結果.

    Attributes:
        pdf_bytes: 翻訳済み PDF バイナリ
        stats: 翻訳統計情報
    """
    pdf_bytes: bytes
    stats: Optional[dict[str, Any]] = None
```

### 4.3 ProgressCallback Protocol（新規）

**配置**: `pipeline/progress.py`

```python
@runtime_checkable
class ProgressCallback(Protocol):
    """進捗報告用プロトコル."""

    def __call__(
        self,
        stage: str,        # ステージ名（下記参照）
        current: int,      # 現在の処理数
        total: int,        # 総数
        message: str = "", # 状態メッセージ
    ) -> None: ...
```

**ステージ一覧と total の計算方法**:

| stage | 説明 | total |
|-------|------|-------|
| `extract` | pdftext でブロック抽出 + Paragraph 変換 | page_count |
| `analyze` | レイアウト解析（Optional） | page_count |
| `categorize` | PP-DocLayout カテゴリ付与 | len(paragraphs) |
| `translate` | バッチ翻訳（`is_translatable` のみ） | len(translatable_paragraphs) |
| `font_adjust` | フォントサイズ調整 | len(translatable_paragraphs) |
| `apply` | PDFに適用 | 1（PDF 1 ファイル） |

**変更点（pdftext 統合）**:
- `extract`: PDFProcessor → pdftext + ParagraphExtractor
- `merge` → `categorize` に名称変更（カテゴリ付与）

---

## 5. コンポーネント設計

### 5.1 ParagraphExtractor（旧 TextMerger）

**ファイル**: `src/pdf_translator/core/paragraph_extractor.py`

#### 5.1.1 目的

pdftext で抽出されたブロックを `Paragraph` リストに変換する。
pdftext がブロック検出を自動で行い、翻訳サービスがハイフネーションを自動処理するため、
このコンポーネントの責務は**極めてシンプル**になる。

**責務**:
1. pdftext ブロック → Paragraph 変換
2. 座標系変換（pdftext → PDF）
3. ブロック内行のテキスト結合（スペース区切り）

**削除された責務**:
- ~~行クラスタリング~~ → pdftext が担当
- ~~列分離（多段組対応）~~ → pdftext が担当
- ~~段落境界検出~~ → pdftext が担当
- ~~ハイフネーション結合~~ → 翻訳サービスが自動処理（§5.1.4 参照）

**API 設計**:
```python
class ParagraphExtractor:
    """pdftext ブロックから Paragraph を抽出."""

    def extract(
        self,
        pdftext_result: list[dict],
        page_range: list[int] | None = None,
    ) -> list[Paragraph]:
        """pdftext の出力から Paragraph リストを生成.

        Args:
            pdftext_result: pdftext.dictionary_output() の戻り値
            page_range: 対象ページ番号リスト（None で全ページ）

        Returns:
            段落のリスト
        """
        ...

    @staticmethod
    def extract_from_pdf(
        pdf_path: str | Path,
        page_range: list[int] | None = None,
    ) -> list[Paragraph]:
        """PDF ファイルから直接 Paragraph を抽出（便利メソッド）.

        Args:
            pdf_path: PDF ファイルパス
            page_range: 対象ページ番号リスト

        Returns:
            段落のリスト
        """
        from pdftext.extraction import dictionary_output
        result = dictionary_output(str(pdf_path), page_range=page_range)
        extractor = ParagraphExtractor()
        return extractor.extract(result, page_range)
```

#### 5.1.2 アルゴリズム概要

```
入力: pdftext_result (list[dict])
出力: list[Paragraph]

1. ページごとにループ
   - page_data = pdftext_result[page_idx]
   - page_height = page_data['bbox'][3]  # 座標変換用

2. ブロックごとにループ
   - block = page_data['blocks'][block_idx]

3. 行テキスト抽出・結合
   - lines = [join(span['text'] for span in line['spans']) for line in block['lines']]
   - merged_text = " ".join(line.strip() for line in lines)

4. 座標変換
   - pdftext bbox → PDF bbox
   - pdf_y0 = page_height - y1_bottom
   - pdf_y1 = page_height - y0_top

5. Paragraph 生成
   - id, page_number, text, block_bbox, line_count を設定
```

> **NOTE**: ハイフネーション処理は不要。翻訳サービスが `founda- tion` のような
> ハイフネーションされたテキストを正しく認識し翻訳する。

#### 5.1.3 実装

```python
import re
from pathlib import Path
from pdf_translator.core.models import Paragraph, BBox


class ParagraphExtractor:
    """pdftext ブロックから Paragraph を抽出."""

    def extract(
        self,
        pdftext_result: list[dict],
        page_range: list[int] | None = None,
    ) -> list[Paragraph]:
        """pdftext の出力から Paragraph リストを生成."""
        paragraphs: list[Paragraph] = []

        for page_idx, page_data in enumerate(pdftext_result):
            # page_range 指定時はフィルタリング
            if page_range is not None and page_idx not in page_range:
                continue

            page_height = page_data['bbox'][3]

            for block_idx, block in enumerate(page_data['blocks']):
                para = self._process_block(
                    block, page_idx, block_idx, page_height
                )
                if para:
                    paragraphs.append(para)

        return paragraphs

    def _process_block(
        self,
        block: dict,
        page_idx: int,
        block_idx: int,
        page_height: float,
    ) -> Paragraph | None:
        """単一ブロックを Paragraph に変換."""
        # 行テキスト抽出
        lines = []
        for line in block['lines']:
            line_text = "".join(span['text'] for span in line['spans'])
            lines.append(line_text.strip())

        if not lines:
            return None

        # テキスト結合（スペース区切り）
        # ハイフネーション処理は不要 - 翻訳サービスが自動処理
        merged_text = " ".join(line for line in lines if line)
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()

        if not merged_text:
            return None

        # 座標変換（pdftext → PDF）
        x0, y0_top, x1, y1_bottom = block['bbox']
        pdf_y0 = page_height - y1_bottom  # PDF の下端
        pdf_y1 = page_height - y0_top     # PDF の上端

        # フォントサイズ推定（最初の span から）
        font_size = 12.0  # デフォルト
        if block['lines'] and block['lines'][0]['spans']:
            first_span = block['lines'][0]['spans'][0]
            font_size = first_span.get('font', {}).get('size', 12.0)

        return Paragraph(
            id=f"para_p{page_idx}_b{block_idx}",
            page_number=page_idx,
            text=merged_text,
            block_bbox=BBox(x0=x0, y0=pdf_y0, x1=x1, y1=pdf_y1),
            line_count=len(lines),
            original_font_size=font_size,
        )
```

> **実装のシンプルさ**: ハイフネーション結合ロジックを削除したことで、
> 約 30 行のコードが削減され、エッジケース対応の複雑さも解消された。

#### 5.1.4 翻訳サービスによるハイフネーション自動処理（検証結果）

**結論**: ハイフネーション前処理は不要。翻訳サービスが自動で処理する。

##### 検証方法

LLaMA 論文から抽出したハイフネーション付きテキストを、3 つの翻訳サービス
（Google Translate、DeepL、OpenAI）でテストし、前処理なしで正しく翻訳されるか検証した。

##### 検証結果

**テスト入力**: `"a collection of founda- tion language models"`（ハイフネーション付き）
**比較対象**: `"a collection of foundation language models"`（ハイフネーションなし）

| サービス | ハイフン付き翻訳 | ハイフンなし翻訳 | 結果 |
|---------|-----------------|-----------------|------|
| Google | 基礎言語モデルのコレクション | 基本言語モデルのコレクション | ✅ 両方正しい |
| DeepL | 基礎言語モデル集 | 基礎言語モデル集 | ✅ 完全一致 |
| OpenAI | 基盤言語モデルのコレクション | 基盤言語モデルのコレクション | ✅ 完全一致 |

**その他のテストケース**:

| テストケース | 入力テキスト | 結果 |
|-------------|-------------|------|
| 固有名詞 | `Hoff- mann et al.` | ✅ Google/OpenAI: ホフマン、DeepL: Hoffmann |
| 一般単語 | `avail- able datasets` | ✅ 全サービスで正しく翻訳 |
| 複合語（数字） | `LLaMA- 65B` | ✅ 全サービスで正しく翻訳 |

##### サービス別評価

| サービス | ハイフネーション処理 | 前処理必要? |
|---------|---------------------|------------|
| **Google** | 優秀（意味は正しく翻訳） | **不要** |
| **DeepL** | 良好（一部アーティファクト残存） | 不要 |
| **OpenAI** | 完璧（4/4 完全一致） | **不要** |

##### 設計への影響

この検証結果により、`ParagraphExtractor` からハイフネーション結合ロジックを削除:

- **削除**: `_merge_hyphenation()` メソッド（約 30 行）
- **削除**: 複合語判定ロジック（小文字開始チェック等）
- **簡略化**: 行結合は単純なスペース結合のみ

> **将来の検討事項**: DeepL で一部アーティファクトが残存する場合があるため、
> 翻訳品質を重視するユースケースではオプションとしてハイフネーション前処理を
> 追加することを検討できる。ただし、現時点では翻訳精度に影響しないため不要。

#### 5.1.5 設計の簡略化

従来の TextMerger と比較した削減内容:

| 従来の責務 | 新設計 | 理由 |
|-----------|--------|------|
| 行クラスタリング | **削除** | pdftext が自動処理 |
| 列分離（多段組対応） | **削除** | pdftext が自動処理 |
| 段落境界検出 | **削除** | pdftext が自動処理 |
| ハイフネーション処理 | **削除** | 翻訳サービスが自動処理（§5.1.4 参照） |

**コード量の削減**: 約 200 行 → 約 50 行（75% 削減）

> **NOTE**: ハイフネーション処理も不要になったことで、当初見積もり（60% 削減）を
> 上回る 75% 削減を達成。`ParagraphExtractor` は座標変換と単純な行結合のみを担当する。

#### 5.1.6 エッジケース対応

| ケース | 対応 |
|--------|------|
| 空のブロック | `None` を返しスキップ |
| 空白のみのブロック | `None` を返しスキップ |
| フォントサイズ不明 | デフォルト 12.0 pt を使用 |
| 単一行ブロック | そのまま Paragraph 生成 |

### 5.2 カテゴリ付与（assign_categories）

**ファイル**: `src/pdf_translator/core/layout_utils.py`

#### 5.2.1 目的

PP-DocLayout で検出した LayoutBlock と pdftext の Paragraph を照合し、
各 Paragraph に適切なカテゴリ（text, formula, table, figure 等）を付与する。

**処理フロー**:
```
pdftext Paragraphs + PP-DocLayout LayoutBlocks
                    ↓
    bbox 重複判定（containment ratio）
                    ↓
    重複する LayoutBlock のカテゴリを Paragraph に付与
                    ↓
    カテゴリ付き Paragraphs（中間 JSON に保存可能）
```

#### 5.2.2 API 設計

```python
def assign_categories(
    paragraphs: list[Paragraph],
    layout_blocks: dict[int, list[LayoutBlock]],
    threshold: float = 0.5,
) -> list[Paragraph]:
    """Paragraph に PP-DocLayout のカテゴリを付与.

    Args:
        paragraphs: pdftext から抽出した Paragraph リスト
        layout_blocks: ページ番号 → LayoutBlock リストのマッピング
        threshold: 重複判定の閾値（containment ratio）

    Returns:
        カテゴリが付与された Paragraph リスト（入力と同じオブジェクト）

    Note:
        - 複数の LayoutBlock と重複する場合は CATEGORY_PRIORITY に基づき選択
        - 翻訳除外カテゴリ（formula, table, image）を優先（fail-safe）
        - どの LayoutBlock とも重複しない場合は category = None のまま
        - LayoutBlock.project_category を Paragraph.category に設定
    """
    for para in paragraphs:
        page_blocks = layout_blocks.get(para.page_number, [])
        best_match = _find_best_matching_block(para.block_bbox, page_blocks, threshold)
        if best_match:
            para.category = best_match.raw_category.value  # raw_category 文字列を使用
            para.category_confidence = best_match.confidence
    return paragraphs


def _find_best_matching_block(
    para_bbox: BBox,
    blocks: list[LayoutBlock],
    threshold: float,
) -> Optional[LayoutBlock]:
    """最適な LayoutBlock を検索（fail-safe: 翻訳除外カテゴリを優先）.

    既存 layout_utils.CATEGORY_PRIORITY を使用してカテゴリ優先度を判定。
    数式・表・図は優先度が高く、少しでも重複すれば翻訳除外となる。
    """
    candidates: list[tuple[LayoutBlock, float]] = []

    for block in blocks:
        overlap = _calculate_overlap_ratio(para_bbox, block.bbox)
        if overlap >= threshold:
            candidates.append((block, overlap))

    if not candidates:
        return None

    # CATEGORY_PRIORITY に基づきソート（優先度昇順 → 重複率降順）
    # 翻訳除外カテゴリ（formula=1, table=3, image=4）が先に来る
    candidates.sort(
        key=lambda x: (
            CATEGORY_PRIORITY.get(x[0].raw_category, DEFAULT_PRIORITY),
            -x[1],  # 同一優先度なら重複率が高い方
        )
    )

    return candidates[0][0]


def _calculate_overlap_ratio(bbox1: BBox, bbox2: BBox) -> float:
    """2つの BBox の重複率を計算（intersection / bbox1 area）."""
    inter_x0 = max(bbox1.x0, bbox2.x0)
    inter_y0 = max(bbox1.y0, bbox2.y0)
    inter_x1 = min(bbox1.x1, bbox2.x1)
    inter_y1 = min(bbox1.y1, bbox2.y1)

    if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
        return 0.0

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    bbox1_area = (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0)

    if bbox1_area <= 0:
        return 0.0

    return inter_area / bbox1_area
```

#### 5.2.3 PP-DocLayout カテゴリ一覧

PP-DocLayoutV2 が検出するカテゴリは `RawLayoutCategory` として定義され、
`ProjectCategory` にマッピングされる（`models.py` 参照）。

**CATEGORY_PRIORITY（layout_utils.py）**:
既存の `CATEGORY_PRIORITY` を使用して fail-safe なマッチングを実現。
翻訳除外カテゴリ（formula=1, table=3, image=4）が優先される。

**翻訳対象判定（raw_category ベース）**:

> **実装方針の簡略化**: 当初設計では `RawLayoutCategory` → `ProjectCategory` のマッピングを
> 行う予定だったが、実装では `raw_category` 文字列を直接使用する簡略化を採用した。

| raw_category | 翻訳対象 |
|--------------|---------|
| `text`, `vertical_text`, `abstract`, `aside_text` | ✅ Yes |
| `paragraph_title`, `doc_title` | ❌ No（タイトルは原文保持）|
| `figure_title` | ❌ No（キャプションは原文保持）|
| `footnote`, `vision_footnote` | ❌ No |
| `inline_formula`, `display_formula`, `formula_number` | ❌ No |
| `algorithm` | ❌ No |
| `table` | ❌ No |
| `image` | ❌ No |
| `chart` | ❌ No |
| `header`, `header_image`, `footer`, `footer_image`, `number` | ❌ No |
| `reference`, `reference_content` | ❌ No |
| `seal`, `content`, `unknown` | ❌ No |

**翻訳対象カテゴリ** (`DEFAULT_TRANSLATABLE_RAW_CATEGORIES`):
```python
DEFAULT_TRANSLATABLE_RAW_CATEGORIES: frozenset[str] = frozenset({
    "text", "vertical_text", "abstract", "aside_text",
})
```

> **NOTE**: タイトル系カテゴリ（`doc_title`, `paragraph_title`, `figure_title`）は
> 原文のまま保持する設計に変更。翻訳が必要な場合は `PipelineConfig.translatable_categories`
> でカスタマイズ可能。

> **NOTE**: `Paragraph.is_translatable()` と `LayoutBlock.is_translatable` は
> 同じ `DEFAULT_TRANSLATABLE_RAW_CATEGORIES` を参照する。

#### 5.2.4 座標系の注意点

PP-DocLayout と pdftext は異なる座標系を使用する可能性がある:

| ライブラリ | 座標系 |
|-----------|--------|
| pdftext | Y: 上→下（ページ左上原点）→ **PDF座標系に変換済み** |
| PP-DocLayout | 画像座標系（Y: 上→下） |
| PDF/pypdfium2 | Y: 下→上（ページ左下原点） |

**対応方針**:
- `ParagraphExtractor` で pdftext bbox を PDF 座標系に変換済み
- `LayoutAnalyzer` で PP-DocLayout bbox を PDF 座標系に変換（既存実装）
- `assign_categories()` では両者が PDF 座標系であることを前提とする

#### 5.2.5 layout_analysis=False の場合

レイアウト解析が無効の場合、`assign_categories()` はスキップされ、
すべての Paragraph は `category = None` のまま翻訳対象となる。

```python
if self._config.layout_analysis:
    layout_blocks = await self._stage_analyze(pdf_path)
    assign_categories(paragraphs, layout_blocks, self._config.layout_containment_threshold)
else:
    logger.warning(
        "Layout analysis disabled. All paragraphs will be translated. "
        "Formulas, tables, and figures may be incorrectly translated."
    )
    # category = None のまま → is_translatable = True
```

### 5.3 TextLayoutEngine

**ファイル**: `src/pdf_translator/core/text_layout.py`

> **実装変更**: 当初設計の `FontSizeAdjuster` は、より高機能な `TextLayoutEngine` に
> 置き換えられた。PDFium のフォントメトリクスを使用した正確なレイアウト計算が可能。

#### 5.3.1 目的

翻訳後テキストが元の BBox に収まるよう、以下の機能を提供する：
- 正確なテキスト幅計算（PDFium `FPDFFont_GetGlyphWidth` 使用）
- 自動改行（単語境界・文字境界対応）
- フォントサイズ自動調整
- 日本語禁則処理（kinsoku）

#### 5.3.2 API 設計

```python
class TextLayoutEngine:
    def __init__(
        self,
        min_font_size: float = 6.0,
        font_size_step: float = 0.5,
        line_height_factor: float = 1.2,
    ) -> None:
        """TextLayoutEngine を初期化.

        Args:
            min_font_size: 最小フォントサイズ（pt）
            font_size_step: 縮小ステップ（pt）
            line_height_factor: 行高さ係数（1.0 = tight, 1.5 = loose）
        """
        ...

    def fit_text_in_bbox(
        self,
        text: str,
        bbox: BBox,
        font_handle: ctypes.c_void_p,
        initial_font_size: float,
        rotation_degrees: float = 0.0,
    ) -> LayoutResult:
        """テキストを bbox に収める.

        Args:
            text: レイアウトするテキスト
            bbox: 収める BBox
            font_handle: PDFium フォントハンドル (FPDF_FONT)
            initial_font_size: 初期フォントサイズ（pt）
            rotation_degrees: 回転角度（0, 90, 180, 270）

        Returns:
            LayoutResult（行リスト、最終フォントサイズ、収まったかのフラグ）
        """
        ...

    def calculate_text_width(
        self,
        text: str,
        font_handle: ctypes.c_void_p,
        font_size: float,
    ) -> float:
        """テキスト幅を計算（PDFium フォントメトリクス使用）."""
        ...

    def wrap_text(
        self,
        text: str,
        max_width: float,
        font_handle: ctypes.c_void_p,
        font_size: float,
    ) -> list[str]:
        """テキストを自動改行."""
        ...
```

#### 5.3.3 LayoutResult データクラス

```python
@dataclass
class LayoutLine:
    """レイアウト済みの1行."""
    text: str
    width: float
    y_position: float

@dataclass
class LayoutResult:
    """レイアウト計算結果."""
    lines: list[LayoutLine]
    font_size: float
    total_height: float
    fits_in_bbox: bool
```

#### 5.3.4 アルゴリズム

```
入力: text, bbox, font_handle, initial_font_size, rotation
出力: LayoutResult

1. font_size = initial_font_size
2. 90° または 270° 回転時は bbox の width/height を交換
3. While font_size >= min_font_size:
   a. wrapped_lines = wrap_text(text, bbox_width, font_handle, font_size)
   b. line_height = get_line_height(font_handle, font_size)
   c. total_height = line_height * len(wrapped_lines)
   d. If total_height <= bbox_height:
      - 各行の y_position を計算
      - return LayoutResult(lines, font_size, total_height, fits=True)
   e. font_size -= font_size_step
4. return LayoutResult(lines, min_font_size, total_height, fits=False)
```

#### 5.3.5 禁則処理（kinsoku）

日本語テキストの改行位置を適切に制御:

```python
# 行頭禁止文字（句読点、閉じ括弧、小書き仮名など）
KINSOKU_NOT_AT_LINE_START = {"。", "、", "）", "」", "ぁ", "っ", "ー", ...}

# 行末禁止文字（開き括弧）
KINSOKU_NOT_AT_LINE_END = {"（", "「", "『", ...}
```

#### 5.3.6 CJK 文字判定

```python
def _is_cjk_char(self, char: str) -> bool:
    """CJK 文字かどうかを判定."""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF    # CJK Unified Ideographs
        or 0x3040 <= code <= 0x309F  # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0xAC00 <= code <= 0xD7AF  # Hangul
        or 0x3000 <= code <= 0x303F  # CJK Punctuation
        or 0xFF00 <= code <= 0xFFEF  # Fullwidth Forms
    )
```

### 5.4 TranslationPipeline

**ファイル**: `src/pdf_translator/pipeline/translation_pipeline.py`

#### 5.4.1 クラス設計

```python
class TranslationPipeline:
    """PDF翻訳パイプライン.

    ワークフロー（pdftext 統合版）:
    1. extract: pdftext でブロック抽出 → Paragraph 変換
    2. analyze: レイアウト解析（Optional, asyncio.to_thread 経由）
    3. categorize: PP-DocLayout カテゴリ付与（assign_categories）
    4. translate: バッチ翻訳（is_translatable のみ）
    5. font_adjust: フォントサイズ調整
    6. apply: PDFに適用（remove_text_in_bbox → insert_text_object）
    """

    def __init__(
        self,
        translator: TranslatorBackend,
        config: PipelineConfig | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """TranslationPipeline を初期化.

        Args:
            translator: 翻訳バックエンド
            config: パイプライン設定（None の場合はデフォルト値）
            progress_callback: 進捗コールバック
        """
        self._translator = translator
        self._config = config or PipelineConfig()
        self._progress_callback = progress_callback
        self._analyzer = LayoutAnalyzer()  # Lazy 初期化でも可

    async def translate(
        self,
        pdf_source: Path | str | bytes,
        output_path: Path | None = None,
    ) -> TranslationResult:
        """PDFを翻訳.

        Args:
            pdf_source: 入力 PDF（パスまたは bytes）
            output_path: 出力先パス（指定時はファイル保存も行う）

        Returns:
            TranslationResult（常に pdf_bytes を含む）
        """
        ...
```

#### 5.4.2 output_path 指定時の動作

`output_path` が指定された場合、ファイル保存を行い、かつ `TranslationResult` も返却する。

```python
async def translate(...) -> TranslationResult:
    ...
    pdf_bytes = processor.to_bytes()

    # output_path が指定されていればファイル保存
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pdf_bytes)

    return TranslationResult(pdf_bytes=pdf_bytes, stats=stats)
```

**理由**: 常に `TranslationResult` を返すことで、呼び出し側の処理を統一できる。

#### 5.4.3 LayoutAnalyzer の非同期対応

`LayoutAnalyzer.analyze_all()` は同期メソッドだが、Pipeline は非同期。
CPU バウンドの処理をブロックしないよう `asyncio.to_thread()` を使用する。

```python
async def _stage_analyze(self, pdf_path: Path) -> dict[int, list[LayoutBlock]]:
    if not self._config.layout_analysis:
        return {}

    # CPU バウンドの処理をスレッドプールで実行
    return await asyncio.to_thread(
        self._analyzer.analyze_all, pdf_path
    )
```

**理由**:
- `LayoutAnalyzer` の既存コードを変更しない
- CPU バウンド処理（PP-DocLayout 推論）をイベントループから分離
- Pipeline 側の責務として適切

#### 5.4.4 layout_analysis=False の動作

レイアウト解析を無効化した場合の動作を定義する。

**動作**: すべての TextObject を翻訳対象（`ProjectCategory.TEXT`）として扱い、警告ログを出力する。

**ユースケース**:
- レイアウト解析が不要な単純な PDF（テキストのみ）
- レイアウト解析が失敗した場合のフォールバック
- 以前の実装（`_archive`）との整合性

**実装イメージ**:
```python
if not self._config.layout_analysis:
    logger.warning(
        "Layout analysis disabled. All text objects will be translated. "
        "Formulas, tables, and other non-text elements may be incorrectly translated."
    )
    # すべての TextObject を TEXT カテゴリとして扱う
    categories = {obj.id: ProjectCategory.TEXT for obj in all_objects}
else:
    # LayoutAnalyzer.analyze_all() は dict[int, list[LayoutBlock]] を返す
    layout_by_page = await self._stage_analyze(pdf_path)

    # ページごとにマッチングを実行し、結果を集約
    categories: dict[str, ProjectCategory] = {}
    for page in pdf_doc.pages:
        page_blocks = layout_by_page.get(page.page_number, [])
        page_categories = match_text_with_layout(
            page.text_objects,
            page_blocks,
            self._config.layout_containment_threshold,
        )
        categories.update(page_categories)
```

**注意点**: 警告メッセージで「数式や表も翻訳される可能性がある」ことを明示する。

#### 5.4.5 bytes 入力時の注意

**重要**: `LayoutAnalyzer` は `Path` を前提としている（内部で pypdfium2 に渡す）。
`bytes` 入力の場合は一時ファイル経由が必要。

```python
async def translate(
    self,
    pdf_source: Union[Path, str, bytes],
    output_path: Optional[Path] = None,
) -> TranslationResult:
    # bytes 入力の場合は一時ファイルに書き出し
    if isinstance(pdf_source, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_source)
            temp_path = Path(f.name)
        try:
            return await self._translate_impl(temp_path, output_path)
        finally:
            temp_path.unlink(missing_ok=True)
    else:
        path = Path(pdf_source)
        return await self._translate_impl(path, output_path)
```

#### 5.4.6 ステージ構成

| ステージ (stage値) | メソッド | 処理内容 |
|-------------------|---------|---------|
| `extract` | `_stage_extract` | pdftext でブロック抽出 + Paragraph 変換 |
| `analyze` | `_stage_analyze` | レイアウト解析（Optional） |
| `categorize` | `_stage_categorize` | カテゴリ付与（`assign_categories()`） |
| `translate` | `_stage_translate` | バッチ翻訳（`is_translatable` のみ） |
| `font_adjust` | `_stage_font_adjust` | フォントサイズ調整 |
| `apply` | `_stage_apply` | PDFに適用 |

**変更点（pdftext 統合）**:
- `extract`: PDFProcessor → pdftext + ParagraphExtractor
- `merge` → `categorize` に名称変更（カテゴリ付与）

> **NOTE**: ステージ名は §4.3 の ProgressCallback stage 値と一致させている。

#### 5.4.7 リトライロジック

**リトライ対象の例外**:
- `TranslationError`: リトライ対象（API 障害、レート制限など一時的エラー）
- `ConfigurationError`: **リトライ対象外**（API キー不正、設定エラーなど即 fail）

```python
from pdf_translator.translators.base import ConfigurationError, TranslationError

async def _translate_with_retry(
    self,
    texts: list[str],
) -> list[str]:
    for attempt in range(self._config.max_retries + 1):
        try:
            return await self._translator.translate_batch(
                texts, self._config.source_lang, self._config.target_lang
            )
        except ConfigurationError:
            # 設定エラーはリトライせず即 fail
            raise
        except TranslationError as e:
            if attempt < self._config.max_retries:
                delay = self._config.retry_delay * (2 ** attempt)  # 指数バックオフ
                await asyncio.sleep(delay)
            else:
                raise PipelineError(
                    f"Translation failed after {self._config.max_retries} retries",
                    stage="translate",
                    cause=e,
                )
```

### 5.5 エラーハンドリング

**ファイル**: `src/pdf_translator/pipeline/errors.py`

```python
class PipelineError(Exception):
    """パイプラインエラー基底クラス."""

    def __init__(
        self,
        message: str,
        stage: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.cause = cause

class ExtractionError(PipelineError):
    """テキスト抽出エラー."""
    pass

class LayoutAnalysisError(PipelineError):
    """レイアウト解析エラー."""
    pass

class MergeError(PipelineError):
    """テキスト結合エラー."""
    pass

class FontAdjustmentError(PipelineError):
    """フォント調整エラー."""
    pass
```

#### 5.5.1 エラー発生時の動作

**方針**: エラー発生時は全体失敗とする。

**理由**:
1. **シンプルさ**: 部分的成功の処理は複雑（どのページが失敗したか追跡、UI への通知など）
2. **ユーザー期待**: 翻訳を依頼したら全ページ翻訳されることを期待する
3. **リトライで回復可能**: 一時的エラー（レート制限、ネットワーク）なら再実行で成功する
4. **デバッグ容易性**: 全体失敗のほうが問題の切り分けが容易

**動作**:
- リトライ上限に達したら `PipelineError` を raise
- エラーメッセージに失敗したステージと原因を含める

**将来拡張案**:
```python
@dataclass
class TranslationResult:
    pdf_bytes: bytes
    stats: Optional[dict[str, Any]] = None
    # 将来追加予定:
    # failed_pages: list[int] = field(default_factory=list)
    # partial_success: bool = False
```

---

## 6. 実装フェーズ

pdftext 統合により、実装フェーズが簡略化される。

### Phase 1: データモデル拡張

**成果物**:
- `src/pdf_translator/core/models.py` に Paragraph 追加

**タスク**:
1. Paragraph dataclass 実装（シンプル版）
2. 既存テストが壊れていないことを確認

### Phase 2: ParagraphExtractor（pdftext 統合）

**成果物**:
- `src/pdf_translator/core/paragraph_extractor.py`
- `tests/test_paragraph_extractor.py`

**タスク**:
1. ParagraphExtractor クラス実装
2. pdftext ブロック → Paragraph 変換
3. 座標系変換（pdftext → PDF）
4. ユニットテスト作成

**削減されたタスク**（pdftext / 翻訳サービスが担当）:
- ~~行クラスタリング~~ → pdftext
- ~~列分離（多段組対応）~~ → pdftext
- ~~段落境界検出~~ → pdftext
- ~~ハイフネーション処理~~ → 翻訳サービスが自動処理（§5.1.4 参照）

### Phase 3: TextLayoutEngine（テキストレイアウト）

**成果物**:
- `src/pdf_translator/core/text_layout.py`
- `tests/test_text_layout.py`

**タスク**:
1. TextLayoutEngine クラス実装
2. PDFium フォントメトリクスを使用した正確なテキスト幅計算
3. 自動改行アルゴリズム（単語境界・文字境界対応）
4. フォントサイズ自動調整
5. 日本語禁則処理（kinsoku）
6. ユニットテスト作成

### Phase 4: PDFProcessor 拡張

**成果物**:
- `src/pdf_translator/core/pdf_processor.py` に `apply_paragraphs()`, `remove_text_in_bbox()` 追加

**タスク**:
1. `remove_text_in_bbox()` メソッド実装（bbox 内テキスト一括削除）
2. `apply_paragraphs()` メソッド実装
3. 既存 apply() との整合性確認
4. ユニットテスト作成

### Phase 5: Pipeline（パイプライン統合）

**成果物**:
- `src/pdf_translator/pipeline/__init__.py`
- `src/pdf_translator/pipeline/translation_pipeline.py`
- `src/pdf_translator/pipeline/progress.py`
- `src/pdf_translator/pipeline/errors.py`
- `tests/test_translation_pipeline.py`

**タスク**:
1. エラークラス定義
2. 進捗コールバック定義
3. TranslationPipeline クラス実装
4. 各ステージ実装（pdftext + ParagraphExtractor ベース）
5. リトライロジック実装
6. 統合テスト作成

### Phase 6: テスト・検証

**成果物**:
- 全テストの実行確認
- サンプルPDFでの動作確認

**タスク**:
1. ユニットテスト全パス確認
2. mypy strict 確認
3. ruff lint 確認
4. サンプルPDF（LLaMA論文等）で E2E テスト
5. 翻訳品質の目視確認（ハイフネーション結合が正しく動作しているか）
6. pdftext ブロック検出精度の確認

### 工数見積もり（pdftext 統合による削減）

| フェーズ | 従来見積もり | pdftext 統合後 | 削減理由 |
|---------|-------------|---------------|----------|
| Phase 2 | 大 | **小** | 段落検出ロジック不要 |
| Phase 4 | 中 | 中 | 変更なし |
| Phase 5 | 大 | 中 | ステージ数削減 |
| 合計 | - | **約 40% 削減** | - |

---

## 7. テスト計画

### 7.1 ParagraphExtractor テスト

```python
class TestParagraphExtractor:
    # 基本機能
    def test_extract_single_block(self): ...
    def test_extract_multiple_blocks(self): ...
    def test_extract_multiple_pages(self): ...
    def test_page_range_filtering(self): ...

    # 座標変換
    def test_coordinate_conversion_top_to_bottom(self): ...
    def test_block_bbox_in_pdf_coordinates(self): ...

    # テキスト結合
    def test_lines_joined_with_space(self): ...
    def test_whitespace_normalized(self): ...

    # エッジケース
    def test_empty_block_skipped(self): ...
    def test_whitespace_only_block_skipped(self): ...
    def test_single_line_block(self): ...
    def test_font_size_extraction(self): ...
    def test_font_size_default_fallback(self): ...

    # 実PDFテスト
    def test_llama_paper_abstract_block(self): ...  # 13行が1ブロック
    def test_llama_paper_two_column(self): ...  # 2段組が分離される
```

**従来テストとの比較**:

| 従来（TextMerger） | 新（ParagraphExtractor） | 理由 |
|-------------------|-------------------------|------|
| `test_cluster_*` | **削除** | pdftext が担当 |
| `test_separate_columns` | **削除** | pdftext が担当 |
| `test_paragraph_boundary_*` | **削除** | pdftext が担当 |
| `test_hyphenation_*` | **削除** | 翻訳サービスが自動処理 |

> **NOTE**: ハイフネーション関連テストは不要。翻訳サービスがハイフネーションを
> 自動処理することは §5.1.4 で検証済み。

### 7.2 カテゴリ付与（assign_categories）テスト

```python
class TestAssignCategories:
    # 基本機能
    def test_assign_text_category(self): ...
    def test_assign_formula_category(self): ...
    def test_assign_table_category(self): ...
    def test_assign_figure_category(self): ...

    # マッチング
    def test_best_overlap_selected(self): ...
    def test_threshold_filtering(self): ...
    def test_no_match_keeps_none(self): ...

    # is_translatable プロパティ
    def test_text_is_translatable(self): ...
    def test_title_is_translatable(self): ...
    def test_formula_not_translatable(self): ...
    def test_table_not_translatable(self): ...
    def test_none_category_is_translatable(self): ...

    # 座標系
    def test_pdf_coordinates_used(self): ...

    # エッジケース
    def test_empty_layout_blocks(self): ...
    def test_multiple_pages(self): ...
```

### 7.3 TextLayoutEngine テスト

```python
class TestTextLayoutEngine:
    def test_text_fits_original_size(self): ...
    def test_text_requires_reduction(self): ...
    def test_minimum_font_size_limit(self): ...
    def test_text_wrapping_word_boundary(self): ...
    def test_text_wrapping_cjk_character(self): ...
    def test_kinsoku_line_start(self): ...  # 行頭禁則
    def test_kinsoku_line_end(self): ...    # 行末禁則
    def test_rotation_90_degrees(self): ... # 回転テキスト
    def test_calculate_text_width_accuracy(self): ...
    def test_long_paragraph_sizing(self): ...  # 段落全体が収まるサイズ計算
```

### 7.4 PDFProcessor 拡張テスト

```python
class TestPDFProcessorParagraphs:
    # remove_text_in_bbox
    def test_remove_text_in_bbox_removes_contained_objects(self): ...
    def test_remove_text_in_bbox_preserves_outside_objects(self): ...
    def test_remove_text_in_bbox_partial_overlap_above_threshold(self): ...  # 50%以上→削除
    def test_remove_text_in_bbox_partial_overlap_below_threshold(self): ...  # 50%未満→保持
    def test_remove_text_in_bbox_adjacent_blocks_preserved(self): ...  # 隣接ブロック保護

    # apply_paragraphs
    def test_apply_paragraphs_removes_text_in_bbox(self): ...
    def test_apply_paragraphs_inserts_translated_text(self): ...
    def test_apply_paragraphs_uses_block_bbox_position(self): ...
    def test_apply_paragraphs_multiple_paragraphs(self): ...
    def test_apply_paragraphs_respects_font_size(self): ...
```

### 7.5 TranslationPipeline テスト

```python
class TestTranslationPipeline:
    # 基本フロー
    async def test_full_pipeline_mocked_translator(self): ...
    async def test_progress_callback_invoked(self): ...
    async def test_output_path_saves_file(self): ...

    # pdftext 統合
    async def test_pdftext_extraction_called(self): ...
    async def test_paragraph_extractor_integration(self): ...

    # エラーハンドリング
    async def test_error_handling_extraction_failure(self): ...
    async def test_retry_on_transient_error(self): ...
    async def test_configuration_error_no_retry(self): ...

    # 翻訳フロー
    async def test_paragraph_based_translation(self): ...
    async def test_translatable_filtering(self): ...  # is_translatable によるフィルタリング
```

### 7.6 E2E テスト（サンプルPDF）

```python
class TestE2ETranslation:
    # LLaMA 論文テスト
    async def test_llama_paper_abstract_translation(self): ...
    async def test_llama_paper_two_column_preserved(self): ...

    # pdftext ブロック検出
    async def test_pdftext_block_detection_accuracy(self): ...
    async def test_coordinate_conversion_correct(self): ...

    # 翻訳品質
    async def test_paragraph_context_preserved(self): ...
    async def test_no_broken_sentences(self): ...
    async def test_hyphenated_text_translated_correctly(self): ...  # 翻訳サービス検証
```

> **NOTE**: `test_hyphenated_text_translated_correctly` は翻訳サービスが
> ハイフネーションを正しく処理することを検証する（§5.1.4 の結果を自動テスト化）。

---

## 8. リスクと対策

### 8.1 pdftext 統合に伴うリスク

| リスク | 影響度 | 対策 |
|--------|-------|------|
| pdftext ブロック検出の精度 | 中 | LLaMA 論文で検証済み（15 ブロック正確検出）、E2E テストで継続検証 |
| pdftext の pypdfium2 バージョン依存 | 低 | `[tool.uv] override-dependencies` で 5.2.0 強制、動作確認済み |
| pdftext の将来的なメンテナンス | 低 | Apache-2.0 ライセンス、必要なら fork 可能 |
| 座標変換ミス | 中 | 座標変換ロジックをユニットテストで検証 |

### 8.2 従来のリスク（軽減）

| リスク | 影響度 | 対策 | 変化 |
|--------|-------|------|------|
| 段落境界の誤検出 | ~~高~~ → **低** | pdftext が自動検出 | **リスク軽減** |
| 多段組の列分離ミス | ~~高~~ → **低** | pdftext が自動処理 | **リスク軽減** |
| ハイフネーション誤結合 | ~~中~~ → **解消** | 翻訳サービスが自動処理（§5.1.4 参照） | **リスク解消** |
| 複雑なレイアウトでの読み順誤り | ~~中~~ → **低** | pdftext が処理 | **リスク軽減** |

### 8.3 残存リスク

| リスク | 影響度 | 対策 |
|--------|-------|------|
| フォント幅推定の不正確さ | 中 | 保守的な推定値を使用 |
| 翻訳APIのレート制限 | 低 | 指数バックオフ、バッチサイズ調整 |
| 大規模PDFでのメモリ使用 | 低 | ページ単位処理 |
| CJK 言語への翻訳時の文字化け | 中 | CJK フォント対応予定（Issue #18、§1.4.1 参照） |
| pdftext 非対応の特殊 PDF | 低 | 必要に応じてフォールバック処理を検討 |

---

## 9. 設定オプション

**設定の方針**: すべての設定パラメータは `PipelineConfig` に統合する。
pdftext 統合により、段落検出関連のパラメータが**削除**される。

```python
@dataclass
class PipelineConfig:
    """パイプライン設定.

    すべての設定パラメータを統合管理する。
    pdftext 統合により、段落検出パラメータは不要になった。
    """

    # 翻訳設定
    source_lang: str = "en"
    target_lang: str = "ja"

    # レイアウト解析（Optional - 数式・表・図フィルタリング用）
    layout_analysis: bool = True
    layout_containment_threshold: float = 0.5

    # フォント調整（TextLayoutEngine 用）
    min_font_size: float = 6.0
    # NOTE: font_size_step (0.5) と line_height_factor (1.2) は
    # TextLayoutEngine のデフォルト値を使用（設定不要）

    # 翻訳リトライ
    max_retries: int = 3
    retry_delay: float = 1.0

    # CJK フォント（Issue #18 で対応予定、§1.4.1 参照）
    # cjk_font_path: Optional[Path] = None
```

**削除されたパラメータ**（pdftext が担当）:

| 削除パラメータ | 理由 |
|---------------|------|
| `line_y_tolerance` | pdftext が行検出 |
| `paragraph_gap_threshold` | pdftext が段落検出 |
| `x_overlap_ratio` | pdftext が列分離 |

**使用例**:
```python
config = PipelineConfig(target_lang="ja")

# TextLayoutEngine は PDFProcessor 内部で生成される（デフォルト値使用）
# min_font_size は apply_paragraphs() に渡される

# ParagraphExtractor はパラメータ不要（pdftext がすべて処理）
extractor = ParagraphExtractor()
```

---

## 10. 出力ファイル一覧

### 10.1 実装ファイル

| ファイル | 種別 | 説明 |
|---------|------|------|
| `src/pdf_translator/core/models.py` | 変更 | `Paragraph` dataclass 追加（`category` フィールド含む） |
| `src/pdf_translator/core/paragraph_extractor.py` | **新規** | pdftext ブロック → Paragraph 変換 |
| `src/pdf_translator/core/text_layout.py` | 新規 | テキストレイアウト（自動改行・フォントサイズ調整） |
| `src/pdf_translator/core/pdf_processor.py` | 変更 | `to_bytes()`, `apply_paragraphs()`, `remove_text_in_bbox()` 追加 |
| `src/pdf_translator/core/layout_utils.py` | 変更 | `assign_categories()` 追加（PP-DocLayout カテゴリ付与） |
| `src/pdf_translator/pipeline/__init__.py` | 新規 | 公開 API エクスポート |
| `src/pdf_translator/pipeline/translation_pipeline.py` | 新規 | PipelineConfig, TranslationResult 含む |
| `src/pdf_translator/pipeline/progress.py` | 新規 | ProgressCallback |
| `src/pdf_translator/pipeline/errors.py` | 新規 | PipelineError 等 |

### 10.2 テストファイル

| ファイル | 種別 | 説明 |
|---------|------|------|
| `tests/test_paragraph_extractor.py` | **新規** | ParagraphExtractor テスト |
| `tests/test_assign_categories.py` | **新規** | カテゴリ付与テスト |
| `tests/test_text_layout.py` | 新規 | TextLayoutEngine テスト |
| `tests/test_translation_pipeline.py` | 新規 | パイプライン統合テスト |

### 10.3 設定ファイル

| ファイル | 種別 | 説明 |
|---------|------|------|
| `pyproject.toml` | 変更 | `pdftext>=0.6.0` 追加、`[tool.uv]` セクション追加 |

### 10.4 変更点サマリー（pdftext 統合 + PP-DocLayout カテゴリ付与）

| 従来 | 新設計 |
|------|--------|
| `text_merger.py` (約200行) | `paragraph_extractor.py` (約50行) |
| 複雑な段落検出ロジック | pdftext に委譲 |
| ハイフネーション結合ロジック | **翻訳サービスが自動処理**（§5.1.4 参照） |
| フィルタリングで段落除外 | **カテゴリ付与**（`Paragraph.category`） |
| 多数の設定パラメータ | シンプルな設定 |

**設計簡略化の経緯**:

1. **pdftext 統合** (§1.4.2): 段落検出・多段組分離を pdftext に委譲 → 60% 削減
2. **翻訳サービス検証** (§5.1.4): ハイフネーション処理も不要と判明 → **75% 削減**
3. **PP-DocLayout カテゴリ付与** (§5.2): 中間データに category 情報を保持

**PP-DocLayout 連携**:
- `Paragraph.category`: `str`（raw_category 文字列を直接使用: "text", "formula", "table" 等）
- `Paragraph.is_translatable()`: `DEFAULT_TRANSLATABLE_RAW_CATEGORIES` を参照
- `assign_categories()`: pdftext bbox と LayoutBlock bbox の重複判定
- `LayoutBlock.is_translatable` と同じ判定ロジックを使用

**検証結果サマリー（翻訳サービス）**:
- Google Translate: ハイフネーション自動処理 ✅
- DeepL: ハイフネーション自動処理 ✅
- OpenAI: ハイフネーション自動処理 ✅（4/4 完全一致）
