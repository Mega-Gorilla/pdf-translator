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

### 1.4 v1 制約事項

v1 では以下の制約を設け、実装範囲を限定する。これらは v1.1 以降で段階的に対応予定。

#### 1.4.1 CJK フォント未対応

**制約**: 翻訳先言語が CJK（日本語・中国語・韓国語）の場合、正しく表示されない。

**理由**:
- `PDFProcessor.insert_text_object()` は標準フォント（Helvetica, Times-Roman 等）を使用
- CJK 文字は標準フォントに含まれないため、フォントファイル（TTF）の指定が必要
- CJK フォントの同梱はライセンス確認・サイズ（数十MB）の検討が必要

**v1 の動作**:
- 翻訳先が CJK 言語の場合、警告ログを出力
- 翻訳処理は実行するが、PDF 出力時に文字化けする可能性あり

**v1.1 以降の対応予定**:
- `PipelineConfig.cjk_font_path: Optional[Path]` を追加
- システムフォント検索、または Noto Sans CJK 同梱を検討

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

##### v1 の対応方針: 段落単位マージ

**v1 では TextMerger で段落単位にマージしてから翻訳する**。

```
抽出(行単位) → マージ(段落単位) → 翻訳 → 配置 → 挿入
```

**v1 で実装するマージ機能**:

| 機能 | 説明 | 優先度 |
|------|------|--------|
| 行クラスタリング | Y座標近接度で同一段落を判定 | 必須 |
| ハイフネーション結合 | 行末 `-` を検出し次行と結合 | 必須 |
| 段落境界検出 | 文末句読点、空行、インデントで段落を分割 | 必須 |
| 多段組対応 | X座標のオーバーラップで列を分離 | 必須 |

**v1 の制約（v2 以降で対応予定）**:

| 制約 | 理由 | v2 対応案 |
|------|------|-----------|
| 翻訳後テキストは最初の行の BBox に配置 | 行ごとの分配は複雑で破綻しやすい | LLM で段落レイアウト再構成 |
| 段落が元の BBox を超過する場合はフォント縮小 | シンプルな対応 | 動的レイアウト調整 |

##### 翻訳後テキスト配置戦略（v1）

マージされた段落の翻訳後、テキストをどこに配置するか：

**方針**: 段落の最初の行の BBox を基準として配置

```
元の PDF:
┌──────────────────────────────────┐
│ We introduce LLaMA, a collection │ ← 行1 (BBox A)
│ of foundation language models    │ ← 行2 (BBox B)
│ ranging from 7B to 65B.         │ ← 行3 (BBox C)
└──────────────────────────────────┘

翻訳後:
┌──────────────────────────────────┐
│ 我々はLLaMAを紹介します。これは  │ ← BBox A を基準に配置
│ 7Bから65Bまでの基盤言語モデル   │    （元の行2,3は削除）
│ のコレクションです。             │
└──────────────────────────────────┘
```

**注意点**:
- 翻訳前の行2, 行3 のテキストは削除される
- 翻訳後テキストは行1の位置から開始
- フォントサイズは BBox A に収まるよう調整

##### 将来拡張（v2 以降）

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
│ PDFProcessor.extract_text_objects()          │
│ → PDFDocument (TextObjects)                  │
│ ※ TextObject は行単位で分離されている        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ LayoutAnalyzer.analyze_all()                 │
│ → dict[int, list[LayoutBlock]]               │
│ ※ asyncio.to_thread() 経由で呼び出し         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ layout_utils.match_text_with_layout()        │
│ → dict[str, ProjectCategory]                 │
│ ※ 数式・表・図をフィルタリング               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ TextMerger.merge()                     [NEW] │
│ → list[Paragraph]                            │
│ ※ 行単位 TextObject を段落単位にマージ       │
│ ※ ハイフネーション結合を含む                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ TranslatorBackend.translate_batch()          │
│ → 翻訳済みテキスト（段落単位）               │
│ ※ 文脈を保持した翻訳が可能                   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ FontSizeAdjuster.calculate_font_size() [NEW] │
│ → 調整済みフォントサイズ                     │
│ ※ 段落の anchor_bbox に収まるよう調整        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ PDFProcessor.apply()                         │
│ → 翻訳済みPDF                                │
│ ※ 元の行を全削除し、翻訳テキストを挿入       │
└─────────────────────────────────────────┘
    │
    ▼
Output PDF
```

### 3.2 ファイル構成

```
src/pdf_translator/
├── core/
│   ├── models.py             # 既存（変更なし）
│   ├── pdf_processor.py      # 既存 + to_bytes() 追加
│   ├── layout_analyzer.py    # 既存
│   ├── layout_utils.py       # 既存
│   ├── text_merger.py        # 新規: 読み順ソート
│   └── font_adjuster.py      # 新規: フォント調整
├── pipeline/                  # 新規ディレクトリ
│   ├── __init__.py           # 公開 API エクスポート
│   ├── translation_pipeline.py  # TranslationPipeline, PipelineConfig, TranslationResult
│   ├── progress.py           # ProgressCallback
│   └── errors.py             # PipelineError 等
└── translators/              # 既存
```

---

## 4. データモデル

### 4.1 Paragraph データ構造（新規）

#### 4.1.1 設計方針: 段落単位でのマージと翻訳

PDF から抽出される TextObject は行単位で分離されているため、翻訳前に段落単位でマージする必要がある。
`Paragraph` dataclass は、複数の TextObject をまとめた翻訳単位を表す。

**配置**: `src/pdf_translator/core/models.py`

```python
@dataclass
class Paragraph:
    """段落（翻訳単位）.

    複数の TextObject をマージした翻訳単位。
    翻訳後テキストは anchor_bbox の位置に配置される。

    Attributes:
        id: 段落ID（"para_p{page}_i{index}" 形式）
        page_number: ページ番号
        text: マージされたテキスト（ハイフネーション結合済み）
        text_object_ids: 元の TextObject ID リスト（削除対象）
        anchor_bbox: 翻訳テキスト配置基準（最初の行の BBox）
        anchor_font: 翻訳テキストのフォント情報（最初の行から取得）
        anchor_transform: 翻訳テキストの Transform（最初の行から取得）
    """
    id: str
    page_number: int
    text: str
    text_object_ids: list[str]
    anchor_bbox: BBox
    anchor_font: Font
    anchor_transform: Transform

    # 翻訳後に設定されるフィールド
    translated_text: Optional[str] = None
    adjusted_font_size: Optional[float] = None
```

#### 4.1.2 Paragraph 生成フロー

```python
# TextMerger が Paragraph リストを生成
paragraphs = merger.merge(text_objects, categories)

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
        para.anchor_bbox,
        para.anchor_font.size,
        target_lang,
    )

# PDF 適用
processor.apply_paragraphs(paragraphs)  # 元の TextObject を削除し、翻訳テキストを挿入
```

#### 4.1.3 PDFProcessor.apply_paragraphs() の動作

```python
def apply_paragraphs(self, paragraphs: list[Paragraph]) -> None:
    """段落単位で翻訳テキストを適用.

    各 Paragraph について:
    1. text_object_ids の TextObject を全て削除
    2. anchor_bbox の位置に translated_text を挿入
    """
    for para in paragraphs:
        # 元の TextObject を全削除
        for obj_id in para.text_object_ids:
            self.remove_text_object(obj_id)

        # 翻訳テキストを挿入
        self.insert_text_object(
            page_number=para.page_number,
            text=para.translated_text,
            bbox=para.anchor_bbox,
            font_size=para.adjusted_font_size,
            transform=para.anchor_transform,
        )
```

#### 4.1.4 将来拡張（v2 以降）

v2 では LLM バックエンドを活用した高度な翻訳を検討:

```python
# v2: LLM での structured output を活用
@dataclass
class Paragraph:
    ...
    # v2 で追加予定
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
| `extract` | PDFからテキスト抽出 | 1（PDF 1 ファイル） |
| `analyze` | レイアウト解析 | page_count |
| `merge` | 読み順ソート | page_count |
| `translate` | バッチ翻訳 | len(translatable_objects) |
| `font_adjust` | フォントサイズ調整 | len(translatable_objects) |
| `apply` | PDFに適用 | 1（PDF 1 ファイル） |

---

## 5. コンポーネント設計

### 5.1 TextMerger

**ファイル**: `src/pdf_translator/core/text_merger.py`

#### 5.1.1 目的

行単位で分離された TextObject を**段落単位にマージ**し、`Paragraph` リストを生成する。
これにより、文脈を保持した翻訳が可能になる。

**責務**:
1. 翻訳対象カテゴリ（TEXT, TITLE, CAPTION）のフィルタリング
2. 行単位 TextObject の段落へのマージ
3. ハイフネーション結合（行末 `-` の処理）
4. 多段組の列分離
5. `Paragraph` リストの生成

**API 設計**:
```python
class TextMerger:
    def __init__(
        self,
        line_y_tolerance: float = 3.0,
        paragraph_gap_threshold: float = 1.5,  # 行高さの倍率
        x_overlap_ratio: float = 0.3,
    ) -> None:
        """TextMerger を初期化.

        Args:
            line_y_tolerance: 同一行判定の y 許容差（pt）
            paragraph_gap_threshold: 段落境界判定の行間閾値（行高さの倍率）
            x_overlap_ratio: 同一列判定に必要な x overlap 比率
        """
        ...

    def merge(
        self,
        text_objects: list[TextObject],
        categories: dict[str, ProjectCategory],
        page_number: int,
    ) -> list[Paragraph]:
        """TextObject を段落にマージして返す.

        Args:
            text_objects: ページ内の全 TextObject
            categories: TextObject.id → ProjectCategory のマッピング
            page_number: ページ番号

        Returns:
            段落のリスト（読み順）
        """
        ...
```

#### 5.1.2 アルゴリズム概要

```
入力: list[TextObject], dict[str, ProjectCategory], page_number
出力: list[Paragraph]

1. カテゴリフィルタリング
   - TEXT, TITLE, CAPTION のみ抽出

2. 行クラスタリング
   - Y座標で行をグループ化（line_y_tolerance 許容）
   - 同一行内は X0 昇順でソート

3. 列分離（多段組対応）
   - X座標のオーバーラップで列を判定
   - 各列を独立して処理

4. 段落検出
   - 段落境界の判定:
     a. 行間が paragraph_gap_threshold × 行高さ を超える
     b. 文末句読点で終わる行
     c. インデントの変化
   - 境界で分割して段落グループを形成

5. テキストマージ
   - 段落内の行を結合
   - ハイフネーション処理（行末 `-` を削除して次行と連結）

6. Paragraph 生成
   - 各段落から Paragraph オブジェクトを生成
   - anchor_bbox, anchor_font, anchor_transform は最初の行から取得
```

#### 5.1.3 行クラスタリング

```python
def _cluster_by_line(
    self,
    text_objects: list[TextObject],
) -> list[list[TextObject]]:
    """Y座標でクラスタリングして行グループを形成."""
    if not text_objects:
        return []

    # Y1（上端）でソート（上から下へ）
    sorted_objs = sorted(text_objects, key=lambda o: -o.bbox.y1)

    lines: list[list[TextObject]] = []
    current_line: list[TextObject] = [sorted_objs[0]]
    current_y = sorted_objs[0].bbox.y1

    for obj in sorted_objs[1:]:
        if abs(obj.bbox.y1 - current_y) <= self._line_y_tolerance:
            current_line.append(obj)
        else:
            current_line.sort(key=lambda o: o.bbox.x0)  # X0 昇順
            lines.append(current_line)
            current_line = [obj]
            current_y = obj.bbox.y1

    if current_line:
        current_line.sort(key=lambda o: o.bbox.x0)
        lines.append(current_line)

    return lines
```

#### 5.1.4 多段組対応（列分離）

学術論文などの多段組レイアウトで、左列と右列を誤って結合しないよう列を分離する。

```python
def _separate_columns(
    self,
    lines: list[list[TextObject]],
) -> list[list[list[TextObject]]]:
    """行リストを列ごとに分離.

    Returns:
        列ごとの行リスト（左から右の順）
    """
    if not lines:
        return []

    # 全行の X 範囲を収集
    x_ranges = []
    for line in lines:
        x0 = min(obj.bbox.x0 for obj in line)
        x1 = max(obj.bbox.x1 for obj in line)
        x_ranges.append((x0, x1))

    # 列境界を検出（大きな X ギャップを探す）
    columns = self._detect_column_boundaries(x_ranges)

    # 各列に属する行を分類
    column_lines: list[list[list[TextObject]]] = [[] for _ in columns]
    for line, (x0, x1) in zip(lines, x_ranges):
        for i, (col_x0, col_x1) in enumerate(columns):
            # X オーバーラップで列を判定
            overlap = min(x1, col_x1) - max(x0, col_x0)
            line_width = x1 - x0
            if overlap >= line_width * self._x_overlap_ratio:
                column_lines[i].append(line)
                break

    return column_lines
```

#### 5.1.5 段落境界検出

```python
def _detect_paragraph_boundaries(
    self,
    lines: list[list[TextObject]],
) -> list[int]:
    """段落境界のインデックスを検出.

    Returns:
        段落境界（新しい段落の開始行インデックス）のリスト
    """
    boundaries = [0]  # 最初の行は常に段落開始

    for i in range(len(lines) - 1):
        current_line = lines[i]
        next_line = lines[i + 1]

        if self._is_paragraph_boundary(current_line, next_line):
            boundaries.append(i + 1)

    return boundaries

def _is_paragraph_boundary(
    self,
    current_line: list[TextObject],
    next_line: list[TextObject],
) -> bool:
    """2行間が段落境界かどうか判定."""
    # 1. 文末判定
    last_text = current_line[-1].text.rstrip()
    if last_text and last_text[-1] in self._sentence_terminals:
        # 文末で終わり、かつ行間が広い場合は境界
        line_height = self._estimate_line_height(current_line)
        y_gap = current_line[-1].bbox.y0 - next_line[0].bbox.y1
        if y_gap > line_height * self._paragraph_gap_threshold:
            return True

    # 2. インデント検出
    current_x0 = min(obj.bbox.x0 for obj in current_line)
    next_x0 = min(obj.bbox.x0 for obj in next_line)
    if next_x0 - current_x0 > 10:  # インデント閾値
        return True

    # 3. 行間が著しく広い場合
    line_height = self._estimate_line_height(current_line)
    y_gap = current_line[-1].bbox.y0 - next_line[0].bbox.y1
    if y_gap > line_height * (self._paragraph_gap_threshold + 0.5):
        return True

    return False
```

#### 5.1.6 ハイフネーション処理

```python
def _merge_lines_to_text(
    self,
    lines: list[list[TextObject]],
) -> str:
    """行リストをテキストにマージ（ハイフネーション処理含む）."""
    merged_parts = []

    for i, line in enumerate(lines):
        # 行内のテキストを結合
        line_text = " ".join(obj.text for obj in line)

        if i > 0 and merged_parts:
            prev_text = merged_parts[-1]

            # ハイフネーション結合
            if prev_text.endswith("-"):
                # 行末ハイフンを削除して連結
                merged_parts[-1] = prev_text[:-1] + line_text
            else:
                # スペースで連結
                merged_parts[-1] = prev_text + " " + line_text
        else:
            merged_parts.append(line_text)

    return merged_parts[0] if merged_parts else ""
```

#### 5.1.7 Paragraph 生成

```python
def _create_paragraph(
    self,
    lines: list[list[TextObject]],
    page_number: int,
    para_index: int,
) -> Paragraph:
    """行グループから Paragraph を生成."""
    # 全 TextObject ID を収集
    text_object_ids = [obj.id for line in lines for obj in line]

    # 最初の行の最初の TextObject から anchor 情報を取得
    anchor_obj = lines[0][0]

    # テキストをマージ
    merged_text = self._merge_lines_to_text(lines)

    return Paragraph(
        id=f"para_p{page_number}_i{para_index}",
        page_number=page_number,
        text=merged_text,
        text_object_ids=text_object_ids,
        anchor_bbox=anchor_obj.bbox,
        anchor_font=anchor_obj.font,
        anchor_transform=anchor_obj.transform,
    )
```

#### 5.1.8 設定パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `line_y_tolerance` | 3.0 pt | 同一行判定の Y 許容差 |
| `paragraph_gap_threshold` | 1.5 | 段落境界判定の行間閾値（行高さの倍率） |
| `x_overlap_ratio` | 0.3 | 同一列判定に必要な X overlap 比率 |

#### 5.1.9 エッジケース対応

| ケース | 対応 |
|--------|------|
| 単一行段落 | そのまま 1 行の Paragraph を生成 |
| 空白のみの行 | フィルタリングで除外 |
| 特殊文字のみ | フィルタリングで除外 |
| 多段組の途中改ページ | 列分離後、各列を独立処理 |

### 5.2 FontSizeAdjuster

**ファイル**: `src/pdf_translator/core/font_adjuster.py`

#### 5.2.1 目的

翻訳後テキストが元の BBox に収まるようフォントサイズを調整する。

#### 5.2.2 API 設計

```python
class FontSizeAdjuster:
    def __init__(
        self,
        min_font_size: float = 6.0,
        font_size_decrement: float = 0.1,
    ) -> None:
        """FontSizeAdjuster を初期化.

        Args:
            min_font_size: 最小フォントサイズ（pt）
            font_size_decrement: 縮小ステップ（pt）
        """
        ...

    def calculate_font_size(
        self,
        text: str,
        bbox: BBox,
        original_font_size: float,
        target_lang: str,
    ) -> float:
        """テキストが bbox に収まるフォントサイズを計算.

        Args:
            text: 翻訳後テキスト
            bbox: 元の TextObject の BBox
            original_font_size: 元のフォントサイズ
            target_lang: 翻訳先言語（文字幅推定に使用）

        Returns:
            調整後のフォントサイズ（min_font_size 以上）
        """
        ...
```

#### 5.2.3 前提条件

**重要**: 現在の `PDFProcessor.insert_text_object()` は**自動改行しない**。
したがって、初期実装では「bbox 幅に収まるまで縮小」を中心に設計する。

- 高さ/行数計算は、実際の改行処理が入ってから対応
- 現段階では単一行として扱い、幅に収まるかを判定

#### 5.2.3 定数

```python
FONT_SIZE_DECREMENT = 0.1   # 縮小ステップ（pt）
MIN_FONT_SIZE = 6.0         # 最小フォントサイズ（pt）
```

#### 5.2.4 アルゴリズム（幅ベース・初期実装）

```
入力: text, bbox, original_font_size, target_lang
出力: adjusted_font_size

1. font_size = original_font_size
2. While font_size >= MIN_FONT_SIZE:
   a. char_width = estimate_char_width(font_size, target_lang)
   b. text_width = len(text) * char_width
   c. If text_width <= bbox.width:
      return font_size
   f. font_size -= FONT_SIZE_DECREMENT
3. return MIN_FONT_SIZE  # 収まらない場合は最小サイズ
```

**将来拡張**: 改行対応が入った場合は以下のアルゴリズムに切り替え:
```
1. font_size = original_font_size
2. While font_size >= MIN_FONT_SIZE:
   a. char_width = estimate_char_width(font_size, target_lang)
   b. chars_per_line = bbox.width / char_width
   c. lines_available = bbox.height / (font_size * LINE_HEIGHT_FACTOR)
   d. capacity = chars_per_line * lines_available
   e. If len(text) <= capacity:
      return font_size
   f. font_size -= FONT_SIZE_DECREMENT
3. return MIN_FONT_SIZE
```

#### 5.2.5 文字幅推定

```python
def _estimate_char_width(self, font_size: float, target_lang: str) -> float:
    if target_lang in ("ja", "zh", "ko"):
        # CJK: ほぼ正方形
        return font_size * 0.9
    else:
        # Latin: 平均的に狭い
        return font_size * 0.55
```

### 5.3 TranslationPipeline

**ファイル**: `src/pdf_translator/pipeline/translation_pipeline.py`

#### 5.3.1 クラス設計

```python
class TranslationPipeline:
    """PDF翻訳パイプライン.

    ワークフロー:
    1. PDFからテキスト抽出
    2. レイアウト解析（asyncio.to_thread 経由）
    3. テキストとレイアウトのマッチング
    4. 翻訳対象フィルタリング + 読み順ソート
    5. バッチ翻訳
    6. フォントサイズ調整
    7. PDFに適用
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

#### 5.3.2 output_path 指定時の動作

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

#### 5.3.3 LayoutAnalyzer の非同期対応

`LayoutAnalyzer.analyze_all()` は同期メソッドだが、Pipeline は非同期。
CPU バウンドの処理をブロックしないよう `asyncio.to_thread()` を使用する。

```python
async def _stage_analyze(self, pdf_path: Path) -> dict[int, list[LayoutBlock]]:
    if not self._config.use_layout_analysis:
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

#### 5.3.4 use_layout_analysis=False の動作

レイアウト解析を無効化した場合の動作を定義する。

**動作**: すべての TextObject を翻訳対象（`ProjectCategory.TEXT`）として扱い、警告ログを出力する。

**ユースケース**:
- レイアウト解析が不要な単純な PDF（テキストのみ）
- レイアウト解析が失敗した場合のフォールバック
- 以前の実装（`_archive`）との整合性

**実装イメージ**:
```python
if not self._config.use_layout_analysis:
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

#### 5.3.5 bytes 入力時の注意

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

#### 5.3.6 ステージ構成

| ステージ (stage値) | メソッド | 処理内容 |
|-------------------|---------|---------|
| `extract` | `_stage_extract` | PDFからテキスト抽出 |
| `analyze` | `_stage_analyze` | レイアウト解析＋マッチング |
| `merge` | `_stage_merge` | 翻訳対象フィルタリング + 読み順ソート |
| `translate` | `_stage_translate` | バッチ翻訳（リトライあり） |
| `font_adjust` | `_stage_font_adjust` | フォントサイズ調整 |
| `apply` | `_stage_apply` | PDFに適用 |

> **NOTE**: ステージ名は §4.3 の ProgressCallback stage 値と一致させている。

#### 5.3.7 リトライロジック

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

### 5.4 エラーハンドリング

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

#### 5.4.1 エラー発生時の動作（v1: 全体失敗）

**v1 方針**: エラー発生時は全体失敗とする。部分的成功（一部ページのみ翻訳）は v2 以降で検討。

**理由**:
1. **シンプルさ**: 部分的成功の処理は複雑（どのページが失敗したか追跡、UI への通知など）
2. **ユーザー期待**: 翻訳を依頼したら全ページ翻訳されることを期待する
3. **リトライで回復可能**: 一時的エラー（レート制限、ネットワーク）なら再実行で成功する
4. **デバッグ容易性**: 全体失敗のほうが問題の切り分けが容易

**v1 の動作**:
- リトライ上限に達したら `PipelineError` を raise
- エラーメッセージに失敗したステージと原因を含める

**v2 での拡張案**:
```python
@dataclass
class TranslationResult:
    pdf_bytes: bytes
    stats: Optional[dict[str, Any]] = None
    # v2 で追加予定:
    # failed_pages: list[int] = field(default_factory=list)
    # partial_success: bool = False
```

---

## 6. 実装フェーズ

### Phase 1: データモデル拡張

**成果物**:
- `src/pdf_translator/core/models.py` に Paragraph 追加

**タスク**:
1. Paragraph dataclass 実装
2. 既存テストが壊れていないことを確認

### Phase 2: TextMerger（段落マージ）

**成果物**:
- `src/pdf_translator/core/text_merger.py`
- `tests/test_text_merger.py`

**タスク**:
1. TextMerger クラス実装
2. 行クラスタリング実装
3. 列分離（多段組対応）実装
4. 段落境界検出実装
5. ハイフネーション結合実装
6. Paragraph 生成実装
7. ユニットテスト作成

### Phase 3: FontSizeAdjuster（フォント調整）

**成果物**:
- `src/pdf_translator/core/font_adjuster.py`
- `tests/test_font_adjuster.py`

**タスク**:
1. FontSizeAdjuster クラス実装
2. 文字幅推定実装
3. フォントサイズ縮小アルゴリズム実装
4. ユニットテスト作成

### Phase 4: PDFProcessor 拡張

**成果物**:
- `src/pdf_translator/core/pdf_processor.py` に apply_paragraphs() 追加

**タスク**:
1. apply_paragraphs() メソッド実装
2. 既存 apply() との整合性確認
3. ユニットテスト作成

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
4. 各ステージ実装（Paragraph ベース）
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

---

## 7. テスト計画

### 7.1 TextMerger テスト

```python
class TestTextMerger:
    # 行クラスタリング
    def test_cluster_single_line(self): ...
    def test_cluster_multiple_lines(self): ...
    def test_cluster_with_y_tolerance(self): ...  # 微小 Y ブレ対策

    # カテゴリフィルタリング
    def test_filters_translatable_categories(self): ...
    def test_excludes_formula_and_table(self): ...

    # 多段組対応
    def test_separate_two_columns(self): ...
    def test_single_column_unchanged(self): ...
    def test_column_reading_order(self): ...  # 左列 → 右列の順

    # 段落境界検出
    def test_paragraph_boundary_by_gap(self): ...
    def test_paragraph_boundary_by_indent(self): ...
    def test_paragraph_boundary_by_sentence_end(self): ...

    # ハイフネーション結合
    def test_hyphenation_merge(self): ...  # "founda-" + "tion" → "foundation"
    def test_no_merge_without_hyphen(self): ...
    def test_preserve_compound_hyphen(self): ...  # "state-of-the-art" は保持

    # Paragraph 生成
    def test_paragraph_has_all_text_object_ids(self): ...
    def test_paragraph_anchor_from_first_line(self): ...
    def test_paragraph_merged_text_correct(self): ...
```

### 7.2 FontSizeAdjuster テスト

```python
class TestFontSizeAdjuster:
    def test_text_fits_original_size(self): ...
    def test_text_requires_reduction(self): ...
    def test_minimum_font_size_limit(self): ...
    def test_cjk_character_width(self): ...
    def test_latin_character_width(self): ...
    def test_long_paragraph_sizing(self): ...  # 段落全体が収まるサイズ計算
```

### 7.3 PDFProcessor 拡張テスト

```python
class TestPDFProcessorParagraphs:
    def test_apply_paragraphs_removes_all_source_objects(self): ...
    def test_apply_paragraphs_inserts_translated_text(self): ...
    def test_apply_paragraphs_uses_anchor_position(self): ...
    def test_apply_paragraphs_multiple_paragraphs(self): ...
```

### 7.4 TranslationPipeline テスト

```python
class TestTranslationPipeline:
    async def test_full_pipeline_mocked_translator(self): ...
    async def test_progress_callback_invoked(self): ...
    async def test_error_handling_extraction_failure(self): ...
    async def test_retry_on_transient_error(self): ...
    async def test_paragraph_based_translation(self): ...  # 段落単位翻訳
    async def test_hyphenation_handled_correctly(self): ...  # E2E ハイフネーション
```

### 7.5 E2E テスト（サンプルPDF）

```python
class TestE2ETranslation:
    async def test_llama_paper_abstract_translation(self): ...
    async def test_two_column_layout_preserved(self): ...
    async def test_hyphenated_words_merged(self): ...
    async def test_paragraph_context_preserved(self): ...
```

---

## 8. リスクと対策

| リスク | 影響度 | 対策 |
|--------|-------|------|
| 段落境界の誤検出 | 高 | `paragraph_gap_threshold` パラメータで調整、E2E テストで検証 |
| 多段組の列分離ミス | 高 | `x_overlap_ratio` パラメータで調整、2段組論文でテスト |
| ハイフネーション誤結合 | 中 | 複合語（state-of-the-art）を保持するロジック実装 |
| 複雑なレイアウトでの読み順誤り | 中 | `line_y_tolerance` 等のパラメータ調整で対応 |
| フォント幅推定の不正確さ | 中 | 保守的な推定値を使用 |
| 翻訳APIのレート制限 | 低 | 指数バックオフ、バッチサイズ調整 |
| 大規模PDFでのメモリ使用 | 低 | ページ単位処理 |
| CJK 言語への翻訳時の文字化け | 中 | v1.1 で CJK フォント対応予定（§1.4.1 参照） |

---

## 9. 設定オプション

**設定の方針**: すべての設定パラメータは `PipelineConfig` に統合する。
個別コンポーネント（TextMerger, FontSizeAdjuster）は `PipelineConfig` から必要な値を受け取る。

```python
@dataclass
class PipelineConfig:
    """パイプライン設定.

    すべての設定パラメータを統合管理する。
    個別コンポーネントはここから必要な値を受け取る。
    """

    # 翻訳設定
    source_lang: str = "en"
    target_lang: str = "ja"

    # レイアウト解析
    use_layout_analysis: bool = True
    layout_containment_threshold: float = 0.5

    # テキストマージ（TextMerger 用）
    line_y_tolerance: float = 3.0           # 同一行判定の Y 許容差（pt）
    paragraph_gap_threshold: float = 1.5    # 段落境界の行間閾値（行高さの倍率）
    x_overlap_ratio: float = 0.3            # 同一列判定に必要な X overlap 比率

    # フォント調整（FontSizeAdjuster 用）
    min_font_size: float = 6.0
    font_size_decrement: float = 0.1

    # 翻訳リトライ
    max_retries: int = 3
    retry_delay: float = 1.0

    # CJK フォント（v1.1 対応予定、§1.4.1 参照）
    # cjk_font_path: Optional[Path] = None
```

**使用例**:
```python
config = PipelineConfig(target_lang="ja", paragraph_gap_threshold=2.0)

# TextMerger に渡す
merger = TextMerger(
    line_y_tolerance=config.line_y_tolerance,
    paragraph_gap_threshold=config.paragraph_gap_threshold,
    x_overlap_ratio=config.x_overlap_ratio,
)

# FontSizeAdjuster に渡す
adjuster = FontSizeAdjuster(
    min_font_size=config.min_font_size,
    font_size_decrement=config.font_size_decrement,
)
```

---

## 10. 出力ファイル一覧

| ファイル | 種別 |
|---------|------|
| `src/pdf_translator/core/models.py` | 変更（`Paragraph` dataclass 追加） |
| `src/pdf_translator/core/text_merger.py` | 新規（段落マージ機能） |
| `src/pdf_translator/core/font_adjuster.py` | 新規 |
| `src/pdf_translator/core/pdf_processor.py` | 変更（`to_bytes()`, `apply_paragraphs()` 追加） |
| `src/pdf_translator/pipeline/__init__.py` | 新規 |
| `src/pdf_translator/pipeline/translation_pipeline.py` | 新規（PipelineConfig, TranslationResult 含む） |
| `src/pdf_translator/pipeline/progress.py` | 新規 |
| `src/pdf_translator/pipeline/errors.py` | 新規 |
| `tests/test_text_merger.py` | 新規 |
| `tests/test_font_adjuster.py` | 新規 |
| `tests/test_translation_pipeline.py` | 新規 |
