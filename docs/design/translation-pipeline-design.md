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

---

## 3. アーキテクチャ

### 3.1 データフロー

```
PDF Input
    │
    ▼
┌─────────────────────────────────┐
│ PDFProcessor.extract_text_objects() │
│ → PDFDocument (TextObjects)         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ LayoutAnalyzer.analyze_all()        │
│ → dict[int, list[LayoutBlock]]      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ layout_utils.match_text_with_layout() │
│ → dict[str, ProjectCategory]          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ TextMerger.merge_text_objects()  [NEW] │
│ → list[TextGroup]                      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ TranslatorBackend.translate_batch()   │
│ → 翻訳済みテキスト                     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ FontSizeAdjuster.calculate_fit_font_size() [NEW] │
│ → 調整済みフォントサイズ                          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ PDFProcessor.apply()                  │
│ → 翻訳済みPDF                          │
└─────────────────────────────────┘
    │
    ▼
Output PDF
```

### 3.2 ファイル構成

```
src/pdf_translator/
├── core/
│   ├── models.py             # 既存 + TextGroup追加
│   ├── pdf_processor.py      # 既存
│   ├── layout_analyzer.py    # 既存
│   ├── layout_utils.py       # 既存
│   ├── text_merger.py        # 新規: テキスト結合
│   └── font_adjuster.py      # 新規: フォント調整
├── pipeline/                  # 新規ディレクトリ
│   ├── __init__.py
│   ├── translation_pipeline.py
│   ├── progress.py
│   └── errors.py
└── translators/              # 既存
```

---

## 4. データモデル

### 4.1 TextGroup（新規）

`core/models.py` に追加:

```python
@dataclass
class TextGroup:
    """読み順でソートされた TextObject のグループ.

    v1 では「読み順ソート・翻訳対象のバッチ化」のために使用。
    翻訳は TextObject 単位で 1:1 に行い、結果は直接 TextObject.text に書き戻す。

    Attributes:
        id: グループ一意識別子
        text_object_ids: 構成する TextObject の ID リスト（読み順）
        page_num: ページ番号
        category: ProjectCategory
    """
    id: str
    text_object_ids: list[str]
    page_num: int
    category: ProjectCategory
```

#### 4.1.1 翻訳単位の設計方針（v1: 1:1 翻訳）

`PDFProcessor.apply()` は `TextObject` 単位で再挿入を行うため、パイプラインの最終成果は TextObject 単位に書き戻せる形である必要がある。

**v1 設計方針（安全側）**:

翻訳結果の分配（スライス）は、翻訳による伸縮・語順変化・記号単体 TextObject（`,` や `∗` など）混在で破綻しやすい。そのため、v1 では以下の方針を採用する:

1. **TextGroup は「読み順ソート・翻訳対象のバッチ化」までに留める**
2. **翻訳は TextObject 単位で 1:1 に行う**（`translate_batch(texts=original_texts)`）
3. **翻訳結果はそのまま各 `TextObject.text` に書き戻す**

```python
# v1: 1:1 翻訳（TextObject 単位）
texts_to_translate = [obj.text for obj in translatable_objects]
translated_texts = await translator.translate_batch(
    texts_to_translate, source_lang, target_lang
)

# 結果を TextObject に書き戻し
for obj, translated in zip(translatable_objects, translated_texts):
    obj.text = translated
```

**TextGroup の役割（v1）**:
- 読み順のソート
- 翻訳対象カテゴリのフィルタリング
- 将来のクロスブロック翻訳のための構造維持

**将来拡張（v2 以降）**:

クロスブロックで文脈を活かす設計は、以下のタイミングで再検討:
- LLM バックエンドでの structured output 対応
- 翻訳品質の評価基盤整備後

その際は以下の方式を検討:
```python
# v2（将来）: LLM での structured output を活用
response = await llm_translator.translate_with_structure(
    texts=original_texts,
    instruction="各テキストを個別に翻訳し、JSON配列で返してください"
)
# response: ["翻訳1", "翻訳2", ...]
```

### 4.2 TranslationResult（新規）

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

**ステージ一覧**:

| stage | 説明 |
|-------|------|
| `extract` | PDFからテキスト抽出 |
| `analyze` | レイアウト解析 |
| `merge` | テキストグループ化（読み順ソート） |
| `translate` | バッチ翻訳 |
| `font_adjust` | フォントサイズ調整 |
| `apply` | PDFに適用 |

---

## 5. コンポーネント設計

### 5.1 TextMerger

**ファイル**: `src/pdf_translator/core/text_merger.py`

#### 5.1.1 目的

TextObject を読み順でソートし、文の連続性を保ってグループ化する。

#### 5.1.2 アルゴリズム

```
入力: list[TextObject], dict[str, ProjectCategory], page_num
出力: list[TextGroup]

1. 翻訳対象カテゴリ（TEXT, TITLE, CAPTION）のみフィルタ
2. 行クラスタリング:
   - y 座標を line_y_tolerance で丸めて行グループを形成
   - 同一行内は x0 昇順でソート
3. 行間ソート:
   - 行グループを y1 降順でソート（上から下へ）
4. 結合ループ:
   a. current_group = 空
   b. For each text_object:
      - current_group が空 → 新グループ開始
      - should_merge(last, current) → グループに追加
      - else → グループ確定、新グループ開始
   c. 最後のグループ確定
5. TextGroup リスト返却
```

#### 5.1.3 行クラスタリング（y 座標の微小ブレ対策）

PDF 抽出では同一行が複数 TextObject に分かれ、y 座標が微小にブレることが一般的。
これに対応するため、`line_y_tolerance` を導入して行クラスタを形成する。

```python
DEFAULT_LINE_Y_TOLERANCE = 3.0  # pt

def _cluster_by_line(
    self,
    text_objects: list[TextObject],
) -> list[list[TextObject]]:
    """y 座標でクラスタリングして行グループを形成."""
    if not text_objects:
        return []

    # y1（上端）でソート
    sorted_objs = sorted(text_objects, key=lambda o: -o.bbox.y1)

    lines: list[list[TextObject]] = []
    current_line: list[TextObject] = [sorted_objs[0]]
    current_y = sorted_objs[0].bbox.y1

    for obj in sorted_objs[1:]:
        if abs(obj.bbox.y1 - current_y) <= self._line_y_tolerance:
            # 同一行
            current_line.append(obj)
        else:
            # 新しい行
            current_line.sort(key=lambda o: o.bbox.x0)  # x0 昇順
            lines.append(current_line)
            current_line = [obj]
            current_y = obj.bbox.y1

    if current_line:
        current_line.sort(key=lambda o: o.bbox.x0)
        lines.append(current_line)

    return lines
```

#### 5.1.4 文末判定

```python
SENTENCE_TERMINALS = frozenset({'.', '!', '?', ':', ';'})
CJK_TERMINALS = frozenset({'。', '！', '？', '：', '；'})

def _is_sentence_end(self, text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] in self.SENTENCE_TERMINALS | self.CJK_TERMINALS
```

#### 5.1.5 結合判定（2系統）

PDF 抽出の現実に合わせ、結合判定を 2 系統に分ける:
- **(a) 同一行内**: x gap で結合判定
- **(b) 次行への結合**: y gap + 列の x-overlap を要求

```python
def _should_merge_same_line(
    self,
    current: TextObject,
    next_obj: TextObject,
) -> bool:
    """同一行内の結合判定（x gap ベース）."""
    # 文末句読点があれば結合しない
    if self._is_sentence_end(current.text):
        return False

    # x 方向の距離が閾値以内
    x_gap = next_obj.bbox.x0 - current.bbox.x1
    return x_gap <= self._merge_threshold_x

def _should_merge_next_line(
    self,
    current: TextObject,
    next_obj: TextObject,
) -> bool:
    """次行への結合判定（y gap + x overlap ベース）."""
    # 文末句読点があれば結合しない
    if self._is_sentence_end(current.text):
        return False

    # y 方向の距離が閾値以内
    y_gap = current.bbox.y0 - next_obj.bbox.y1
    if y_gap > self._merge_threshold_y:
        return False

    # x 方向のオーバーラップを要求（多段組の誤結合防止）
    x_overlap = min(current.bbox.x1, next_obj.bbox.x1) - max(current.bbox.x0, next_obj.bbox.x0)
    min_width = min(current.bbox.width, next_obj.bbox.width)

    # 少なくとも 50% のオーバーラップを要求
    return x_overlap >= min_width * 0.5
```

#### 5.1.6 設定パラメータ

TextMerger の設定パラメータは `PipelineConfig` に統合（§9 参照）。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `line_y_tolerance` | 3.0 pt | 同一行判定の y 許容差 |
| `merge_threshold_x` | 20.0 pt | 同一行内の x gap 閾値 |
| `merge_threshold_y` | 5.0 pt | 次行への y gap 閾値 |
| `x_overlap_ratio` | 0.5 | 次行結合に必要な x overlap 比率 |

### 5.2 FontSizeAdjuster

**ファイル**: `src/pdf_translator/core/font_adjuster.py`

#### 5.2.1 目的

翻訳後テキストが元の BBox に収まるようフォントサイズを調整する。

#### 5.2.2 前提条件

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
    2. レイアウト解析
    3. テキストとレイアウトのマッチング
    4. 翻訳対象フィルタリング
    5. テキストグループ化
    6. バッチ翻訳
    7. フォントサイズ調整
    8. PDFに適用
    """

    def __init__(
        self,
        translator: TranslatorBackend,
        source_lang: str = "en",
        target_lang: str = "ja",
        use_layout_analysis: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ): ...

    async def translate(
        self,
        pdf_source: Union[Path, str, bytes],
        output_path: Optional[Path] = None,
    ) -> TranslationResult: ...
```

#### 5.3.2 bytes 入力時の注意

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

#### 5.3.3 ステージ構成

| ステージ | メソッド | 処理内容 |
|---------|---------|---------|
| Extract | `_stage_extract` | PDFからテキスト抽出 |
| Analyze | `_stage_analyze` | レイアウト解析＋マッチング |
| Merge | `_stage_merge` | TextGroup作成 |
| Translate | `_stage_translate` | バッチ翻訳（リトライあり） |
| Apply | `_stage_apply` | PDFに適用 |

#### 5.3.4 リトライロジック

**リトライ対象の例外**:
- `TranslationError`: リトライ対象（API 障害、レート制限など一時的エラー）
- `ConfigurationError`: **リトライ対象外**（API キー不正、設定エラーなど即 fail）

```python
from pdf_translator.translators.base import ConfigurationError, TranslationError

async def _translate_with_retry(
    self,
    texts: list[str],
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> list[str]:
    for attempt in range(max_retries + 1):
        try:
            return await self._translator.translate_batch(
                texts, self._source_lang, self._target_lang
            )
        except ConfigurationError:
            # 設定エラーはリトライせず即 fail
            raise
        except TranslationError as e:
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)  # 指数バックオフ
                await asyncio.sleep(delay)
            else:
                raise PipelineError(
                    f"Translation failed after {max_retries} retries",
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

---

## 6. 実装フェーズ

### Phase 1: TextMerger（テキスト結合）

**成果物**:
- `src/pdf_translator/core/text_merger.py`
- `src/pdf_translator/core/models.py` への TextGroup 追加
- `tests/test_text_merger.py`

**タスク**:
1. TextGroup dataclass 追加
2. TextMerger クラス実装
3. 読み順ソート実装
4. 文末判定実装
5. 結合ロジック実装
6. ユニットテスト作成

### Phase 2: FontSizeAdjuster（フォント調整）

**成果物**:
- `src/pdf_translator/core/font_adjuster.py`
- `tests/test_font_adjuster.py`

**タスク**:
1. FontSizeAdjuster クラス実装
2. 文字幅推定実装
3. フォントサイズ縮小アルゴリズム実装
4. ユニットテスト作成

### Phase 3: Pipeline（パイプライン統合）

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
4. 各ステージ実装
5. リトライロジック実装
6. 統合テスト作成

### Phase 4: テスト・検証

**成果物**:
- 全テストの実行確認
- サンプルPDFでの動作確認

**タスク**:
1. ユニットテスト全パス確認
2. mypy strict 確認
3. ruff lint 確認
4. サンプルPDFで E2E テスト

---

## 7. テスト計画

### 7.1 TextMerger テスト

```python
class TestTextMerger:
    def test_reading_order_sorting(self): ...
    def test_sentence_end_detection_english(self): ...
    def test_sentence_end_detection_japanese(self): ...
    def test_cross_block_merging(self): ...
    def test_no_merge_at_sentence_boundary(self): ...
    def test_single_object_becomes_single_group(self): ...
```

### 7.2 FontSizeAdjuster テスト

```python
class TestFontSizeAdjuster:
    def test_text_fits_original_size(self): ...
    def test_text_requires_reduction(self): ...
    def test_minimum_font_size_limit(self): ...
    def test_cjk_character_width(self): ...
    def test_latin_character_width(self): ...
```

### 7.3 TranslationPipeline テスト

```python
class TestTranslationPipeline:
    async def test_full_pipeline_mocked_translator(self): ...
    async def test_progress_callback_invoked(self): ...
    async def test_error_handling_extraction_failure(self): ...
    async def test_retry_on_transient_error(self): ...
```

---

## 8. リスクと対策

| リスク | 影響度 | 対策 |
|--------|-------|------|
| 複雑なレイアウトでの読み順誤り | 中 | 設定でマージ無効化可能に |
| フォント幅推定の不正確さ | 中 | 保守的な推定値を使用 |
| 翻訳APIのレート制限 | 低 | 指数バックオフ、バッチサイズ調整 |
| 大規模PDFでのメモリ使用 | 低 | ページ単位処理 |

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

    # テキスト結合（TextMerger 用）
    line_y_tolerance: float = 3.0    # 同一行判定の y 許容差（pt）
    merge_threshold_x: float = 20.0  # 同一行内の x gap 閾値（pt）
    merge_threshold_y: float = 5.0   # 次行への y gap 閾値（pt）
    x_overlap_ratio: float = 0.5     # 次行結合に必要な x overlap 比率

    # フォント調整（FontSizeAdjuster 用）
    min_font_size: float = 6.0
    font_size_decrement: float = 0.1

    # 翻訳リトライ
    max_retries: int = 3
    retry_delay: float = 1.0
```

**使用例**:
```python
config = PipelineConfig(target_lang="ja", line_y_tolerance=5.0)

# TextMerger に渡す
merger = TextMerger(
    line_y_tolerance=config.line_y_tolerance,
    merge_threshold_x=config.merge_threshold_x,
    merge_threshold_y=config.merge_threshold_y,
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
| `src/pdf_translator/core/text_merger.py` | 新規 |
| `src/pdf_translator/core/font_adjuster.py` | 新規 |
| `src/pdf_translator/core/models.py` | 変更（TextGroup追加） |
| `src/pdf_translator/pipeline/__init__.py` | 新規 |
| `src/pdf_translator/pipeline/translation_pipeline.py` | 新規 |
| `src/pdf_translator/pipeline/progress.py` | 新規 |
| `src/pdf_translator/pipeline/errors.py` | 新規 |
| `tests/test_text_merger.py` | 新規 |
| `tests/test_font_adjuster.py` | 新規 |
| `tests/test_translation_pipeline.py` | 新規 |
