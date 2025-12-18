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
    """文の連続性を保った TextObject のグループ.

    Attributes:
        id: グループ一意識別子
        text_object_ids: 構成する TextObject の ID リスト（読み順）
        merged_text: 結合されたテキスト
        bboxes: 各 TextObject の BBox リスト
        fonts: 各 TextObject の Font リスト
        colors: 各 TextObject の Color リスト
        page_num: ページ番号
        category: ProjectCategory
        translated_text: 翻訳後テキスト（翻訳前は None）
    """
    id: str
    text_object_ids: list[str]
    merged_text: str
    bboxes: list[BBox]
    fonts: list[Optional[Font]]
    colors: list[Optional[Color]]
    page_num: int
    category: ProjectCategory
    translated_text: Optional[str] = None
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
        stage: str,        # "extract", "analyze", "translate", "apply"
        current: int,      # 現在の処理数
        total: int,        # 総数
        message: str = "", # 状態メッセージ
    ) -> None: ...
```

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
2. 読み順でソート:
   - PDF座標系（左下原点）のため y1 降順、x0 昇順
   - key=lambda obj: (-obj.bbox.y1, obj.bbox.x0)
3. 結合ループ:
   a. current_group = 空
   b. For each text_object:
      - current_group が空 → 新グループ開始
      - should_merge(last, current) → グループに追加
      - else → グループ確定、新グループ開始
   c. 最後のグループ確定
4. TextGroup リスト返却
```

#### 5.1.3 文末判定

```python
SENTENCE_TERMINALS = frozenset({'.', '!', '?', ':', ';'})
CJK_TERMINALS = frozenset({'。', '！', '？', '：', '；'})

def _is_sentence_end(self, text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] in self.SENTENCE_TERMINALS | self.CJK_TERMINALS
```

#### 5.1.4 結合判定

```python
def _should_merge_with_next(
    self,
    current: TextObject,
    next_obj: TextObject,
) -> bool:
    # 文末句読点があれば結合しない
    if self._is_sentence_end(current.text):
        return False

    # Y方向の距離が閾値以内
    y_gap = abs(current.bbox.y0 - next_obj.bbox.y1)
    if y_gap > self._merge_threshold_y + current.bbox.height:
        return False

    return True
```

### 5.2 FontSizeAdjuster

**ファイル**: `src/pdf_translator/core/font_adjuster.py`

#### 5.2.1 目的

翻訳後テキストが元の BBox に収まるようフォントサイズを調整する。

#### 5.2.2 定数

```python
LINE_HEIGHT_FACTOR = 1.5    # 行高さ係数
FONT_SIZE_DECREMENT = 0.1   # 縮小ステップ（pt）
MIN_FONT_SIZE = 6.0         # 最小フォントサイズ（pt）
```

#### 5.2.3 アルゴリズム

```
入力: text, bbox, original_font_size, target_lang
出力: adjusted_font_size

1. font_size = original_font_size
2. While font_size >= MIN_FONT_SIZE:
   a. char_width = estimate_char_width(font_size, target_lang)
   b. chars_per_line = bbox.width / char_width
   c. lines_available = bbox.height / (font_size * LINE_HEIGHT_FACTOR)
   d. capacity = chars_per_line * lines_available
   e. If len(text) <= capacity:
      return font_size
   f. font_size -= FONT_SIZE_DECREMENT
3. return MIN_FONT_SIZE  # 収まらない場合は最小サイズ
```

#### 5.2.4 文字幅推定

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

#### 5.3.2 ステージ構成

| ステージ | メソッド | 処理内容 |
|---------|---------|---------|
| Extract | `_stage_extract` | PDFからテキスト抽出 |
| Analyze | `_stage_analyze` | レイアウト解析＋マッチング |
| Merge | `_stage_merge` | TextGroup作成 |
| Translate | `_stage_translate` | バッチ翻訳（リトライあり） |
| Apply | `_stage_apply` | PDFに適用 |

#### 5.3.3 リトライロジック

```python
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

```python
@dataclass
class PipelineConfig:
    """パイプライン設定."""

    # 翻訳設定
    source_lang: str = "en"
    target_lang: str = "ja"

    # レイアウト解析
    use_layout_analysis: bool = True
    layout_containment_threshold: float = 0.5

    # テキスト結合
    merge_cross_block: bool = True
    merge_threshold_y: float = 5.0

    # フォント調整
    min_font_size: float = 6.0
    font_size_decrement: float = 0.1
    line_height_factor: float = 1.5

    # 翻訳リトライ
    max_retries: int = 3
    retry_delay: float = 1.0
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
