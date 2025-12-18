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

#### 1.4.2 クロスブロック翻訳未対応

**制約**: 翻訳は TextObject 単位で 1:1 に行う。複数 TextObject を結合した翻訳は行わない。

**理由**:
- 文字数比率による分配は翻訳の伸縮・語順変化で破綻しやすい
- 安全側の設計として v1 では単純な 1:1 翻訳を採用

**v2 以降の対応予定**:
- LLM バックエンドでの structured output を活用したクロスブロック翻訳
- 文脈を考慮した翻訳品質向上

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
┌─────────────────────────────────┐
│ PDFProcessor.extract_text_objects() │
│ → PDFDocument (TextObjects)         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ LayoutAnalyzer.analyze_all()        │
│ → dict[int, list[LayoutBlock]]      │
│ ※ asyncio.to_thread() 経由で呼び出し │
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
│ TextMerger.merge()              [NEW] │
│ → list[TextObject] (読み順ソート済み) │
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
│ FontSizeAdjuster.calculate_font_size() [NEW] │
│ → 調整済みフォントサイズ                      │
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

### 4.1 v1 設計方針: TextGroup は実装しない

#### 4.1.1 翻訳単位の設計方針（v1: 1:1 翻訳）

`PDFProcessor.apply()` は `TextObject` 単位で再挿入を行うため、パイプラインの最終成果は TextObject 単位に書き戻せる形である必要がある。

**v1 設計方針（YAGNI）**:

v1 では 1:1 翻訳のため、`TextGroup` dataclass は実装しない。TextMerger は読み順でソートした `list[TextObject]` を返す。

**理由**:
1. **YAGNI**: v1 では TextGroup を使う場面がない（1:1 翻訳のため）
2. **シンプルさ**: 不要な抽象化を避ける
3. **将来の柔軟性**: v2 で本当に必要になった際に、適切な設計で追加できる

```python
# v1: TextMerger は読み順ソート済み list[TextObject] を返す
sorted_objects = merger.merge(text_objects, categories, page_num)

# 1:1 翻訳（TextObject 単位）
texts_to_translate = [obj.text for obj in sorted_objects]
translated_texts = await translator.translate_batch(
    texts_to_translate, source_lang, target_lang
)

# 結果を TextObject に書き戻し
for obj, translated in zip(sorted_objects, translated_texts):
    obj.text = translated
```

**将来拡張（v2 以降）**:

クロスブロックで文脈を活かす設計は、以下のタイミングで再検討:
- LLM バックエンドでの structured output 対応
- 翻訳品質の評価基盤整備後

その際に TextGroup dataclass を導入し、以下の方式を検討:
```python
# v2（将来）: TextGroup を導入し、LLM での structured output を活用
@dataclass
class TextGroup:
    id: str
    text_object_ids: list[str]
    page_num: int

response = await llm_translator.translate_with_structure(
    texts=original_texts,
    instruction="各テキストを個別に翻訳し、JSON配列で返してください"
)
# response: ["翻訳1", "翻訳2", ...]
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

翻訳対象の TextObject を読み順でソートする。

**責務の集約**: TextMerger 内部で翻訳対象のフィルタリングを行う。呼び出し側は `categories` だけ渡せばよい。

**API 設計**:
```python
class TextMerger:
    def __init__(
        self,
        line_y_tolerance: float = 3.0,
        merge_threshold_x: float = 20.0,
        merge_threshold_y: float = 5.0,
        x_overlap_ratio: float = 0.5,
    ) -> None:
        """TextMerger を初期化.

        Args:
            line_y_tolerance: 同一行判定の y 許容差（pt）
            merge_threshold_x: 同一行内の x gap 閾値（pt）
            merge_threshold_y: 次行への y gap 閾値（pt）
            x_overlap_ratio: 次行結合に必要な x overlap 比率
        """
        ...

    def merge(
        self,
        text_objects: list[TextObject],
        categories: dict[str, ProjectCategory],
    ) -> list[TextObject]:
        """翻訳対象 TextObject を読み順でソートして返す.

        内部で翻訳対象のフィルタリングを行い、読み順でソートする。

        Args:
            text_objects: ページ内の全 TextObject
            categories: TextObject.id → ProjectCategory のマッピング

        Returns:
            読み順でソートされた翻訳対象 TextObject のリスト
        """
        # 内部で翻訳対象フィルタリング
        translatable = [
            obj for obj in text_objects
            if categories.get(obj.id) in TRANSLATABLE_CATEGORIES
        ]
        # 読み順ソート
        ...
```

#### 5.1.2 アルゴリズム

```
入力: list[TextObject], dict[str, ProjectCategory]
出力: list[TextObject] (読み順ソート済み)

1. 内部で翻訳対象カテゴリ（TEXT, TITLE, CAPTION）のみフィルタ
2. 行クラスタリング:
   - y 座標を line_y_tolerance で丸めて行グループを形成
   - 同一行内は x0 昇順でソート
3. 行間ソート:
   - 行グループを y1 降順でソート（上から下へ）
4. フラット化:
   - 行グループを展開してソート済みリストを作成
5. list[TextObject] 返却
```

> **NOTE**: v1 では「結合」は行わず、読み順ソートのみ。「merge」という名前だが、
> v2 でクロスブロック翻訳を実装する際の拡張を見据えた命名。

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

    # 設定された比率以上のオーバーラップを要求
    return x_overlap >= min_width * self._x_overlap_ratio
```

#### 5.1.6 設定パラメータ

TextMerger の設定パラメータは `PipelineConfig` に統合（§9 参照）。

| パラメータ | デフォルト | 説明 | v1 での用途 |
|-----------|-----------|------|-------------|
| `line_y_tolerance` | 3.0 pt | 同一行判定の y 許容差 | 行クラスタリング |
| `merge_threshold_x` | 20.0 pt | 同一行内の x gap 閾値 | 将来用（v2 結合判定） |
| `merge_threshold_y` | 5.0 pt | 次行への y gap 閾値 | 将来用（v2 結合判定） |
| `x_overlap_ratio` | 0.5 | 次行結合に必要な x overlap 比率 | 将来用（v2 結合判定） |

> **NOTE**: v1 では `line_y_tolerance` のみ実際に使用されます。他のパラメータは v2 でのクロスブロック結合に備えて定義しています。実装時に不要と判断した場合は削除可能です。

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

### Phase 1: TextMerger（読み順ソート）

**成果物**:
- `src/pdf_translator/core/text_merger.py`
- `tests/test_text_merger.py`

**タスク**:
1. TextMerger クラス実装
2. 翻訳対象フィルタリング実装
3. 読み順ソート実装（行クラスタリング + x ソート）
4. 文末判定実装
5. ユニットテスト作成

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
    # 読み順ソート
    def test_reading_order_single_line(self): ...
    def test_reading_order_multiple_lines(self): ...
    def test_reading_order_with_y_tolerance(self): ...  # 微小 y ブレ対策

    # カテゴリフィルタリング
    def test_filters_translatable_categories(self): ...
    def test_excludes_formula_and_table(self): ...

    # ページ単位処理
    def test_preserves_page_order(self): ...

    # 文末判定（将来の結合用だが v1 でも実装）
    def test_sentence_end_detection_english(self): ...
    def test_sentence_end_detection_japanese(self): ...
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
| `src/pdf_translator/core/pdf_processor.py` | 変更（`to_bytes()` 追加） |
| `src/pdf_translator/pipeline/__init__.py` | 新規 |
| `src/pdf_translator/pipeline/translation_pipeline.py` | 新規（PipelineConfig, TranslationResult 含む） |
| `src/pdf_translator/pipeline/progress.py` | 新規 |
| `src/pdf_translator/pipeline/errors.py` | 新規 |
| `tests/test_text_merger.py` | 新規 |
| `tests/test_font_adjuster.py` | 新規 |
| `tests/test_translation_pipeline.py` | 新規 |
