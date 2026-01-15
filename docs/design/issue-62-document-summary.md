# Issue #62: Webサービス向けドキュメントサマリー機能

## 概要

PDF翻訳Webサービス構築に向けて、翻訳結果からドキュメントのサマリー情報（タイトル、Abstract、サムネイル、LLM要約）を抽出・出力する機能を追加する。

**関連 Issue**: [#62](https://github.com/Mega-Gorilla/pdf-translator/issues/62)

### 主要機能

| 機能 | 説明 |
|------|------|
| **メタデータ抽出** | タイトル、Abstract、Organization をレイアウト解析 + LLMフォールバックで取得 |
| **LLM要約生成** | 原文Markdown全文からLLMで要約を生成（LiteLLM経由で複数プロバイダー対応） |
| **サムネイル生成** | PDF 1ページ目のサムネイル画像 |
| **Markdown二重生成** | 原文Markdown + 翻訳Markdownの両方を出力 |
| **統一LLMインターフェース** | LiteLLMによりGemini/OpenAI/Anthropic等を統一的に利用可能 |

---

## 背景

### Webサービスとしての要件

PDF翻訳機能をWebサービスとして提供する際、以下の情報がダッシュボード表示や検索・管理に必要となる:

| 要件 | 用途 |
|------|------|
| ドキュメント識別 | 翻訳履歴から目的のドキュメントを探す |
| 内容プレビュー | 翻訳品質の確認、内容の把握 |
| ビジュアル識別 | サムネイル表示でUIを向上 |

### 現在のレイアウト解析で取得可能な情報

PP-DocLayoutV2のカテゴリを分析した結果、以下が**明確かつ確実に取得可能**:

| 情報 | カテゴリ | 信頼性 |
|------|---------|--------|
| **ドキュメントタイトル** | `doc_title` | ◎ 高い |
| **Abstract** | `abstract` | ◎ 高い |
| 著者情報 | `text` (汎用) | △ 追加ロジック必要 |

著者情報は専用カテゴリが存在しないため、レイアウト解析では取得不可。

### LLM統合の必要性

| 課題 | LLMによる解決 |
|------|--------------|
| タイトル/Abstract未検出時のフォールバック | 1ページ目テキストからLLMで抽出 |
| Organization（団体名）の取得 | レイアウト解析では取得不可→LLMで抽出 |
| ユーザーフレンドリーな要約 | Abstractは技術的すぎる場合あり→LLMで要約生成 |

### LLMライブラリ選定: LiteLLM

複数のLLMプロバイダーを統一的に扱うため、[LiteLLM](https://github.com/BerriAI/litellm) を採用する。

**選定理由:**

| 観点 | LiteLLM |
|------|---------|
| **プロバイダー対応** | 100+ (Gemini, OpenAI, Anthropic, AWS Bedrock, Azure等) |
| **非同期サポート** | ✅ `acompletion()` で完全対応 |
| **ライセンス** | MIT License (Apache-2.0互換) |
| **メンテナンス** | 活発 (33.8k stars, 1,179 contributors) |
| **新モデル対応** | ライブラリ更新のみで対応可能 |

**比較検討した他ライブラリ:**

| ライブラリ | 不採用理由 |
|-----------|-----------|
| aisuite (Andrew Ng) | 非同期非対応、メンテナンス懸念（最終更新2024/12） |
| any-llm (Mozilla) | 比較的新しく、プロバイダー数が少ない |

**デフォルトモデル**: `gemini-3-flash-preview`（provider: gemini、コスト効率・速度のバランス）

### 翻訳対象カテゴリの現状

現在の `DEFAULT_TRANSLATABLE_RAW_CATEGORIES`:

```python
DEFAULT_TRANSLATABLE_RAW_CATEGORIES = frozenset({
    "text",
    "vertical_text",
    "abstract",      # ← 翻訳対象に含まれている
    "aside_text",
    "figure_title",
})
```

| カテゴリ | 翻訳対象 | 備考 |
|---------|---------|------|
| `doc_title` | **対象外** | タイトルは翻訳されない |
| `abstract` | **対象** | Abstractは翻訳される |

この設定により、現状では `title_translated` は常に `None` となる。

### 検証データ

`sample_llama_translated.json` からの抽出例:

```json
{
  "category": "doc_title",
  "text": "LLaMA: Open and Efficient Foundation Language Models",
  "category_confidence": 0.9415737986564636
}

{
  "category": "abstract",
  "text": "We introduce LLaMA, a collection of foundation language models...",
  "category_confidence": 0.9836386442184448
}
```

---

## 設計方針

### 1. Markdown二重生成

**課題**: 現在のMarkdown生成は翻訳後テキストのみ出力。LLM要約には原文が必要。

**決定: 原文Markdownと翻訳Markdownの両方を生成**

```
output/
├── paper_translated.pdf
├── paper_translated.json
├── paper_original.md          # NEW - 原文Markdown
├── paper_translated.md        # 翻訳Markdown（既存動作を維持）
└── paper_translated_thumbnail.png
```

**実装方針:**

```python
# MarkdownWriter の拡張
class MarkdownWriter:
    def write(
        self,
        paragraphs: list[Paragraph],
        use_translated: bool = True,  # NEW parameter
    ) -> str:
        """Generate Markdown from paragraphs.

        Args:
            paragraphs: List of paragraphs.
            use_translated: If True, use translated_text. If False, use original text.
        """
        ...
```

**理由:**
- LLM要約生成には原文Markdownが必要（翻訳品質に依存しない）
- 論文は序論・実験・結論すべてが重要、ページ限定は不適切
- Markdownは既に無駄な文字列（ヘッダー/フッター等）を除去済み
- 原文Markdownはユーザーにも有用（翻訳前後の比較）

### 2. LLM要約生成

**決定: 原文Markdown全文をLLMに渡して要約を生成（デフォルト: Gemini 3.0 Flash）**

```
処理フロー:
1. 原文Markdown生成（画像参照は除外）
2. LLM に原文Markdown全文を送信（--llm-provider/--llm-model で指定）
3. 要約（原文言語）を取得
4. 要約を翻訳（通常の翻訳パイプラインを使用）
```

**入力仕様:**
- 原文Markdown全文（`paper_original.md` の内容）
- 画像参照 (`![...]()`) は除外してテキストのみ
- ページ数制限なし（論文全体の文脈が要約に必要）

**入力サイズ制御:**

| プロバイダー | モデル | コンテキストウィンドウ |
|-------------|--------|---------------------|
| Gemini | gemini-3-flash-preview | 1,048,576 tokens |
| OpenAI | gpt-4o-mini | 128,000 tokens |
| Anthropic | claude-sonnet-4-5-20250514 | 200,000 tokens |

- 典型的な論文（10-50ページ）: 約 10,000-50,000 tokens（全プロバイダーで十分収まる）
- 超長文書（100ページ超）の場合: 警告ログを出力し、先頭 500,000 文字で切り捨て
- **デフォルト動作**: Geminiの大容量コンテキストを前提とした設計
- **他プロバイダー使用時の注意**: 非常に長い文書でOpenAI/Anthropicを使用する場合、コンテキスト上限に注意が必要。必要に応じて切り捨て閾値を調整可能（将来拡張）

**出力仕様:**
- `summary`: 原文言語での要約（3-5文）
- `summary_translated`: 翻訳言語での要約

**プロンプト設計:**

```
Summarize this academic paper in 3-5 sentences, covering:
- Main research objective
- Key methodology
- Important findings/conclusions

Paper content:
{markdown_content}
```

### 3. LLMフォールバック（メタデータ抽出）

**決定: レイアウト解析失敗時、1ページ目テキストからLLMでメタデータを抽出**

**トリガー条件:**
- `doc_title` カテゴリが検出されなかった場合 → title をLLMで抽出
- `abstract` カテゴリが検出されなかった場合 → abstract をLLMで抽出
- `organization` は常にLLMで抽出（レイアウト解析では取得不可）

**入力仕様:**
- **1ページ目のテキストのみ**（メタデータは通常1ページ目に存在）
- トークン制限を考慮し、最小限の入力で効率化

**プロンプト設計:**

```
Extract the following information from this academic paper's first page.
Return JSON format with null for missing fields.

Required fields:
- title: The main title of the paper
- abstract: The abstract/summary section (if present on first page)
- organization: The institution/company names (e.g., "Meta AI", "Google Research")

First page content:
{first_page_text}
```

**出力形式:**

```json
{
  "title": "LLaMA: Open and Efficient Foundation Language Models",
  "abstract": "We introduce LLaMA...",
  "organization": "Meta AI"
}
```

### 4. 処理フロー

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Translation Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. PDF Processing (existing)                                       │
│     └── Extract paragraphs with categories                          │
│                                                                     │
│  2. Layout Analysis (existing)                                      │
│     └── Assign doc_title, abstract categories                       │
│                                                                     │
│  3. Translation (existing)                                          │
│     └── Translate text, abstract (NOT doc_title)                    │
│                                                                     │
│  4. Markdown Generation (UPDATED)                                   │
│     ├── paper_original.md    ← NEW: use_translated=False            │
│     └── paper_translated.md  ← existing: use_translated=True        │
│                                                                     │
│  5. Summary Extraction (NEW)                                        │
│     │                                                               │
│     ├── Step 1: Layout-based extraction                             │
│     │   ├── title ← doc_title category                              │
│     │   └── abstract ← abstract category                            │
│     │                                                               │
│     ├── Step 2: LLM Fallback (if title/abstract missing)            │
│     │   ├── Input: First page text only                             │
│     │   ├── Output: title, abstract, organization                   │
│     │   └── Model: --llm-provider/--llm-model (default: gemini)     │
│     │                                                               │
│     ├── Step 3: Translate title (if not translated)                 │
│     │                                                               │
│     ├── Step 4: LLM Summary Generation                              │
│     │   ├── Input: paper_original.md (full, images excluded)        │
│     │   ├── Output: summary (3-5 sentences)                         │
│     │   └── Model: --llm-provider/--llm-model (default: gemini)     │
│     │                                                               │
│     ├── Step 5: Translate summary                                   │
│     │                                                               │
│     └── Step 6: Generate thumbnail (first page)                     │
│                                                                     │
│  6. Output Generation                                               │
│     ├── paper_translated.pdf                                        │
│     ├── paper_translated.json   ← includes summary section          │
│     ├── paper_original.md       ← NEW                               │
│     ├── paper_translated.md                                         │
│     └── paper_translated_thumbnail.png                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5. タイトル/Abstractの翻訳経路

**課題**: `doc_title` はデフォルトで翻訳対象外のため、`title_translated` が取得できない。

**決定: サマリー抽出時に `doc_title` を追加で翻訳する**

```
翻訳パイプライン
    ├── _stage_translate()
    │   └── DEFAULT_TRANSLATABLE_RAW_CATEGORIES に基づき翻訳
    │       → abstract は翻訳される、doc_title は翻訳されない
    │
    └── _stage_summary() [新規]
        └── doc_title の translated_text が None の場合、追加で翻訳
            → title_translated を取得
```

**理由:**
- `doc_title` を `DEFAULT_TRANSLATABLE_RAW_CATEGORIES` に追加すると、PDFにも翻訳が反映される
- タイトルは原文のまま残したいケースが多い（論文検索等）
- サマリー専用に翻訳することで、PDFへの影響を回避
- 翻訳API呼び出しは1回追加のみ（タイトルは短文）

**代替案（不採用）:**
- `doc_title` を翻訳対象に追加 → PDFのタイトルも翻訳されてしまう
- 翻訳文なしで運用 → Webサービスでの検索・表示に支障

### 2. 抽出対象の明確化

| 情報 | ソース | 翻訳経路 | 出力形式 |
|------|--------|----------|----------|
| タイトル（原文） | `category: "doc_title"` | - | `str \| None` |
| タイトル（翻訳） | サマリー抽出時に追加翻訳 | `_stage_summary()` | `str \| None` |
| Abstract（原文） | `category: "abstract"` | - | `str \| None` |
| Abstract（翻訳） | `_stage_translate()` で翻訳済み | パイプライン標準 | `str \| None` |
| サムネイル | pypdfium2 render（1ページ目） | - | `bytes` (PNG) |

### 3. 複数段落の結合ルール

**課題**: Abstract が複数段落に分かれている場合、先頭1件のみの抽出では欠落する。

**決定: 同一カテゴリの段落を結合して抽出**

```python
def _find_and_merge_by_category(
    paragraphs: list[Paragraph],
    category: str,
) -> tuple[str | None, str | None]:
    """Find all paragraphs with category and merge them.

    Args:
        paragraphs: List of paragraphs to search.
        category: Category to find.

    Returns:
        Tuple of (merged_original, merged_translated).
    """
    # Filter by category
    matched = [p for p in paragraphs if p.category == category]

    if not matched:
        return None, None

    # Sort by page_number, then by y-coordinate (descending, PDF coordinates)
    matched.sort(key=lambda p: (p.page_number, -p.block_bbox.y1))

    # Merge with double newline
    original = "\n\n".join(p.text for p in matched if p.text)
    translated = "\n\n".join(
        p.translated_text for p in matched if p.translated_text
    )

    return original or None, translated or None
```

**結合ルール:**
- 同一カテゴリの全段落を収集
- ページ番号順、Y座標降順（PDF座標系で上から下）でソート
- `\n\n` で結合

### 4. サムネイル出力仕様

**課題**: サムネイルの出力形式（base64埋め込み vs ファイルパス）が曖昧。

**決定: ファイル保存を正とし、JSONには相対パスを記録**

```
output/
├── paper_translated.pdf
├── paper_translated.json          # summary.thumbnail_path を含む
├── paper_translated.md
└── paper_translated_thumbnail.png # サムネイル画像
```

**JSON出力:**
```json
{
  "summary": {
    "title": "LLaMA: Open and Efficient...",
    "title_translated": "LLaMA: オープンで効率的な...",
    "abstract": "We introduce LLaMA...",
    "abstract_translated": "LLaMAを紹介します...",
    "thumbnail_path": "paper_translated_thumbnail.png",
    "thumbnail_width": 400,
    "thumbnail_height": 518,
    "page_count": 1,
    "source_lang": "en",
    "target_lang": "ja"
  }
}
```

**API向け（オプション）:**
- `DocumentSummary.to_dict(include_thumbnail_base64=True)` でbase64埋め込み可能
- デフォルトはファイルパス方式（JSON肥大化防止）

### 5. サムネイル生成の方針

- **対象**: 翻訳前のPDF 1ページ目のみ
- **理由**:
  - 翻訳後サムネイルは翻訳品質に依存し、プレビュー価値が低い
  - 翻訳前サムネイルでドキュメントの識別は十分可能
  - 処理コストの削減

### 6. 既存機能の再利用

| 機能 | 再利用元 |
|------|---------|
| ページレンダリング | `ImageExtractor` (pypdfium2使用) |
| JSON出力 | `TranslatedDocument` |

---

## 技術的決定事項

### 1. データモデル

**決定: `DocumentSummary` dataclass を新規作成**

```python
# src/pdf_translator/output/document_summary.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DocumentSummary:
    """Document summary for web service integration.

    Contains essential metadata extracted from translated documents
    for dashboard display, search, and document management.

    Attributes:
        title: Original document title (from doc_title category or LLM fallback).
        title_translated: Translated document title.
        abstract: Original abstract text (from abstract category or LLM fallback).
        abstract_translated: Translated abstract text.
        organization: Institution/company name (LLM extracted).
        summary: LLM-generated summary of the document.
        summary_translated: Translated LLM-generated summary.
        thumbnail_path: Relative path to thumbnail file (primary reference).
        thumbnail_width: Thumbnail width in pixels.
        thumbnail_height: Thumbnail height in pixels.
        page_count: Total number of pages in the document.
        source_lang: Source language code (e.g., "en").
        target_lang: Target language code (e.g., "ja").
        title_source: Source of title extraction ("layout" or "llm").
        abstract_source: Source of abstract extraction ("layout" or "llm").
    """

    # Title (from doc_title category or LLM fallback)
    title: str | None = None
    title_translated: str | None = None

    # Abstract (from abstract category or LLM fallback, may be merged from multiple paragraphs)
    abstract: str | None = None
    abstract_translated: str | None = None

    # Organization (LLM extracted only - not available via layout analysis)
    organization: str | None = None

    # LLM-generated summary (from full original Markdown)
    summary: str | None = None
    summary_translated: str | None = None

    # Thumbnail (first page of original PDF)
    thumbnail_path: str | None = None  # Relative path to thumbnail file
    thumbnail_width: int = 0
    thumbnail_height: int = 0

    # Metadata
    page_count: int = 0
    source_lang: str = ""
    target_lang: str = ""

    # Extraction source tracking (for debugging/quality monitoring)
    title_source: Literal["layout", "llm"] = "layout"
    abstract_source: Literal["layout", "llm"] = "layout"

    # Internal: thumbnail bytes (not serialized by default)
    _thumbnail_bytes: bytes | None = field(default=None, repr=False)

    def to_dict(self, include_thumbnail_base64: bool = False) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Args:
            include_thumbnail_base64: If True, include base64-encoded thumbnail.
                Default False - use thumbnail_path instead.

        Returns:
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "title": self.title,
            "title_translated": self.title_translated,
            "abstract": self.abstract,
            "abstract_translated": self.abstract_translated,
            "organization": self.organization,
            "summary": self.summary,
            "summary_translated": self.summary_translated,
            "thumbnail_path": self.thumbnail_path,
            "thumbnail_width": self.thumbnail_width,
            "thumbnail_height": self.thumbnail_height,
            "page_count": self.page_count,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "title_source": self.title_source,
            "abstract_source": self.abstract_source,
        }

        if include_thumbnail_base64 and self._thumbnail_bytes:
            import base64
            result["thumbnail_base64"] = base64.b64encode(
                self._thumbnail_bytes
            ).decode("ascii")

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentSummary:
        """Create from dictionary.

        Args:
            data: Dictionary with summary fields.

        Returns:
            DocumentSummary instance.
        """
        thumbnail_bytes = None
        if "thumbnail_base64" in data:
            import base64
            thumbnail_bytes = base64.b64decode(data["thumbnail_base64"])

        return cls(
            title=data.get("title"),
            title_translated=data.get("title_translated"),
            abstract=data.get("abstract"),
            abstract_translated=data.get("abstract_translated"),
            organization=data.get("organization"),
            summary=data.get("summary"),
            summary_translated=data.get("summary_translated"),
            thumbnail_path=data.get("thumbnail_path"),
            thumbnail_width=data.get("thumbnail_width", 0),
            thumbnail_height=data.get("thumbnail_height", 0),
            page_count=data.get("page_count", 0),
            source_lang=data.get("source_lang", ""),
            target_lang=data.get("target_lang", ""),
            title_source=data.get("title_source", "layout"),
            abstract_source=data.get("abstract_source", "layout"),
            _thumbnail_bytes=thumbnail_bytes,
        )

    def has_content(self) -> bool:
        """Check if summary has any meaningful content.

        Returns:
            True if at least title, abstract, summary, or thumbnail is present.
        """
        return bool(
            self.title or self.abstract or self.summary or self.thumbnail_path
        )
```

**理由:**
- 独立したモジュールで責務を明確化
- `to_dict()` でサムネイルの含有を制御可能（JSON肥大化防止）
- Webサービス連携に必要な情報を集約

### 2. サムネイル生成

**決定: `ThumbnailGenerator` クラスを新規作成**

```python
# src/pdf_translator/output/thumbnail_generator.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pypdfium2 as pdfium

logger = logging.getLogger(__name__)


@dataclass
class ThumbnailConfig:
    """Configuration for thumbnail generation.

    Attributes:
        width: Target thumbnail width in pixels.
            Height is calculated to maintain aspect ratio.
        page_number: Page to render (0-indexed). Default: 0 (first page).

    Note:
        Output format is fixed to PNG for simplicity and transparency support.
    """

    width: int = 400
    page_number: int = 0


class ThumbnailGenerator:
    """Generate thumbnail images from PDF pages.

    Uses pypdfium2 for rendering, similar to ImageExtractor.
    """

    def __init__(self, config: ThumbnailConfig | None = None) -> None:
        """Initialize ThumbnailGenerator.

        Args:
            config: Thumbnail generation configuration.
        """
        self._config = config or ThumbnailConfig()

    def generate(self, pdf_path: Path) -> tuple[bytes, int, int]:
        """Generate thumbnail from PDF first page.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Tuple of (image_bytes, width, height).

        Raises:
            FileNotFoundError: If PDF file not found.
            ValueError: If PDF has no pages.
        """
        import pypdfium2 as pdfium

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = pdfium.PdfDocument(pdf_path)
        try:
            if len(doc) == 0:
                raise ValueError("PDF has no pages")

            page_num = min(self._config.page_number, len(doc) - 1)
            page = doc[page_num]

            # Calculate scale to achieve target width
            page_width = page.get_width()
            page_height = page.get_height()
            scale = self._config.width / page_width

            # Render page
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()

            # Get actual dimensions
            actual_width = pil_image.width
            actual_height = pil_image.height

            # Convert to PNG bytes
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")

            return buffer.getvalue(), actual_width, actual_height

        finally:
            doc.close()

    def generate_to_file(self, pdf_path: Path, output_path: Path) -> tuple[int, int]:
        """Generate thumbnail and save to file.

        Args:
            pdf_path: Path to PDF file.
            output_path: Path to save thumbnail.

        Returns:
            Tuple of (width, height).
        """
        image_bytes, width, height = self.generate(pdf_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        return width, height
```

**理由:**
- `ImageExtractor` と同様のpypdfium2ベースの実装
- 設定可能なサイズ（フォーマットはPNG固定）
- メモリ（bytes）とファイル出力の両方に対応

### 3. LLM統合モジュール（LiteLLM）

**決定: LiteLLMを使用した統一LLMクライアントを実装**

```python
# src/pdf_translator/llm/client.py

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM integration via LiteLLM.

    Attributes:
        provider: LLM provider ("gemini", "openai", "anthropic", etc.).
        model: Model name within provider. If None, uses PROVIDER_DEFAULTS.
        api_key: API key (optional, can use environment variables).
        use_summary: Enable LLM summary generation.
        use_fallback: Enable LLM fallback for metadata extraction.
    """

    provider: str = "gemini"
    model: str | None = None  # None = use PROVIDER_DEFAULTS[provider]
    api_key: str | None = None
    use_summary: bool = False
    use_fallback: bool = True

    # Supported providers and their default models
    PROVIDER_DEFAULTS: ClassVar[dict[str, str]] = {
        "gemini": "gemini-3-flash-preview",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-5",
    }

    # Environment variable names for API keys
    API_KEY_ENV_VARS: ClassVar[dict[str, str]] = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    @property
    def effective_model(self) -> str:
        """Get effective model name (resolves None to provider default)."""
        if self.model is not None:
            return self.model
        return self.PROVIDER_DEFAULTS.get(self.provider, "gemini-3-flash-preview")

    @property
    def litellm_model(self) -> str:
        """Get LiteLLM model string (provider/model format)."""
        return f"{self.provider}/{self.effective_model}"

    def get_api_key_env_var(self) -> str:
        """Get environment variable name for API key."""
        return self.API_KEY_ENV_VARS.get(
            self.provider,
            f"{self.provider.upper()}_API_KEY"
        )


class LLMClient:
    """Unified LLM client using LiteLLM.

    Provides a simple interface for text generation across multiple providers.
    Supports: Gemini, OpenAI, Anthropic, AWS Bedrock, Azure, and 100+ others.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLMClient.

        Args:
            config: LLM configuration.

        Raises:
            ImportError: If litellm is not installed.
        """
        self._config = config
        self._setup_api_key()

    def _setup_api_key(self) -> None:
        """Set up API key in environment if provided."""
        if self._config.api_key:
            env_var = self._config.get_api_key_env_var()
            os.environ[env_var] = self._config.api_key

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Generate text from prompt using LiteLLM.

        Args:
            prompt: User prompt.
            system: Optional system prompt.

        Returns:
            Generated text.

        Raises:
            Exception: On LLM API errors.
        """
        from litellm import acompletion

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await acompletion(
            model=self._config.litellm_model,
            messages=messages,
        )

        return response.choices[0].message.content
```

```python
# src/pdf_translator/llm/summary_generator.py

from __future__ import annotations

import json
import logging
import re

from pdf_translator.llm.client import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class LLMSummaryGenerator:
    """Generate document summaries and extract metadata using LLM.

    Uses LiteLLM for unified access to multiple LLM providers:
    - Gemini (default): gemini/gemini-3-flash-preview
    - OpenAI: openai/gpt-4o-mini
    - Anthropic: anthropic/claude-sonnet-4-5
    """

    SUMMARY_PROMPT = """Summarize this academic paper in 3-5 sentences, covering:
- Main research objective
- Key methodology
- Important findings/conclusions

Paper content:
{content}"""

    METADATA_PROMPT = """Extract the following information from this academic paper's first page.
Return JSON format with null for missing fields.

Required fields:
- title: The main title of the paper
- abstract: The abstract/summary section (if present on first page)
- organization: The institution/company names (e.g., "Meta AI", "Google Research")

First page content:
{content}"""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLMSummaryGenerator.

        Args:
            config: LLM configuration.
        """
        self._config = config
        self._client = LLMClient(config) if (config.use_summary or config.use_fallback) else None

    async def generate_summary(
        self,
        markdown_content: str,
    ) -> str | None:
        """Generate summary from original Markdown content.

        Args:
            markdown_content: Full original Markdown (images excluded).

        Returns:
            Summary text in original language, or None if disabled/failed.
        """
        if not self._config.use_summary or not self._client:
            return None

        # Remove image references from Markdown
        content = re.sub(r"!\[.*?\]\(.*?\)", "", markdown_content)

        # Truncate if too long (500K chars safety limit)
        if len(content) > 500_000:
            logger.warning("Markdown content truncated from %d to 500K chars", len(content))
            content = content[:500_000]

        prompt = self.SUMMARY_PROMPT.format(content=content)

        try:
            return await self._client.generate(prompt)
        except Exception as e:
            logger.warning("Failed to generate LLM summary: %s", e)
            return None

    async def extract_metadata_fallback(
        self,
        first_page_text: str,
    ) -> dict[str, str | None]:
        """Extract metadata from first page when layout analysis fails.

        Args:
            first_page_text: Text content of first page only.

        Returns:
            Dict with title, abstract, organization (any may be None).
        """
        if not self._config.use_fallback or not self._client:
            return {"title": None, "abstract": None, "organization": None}

        prompt = self.METADATA_PROMPT.format(content=first_page_text)

        try:
            text = await self._client.generate(prompt)

            # Parse JSON from response
            # Handle markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text)
        except Exception as e:
            logger.warning("Failed to extract metadata via LLM: %s", e)
            return {"title": None, "abstract": None, "organization": None}
```

**理由:**
- **LiteLLM採用**: 100+プロバイダーを統一インターフェースで利用可能
- **プロバイダー切り替え**: `provider` + `model` の組み合わせで任意のモデルを指定
- **後方互換**: デフォルトは `gemini/gemini-3-flash-preview`（既存設計と同じ動作）
- **要約生成**: 原文Markdown全文を使用（論文全体の文脈が必要）
- **フォールバック**: 1ページ目のみ使用（メタデータは通常1ページ目に存在）
- Gemini 3.0 Flash はコスト効率と速度のバランスが良い
- API呼び出しエラーは吸収してNoneを返す（オプショナル機能）

### 4. サマリー抽出ロジック

**決定: `SummaryExtractor` クラスでレイアウト抽出 + LLM統合を管理**

```python
# src/pdf_translator/output/summary_extractor.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pdf_translator.core.models import Paragraph
from pdf_translator.output.document_summary import DocumentSummary
from pdf_translator.output.thumbnail_generator import ThumbnailConfig, ThumbnailGenerator
from pdf_translator.llm.summary_generator import LLMConfig, LLMSummaryGenerator

if TYPE_CHECKING:
    from pdf_translator.translators.base import TranslatorBackend

logger = logging.getLogger(__name__)


class SummaryExtractor:
    """Extract document summary from translated paragraphs.

    Handles:
    - Title extraction from doc_title category (with LLM fallback)
    - Abstract extraction from abstract category (with LLM fallback)
    - Organization extraction (LLM only)
    - LLM summary generation from original Markdown
    - Thumbnail generation from first page
    """

    TITLE_CATEGORY = "doc_title"
    ABSTRACT_CATEGORY = "abstract"

    def __init__(
        self,
        thumbnail_config: ThumbnailConfig | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        """Initialize SummaryExtractor.

        Args:
            thumbnail_config: Configuration for thumbnail generation.
            llm_config: Configuration for LLM integration.
        """
        self._thumbnail_config = thumbnail_config or ThumbnailConfig()
        self._llm_config = llm_config
        self._llm_generator = (
            LLMSummaryGenerator(llm_config) if llm_config else None
        )

    async def extract(
        self,
        paragraphs: list[Paragraph],
        pdf_path: Path,
        output_dir: Path,
        output_stem: str,
        source_lang: str = "",
        target_lang: str = "",
        page_count: int = 0,
        generate_thumbnail: bool = True,
        translator: TranslatorBackend | None = None,
        original_markdown: str | None = None,  # NEW: for LLM summary
    ) -> DocumentSummary:
        """Extract document summary from paragraphs.

        Args:
            paragraphs: List of translated paragraphs.
            pdf_path: Path to original PDF (for thumbnail).
            output_dir: Directory to save thumbnail.
            output_stem: Base filename for outputs.
            source_lang: Source language code.
            target_lang: Target language code.
            page_count: Total page count.
            generate_thumbnail: Whether to generate thumbnail.
            translator: Translator backend for title/summary translation.
            original_markdown: Original Markdown content for LLM summary.

        Returns:
            DocumentSummary with extracted information.
        """
        # Step 1: Extract from layout analysis
        title, title_translated = self._find_and_merge_by_category(
            paragraphs, self.TITLE_CATEGORY
        )
        abstract, abstract_translated = self._find_and_merge_by_category(
            paragraphs, self.ABSTRACT_CATEGORY
        )

        title_source: Literal["layout", "llm"] = "layout"
        abstract_source: Literal["layout", "llm"] = "layout"
        organization = None

        # Step 2: LLM metadata extraction
        # - Organization: Always extracted via LLM (not available in layout)
        # - Title/Abstract: Fallback when layout analysis fails
        if self._llm_generator:
            first_page_text = self._get_first_page_text(paragraphs)
            if first_page_text:
                llm_metadata = await self._llm_generator.extract_metadata_fallback(
                    first_page_text
                )

                # Organization is always from LLM (no layout category exists)
                organization = llm_metadata.get("organization")

                # Use LLM result only if layout failed
                if not title and llm_metadata.get("title"):
                    title = llm_metadata["title"]
                    title_source = "llm"

                if not abstract and llm_metadata.get("abstract"):
                    abstract = llm_metadata["abstract"]
                    abstract_source = "llm"

        # Step 3: Translate title if needed
        if title and not title_translated and translator:
            try:
                title_translated = await translator.translate(
                    title, source_lang, target_lang
                )
            except Exception as e:
                logger.warning("Failed to translate title: %s", e)

        # Step 4: Translate abstract if from LLM fallback
        if abstract and not abstract_translated and abstract_source == "llm" and translator:
            try:
                abstract_translated = await translator.translate(
                    abstract, source_lang, target_lang
                )
            except Exception as e:
                logger.warning("Failed to translate abstract: %s", e)

        # Step 5: Generate LLM summary from original Markdown
        summary = None
        summary_translated = None

        if self._llm_generator and original_markdown:
            summary = await self._llm_generator.generate_summary(original_markdown)
            if summary and translator:
                try:
                    summary_translated = await translator.translate(
                        summary, source_lang, target_lang
                    )
                except Exception as e:
                    logger.warning("Failed to translate summary: %s", e)

        # Step 6: Generate thumbnail
        thumbnail_path, thumbnail_bytes, thumb_width, thumb_height = (
            await self._generate_thumbnail(pdf_path, output_dir, output_stem, generate_thumbnail)
        )

        return DocumentSummary(
            title=title,
            title_translated=title_translated,
            abstract=abstract,
            abstract_translated=abstract_translated,
            organization=organization,
            summary=summary,
            summary_translated=summary_translated,
            thumbnail_path=thumbnail_path,
            thumbnail_width=thumb_width,
            thumbnail_height=thumb_height,
            page_count=page_count,
            source_lang=source_lang,
            target_lang=target_lang,
            title_source=title_source,
            abstract_source=abstract_source,
            _thumbnail_bytes=thumbnail_bytes,
        )

    @staticmethod
    def _find_and_merge_by_category(
        paragraphs: list[Paragraph],
        category: str,
    ) -> tuple[str | None, str | None]:
        """Find all paragraphs with category and merge them."""
        matched = [p for p in paragraphs if p.category == category]

        if not matched:
            return None, None

        matched.sort(key=lambda p: (p.page_number, -p.block_bbox.y1))

        original_parts = [p.text for p in matched if p.text]
        translated_parts = [p.translated_text for p in matched if p.translated_text]

        original = "\n\n".join(original_parts) if original_parts else None
        translated = "\n\n".join(translated_parts) if translated_parts else None

        return original, translated

    @staticmethod
    def _get_first_page_text(paragraphs: list[Paragraph]) -> str:
        """Get concatenated text from first page for LLM fallback."""
        first_page = [p for p in paragraphs if p.page_number == 0]
        first_page.sort(key=lambda p: -p.block_bbox.y1)
        return "\n\n".join(p.text for p in first_page if p.text)

    async def _generate_thumbnail(
        self,
        pdf_path: Path,
        output_dir: Path,
        output_stem: str,
        generate_thumbnail: bool,
    ) -> tuple[str | None, bytes | None, int, int]:
        """Generate thumbnail from PDF first page."""
        if not generate_thumbnail or not pdf_path.exists():
            return None, None, 0, 0

        try:
            generator = ThumbnailGenerator(self._thumbnail_config)
            thumbnail_bytes, width, height = generator.generate(pdf_path)

            thumbnail_filename = f"{output_stem}_thumbnail.png"
            thumbnail_file = output_dir / thumbnail_filename
            thumbnail_file.write_bytes(thumbnail_bytes)

            logger.debug("Generated thumbnail: %s", thumbnail_file)
            return thumbnail_filename, thumbnail_bytes, width, height
        except Exception as e:
            logger.warning("Failed to generate thumbnail: %s", e)
            return None, None, 0, 0
```

**理由:**
- レイアウト抽出 + LLMフォールバック + LLM要約生成を統合管理
- **抽出優先順位**: レイアウト解析 → LLMフォールバック
- **extraction source tracking**: デバッグ・品質管理用
- **organization**: レイアウト解析では取得不可のため、常にLLMで抽出

### 5. TranslationResult への統合

**決定: `summary` および `markdown_original` フィールドを追加**

```python
# src/pdf_translator/pipeline/translation_pipeline.py

@dataclass
class TranslationResult:
    """Translation pipeline result."""

    pdf_bytes: bytes
    stats: dict[str, Any] | None = None
    side_by_side_pdf_bytes: bytes | None = None
    markdown: str | None = None              # 翻訳Markdown
    markdown_original: str | None = None     # NEW: 原文Markdown
    paragraphs: list[Paragraph] | None = None
    summary: DocumentSummary | None = None   # NEW: サマリー情報
```

### 6. JSON出力拡張

**決定: `TranslatedDocument` に `summary` セクションを追加（LLM強化版）**

```json
{
  "version": "1.0.0",
  "metadata": {
    "source_file": "sample_llama.pdf",
    "source_lang": "en",
    "target_lang": "ja",
    "translated_at": "2026-01-14T07:05:05",
    "translator_backend": "google",
    "page_count": 1,
    "paragraph_count": 13,
    "translated_count": 9
  },
  "summary": {
    "title": "LLaMA: Open and Efficient Foundation Language Models",
    "title_translated": "LLaMA: オープンで効率的な基盤言語モデル",
    "abstract": "We introduce LLaMA, a collection of foundation language models...",
    "abstract_translated": "LLaMAを紹介します。これは7Bから65Bのパラメータを持つ...",
    "organization": "Meta AI",
    "summary": "This paper introduces LLaMA, a series of foundation language models...",
    "summary_translated": "本論文はLLaMAを紹介し、7Bから65Bのパラメータを持つ基盤言語モデルシリーズです...",
    "thumbnail_path": "sample_llama_translated_thumbnail.png",
    "thumbnail_width": 400,
    "thumbnail_height": 518,
    "page_count": 1,
    "source_lang": "en",
    "target_lang": "ja",
    "title_source": "layout",
    "abstract_source": "layout"
  },
  "paragraphs": [ ... ]
}
```

**出力ファイル構成:**

```
output/
├── paper_translated.pdf
├── paper_translated.json              # summary を含む
├── paper_original.md                  # NEW: 原文Markdown
├── paper_translated.md                # 翻訳Markdown
└── paper_translated_thumbnail.png     # サムネイル画像
```

**Webサービス側での参照:**
- JSON から `summary.*` フィールドを取得
- `summary.thumbnail_path` でサムネイル画像を参照
- `summary.title_source` / `abstract_source` で抽出品質を確認

### 7. CLI オプション

**決定: サムネイル + LLM関連オプションを追加（LiteLLM経由でプロバイダー選択可能）**

```bash
# 基本使用（サムネイル・LLMなし、後方互換性）
uv run translate-pdf paper.pdf --save-intermediate

# サムネイル生成あり
uv run translate-pdf paper.pdf --save-intermediate --thumbnail

# サムネイルサイズ指定
uv run translate-pdf paper.pdf --save-intermediate --thumbnail --thumbnail-width 600

# LLM要約生成あり（デフォルト: Gemini 3.0 Flash）
uv run translate-pdf paper.pdf --save-intermediate --llm-summary

# LLMプロバイダー・モデル選択
uv run translate-pdf paper.pdf --save-intermediate --llm-summary --llm-provider openai --llm-model gpt-4o-mini
uv run translate-pdf paper.pdf --save-intermediate --llm-summary --llm-provider anthropic --llm-model claude-sonnet-4-5

# LLMフォールバック有効化（メタデータ抽出にLLMを使用）
uv run translate-pdf paper.pdf --save-intermediate --llm-fallback

# フル機能（Webサービス向け）
uv run translate-pdf paper.pdf --save-intermediate --thumbnail --llm-summary
```

**CLIオプション一覧:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--thumbnail` | サムネイル生成を有効化 | 無効 |
| `--thumbnail-width` | サムネイル幅（ピクセル） | 400 |
| `--llm-summary` | LLM要約生成を有効化 | 無効 |
| `--llm-provider` | LLMプロバイダー (gemini, openai, anthropic, etc.) | gemini |
| `--llm-model` | LLMモデル名（provider抜き）。省略時はproviderごとのデフォルトを使用 | (providerによる) |
| `--llm-fallback` | LLMメタデータフォールバックを有効化 | 無効 |

**モデル自動選択ルール:**
- `--llm-model` を省略した場合、`--llm-provider` に応じてデフォルトモデルを自動選択
- 例: `--llm-provider openai` → `gpt-4o-mini` が自動設定

| --llm-provider | --llm-model 省略時のデフォルト |
|----------------|------------------------------|
| gemini | gemini-3-flash-preview |
| openai | gpt-4o-mini |
| anthropic | claude-sonnet-4-5 |

**環境変数:**

```bash
# プロバイダーに応じたAPIキー（LLM機能使用時に必須）
export GEMINI_API_KEY="your-gemini-key"      # Gemini使用時
export OPENAI_API_KEY="your-openai-key"      # OpenAI使用時
export ANTHROPIC_API_KEY="your-anthropic-key"  # Anthropic使用時
```

**LiteLLMサポートプロバイダー例:**

| プロバイダー | --llm-provider | 主要モデル例 |
|-------------|---------------|-------------|
| Google Gemini | gemini | gemini-3-flash-preview, gemini-1.5-pro |
| OpenAI | openai | gpt-4o-mini, gpt-4o |
| Anthropic | anthropic | claude-sonnet-4-5 |
| AWS Bedrock | bedrock | anthropic.claude-v2 |
| Azure OpenAI | azure | gpt-4 |

※ LiteLLMは100+プロバイダーをサポート。詳細は [LiteLLM Providers](https://docs.litellm.ai/docs/providers) 参照。

**理由:**
- 後方互換性のためデフォルトはすべてオフ
- LLM機能はオプトイン（コスト・プライバシー考慮）
- LiteLLM経由で任意のLLMプロバイダー/モデルを選択可能
- Webサービス用途では明示的に有効化

---

## 実装計画

### Phase 1: データモデル

**ファイル:** `src/pdf_translator/output/document_summary.py` (新規)

1. `DocumentSummary` dataclass 作成
   - 全フィールド定義（organization, summary, title_source 等を含む）
   - `to_dict()` / `from_dict()` メソッド
   - `has_content()` ヘルパーメソッド

### Phase 2: サムネイル生成

**ファイル:** `src/pdf_translator/output/thumbnail_generator.py` (新規)

1. `ThumbnailConfig` dataclass 作成
2. `ThumbnailGenerator` クラス作成
   - `generate()`: bytes出力
   - `generate_to_file()`: ファイル出力

### Phase 3: LLM統合モジュール（LiteLLM）

**ファイル:**
- `src/pdf_translator/llm/__init__.py` (新規ディレクトリ)
- `src/pdf_translator/llm/client.py` (新規) - LiteLLM統一クライアント
- `src/pdf_translator/llm/summary_generator.py` (新規) - 要約・メタデータ抽出

1. `LLMConfig` dataclass 作成（provider/model 分離設計）
2. `LLMClient` クラス作成
   - `generate()`: LiteLLM経由でテキスト生成
   - プロバイダー自動切り替え（gemini, openai, anthropic 等）
3. `LLMSummaryGenerator` クラス作成
   - `generate_summary()`: 原文Markdownから要約生成
   - `extract_metadata_fallback()`: 1ページ目からメタデータ抽出

**依存パッケージ追加:**
```toml
# pyproject.toml
[project.optional-dependencies]
llm = ["litellm>=1.80.0"]
```

**LiteLLM採用理由:**
- 100+プロバイダーを統一APIで利用可能
- `acompletion()` で非同期対応
- MIT License（Apache-2.0互換）
- 新モデル追加時もライブラリ更新のみで対応

### Phase 4: Markdown二重生成

**ファイル:** `src/pdf_translator/output/markdown_writer.py`

1. `write()` メソッドに `use_translated: bool = True` パラメータ追加
2. `use_translated=False` で原文Markdown生成

### Phase 5: サマリー抽出

**ファイル:** `src/pdf_translator/output/summary_extractor.py` (新規)

1. `SummaryExtractor` クラス作成
   - `extract()`: paragraphs からサマリー抽出
   - `_find_and_merge_by_category()`: カテゴリ検索・結合
   - `_get_first_page_text()`: LLMフォールバック用
   - LLM統合（フォールバック + 要約生成）

### Phase 6: パイプライン統合

**ファイル:** `src/pdf_translator/pipeline/translation_pipeline.py`

1. `TranslationResult` に `summary`, `markdown_original` フィールド追加
2. `PipelineConfig` に以下を追加:
   - `generate_thumbnail: bool`
   - `thumbnail_width: int`
   - `llm_summary: bool`
   - `llm_fallback: bool`
3. `_stage_markdown()` で原文・翻訳両方のMarkdown生成
4. `_stage_summary()` メソッド追加
5. `_translate_impl()` でサマリー生成を呼び出し

### Phase 7: JSON出力拡張

**ファイル:** `src/pdf_translator/output/translated_document.py`

1. `summary` セクションを JSON に含める
2. サムネイルファイルの別途保存
3. 原文Markdownファイルの保存

### Phase 8: CLI対応

**ファイル:** `src/pdf_translator/cli.py`

1. `--thumbnail` フラグ追加
2. `--thumbnail-width` オプション追加
3. `--llm-summary` フラグ追加
4. `--llm-provider` オプション追加（デフォルト: gemini）
5. `--llm-model` オプション追加（省略時: providerごとのデフォルトを自動選択）
6. `--llm-fallback` フラグ追加
7. プロバイダー対応APIキー環境変数読み取り（GEMINI_API_KEY, OPENAI_API_KEY, etc.）
8. `PipelineConfig` への伝播

### Phase 9: テスト

**ファイル:**
- `tests/test_document_summary.py` (新規)
- `tests/test_thumbnail_generator.py` (新規)
- `tests/test_summary_extractor.py` (新規)
- `tests/test_llm_summary_generator.py` (新規)

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/pdf_translator/output/document_summary.py` | **新規** - `DocumentSummary` dataclass（LLM強化版） |
| `src/pdf_translator/output/thumbnail_generator.py` | **新規** - `ThumbnailGenerator` クラス |
| `src/pdf_translator/output/summary_extractor.py` | **新規** - `SummaryExtractor` クラス（LLM統合） |
| `src/pdf_translator/llm/__init__.py` | **新規** - LLMモジュール初期化 |
| `src/pdf_translator/llm/client.py` | **新規** - `LLMConfig`, `LLMClient` クラス（LiteLLM統一インターフェース） |
| `src/pdf_translator/llm/summary_generator.py` | **新規** - `LLMSummaryGenerator` クラス |
| `src/pdf_translator/output/markdown_writer.py` | `use_translated` パラメータ追加 |
| `src/pdf_translator/pipeline/translation_pipeline.py` | `TranslationResult.summary/markdown_original`, `PipelineConfig` LLM設定追加 |
| `src/pdf_translator/output/translated_document.py` | `summary` セクション追加、原文Markdown保存 |
| `src/pdf_translator/cli.py` | `--thumbnail`, `--llm-summary`, `--llm-provider`, `--llm-model`, `--llm-fallback` オプション追加 |
| `pyproject.toml` | `[project.optional-dependencies]` に `llm = ["litellm>=1.80.0"]` 追加 |
| `tests/test_document_summary.py` | **新規** - DocumentSummary テスト |
| `tests/test_thumbnail_generator.py` | **新規** - サムネイル生成テスト |
| `tests/test_summary_extractor.py` | **新規** - サマリー抽出テスト |
| `tests/test_llm_summary_generator.py` | **新規** - LLM要約生成テスト |
| `tests/test_llm_client.py` | **新規** - LLMClient テスト |

---

## リスクと対策

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|-------|------|
| タイトル/Abstract未検出（レイアウト） | 中 | 低 | LLMフォールバックで補完 |
| LLM API呼び出し失敗 | 低 | 低 | エラーをログし、`None` で続行（オプショナル機能） |
| LLM APIコスト | 中 | 中 | オプトイン制、summary/fallback 個別制御可能 |
| サムネイル生成失敗 | 低 | 低 | エラーをログし、`thumbnail=None` で続行 |
| JSON肥大化（サムネイル埋め込み時） | 中 | 中 | `include_thumbnail=False` をデフォルトに |
| 既存テスト破壊 | 低 | 低 | 新規フィールドはオプショナル |
| LLM APIキー未設定 | 中 | 低 | LLM機能使用時のみエラー、それ以外は正常動作。プロバイダーに応じた環境変数名を案内 |
| LiteLLMバージョン互換性 | 低 | 低 | `>=1.80.0` でバージョン固定、CIでテスト |

**後方互換性:**
- `DocumentSummary` は新規追加のため影響なし
- `TranslationResult.summary`, `markdown_original` はオプショナル（`None` がデフォルト）
- CLI オプションはデフォルトオフ
- LLM機能は `litellm` をオプショナル依存として追加（MIT License、Apache-2.0互換）

---

## 参考

- [Issue #62](https://github.com/Mega-Gorilla/pdf-translator/issues/62)
- [Issue #61](https://github.com/Mega-Gorilla/pdf-translator/issues/61) - ベンチマーク結果
- `examples/outputs/sample_llama_translated.json` - 現在のJSON出力例
- `src/pdf_translator/output/image_extractor.py` - pypdfium2レンダリング参考
- [LiteLLM Documentation](https://docs.litellm.ai/) - 統一LLMライブラリ
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) - サポートプロバイダー一覧
- [LiteLLM GitHub](https://github.com/BerriAI/litellm) - MIT License

---

## 将来対応（別 Issue）

| 機能 | 概要 | 優先度 |
|------|------|--------|
| 著者情報抽出 | LLMで著者名リストを抽出 | 中 |
| キーワード抽出 | LLMでキーワードを自動抽出 | 低 |
| 複数タイトル対応 | 副題がある場合の結合 | 低 |

※ LLMモデル選択はLiteLLM採用により実装済み（100+プロバイダー対応）

---

## 変更履歴

| 日付 | 変更内容 |
|------|---------|
| 2026-01-14 | 初版作成 |
| 2026-01-14 | レビューFB対応: タイトル翻訳経路、サムネイル出力仕様、複数段落結合ルールを明記 |
| 2026-01-15 | LLM統合追加: Gemini 3.0 Flashによる要約生成、メタデータフォールバック、Markdown二重生成 |
| 2026-01-15 | Re-Review対応: Organization常時抽出、入力サイズ制御追記、summary_max_tokens削除、サムネイルPNG固定、モデル名統一 |
| 2026-01-15 | LiteLLM採用: 統一LLMインターフェース導入、100+プロバイダー対応、provider/model CLI引数追加 |
| 2026-01-15 | Re-Review 2対応: モデル自動選択ルール追加、provider/model表記明確化、マルチプロバイダー入力サイズ制御 |
| 2026-01-15 | Re-Review 3対応: Phase 8 CLI記述統一、LLMConfig import修正、LLM表現汎化 |
| 2026-01-15 | Re-Review 4対応: サムネイル理由文言修正(PNG固定)、DocumentSummary Literal import追加 |
| 2026-01-15 | **Issue #66 により構造変更**: DocumentSummary → BaseSummary/TranslatedSummary に分離。詳細は [issue-66-multilingual-json.md](./issue-66-multilingual-json.md) 参照 |
