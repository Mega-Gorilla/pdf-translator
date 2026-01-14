# Issue #62: Webサービス向けドキュメントサマリー機能

## 概要

PDF翻訳Webサービス構築に向けて、翻訳結果からドキュメントのサマリー情報（タイトル、Abstract、サムネイル）を抽出・出力する機能を追加する。

**関連 Issue**: [#62](https://github.com/Mega-Gorilla/pdf-translator/issues/62)

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

著者情報は専用カテゴリが存在しないため、本Issueのスコープ外とする。

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

### 1. タイトル/Abstractの翻訳経路

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
from typing import Any


@dataclass
class DocumentSummary:
    """Document summary for web service integration.

    Contains essential metadata extracted from translated documents
    for dashboard display, search, and document management.

    Attributes:
        title: Original document title (from doc_title category).
        title_translated: Translated document title.
        abstract: Original abstract text (from abstract category).
        abstract_translated: Translated abstract text.
        thumbnail_path: Relative path to thumbnail file (primary reference).
        thumbnail_width: Thumbnail width in pixels.
        thumbnail_height: Thumbnail height in pixels.
        page_count: Total number of pages in the document.
        source_lang: Source language code (e.g., "en").
        target_lang: Target language code (e.g., "ja").
    """

    # Title (from doc_title category)
    title: str | None = None
    title_translated: str | None = None

    # Abstract (from abstract category, may be merged from multiple paragraphs)
    abstract: str | None = None
    abstract_translated: str | None = None

    # Thumbnail (first page of original PDF)
    thumbnail_path: str | None = None  # Relative path to thumbnail file
    thumbnail_width: int = 0
    thumbnail_height: int = 0

    # Metadata
    page_count: int = 0
    source_lang: str = ""
    target_lang: str = ""

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
            "thumbnail_path": self.thumbnail_path,
            "thumbnail_width": self.thumbnail_width,
            "thumbnail_height": self.thumbnail_height,
            "page_count": self.page_count,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
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
            thumbnail_path=data.get("thumbnail_path"),
            thumbnail_width=data.get("thumbnail_width", 0),
            thumbnail_height=data.get("thumbnail_height", 0),
            page_count=data.get("page_count", 0),
            source_lang=data.get("source_lang", ""),
            target_lang=data.get("target_lang", ""),
            _thumbnail_bytes=thumbnail_bytes,
        )

    def has_content(self) -> bool:
        """Check if summary has any meaningful content.

        Returns:
            True if at least title, abstract, or thumbnail is present.
        """
        return bool(self.title or self.abstract or self.thumbnail_path)
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
        format: Output format ("png" or "jpeg").
        quality: JPEG quality (1-100). Ignored for PNG.
        page_number: Page to render (0-indexed). Default: 0 (first page).
    """

    width: int = 400
    format: str = "png"
    quality: int = 85
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

            # Convert to bytes
            import io
            buffer = io.BytesIO()

            if self._config.format.lower() == "jpeg":
                pil_image = pil_image.convert("RGB")  # JPEG doesn't support alpha
                pil_image.save(buffer, format="JPEG", quality=self._config.quality)
            else:
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
- 設定可能なサイズ・フォーマット
- メモリ（bytes）とファイル出力の両方に対応

### 3. サマリー抽出ロジック

**決定: `SummaryExtractor` クラスで抽出ロジックを集約**

```python
# src/pdf_translator/output/summary_extractor.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pdf_translator.core.models import Paragraph
from pdf_translator.output.document_summary import DocumentSummary
from pdf_translator.output.thumbnail_generator import ThumbnailConfig, ThumbnailGenerator

if TYPE_CHECKING:
    from pdf_translator.translators.base import TranslatorBackend

logger = logging.getLogger(__name__)


class SummaryExtractor:
    """Extract document summary from translated paragraphs.

    Handles:
    - Title extraction from doc_title category (with additional translation)
    - Abstract extraction from abstract category (merged if multiple)
    - Thumbnail generation from first page
    """

    # Categories to extract
    TITLE_CATEGORY = "doc_title"
    ABSTRACT_CATEGORY = "abstract"

    def __init__(
        self,
        thumbnail_config: ThumbnailConfig | None = None,
    ) -> None:
        """Initialize SummaryExtractor.

        Args:
            thumbnail_config: Configuration for thumbnail generation.
        """
        self._thumbnail_config = thumbnail_config or ThumbnailConfig()

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
    ) -> DocumentSummary:
        """Extract document summary from paragraphs.

        Args:
            paragraphs: List of translated paragraphs.
            pdf_path: Path to original PDF (for thumbnail).
            output_dir: Directory to save thumbnail.
            output_stem: Base filename for outputs (e.g., "paper_translated").
            source_lang: Source language code.
            target_lang: Target language code.
            page_count: Total page count.
            generate_thumbnail: Whether to generate thumbnail.
            translator: Translator backend for title translation.

        Returns:
            DocumentSummary with extracted information.
        """
        # Extract and merge title (may be multiple paragraphs)
        title, title_translated = self._find_and_merge_by_category(
            paragraphs, self.TITLE_CATEGORY
        )

        # If title exists but not translated, translate it now
        if title and not title_translated and translator:
            try:
                title_translated = await translator.translate(
                    title, source_lang, target_lang
                )
            except Exception as e:
                logger.warning("Failed to translate title: %s", e)

        # Extract and merge abstract (may be multiple paragraphs)
        abstract, abstract_translated = self._find_and_merge_by_category(
            paragraphs, self.ABSTRACT_CATEGORY
        )

        # Generate thumbnail
        thumbnail_path = None
        thumbnail_bytes = None
        thumb_width = 0
        thumb_height = 0

        if generate_thumbnail and pdf_path.exists():
            try:
                generator = ThumbnailGenerator(self._thumbnail_config)
                thumbnail_bytes, thumb_width, thumb_height = generator.generate(pdf_path)

                # Save thumbnail file
                thumbnail_filename = f"{output_stem}_thumbnail.png"
                thumbnail_file = output_dir / thumbnail_filename
                thumbnail_file.write_bytes(thumbnail_bytes)
                thumbnail_path = thumbnail_filename  # Relative path

                logger.debug("Generated thumbnail: %s", thumbnail_file)
            except Exception as e:
                logger.warning("Failed to generate thumbnail: %s", e)

        return DocumentSummary(
            title=title,
            title_translated=title_translated,
            abstract=abstract,
            abstract_translated=abstract_translated,
            thumbnail_path=thumbnail_path,
            thumbnail_width=thumb_width,
            thumbnail_height=thumb_height,
            page_count=page_count,
            source_lang=source_lang,
            target_lang=target_lang,
            _thumbnail_bytes=thumbnail_bytes,
        )

    @staticmethod
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
        original_parts = [p.text for p in matched if p.text]
        translated_parts = [p.translated_text for p in matched if p.translated_text]

        original = "\n\n".join(original_parts) if original_parts else None
        translated = "\n\n".join(translated_parts) if translated_parts else None

        return original, translated
```

**理由:**
- 抽出ロジックを独立したクラスに集約
- **複数段落の結合**: 同一カテゴリの段落を `\n\n` で結合
- **タイトルの追加翻訳**: `doc_title` は翻訳対象外のため、サマリー抽出時に追加で翻訳
- サムネイル生成のエラーを吸収（オプショナル機能）
- サムネイルはファイル保存し、相対パスを `thumbnail_path` に記録

### 4. TranslationResult への統合

**決定: `summary` フィールドを追加**

```python
# src/pdf_translator/pipeline/translation_pipeline.py

@dataclass
class TranslationResult:
    """Translation pipeline result."""

    pdf_bytes: bytes
    stats: dict[str, Any] | None = None
    side_by_side_pdf_bytes: bytes | None = None
    markdown: str | None = None
    paragraphs: list[Paragraph] | None = None
    summary: DocumentSummary | None = None  # 新規追加
```

### 5. JSON出力拡張

**決定: `TranslatedDocument` に `summary` セクションを追加**

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
    "thumbnail_path": "sample_llama_translated_thumbnail.png",
    "thumbnail_width": 400,
    "thumbnail_height": 518,
    "page_count": 1,
    "source_lang": "en",
    "target_lang": "ja"
  },
  "paragraphs": [ ... ]
}
```

**サムネイルファイルの保存:**

```
output/
├── paper_translated.pdf
├── paper_translated.json              # summary.thumbnail_path を含む
├── paper_translated.md
└── paper_translated_thumbnail.png     # サムネイル画像ファイル
```

**Webサービス側での参照:**
- JSON から `summary.thumbnail_path` を取得
- 同一ディレクトリ内のファイルとして画像を取得

### 6. CLI オプション

**決定: `--thumbnail` オプションを追加**

```bash
# サムネイル生成なし（デフォルト、後方互換性）
uv run translate-pdf paper.pdf --save-intermediate

# サムネイル生成あり
uv run translate-pdf paper.pdf --save-intermediate --thumbnail

# サムネイルサイズ指定
uv run translate-pdf paper.pdf --save-intermediate --thumbnail --thumbnail-width 600
```

**理由:**
- 後方互換性のためデフォルトはオフ
- Webサービス用途では明示的に有効化

---

## 実装計画

### Phase 1: データモデル

**ファイル:** `src/pdf_translator/output/document_summary.py` (新規)

1. `DocumentSummary` dataclass 作成
   - フィールド定義
   - `to_dict()` / `from_dict()` メソッド
   - `has_content()` ヘルパーメソッド

### Phase 2: サムネイル生成

**ファイル:** `src/pdf_translator/output/thumbnail_generator.py` (新規)

1. `ThumbnailConfig` dataclass 作成
2. `ThumbnailGenerator` クラス作成
   - `generate()`: bytes出力
   - `generate_to_file()`: ファイル出力

### Phase 3: サマリー抽出

**ファイル:** `src/pdf_translator/output/summary_extractor.py` (新規)

1. `SummaryExtractor` クラス作成
   - `extract()`: paragraphs からサマリー抽出
   - `_find_first_by_category()`: カテゴリ検索

### Phase 4: パイプライン統合

**ファイル:** `src/pdf_translator/pipeline/translation_pipeline.py`

1. `TranslationResult` に `summary` フィールド追加
2. `PipelineConfig` に `generate_thumbnail` フィールド追加
3. `_stage_summary()` メソッド追加
4. `_translate_impl()` でサマリー生成を呼び出し

### Phase 5: JSON出力拡張

**ファイル:** `src/pdf_translator/output/translated_document.py`

1. `summary` セクションを JSON に含める
2. サムネイルファイルの別途保存

### Phase 6: CLI対応

**ファイル:** `src/pdf_translator/cli.py`

1. `--thumbnail` フラグ追加
2. `--thumbnail-width` オプション追加
3. `PipelineConfig` への伝播

### Phase 7: テスト

**ファイル:**
- `tests/test_document_summary.py` (新規)
- `tests/test_thumbnail_generator.py` (新規)
- `tests/test_summary_extractor.py` (新規)

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/pdf_translator/output/document_summary.py` | **新規** - `DocumentSummary` dataclass |
| `src/pdf_translator/output/thumbnail_generator.py` | **新規** - `ThumbnailGenerator` クラス |
| `src/pdf_translator/output/summary_extractor.py` | **新規** - `SummaryExtractor` クラス |
| `src/pdf_translator/pipeline/translation_pipeline.py` | `TranslationResult.summary`, `PipelineConfig.generate_thumbnail` 追加 |
| `src/pdf_translator/output/translated_document.py` | `summary` セクション追加 |
| `src/pdf_translator/cli.py` | `--thumbnail`, `--thumbnail-width` オプション追加 |
| `tests/test_document_summary.py` | **新規** - DocumentSummary テスト |
| `tests/test_thumbnail_generator.py` | **新規** - サムネイル生成テスト |
| `tests/test_summary_extractor.py` | **新規** - サマリー抽出テスト |

---

## リスクと対策

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|-------|------|
| タイトル/Abstract未検出 | 中 | 低 | `None` を返し、Webサービス側でフォールバック |
| サムネイル生成失敗 | 低 | 低 | エラーをログし、`thumbnail=None` で続行 |
| JSON肥大化（サムネイル埋め込み時） | 中 | 中 | `include_thumbnail=False` をデフォルトに |
| 既存テスト破壊 | 低 | 低 | 新規フィールドはオプショナル |

**後方互換性:**
- `DocumentSummary` は新規追加のため影響なし
- `TranslationResult.summary` はオプショナル（`None` がデフォルト）
- CLI オプションはデフォルトオフ

---

## 参考

- [Issue #62](https://github.com/Mega-Gorilla/pdf-translator/issues/62)
- [Issue #61](https://github.com/Mega-Gorilla/pdf-translator/issues/61) - ベンチマーク結果
- `examples/outputs/sample_llama_translated.json` - 現在のJSON出力例
- `src/pdf_translator/output/image_extractor.py` - pypdfium2レンダリング参考

---

## 将来対応（別 Issue）

| 機能 | 概要 | 優先度 |
|------|------|--------|
| 著者情報抽出 | `doc_title` 直後の `text` カテゴリからヒューリスティック抽出 | 低 |
| キーワード抽出 | Abstract からの自動キーワード抽出 | 低 |
| 複数タイトル対応 | 副題がある場合の結合 | 低 |

---

## 変更履歴

| 日付 | 変更内容 |
|------|---------|
| 2026-01-14 | 初版作成 |
| 2026-01-14 | レビューFB対応: タイトル翻訳経路、サムネイル出力仕様、複数段落結合ルールを明記 |
