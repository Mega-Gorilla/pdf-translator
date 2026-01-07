# Markdown出力機能 設計書

Issue: #5

## 1. 概要

### 1.1 目的

翻訳結果をMarkdown形式で出力する機能を実装し、編集・再利用しやすい形式でのエクスポートを可能にする。

### 1.2 ゴール

- PDF翻訳結果をMarkdown形式で出力
- レイアウト解析結果を活用した構造化出力
- 複数の出力オプションをサポート

### 1.3 スコープ外

- 画像の抽出・埋め込み（将来の拡張）
- 表のMarkdown変換（将来の拡張）
- 数式のLaTeX変換（将来の拡張）
- リスト・箇条書きの自動検出（将来の拡張）
  - Issue #5 に記載あり、現時点では段落として出力
  - PDFテキストからリスト構造を正確に検出するには、先頭文字パターン（「・」「-」「1.」等）の解析が必要

---

## 2. データフロー

### 2.1 基本フロー（単一パイプライン実行で複数出力）

**設計方針**: 1回のパイプライン実行で PDF + Markdown + JSON（オプション）を同時出力。
翻訳結果を使い回すことで、再翻訳なしに複数形式の出力が可能。

```
PDF
 │
 ▼
┌─────────────────────────────────────────────────────────────┐
│ TranslationPipeline._translate_impl()                        │
│  ├─ _stage_extract() → Paragraph[]                          │
│  ├─ _stage_analyze() → LayoutBlock[]                        │
│  ├─ _stage_categorize()                                     │
│  ├─ _stage_merge()                                          │
│  ├─ _stage_translate()  ← 翻訳は1回のみ                      │
│  │                                                          │
│  ├─ _stage_apply() → pdf_bytes (翻訳済みPDF)                 │
│  │                                                          │
│  └─ if markdown_output:                                     │
│       MarkdownWriter.write(paragraphs) → markdown           │
└─────────────────────────────────────────────────────────────┘
 │
 │ TranslationResult (単一オブジェクト)
 │   - pdf_bytes       → output.pdf
 │   - markdown        → output.md (オプション)
 │   - paragraphs      → output.json (オプション、TranslatedDocument経由)
 ▼
┌─────────────────────────────────────────────────────────────┐
│ 出力ファイル (output_pathから自動決定)                        │
│  ├─ output.pdf         (常に出力)                            │
│  ├─ output.md          (markdown_output=True時)             │
│  └─ output.json        (save_intermediate=True時)            │
└─────────────────────────────────────────────────────────────┘
```

**ポイント**:
- 翻訳処理は1回のみ実行（PDF生成とMarkdown生成で再翻訳しない）
- `paragraphs`を保持することで、同一の翻訳結果から複数形式を生成
- ファイル出力は`output_path`から自動的に`.md`/`.json`パスを決定

### 2.2 再生成フロー（中間データからMarkdown生成）

```
.json file (TranslatedDocument)
 │
 ▼
┌─────────────────────────────────────────┐
│ TranslatedDocument.from_json()           │
│  → paragraphs: list[Paragraph]          │
└─────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────┐
│ MarkdownWriter                           │
│  ├─ write() → Markdown string           │
│  └─ write_to_file() → .md file          │
└─────────────────────────────────────────┘
 │
 ▼
Markdown File (再翻訳なしで生成)
```

**ポイント**: 中間データ（JSON）を保存することで、再翻訳なしにMarkdown再生成が可能

---

## 3. API設計

### 3.1 MarkdownWriter クラス

```python
# src/pdf_translator/output/markdown_writer.py

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime

from pdf_translator.core.models import Paragraph


class MarkdownOutputMode(Enum):
    """Markdown出力モード"""
    TRANSLATED_ONLY = "translated_only"      # 訳文のみ（フォールバック: 原文）
    ORIGINAL_ONLY = "original_only"          # 原文のみ
    PARALLEL = "parallel"                    # 原文と訳文を並列


@dataclass
class MarkdownConfig:
    """Markdown出力設定"""
    output_mode: MarkdownOutputMode = MarkdownOutputMode.TRANSLATED_ONLY
    include_metadata: bool = True            # メタデータを含める
    include_page_breaks: bool = True         # ページ区切りを含める
    heading_offset: int = 0                  # 見出しレベルのオフセット (0-5)
    source_lang: Optional[str] = None        # 原文言語
    target_lang: Optional[str] = None        # 翻訳先言語
    source_filename: Optional[str] = None    # 元PDFファイル名
    # カテゴリマッピングのオーバーライド（詳細は Section 4.3 参照）
    category_mapping_overrides: Optional[dict[str, str]] = None


class MarkdownWriter:
    """Markdown出力ライター"""

    def __init__(self, config: Optional[MarkdownConfig] = None) -> None:
        self._config = config or MarkdownConfig()

    def write(self, paragraphs: list[Paragraph]) -> str:
        """ParagraphリストからMarkdown文字列を生成"""
        ...

    def write_to_file(
        self,
        paragraphs: list[Paragraph],
        output_path: Path
    ) -> None:
        """Markdown文字列をファイルに出力"""
        ...

    def _generate_metadata(self) -> str:
        """YAMLフロントマター形式のメタデータを生成"""
        ...

    def _paragraph_to_markdown(self, paragraph: Paragraph) -> str:
        """単一のParagraphをMarkdownに変換"""
        ...

    def _get_element_type(self, category: Optional[str]) -> str:
        """カテゴリからMarkdown要素タイプを取得（Section 4.4 参照）"""
        ...
```

### 3.2 TranslationResult の拡張

```python
# src/pdf_translator/pipeline/translation_pipeline.py

@dataclass
class TranslationResult:
    """Translation pipeline result.

    単一のパイプライン実行で生成される全出力を保持。
    """
    pdf_bytes: bytes                             # 翻訳済みPDF
    stats: dict[str, Any] | None = None
    side_by_side_pdf_bytes: bytes | None = None
    markdown: str | None = None                  # NEW: 生成されたMarkdown
    paragraphs: list[Paragraph] | None = None    # NEW: 中間データ（JSON保存用）
```

### 3.3 PipelineConfig の拡張

```python
@dataclass
class PipelineConfig:
    # 既存フィールド...

    # Markdown出力オプション
    markdown_output: bool = False
    markdown_mode: MarkdownOutputMode = MarkdownOutputMode.TRANSLATED_ONLY
    markdown_include_metadata: bool = True
    markdown_include_page_breaks: bool = True
    save_intermediate: bool = False  # 中間データ(JSON)を保存するか
```

### 3.3.1 _translate_impl での実装イメージ

```python
async def _translate_impl(
    self,
    pdf_path: Path,
    output_path: Path | None = None,
) -> TranslationResult:
    # ... 既存のパイプラインステージ ...

    # PDF生成
    pdf_bytes = await self._stage_apply(paragraphs, original_pdf)

    # Markdown生成（有効時）
    markdown: str | None = None
    if self._config.markdown_output:
        writer = MarkdownWriter(MarkdownConfig(
            output_mode=self._config.markdown_mode,
            include_metadata=self._config.markdown_include_metadata,
            include_page_breaks=self._config.markdown_include_page_breaks,
            source_lang=self._config.source_lang,
            target_lang=self._config.target_lang,
            source_filename=pdf_path.name,
        ))
        markdown = writer.write(paragraphs)

    # ファイル出力
    if output_path is not None:
        # PDF出力（常に）
        output_path.write_bytes(pdf_bytes)

        # Markdown出力（有効時）
        if markdown:
            md_path = output_path.with_suffix(".md")
            md_path.write_text(markdown, encoding="utf-8")

        # 中間データ出力（有効時）
        if self._config.save_intermediate:
            json_path = output_path.with_suffix(".json")
            doc = TranslatedDocument.from_pipeline_result(
                paragraphs=paragraphs,
                source_file=pdf_path.name,
                source_lang=self._config.source_lang,
                target_lang=self._config.target_lang,
                translator_backend=self._translator.name,
                total_pages=len(original_pdf),  # PDF実ページ数
            )
            doc.save(json_path)

    # 結果を返す（paragraphsはsave_intermediate時のみ保持）
    return TranslationResult(
        pdf_bytes=pdf_bytes,
        stats=stats,
        markdown=markdown,
        paragraphs=paragraphs if self._config.save_intermediate else None,
    )
```

### 3.4 中間データ形式（再翻訳なしMarkdown生成用）

#### Paragraph シリアライズ拡張

```python
# src/pdf_translator/core/models.py に追加

@dataclass
class Paragraph:
    # 既存フィールド...

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "page_number": self.page_number,
            "text": self.text,
            "block_bbox": self.block_bbox.to_dict(),
            "line_count": self.line_count,
            "original_font_size": self.original_font_size,
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
            "rotation": self.rotation,
            "alignment": self.alignment,
        }
        # Optional fields
        if self.category is not None:
            result["category"] = self.category
        if self.category_confidence is not None:
            result["category_confidence"] = self.category_confidence
        if self.translated_text is not None:
            result["translated_text"] = self.translated_text
        if self.adjusted_font_size is not None:
            result["adjusted_font_size"] = self.adjusted_font_size
        if self.font_name is not None:
            result["font_name"] = self.font_name
        if self.text_color is not None:
            result["text_color"] = self.text_color.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paragraph:
        """Create from dictionary."""
        text_color = (
            Color.from_dict(data["text_color"])
            if "text_color" in data
            else None
        )
        return cls(
            id=data["id"],
            page_number=int(data["page_number"]),
            text=data["text"],
            block_bbox=BBox.from_dict(data["block_bbox"]),
            line_count=int(data["line_count"]),
            original_font_size=float(data.get("original_font_size", 12.0)),
            category=data.get("category"),
            category_confidence=data.get("category_confidence"),
            translated_text=data.get("translated_text"),
            adjusted_font_size=data.get("adjusted_font_size"),
            is_bold=data.get("is_bold", False),
            is_italic=data.get("is_italic", False),
            font_name=data.get("font_name"),
            text_color=text_color,
            rotation=float(data.get("rotation", 0.0)),
            alignment=data.get("alignment", "left"),
        )
```

#### TranslatedDocument クラス

```python
# src/pdf_translator/output/translated_document.py

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pdf_translator.core.models import Paragraph

TRANSLATED_DOC_VERSION = "1.0.0"


@dataclass
class TranslatedDocumentMetadata:
    """翻訳済みドキュメントのメタデータ"""
    source_file: str
    source_lang: str
    target_lang: str
    translated_at: str
    translator_backend: str
    page_count: int
    paragraph_count: int
    translated_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "translated_at": self.translated_at,
            "translator_backend": self.translator_backend,
            "page_count": self.page_count,
            "paragraph_count": self.paragraph_count,
            "translated_count": self.translated_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslatedDocumentMetadata":
        return cls(**data)


@dataclass
class TranslatedDocument:
    """翻訳済みドキュメント（中間データ形式）

    翻訳結果を保存し、再翻訳なしでMarkdown等を再生成可能にする。
    """
    metadata: TranslatedDocumentMetadata
    paragraphs: list[Paragraph]

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        data = {
            "version": TRANSLATED_DOC_VERSION,
            "metadata": self.metadata.to_dict(),
            "paragraphs": [p.to_dict() for p in self.paragraphs],
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> "TranslatedDocument":
        """Create from JSON string."""
        data = json.loads(json_str)
        version = data.get("version", "unknown")
        if version != TRANSLATED_DOC_VERSION:
            raise ValueError(
                f"Unsupported version: {version} (expected {TRANSLATED_DOC_VERSION})"
            )
        return cls(
            metadata=TranslatedDocumentMetadata.from_dict(data["metadata"]),
            paragraphs=[Paragraph.from_dict(p) for p in data["paragraphs"]],
        )

    @classmethod
    def load(cls, path: Path) -> "TranslatedDocument":
        """Load from JSON file."""
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_pipeline_result(
        cls,
        paragraphs: list[Paragraph],
        source_file: str,
        source_lang: str,
        target_lang: str,
        translator_backend: str,
        total_pages: int,  # PDF実ページ数を明示的に渡す
    ) -> "TranslatedDocument":
        """Create from pipeline result.

        Args:
            paragraphs: 翻訳済みParagraphリスト
            source_file: 元PDFファイル名
            source_lang: 原文言語コード
            target_lang: 翻訳先言語コード
            translator_backend: 使用した翻訳バックエンド名
            total_pages: PDF実ページ数（空ページ含む）

        Note:
            page_countはparagraphsから算出せず、PDF実ページ数を使用する。
            空ページ（テキストなし）があってもページ数が欠落しないようにするため。
        """
        translated_count = sum(1 for p in paragraphs if p.translated_text)
        metadata = TranslatedDocumentMetadata(
            source_file=source_file,
            source_lang=source_lang,
            target_lang=target_lang,
            translated_at=datetime.now().isoformat(),
            translator_backend=translator_backend,
            page_count=total_pages,  # PDF実ページ数を使用
            paragraph_count=len(paragraphs),
            translated_count=translated_count,
        )
        return cls(metadata=metadata, paragraphs=paragraphs)
```

#### 使用例

```python
# 基本的な使い方：PDF + Markdown を同時出力
config = PipelineConfig(
    target_lang="ja",
    markdown_output=True,           # Markdown出力を有効化
    markdown_mode=MarkdownOutputMode.TRANSLATED_ONLY,
)
pipeline = TranslationPipeline(config)
result = await pipeline.translate(
    Path("sample.pdf"),
    output_path=Path("output/sample_ja.pdf"),
)
# 自動的に output/sample_ja.pdf と output/sample_ja.md が出力される

# 中間データも保存する場合
config = PipelineConfig(
    target_lang="ja",
    markdown_output=True,
    save_intermediate=True,         # JSONも保存
)
# sample_ja.pdf, sample_ja.md, sample_ja.json が出力される

# 後からMarkdownを再生成（再翻訳なし）
doc = TranslatedDocument.load(Path("output/sample_ja.json"))
writer = MarkdownWriter(MarkdownConfig(
    output_mode=MarkdownOutputMode.PARALLEL,  # 今度は並列モードで
    source_lang=doc.metadata.source_lang,
    target_lang=doc.metadata.target_lang,
    source_filename=doc.metadata.source_file,
))
markdown = writer.write(doc.paragraphs)
Path("output/sample_ja_parallel.md").write_text(markdown, encoding="utf-8")
```

---

## 4. カテゴリ → Markdown マッピング

### 4.1 要素タイプ文字列

カテゴリから Markdown 要素へのマッピングは文字列で指定する。

| 要素タイプ | 文字列 | Markdown出力 |
|------------|--------|--------------|
| 見出し1 | `"h1"` | `# テキスト` |
| 見出し2 | `"h2"` | `## テキスト` |
| 見出し3 | `"h3"` | `### テキスト` |
| 見出し4 | `"h4"` | `#### テキスト` |
| 見出し5 | `"h5"` | `##### テキスト` |
| 見出し6 | `"h6"` | `###### テキスト` |
| 段落 | `"p"` | `テキスト` |
| 引用 | `"blockquote"` | `> テキスト` |
| コード(インライン) | `"code"` | `` `テキスト` `` |
| コード(ブロック) | `"code_block"` | ` ```テキスト``` ` |
| イタリック | `"italic"` | `*テキスト*` |
| スキップ | `"skip"` | (出力しない) |

### 4.2 デフォルトカテゴリマッピング（全25カテゴリ）

PP-DocLayoutV2 の全カテゴリに対するデフォルトマッピング：

```python
# src/pdf_translator/output/markdown_writer.py

DEFAULT_CATEGORY_MAPPING: dict[str, str] = {
    # === テキスト系 ===
    "doc_title": "h1",           # 文書タイトル → H1
    "paragraph_title": "h2",     # セクション見出し → H2
    "text": "p",                 # 本文 → 段落
    "vertical_text": "p",        # 縦書きテキスト → 段落
    "abstract": "blockquote",    # 概要 → 引用
    "aside_text": "blockquote",  # 補足テキスト → 引用

    # === 図表系 ===
    "figure_title": "italic",    # 図キャプション → イタリック
    "table": "p",                # 表 → 段落（将来: Markdown表）
    "chart": "skip",             # グラフ → スキップ
    "image": "skip",             # 画像 → スキップ

    # === 数式系 ===
    "inline_formula": "code",        # インライン数式 → コード
    "display_formula": "code_block", # ディスプレイ数式 → コードブロック
    "algorithm": "code_block",       # アルゴリズム → コードブロック
    "formula_number": "skip",        # 数式番号 → スキップ

    # === 参照系 ===
    "reference": "p",            # 参考文献 → 段落
    "reference_content": "p",    # 参考文献内容 → 段落
    "footnote": "p",             # 脚注 → 段落
    "vision_footnote": "p",      # 視覚的脚注 → 段落

    # === ナビゲーション系（通常スキップ） ===
    "header": "skip",            # ヘッダー → スキップ
    "header_image": "skip",      # ヘッダー内画像 → スキップ
    "footer": "skip",            # フッター → スキップ
    "footer_image": "skip",      # フッター内画像 → スキップ
    "number": "skip",            # ページ番号等 → スキップ

    # === その他 ===
    "seal": "skip",              # 印鑑 → スキップ
    "content": "p",              # コンテンツ → 段落
    "unknown": "p",              # 未知 → 段落
}

# カテゴリなし（None）の場合のデフォルト
DEFAULT_NONE_CATEGORY_MAPPING: str = "p"  # 段落として出力
```

### 4.3 カテゴリマッピングのカスタマイズ

ユーザーは `category_mapping_overrides` で部分的にマッピングを上書きできる：

```python
@dataclass
class MarkdownConfig:
    # ... 既存フィールド ...

    # カテゴリマッピングのオーバーライド（部分指定可能）
    category_mapping_overrides: dict[str, str] | None = None
```

**使用例**:

```python
# デフォルト: doc_title → h1, paragraph_title → h2
config = MarkdownConfig()

# ユースケース1: 見出しレベルを変更（H2から開始）
config = MarkdownConfig(
    category_mapping_overrides={
        "doc_title": "h2",
        "paragraph_title": "h3",
    }
)

# ユースケース2: abstract を見出しに変更
config = MarkdownConfig(
    category_mapping_overrides={
        "abstract": "h3",  # blockquote → h3
    }
)

# ユースケース3: heading_offset との組み合わせ
config = MarkdownConfig(
    heading_offset=1,  # 全見出しを +1
)
# → doc_title: h1+1=h2, paragraph_title: h2+1=h3

# ユースケース4: ヘッダー/フッターも出力したい
config = MarkdownConfig(
    category_mapping_overrides={
        "header": "p",
        "footer": "p",
    }
)

# ユースケース5: 数式をLaTeX形式で出力（将来拡張用）
config = MarkdownConfig(
    category_mapping_overrides={
        "inline_formula": "latex_inline",   # 将来: $...$
        "display_formula": "latex_block",   # 将来: $$...$$
    }
)
```

### 4.4 マッピング解決ロジック

```python
def _get_element_type(self, category: str | None) -> str:
    """カテゴリからMarkdown要素タイプを取得"""
    if category is None:
        return DEFAULT_NONE_CATEGORY_MAPPING  # "p"

    # 1. オーバーライドを優先チェック
    if self._config.category_mapping_overrides:
        if category in self._config.category_mapping_overrides:
            return self._config.category_mapping_overrides[category]

    # 2. デフォルトマッピングを使用
    return DEFAULT_CATEGORY_MAPPING.get(category, "p")

def _apply_heading_offset(self, element_type: str) -> str:
    """見出しにオフセットを適用"""
    if not element_type.startswith("h") or len(element_type) != 2:
        return element_type

    try:
        level = int(element_type[1])
        new_level = min(6, max(1, level + self._config.heading_offset))
        return f"h{new_level}"
    except ValueError:
        return element_type
```

**脚注について**: 現行データモデル（`Paragraph`）には脚注番号フィールドがないため、
Markdown脚注記法 `[^n]` ではなく、通常の段落として出力する。
本格的な脚注対応は将来の拡張とする（データモデルへの `footnote_number` 追加が必要）。

### 4.5 テキスト選択ルール（フォールバック）

`TRANSLATED_ONLY` モードでは以下の優先順位でテキストを選択:

1. `translated_text` があれば使用
2. `translated_text` がなければ `text`（原文）にフォールバック

**理由**: 非翻訳カテゴリ（`doc_title`, `paragraph_title`, `footnote`, `reference` 等）は
翻訳対象外のため `translated_text` が `None` となる。空見出しや欠落を防ぐため、
原文をそのまま出力する。

```python
def _get_display_text(self, paragraph: Paragraph) -> str:
    """出力モードに応じたテキストを取得"""
    if self._config.output_mode == MarkdownOutputMode.TRANSLATED_ONLY:
        # 訳文優先、なければ原文にフォールバック
        return paragraph.translated_text or paragraph.text
    elif self._config.output_mode == MarkdownOutputMode.ORIGINAL_ONLY:
        return paragraph.text
    else:  # PARALLEL
        # 両方返す（呼び出し元で処理）
        return paragraph.text
```

### 4.6 スタイル変換

| Paragraph属性 | Markdown |
|---------------|----------|
| `is_bold=True` | `**text**` |
| `is_italic=True` | `*text*` |
| `is_bold=True, is_italic=True` | `***text***` |

### 4.7 段落の読み順

現行実装では、段落は pdftext の出力順（ブロック順）で処理される。
多段組（マルチカラム）レイアウトの読み順は pdftext の解析に依存する。

**現状の制限**:
- pdftext はY座標順でブロックを出力する傾向があり、多段組では列が交互に並ぶ可能性がある
- 正確な列順序の検出には、レイアウト解析（PP-DocLayout等）との連携が必要

**将来の拡張案**:
- `_stage_analyze()` で検出した列構造を使用した並べ替え
- `LayoutBlock` の位置情報に基づく列グループ化

現時点では、読み順の問題は許容し、将来のレイアウト解析強化で対応する。

### 4.8 ページ区切り

ページ番号は `Paragraph.page_number`（0始まり）を1始まりに変換して表示:

```python
# page_number は 0-indexed なので +1 して表示
display_page = paragraph.page_number + 1
page_break = f"---\n<!-- Page {display_page} -->\n"
```

```markdown
---
<!-- Page 2 -->

```

---

## 5. 出力フォーマット例

### 5.1 訳文のみ (`TRANSLATED_ONLY`)

```markdown
---
title: Sample Document
source_lang: en
target_lang: ja
translated_at: 2026-01-06T12:00:00
source_file: sample.pdf
---

# 文書タイトル

> これは概要です。この文書では...

## はじめに

これは本文の段落です。翻訳されたテキストがここに表示されます。

---
<!-- Page 2 -->

## 方法

次のセクションの内容...
```

### 5.2 並列出力 (`PARALLEL`)

```markdown
---
title: Sample Document
source_lang: en
target_lang: ja
---

# Document Title
# 文書タイトル

> This is the abstract. This document describes...
> これは概要です。この文書では...

## Introduction
## はじめに

This is a body paragraph. The translated text appears here.
これは本文の段落です。翻訳されたテキストがここに表示されます。
```

---

## 6. ファイル構成

```
src/pdf_translator/
├── output/                          # NEW
│   ├── __init__.py
│   ├── markdown_writer.py           # MarkdownWriter, MarkdownConfig
│   └── translated_document.py       # TranslatedDocument (中間データ)
├── pipeline/
│   └── translation_pipeline.py      # 変更: TranslationResult拡張
└── core/
    └── models.py                    # 変更: Paragraph.to_dict/from_dict追加

tests/
├── test_markdown_writer.py          # NEW
└── test_translated_document.py      # NEW
```

---

## 7. 実装ステップ

### Phase 0: 中間データ形式（前提条件）

0. **Paragraph シリアライズ対応**
   - `models.py` に `Paragraph.to_dict()` / `from_dict()` 追加
   - 既存のシリアライズパターン（TextObject, LayoutBlock等）に準拠

1. **TranslatedDocument 実装**
   - `output/translated_document.py` 新規作成
   - `TranslatedDocumentMetadata` クラス
   - JSON保存/読み込み機能

2. **中間データテスト**
   - シリアライズ/デシリアライズ往復テスト
   - バージョン互換性テスト

### Phase 1: 基本実装

3. **`output/` モジュール作成**
   - `__init__.py`
   - `markdown_writer.py` (MarkdownWriter, MarkdownConfig, MarkdownOutputMode)

4. **カテゴリ→Markdown変換ロジック**
   - `_category_to_heading_level()`
   - `_paragraph_to_markdown()`

5. **基本テスト**
   - 単一段落の変換
   - 複数段落の変換
   - カテゴリ別変換

### Phase 2: パイプライン統合

6. **TranslationResult 拡張**
   - `markdown` フィールド追加
   - `paragraphs` フィールド追加

7. **PipelineConfig 拡張**
   - `markdown_output` オプション追加
   - `markdown_mode` オプション追加
   - `save_intermediate` オプション追加（中間データ保存）

8. **_translate_impl 統合**
   - PDF生成後にMarkdown生成を実行
   - 単一パイプライン実行で複数出力（PDF + MD + JSON）
   - ファイル出力の自動化（output_pathから派生）

9. **パイプライン統合テスト**

### Phase 3: 出力オプション

10. **メタデータ生成**
    - YAMLフロントマター
    - 翻訳情報

11. **ページ区切り**
    - ページ番号に基づく区切り挿入

12. **並列出力モード**
    - 原文/訳文の並列表示

### Phase 4: CLI統合

13. **CLIオプション追加**
    - `--markdown` フラグ（Markdown出力を有効化）
    - `--markdown-mode` オプション（translated_only / parallel）
    - `--heading-offset` オプション（見出しレベル一括オフセット）
    - `--category-map` オプション（カテゴリマッピング上書き）
    - `--save-json` フラグ（中間データ保存を有効化）
    - `--from-json` オプション（中間データからMarkdown生成、再翻訳なし）

    **CLIオプション使用例**:
    ```bash
    # 基本的なMarkdown出力
    translate-pdf input.pdf --markdown

    # 見出しレベルを変更（H2から開始）
    translate-pdf input.pdf --markdown \
        --category-map "doc_title=h2,paragraph_title=h3"

    # 一括オフセット
    translate-pdf input.pdf --markdown --heading-offset 1

    # abstract を見出しに変更
    translate-pdf input.pdf --markdown --category-map "abstract=h3"

    # 中間データも保存
    translate-pdf input.pdf --markdown --save-json
    ```

---

## 8. テスト計画

### 8.1 中間データ形式テスト

```python
class TestParagraphSerialization:
    def test_paragraph_to_dict(self): ...
    def test_paragraph_from_dict(self): ...
    def test_paragraph_roundtrip(self): ...
    def test_paragraph_optional_fields(self): ...

class TestTranslatedDocument:
    def test_to_json(self): ...
    def test_from_json(self): ...
    def test_save_and_load(self): ...
    def test_from_pipeline_result(self): ...
    def test_version_mismatch_raises_error(self): ...
```

### 8.2 Markdown Writer テスト

```python
class TestMarkdownWriter:
    def test_paragraph_to_markdown_text(self): ...
    def test_paragraph_to_markdown_title(self): ...
    def test_paragraph_to_markdown_with_bold(self): ...
    def test_paragraph_to_markdown_with_italic(self): ...
    def test_write_single_paragraph(self): ...
    def test_write_multiple_paragraphs(self): ...
    def test_write_with_page_breaks(self): ...
    def test_generate_metadata(self): ...
    def test_parallel_mode(self): ...
    def test_translated_only_mode(self): ...

class TestCategoryMapping:
    """カテゴリマッピングのテスト"""
    def test_default_mapping_doc_title(self):
        """doc_title → h1 のデフォルトマッピング"""
        ...
    def test_default_mapping_paragraph_title(self):
        """paragraph_title → h2 のデフォルトマッピング"""
        ...
    def test_default_mapping_text(self):
        """text → p のデフォルトマッピング"""
        ...
    def test_default_mapping_skip_categories(self):
        """header, footer, image 等が skip されること"""
        ...
    def test_category_override_heading_level(self):
        """doc_title=h2 等のオーバーライド"""
        ...
    def test_category_override_element_type(self):
        """abstract=h3 等の要素タイプ変更"""
        ...
    def test_heading_offset_applies_to_all(self):
        """heading_offset が全見出しに適用されること"""
        ...
    def test_heading_offset_with_override(self):
        """heading_offset と override の組み合わせ"""
        ...
    def test_none_category_uses_default(self):
        """category=None の場合 "p" が使用されること"""
        ...
    def test_unknown_category_uses_default(self):
        """未知のカテゴリは "p" になること"""
        ...
```

### 8.3 統合テスト

```python
class TestMarkdownIntegration:
    def test_pipeline_with_markdown_output(self):
        """markdown_output=True時にresult.markdownが生成される"""
        ...

    def test_single_run_outputs_pdf_and_markdown(self):
        """1回のパイプライン実行でPDFとMarkdownが同時出力される"""
        ...

    def test_markdown_file_auto_generated(self):
        """output_path指定時に.mdファイルが自動生成される"""
        ...

    def test_save_intermediate_creates_json(self):
        """save_intermediate=True時に.jsonファイルが作成される"""
        ...

    def test_regenerate_from_json_without_retranslation(self):
        """JSONから再生成時に翻訳APIが呼ばれない"""
        ...

    def test_different_markdown_modes_from_same_json(self):
        """同一JSONから異なるモードでMarkdown生成"""
        ...
```

---

## 9. 将来の拡張

### 9.1 画像抽出

```markdown
![Figure 1](./images/figure1.png)
*図1: サンプル画像*
```

### 9.2 表のMarkdown変換

```markdown
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

### 9.3 数式のLaTeX変換

```markdown
$$E = mc^2$$
```

---

## 10. 参考

- Issue #5: feat: Markdown出力機能の実装
- Issue #4: 翻訳パイプライン（依存Issue、完了済み）
- `src/pdf_translator/core/models.py`: Paragraph, RawLayoutCategory
- `src/pdf_translator/pipeline/translation_pipeline.py`: TranslationPipeline
