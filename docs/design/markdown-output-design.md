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

---

## 2. データフロー

```
PDF
 │
 ▼
┌─────────────────────────────────────────┐
│ TranslationPipeline                      │
│  ├─ _stage_extract() → Paragraph[]      │
│  ├─ _stage_analyze() → LayoutBlock[]    │
│  ├─ _stage_categorize()                 │
│  ├─ _stage_merge()                      │
│  ├─ _stage_translate()                  │
│  └─ _stage_apply() → PDF bytes          │
└─────────────────────────────────────────┘
 │
 │  paragraphs (with translated_text, category)
 ▼
┌─────────────────────────────────────────┐
│ MarkdownWriter                           │
│  ├─ write() → Markdown string           │
│  └─ write_to_file() → .md file          │
└─────────────────────────────────────────┘
 │
 ▼
Markdown File
```

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
    TRANSLATED_ONLY = "translated_only"      # 訳文のみ
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

    def _category_to_heading_level(self, category: Optional[str]) -> Optional[int]:
        """カテゴリから見出しレベルを決定"""
        ...
```

### 3.2 TranslationResult の拡張

```python
# src/pdf_translator/pipeline/translation_pipeline.py

@dataclass
class TranslationResult:
    """Translation pipeline result."""
    pdf_bytes: bytes
    stats: dict[str, Any] | None = None
    side_by_side_pdf_bytes: bytes | None = None
    paragraphs: list[Paragraph] | None = None  # NEW: Markdown生成用
    markdown: str | None = None                 # NEW: 生成されたMarkdown
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
```

---

## 4. カテゴリ → Markdown マッピング

### 4.1 見出しレベル

| RawLayoutCategory | Markdown | 説明 |
|-------------------|----------|------|
| `doc_title` | `# H1` | 文書タイトル |
| `paragraph_title` | `## H2` | セクション見出し |
| `abstract` | `> blockquote` | 概要（引用形式） |
| `text` | paragraph | 本文 |
| `vertical_text` | paragraph | 縦書きテキスト |
| `aside_text` | `> blockquote` | 補足テキスト |
| `figure_title` | `*italic*` | 図キャプション |
| `footnote` | `[^n]` | 脚注 |
| `reference` | paragraph (small) | 参考文献 |

### 4.2 スタイル変換

| Paragraph属性 | Markdown |
|---------------|----------|
| `is_bold=True` | `**text**` |
| `is_italic=True` | `*text*` |
| `is_bold=True, is_italic=True` | `***text***` |

### 4.3 ページ区切り

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
│   └── markdown_writer.py
├── pipeline/
│   └── translation_pipeline.py      # 変更: TranslationResult拡張
└── core/
    └── models.py                    # 参照のみ

tests/
└── test_markdown_writer.py          # NEW
```

---

## 7. 実装ステップ

### Phase 1: 基本実装

1. **`output/` モジュール作成**
   - `__init__.py`
   - `markdown_writer.py` (MarkdownWriter, MarkdownConfig, MarkdownOutputMode)

2. **カテゴリ→Markdown変換ロジック**
   - `_category_to_heading_level()`
   - `_paragraph_to_markdown()`

3. **基本テスト**
   - 単一段落の変換
   - 複数段落の変換
   - カテゴリ別変換

### Phase 2: パイプライン統合

4. **TranslationResult 拡張**
   - `paragraphs` フィールド追加
   - `markdown` フィールド追加

5. **PipelineConfig 拡張**
   - `markdown_output` オプション追加
   - `markdown_mode` オプション追加

6. **パイプライン統合テスト**

### Phase 3: 出力オプション

7. **メタデータ生成**
   - YAMLフロントマター
   - 翻訳情報

8. **ページ区切り**
   - ページ番号に基づく区切り挿入

9. **並列出力モード**
   - 原文/訳文の並列表示

### Phase 4: CLI統合

10. **CLIオプション追加**
    - `--markdown` フラグ
    - `--markdown-mode` オプション

---

## 8. テスト計画

### 8.1 単体テスト

```python
class TestMarkdownWriter:
    def test_paragraph_to_markdown_text(self): ...
    def test_paragraph_to_markdown_title(self): ...
    def test_paragraph_to_markdown_with_bold(self): ...
    def test_paragraph_to_markdown_with_italic(self): ...
    def test_category_to_heading_level(self): ...
    def test_write_single_paragraph(self): ...
    def test_write_multiple_paragraphs(self): ...
    def test_write_with_page_breaks(self): ...
    def test_generate_metadata(self): ...
    def test_parallel_mode(self): ...
    def test_translated_only_mode(self): ...
```

### 8.2 統合テスト

```python
class TestMarkdownIntegration:
    def test_pipeline_with_markdown_output(self): ...
    def test_markdown_file_output(self): ...
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
