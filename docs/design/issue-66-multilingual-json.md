# Issue #66: 多言語対応JSON構造

## 概要

Issue #62 で実装されたドキュメントサマリー機能を拡張し、多言語対応のJSON構造を実装する。ベースファイル（原文）と翻訳ファイル（言語別差分）を分離し、Webサービス向けの効率的なデータ構造を実現する。

**関連 Issue**: [#66](https://github.com/Mega-Gorilla/pdf-translator/issues/66)

## 設計決定事項

- **v1サポート・後方互換性: なし**（開発初期段階、シンプルさ優先）
- **デフォルト形式: v2のみ**
- **サムネイル: 外部ファイル参照のみ**（base64埋め込みはオプション）
- **summary言語: target_langで直接LLM生成**

## ファイル構造

### 旧構造（Issue #62）

```
output/
├── paper_translated.pdf
├── paper_translated.json        # 全データが1ファイルに
├── paper_original.md
├── paper_translated.md
└── paper_translated_thumbnail.png
```

### 新構造（Issue #66）

```
output/
├── paper.json                   # ベースドキュメント（原文 + 共通メタデータ）
├── paper.ja.json                # 日本語翻訳差分
├── paper.zh.json                # 中国語翻訳差分（追加翻訳時）
├── paper_thumbnail.png          # サムネイル
├── paper.md                     # 原文Markdown
├── paper.ja.md                  # 日本語Markdown
├── paper.ja.pdf                 # 日本語PDF
└── paper.zh.pdf                 # 中国語PDF（追加翻訳時）
```

## データモデル

### BaseSummary

原文言語でのサマリー情報を保持。

```python
@dataclass
class BaseSummary:
    title: str | None = None
    abstract: str | None = None
    organization: str | None = None
    summary: str | None = None  # LLM生成（source_lang）
    thumbnail_path: str | None = None
    thumbnail_width: int = 0
    thumbnail_height: int = 0
    page_count: int = 0
    title_source: Literal["layout", "llm"] = "layout"
    abstract_source: Literal["layout", "llm"] = "layout"
    _thumbnail_bytes: bytes | None = None
```

### TranslatedSummary

翻訳言語でのサマリー情報を保持。

```python
@dataclass
class TranslatedSummary:
    title: str | None = None
    abstract: str | None = None
    summary: str | None = None  # LLM生成（target_lang）
```

### BaseDocument (paper.json)

```python
@dataclass
class BaseDocumentMetadata:
    source_file: str
    source_lang: str
    page_count: int
    paragraph_count: int

@dataclass
class BaseDocument:
    metadata: BaseDocumentMetadata
    paragraphs: list[Paragraph]  # translated_text を除外
    summary: BaseSummary | None = None
```

**JSON出力例:**

```json
{
  "schema_version": "2.0.0",
  "metadata": {
    "source_file": "paper.pdf",
    "source_lang": "en",
    "page_count": 10,
    "paragraph_count": 145
  },
  "summary": {
    "title": "LLaMA: Open and Efficient Foundation Language Models",
    "abstract": "We introduce LLaMA...",
    "organization": "Meta AI",
    "summary": "This paper introduces LLaMA...",
    "thumbnail_path": "paper_thumbnail.png",
    "thumbnail_width": 400,
    "thumbnail_height": 518,
    "page_count": 10,
    "title_source": "layout",
    "abstract_source": "layout"
  },
  "paragraphs": [
    {
      "id": "para_p0_b0",
      "page_number": 0,
      "text": "LLaMA: Open and Efficient...",
      "category": "doc_title"
    }
  ]
}
```

### TranslationDocument (paper.ja.json)

```python
@dataclass
class TranslationDocument:
    target_lang: str
    base_file: str
    translated_at: str
    translator_backend: str
    translated_count: int
    paragraphs: dict[str, str]  # {id: translated_text}
    summary: TranslatedSummary | None = None
```

**JSON出力例:**

```json
{
  "schema_version": "2.0.0",
  "target_lang": "ja",
  "base_file": "paper.json",
  "translated_at": "2026-01-15T10:30:00",
  "translator_backend": "google",
  "translated_count": 120,
  "summary": {
    "title": "LLaMA: オープンで効率的な基盤言語モデル",
    "abstract": "LLaMAを紹介します...",
    "summary": "本論文はLLaMAを紹介し..."
  },
  "paragraphs": {
    "para_p0_b1": "翻訳文1",
    "para_p0_b2": "翻訳文2"
  }
}
```

## 翻訳差分の表現ルール

### キー欠落 = 翻訳不要

```json
{
  "translated_count": 120,
  "paragraphs": {
    "para_p0_b1": "翻訳文1",
    "para_p0_b2": "翻訳文2"
    // para_p0_b0 (doc_title) はキー欠落 = 翻訳対象外カテゴリ
  }
}
```

### 整合性ルール

- `translated_count == len(paragraphs)`
- 未翻訳カテゴリ（doc_title, paragraph_title等）はキー欠落

## サマリー翻訳フロー

```
1. BaseSummary抽出（原文言語）
   - レイアウト解析 → title, abstract
   - LLMフォールバック → organization, (title), (abstract)
   - LLM要約生成 → summary (source_lang)

2. TranslatedSummary生成（翻訳言語）
   - 翻訳API → title, abstract
   - LLM要約生成 → summary (target_lang)
```

## 処理フロー

### 初回翻訳

```
PDF → レイアウト解析 → 翻訳 → BaseDocument + TranslationDocument 出力
```

### 追加翻訳（--base-file）

```
PDF + 既存BaseDocument → 翻訳のみ → TranslationDocument 出力
```

**--base-file 処理:**

| 処理 | 通常 | --base-file |
|------|------|-------------|
| レイアウト解析 | 実行 | スキップ |
| サムネイル生成 | 実行 | スキップ |
| 原文要約(LLM) | 実行 | スキップ |
| 翻訳API | 実行 | 実行 |
| PDF生成 | --pdf時 | --pdf時 |
| Markdown生成 | --markdown時 | --markdown時 |

## 実装ファイル一覧

| ファイル | 種別 | 内容 |
|---------|------|------|
| `src/pdf_translator/output/base_summary.py` | 新規 | BaseSummary dataclass |
| `src/pdf_translator/output/translated_summary.py` | 新規 | TranslatedSummary dataclass |
| `src/pdf_translator/output/base_document.py` | 新規 | BaseDocument dataclass |
| `src/pdf_translator/output/translation_document.py` | 新規 | TranslationDocument dataclass |
| `src/pdf_translator/output/base_document_writer.py` | 新規 | ベースJSON出力 |
| `src/pdf_translator/output/translation_writer.py` | 新規 | 翻訳JSON出力 |
| `src/pdf_translator/output/summary_extractor.py` | 修正 | BaseSummary返却 |
| `src/pdf_translator/llm/summary_generator.py` | 修正 | target_lang対応 |
| `src/pdf_translator/pipeline/translation_pipeline.py` | 修正 | 新構造対応 |
| `src/pdf_translator/cli.py` | 修正 | 命名規則変更 |
| `src/pdf_translator/output/document_summary.py` | 削除 | 分離完了 |
| `src/pdf_translator/output/translated_document.py` | 削除 | 分離完了 |

## 検証方法

```bash
# 初回翻訳（日本語）
uv run translate-pdf tests/fixtures/sample_llama.pdf -t ja --thumbnail --llm-summary
# → sample_llama.json + sample_llama.ja.json + sample_llama_thumbnail.png

# テスト実行
uv run pytest tests/test_base_summary.py tests/test_base_document.py -v
uv run pytest tests/test_translation_document.py tests/test_translated_summary.py -v

# 型チェック
uv run mypy src/pdf_translator/output/
```

## Issue #62 との関係

本設計はIssue #62の実装を基盤とし、以下を変更:

| 項目 | Issue #62 | Issue #66 |
|------|-----------|-----------|
| サマリー構造 | `DocumentSummary`（統合） | `BaseSummary` + `TranslatedSummary`（分離） |
| ドキュメント構造 | `TranslatedDocument`（統合） | `BaseDocument` + `TranslationDocument`（分離） |
| ファイル命名 | `*_translated.json` | `*.json` + `*.{lang}.json` |
| 多言語対応 | なし | 言語別差分ファイル |

## 変更履歴

| 日付 | 変更内容 |
|------|---------|
| 2026-01-15 | 初版作成 - Issue #66 実装完了 |
