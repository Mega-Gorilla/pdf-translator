# PDF Processor Module Design Document

## 1. 概要

本ドキュメントは、pypdfium2を使用したPDF処理ラッパーモジュール（Issue #1）の設計を定義する。

### 1.1 目的

- pypdfium2によるPDFオブジェクトの取得・操作・削除・挿入機能を提供
- PP-DocLayout（Issue #2）との連携のための中間データ形式を定義
- モジュール間の明確なインターフェースを確立

### 1.2 スコープ

| 対象 | 内容 |
|------|------|
| IN | テキストオブジェクトの抽出・削除・挿入 |
| IN | バウンディングボックス、フォント情報の取得 |
| IN | 中間データ形式（JSON）の入出力 |
| IN | 元PDFをテンプレートとしたテキスト層の編集 |
| OUT | 画像・図形オブジェクトの操作（元PDFから継承） |
| OUT | OCR機能 |

### 1.3 設計方針：テンプレートPDF方式

本モジュールは**元PDFをテンプレートとして保持し、テキスト層のみを編集する**方式を採用する。

```
[元PDF] ─── extract() ───→ [中間データJSON]
   │                              │
   │ (テンプレートとして保持)        │ (テキスト情報のみ)
   ↓                              ↓
[編集後PDF] ←── apply() ─────────┘
   │
   └─ 画像・図形・レイアウトは元PDFから継承
```

**理由**:
- 画像・図形・複雑なレイアウトの完全再現は困難かつスコープ外
- 翻訳ユースケースでは元PDFの非テキスト要素を保持することが望ましい
- 実装の複雑さを抑えつつ、実用的な機能を提供

---

## 2. 中間データ形式（Intermediate Data Schema）

### 2.1 設計原則

1. **テキスト情報の完全性**: テキスト層の編集に必要な全情報を保持
2. **拡張性**: PP-DocLayoutの分類情報を追加可能
3. **可読性**: 人間が読めるJSON形式

> **注**: 中間データはテキスト情報のみを保持し、画像・図形は元PDFから継承される（1.3節参照）

### 2.2 JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PDFIntermediateData",
  "type": "object",
  "required": ["version", "metadata", "pages"],
  "properties": {
    "version": {
      "type": "string",
      "description": "スキーマバージョン",
      "const": "1.0.0"
    },
    "metadata": {
      "$ref": "#/definitions/Metadata"
    },
    "pages": {
      "type": "array",
      "items": { "$ref": "#/definitions/Page" }
    }
  },
  "definitions": {
    "Metadata": {
      "type": "object",
      "required": ["source_file", "created_at"],
      "properties": {
        "source_file": { "type": "string" },
        "created_at": { "type": "string", "format": "date-time" },
        "page_count": { "type": "integer" },
        "pdf_version": { "type": "string" }
      }
    },
    "Page": {
      "type": "object",
      "required": ["page_number", "width", "height", "text_objects"],
      "properties": {
        "page_number": { "type": "integer", "minimum": 0 },
        "width": { "type": "number" },
        "height": { "type": "number" },
        "rotation": { "type": "integer", "default": 0 },
        "text_objects": {
          "type": "array",
          "items": { "$ref": "#/definitions/TextObject" }
        },
        "layout_blocks": {
          "type": "array",
          "items": { "$ref": "#/definitions/LayoutBlock" },
          "description": "PP-DocLayoutからの分類結果（オプション）"
        }
      }
    },
    "TextObject": {
      "type": "object",
      "required": ["id", "bbox", "text"],
      "properties": {
        "id": { "type": "string", "description": "一意識別子" },
        "bbox": { "$ref": "#/definitions/BBox" },
        "text": { "type": "string" },
        "font": { "$ref": "#/definitions/Font" },
        "color": { "$ref": "#/definitions/Color" },
        "transform": { "$ref": "#/definitions/Transform" },
        "char_positions": {
          "type": "array",
          "items": { "$ref": "#/definitions/CharPosition" },
          "description": "文字単位の位置情報（オプション）"
        }
      }
    },
    "Transform": {
      "type": "object",
      "description": "アフィン変換行列 [a, b, c, d, e, f] - 回転・スケール・傾斜を表現",
      "properties": {
        "a": { "type": "number", "default": 1.0, "description": "水平スケール" },
        "b": { "type": "number", "default": 0.0, "description": "垂直傾斜" },
        "c": { "type": "number", "default": 0.0, "description": "水平傾斜" },
        "d": { "type": "number", "default": 1.0, "description": "垂直スケール" },
        "e": { "type": "number", "default": 0.0, "description": "水平移動" },
        "f": { "type": "number", "default": 0.0, "description": "垂直移動" }
      }
    },
    "BBox": {
      "type": "object",
      "required": ["x0", "y0", "x1", "y1"],
      "properties": {
        "x0": { "type": "number", "description": "左下X座標" },
        "y0": { "type": "number", "description": "左下Y座標" },
        "x1": { "type": "number", "description": "右上X座標" },
        "y1": { "type": "number", "description": "右上Y座標" }
      },
      "description": "PDF座標系（原点は左下）"
    },
    "Font": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "size": { "type": "number" },
        "is_bold": { "type": "boolean", "default": false },
        "is_italic": { "type": "boolean", "default": false }
      }
    },
    "Color": {
      "type": "object",
      "properties": {
        "r": { "type": "integer", "minimum": 0, "maximum": 255 },
        "g": { "type": "integer", "minimum": 0, "maximum": 255 },
        "b": { "type": "integer", "minimum": 0, "maximum": 255 }
      }
    },
    "CharPosition": {
      "type": "object",
      "properties": {
        "char": { "type": "string", "maxLength": 1 },
        "bbox": { "$ref": "#/definitions/BBox" }
      }
    },
    "LayoutBlock": {
      "type": "object",
      "required": ["id", "bbox", "type"],
      "properties": {
        "id": { "type": "string" },
        "bbox": { "$ref": "#/definitions/BBox" },
        "type": {
          "type": "string",
          "enum": ["text", "title", "figure", "table", "caption", "formula", "header", "footer", "list"]
        },
        "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
        "text_object_ids": {
          "type": "array",
          "items": { "type": "string" },
          "description": "このブロックに含まれるTextObjectのID"
        }
      }
    }
  }
}
```

### 2.3 サンプルデータ

```json
{
  "version": "1.0.0",
  "metadata": {
    "source_file": "sample.pdf",
    "created_at": "2025-01-15T10:30:00Z",
    "page_count": 1,
    "pdf_version": "1.7"
  },
  "pages": [
    {
      "page_number": 0,
      "width": 595.0,
      "height": 842.0,
      "rotation": 0,
      "text_objects": [
        {
          "id": "text_001",
          "bbox": { "x0": 72.0, "y0": 750.0, "x1": 300.0, "y1": 770.0 },
          "text": "Sample Document Title",
          "font": { "name": "Helvetica-Bold", "size": 18.0, "is_bold": true },
          "color": { "r": 0, "g": 0, "b": 0 }
        },
        {
          "id": "text_002",
          "bbox": { "x0": 72.0, "y0": 700.0, "x1": 523.0, "y1": 720.0 },
          "text": "This is the body text of the document.",
          "font": { "name": "Times-Roman", "size": 12.0 },
          "color": { "r": 0, "g": 0, "b": 0 }
        }
      ],
      "layout_blocks": [
        {
          "id": "block_001",
          "bbox": { "x0": 72.0, "y0": 750.0, "x1": 300.0, "y1": 770.0 },
          "type": "title",
          "confidence": 0.95,
          "text_object_ids": ["text_001"]
        },
        {
          "id": "block_002",
          "bbox": { "x0": 72.0, "y0": 700.0, "x1": 523.0, "y1": 720.0 },
          "type": "text",
          "confidence": 0.98,
          "text_object_ids": ["text_002"]
        }
      ]
    }
  ]
}
```

---

## 3. 成功基準（Success Criteria）

### 3.1 テキスト層編集テスト

**定義**: 元PDFをテンプレートとして、テキスト層の削除・再挿入が正しく行われること。

```
[元PDF] ─── extract() ───→ [中間データJSON]
   │                              │
   │ (テンプレート)                 │ (編集)
   ↓                              ↓
[編集後PDF] ←── apply() ─────────┘
```

**検証ポイント**:
- 非テキスト要素（画像・図形）は元PDFから継承されている
- テキストの削除・再挿入が正しく行われている
- フォント・位置・変換行列が保持されている

### 3.2 検証項目

| 項目 | 検証方法 | 許容誤差 | 備考 |
|------|----------|----------|------|
| テキスト内容 | 文字列完全一致 | 0% | 必須 |
| テキスト位置 | bbox座標比較 | ±1.0pt | 必須 |
| フォント名 | 文字列一致 | - | 必須 |
| フォントサイズ | 数値比較 | ±0.5pt | 必須 |
| 変換行列 | 行列要素比較 | ±0.01 | 回転・傾斜がある場合 |
| ページサイズ | 数値比較 | ±0.1pt | 元PDFから継承 |
| 非テキスト要素 | 視覚的確認 | - | 元PDFから継承されること |

> **注**: 回転・傾斜テキストの完全再現が困難な場合、翻訳ユースケースでは「近似的な位置・サイズでの再挿入」を許容する。この場合、成功基準は「翻訳後テキストが読みやすい位置に配置されること」とする。

### 3.3 テストケース

#### TC-001: シンプルなテキストPDF
- **入力**: 単一ページ、テキストのみ（英語）
- **期待結果**: 全テキストオブジェクトが抽出・再挿入される

#### TC-002: 日本語テキストPDF
- **入力**: 日本語テキストを含むPDF
- **期待結果**: CIDフォントで正しく再現される

#### TC-003: 複数フォント・サイズ
- **入力**: タイトル（大）+ 本文（小）+ キャプション（イタリック）
- **期待結果**: 各フォント属性が保持される

#### TC-004: 複数ページ
- **入力**: 5ページ以上のPDF
- **期待結果**: 全ページが正しく処理される

#### TC-005: 学術論文
- **入力**: `tests/fixtures/sample_llama.pdf`（アーカイブ内）
- **期待結果**: 本文テキストの抽出・再挿入が成功

---

## 4. APIインターフェース設計

### 4.1 クラス構造

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

@dataclass
class Font:
    name: str
    size: float
    is_bold: bool = False
    is_italic: bool = False

@dataclass
class Transform:
    a: float = 1.0  # 水平スケール
    b: float = 0.0  # 垂直傾斜
    c: float = 0.0  # 水平傾斜
    d: float = 1.0  # 垂直スケール
    e: float = 0.0  # 水平移動
    f: float = 0.0  # 垂直移動

@dataclass
class TextObject:
    id: str
    bbox: BBox
    text: str
    font: Optional[Font] = None
    transform: Optional[Transform] = None

@dataclass
class Page:
    page_number: int
    width: float
    height: float
    text_objects: list[TextObject]
    rotation: int = 0

@dataclass
class PDFDocument:
    pages: list[Page]
    metadata: dict
```

### 4.2 主要関数

```python
class PDFProcessor:
    """pypdfium2ベースのPDF処理クラス"""

    def __init__(self, pdf_path: Path | bytes):
        """PDFを読み込む"""
        ...

    def extract_text_objects(self) -> PDFDocument:
        """全テキストオブジェクトを抽出して中間データに変換"""
        ...

    def remove_text_objects(
        self,
        page_num: int,
        object_ids: list[str]
    ) -> None:
        """指定したテキストオブジェクトを削除"""
        ...

    def insert_text_object(
        self,
        page_num: int,
        text: str,
        bbox: BBox,
        font: Font
    ) -> str:
        """テキストオブジェクトを挿入し、IDを返す"""
        ...

    def save(self, output_path: Path) -> None:
        """PDFを保存"""
        ...

    def to_json(self) -> str:
        """中間データをJSON文字列として出力"""
        ...

    def apply(self, doc: PDFDocument) -> None:
        """中間データを元にテキスト層を編集（テンプレートPDF方式）

        元PDFをテンプレートとして保持し、指定されたテキストオブジェクトで
        テキスト層を置換する。画像・図形は元PDFから継承される。
        """
        ...
```

### 4.3 使用例

```python
from pdf_translator.core.pdf_processor import PDFProcessor, BBox, Font

# 抽出
processor = PDFProcessor("input.pdf")
doc = processor.extract_text_objects()

# JSONエクスポート
json_data = processor.to_json()
with open("intermediate.json", "w") as f:
    f.write(json_data)

# 操作
processor.remove_text_objects(0, ["text_001"])
processor.insert_text_object(
    page_num=0,
    text="翻訳されたテキスト",
    bbox=BBox(72.0, 750.0, 300.0, 770.0),
    font=Font("IPAMincho", 18.0)
)

# 保存
processor.save("output.pdf")

# テンプレートPDF方式での編集
with open("intermediate.json") as f:
    json_data = f.read()
doc = PDFDocument.from_json(json_data)  # 中間データを読み込み
doc.pages[0].text_objects[0].text = "翻訳されたタイトル"  # テキストを編集
processor.apply(doc)  # 元PDFをテンプレートとして編集を適用
processor.save("edited.pdf")
```

---

## 5. 制約事項と対処方針

### 5.1 pypdfium2の既知の制限

| 制限 | 影響 | 対処方針 |
|------|------|----------|
| 高レベルAPIにフォント読み込みがない | テキスト挿入に raw API が必要 | `pdfium.raw.FPDFText_LoadFont` を使用 |
| ctypes変換が必要 | バイト配列、ワイド文字列の変換 | ヘルパー関数 `to_widestring()`, `to_byte_array()` を実装 |
| フォント埋め込み | 一部フォントで文字化け | TrueType/CIDフォントを明示的に指定 |

### 5.2 座標系

- **PDF座標系**: 原点は左下、Y軸は上向き
- **画像座標系**: 原点は左上、Y軸は下向き
- **変換**: `y_pdf = page_height - y_image`

---

## 6. 実装計画

### 6.1 フェーズ

| Phase | 内容 | 成果物 |
|-------|------|--------|
| 1 | データクラス定義 | `models.py` |
| 2 | テキスト抽出 | `extract_text_objects()` |
| 3 | テキスト削除 | `remove_text_objects()` |
| 4 | テキスト挿入 | `insert_text_object()` |
| 5 | JSON入出力・適用 | `to_json()`, `apply()` |
| 6 | ラウンドトリップテスト | `tests/test_pdf_processor.py` |

### 6.2 ファイル構成

```
src/pdf_translator/core/
├── __init__.py
├── models.py          # データクラス定義
├── pdf_processor.py   # PDFProcessorクラス
└── helpers.py         # ctypes変換ヘルパー
```

---

## 7. 参照

- pypdfium2検証スクリプト: `_archive/Index_PDF_Translation/tests/evaluation/verify_pypdfium2_text_insertion_v2.py`
- PyMuPDF代替調査: `_archive/Index_PDF_Translation/docs/research/pymupdf-alternative-investigation.md`
- Issue #1: https://github.com/Mega-Gorilla/pdf-translator/issues/1
- Issue #2: https://github.com/Mega-Gorilla/pdf-translator/issues/2

---

## 8. ライセンスに関する注意事項

### 8.1 アーカイブコードの参照について

`_archive/Index_PDF_Translation/` に格納されているコードは **AGPL-3.0-only** でライセンスされています。

**実装時の注意**:
- アーカイブコードを**そのままコピーしない**こと
- API呼び出しの「挙動確認」として参照することは許容される
- 新規実装は Apache-2.0 ライセンスで作成すること

**許容される参照方法**:
- pypdfium2のAPI呼び出し方法の確認
- データ構造の設計参考（ただし独自に再設計）
- テストケースの参考

**禁止される行為**:
- 関数・クラスのコピー＆ペースト
- ロジックの直接移植
- コメントを含むコードのコピー
