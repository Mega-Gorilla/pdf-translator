# Issue #57: リスト構造の保持と Markdown 変換

## 概要

PDF から抽出された箇条書き（ビュレットリスト）および番号付きリストの構造が段落抽出時に失われる問題を修正し、PDF 出力および Markdown 出力の両方で正しく表示されるようにする。

**関連 Issue**: [#57](https://github.com/Mega-Gorilla/pdf-translator/issues/57)

---

## 問題の分析

### 現象

**元の PDF 構造:**
```
• Privacy and Data Protection: The framework allows...
• Bias and Fairness: LLMs have been shown to exhibit...

1. Input the problem:
2. We expect the system...
3. Final answer is...
```

**現在の出力（PDF/Markdown 両方）:**
```
• Privacy and Data Protection: The framework allows... • Bias and Fairness: LLMs have been shown to exhibit...

1. Input the problem: 2. We expect the system... 3. Final answer is...
```

### 根本原因

`paragraph_extractor.py` の `_process_block` メソッドで行がスペース結合される:

```python
merged_text = " ".join(lines)  # ← リスト構造が失われる
```

### PDF の内部構造（重要な発見）

pdftext の出力を分析した結果、リストマーカーは **独立した span として分離** されていることが判明:

```
=== 箇条書きの構造 ===
Line 0:
  Span 0: bbox=[108.0, ...], text='•'           ← マーカー（独立span）
  Span 1: bbox=[111.4, ...], text=' '           ← スペース
  Span 2: bbox=[116.4, ...], text='Privacy...'  ← 本文（異なるx座標）

=== 番号付きリストの構造 ===
Block 1: Span 0: x0=131.4, text='1.'  → Span 2: x0=143.9, text='Input...'
Block 3: Span 0: x0=131.4, text='3.'  → Span 2: x0=143.9, text='We expect...'
Block 4: Span 0: x0=131.4, text='4.'  → Span 2: x0=143.9, text='If the...'
Block 5: Span 0: x0=131.4, text='5.'  → Span 2: x0=143.9, text='Final...'
```

この構造を活用することで、テキストパースより信頼性の高い検出が可能。

---

## 設計方針

### 1. Span 構造ベースのリスト検出

テキストの先頭文字をパースする代わりに、**span の bbox 構造を分析**:

| 検出方法 | テキストパース | Span構造分析 |
|---------|---------------|--------------|
| 信頼性 | `"•Item"` vs `"• Item"` の区別必要 | 物理的に分離されたspan |
| 誤検知 | 文中の `-` を検出する可能性 | マーカーが独立spanなので安全 |
| 番号検出 | 正規表現で推測 | 連続ブロックのパターン検出可能 |
| インデント | `strip()` との競合 | bbox x0 から正確に判定 |

### 2. 段落区切りと行区切りの区別

| 用途 | マーカー | PDF 出力 | Markdown 出力 |
|------|----------|----------|---------------|
| 段落間 | `\n\n` | 段落スペース挿入 | 空行で段落分離 |
| リスト項目間 | `\n` | 単純な改行 | 改行 |

### 3. リストマーカーの分離保持

リストマーカーを本文とは別に保持し、出力時に適切な形式に変換:

```python
@dataclass
class ListMarker:
    """リストマーカー情報"""
    marker_type: Literal["bullet", "numbered"]
    marker_text: str      # "•", "1.", "2." など
    number: int | None    # 番号付きリストの場合の数値
```

---

## 技術的決定事項

### 1. リストマーカー検出ロジック

**決定: 最初の span を分析してリストマーカーを検出**

```python
# 箇条書き記号
BULLET_CHARS = frozenset("•◦○●◆◇▸▹‣⁃")

# 番号付きリストのパターン
NUMBERED_PATTERN = re.compile(r"^(\d+)\.$")

def _detect_list_marker(line: dict[str, Any]) -> ListMarker | None:
    """行の最初のspanからリストマーカーを検出"""
    spans = line.get("spans", [])
    if len(spans) < 2:
        return None

    first_span = spans[0]
    first_text = first_span.get("text", "").strip()
    first_bbox = first_span.get("bbox", [])

    # span幅が狭い（マーカーとして妥当）かチェック
    if first_bbox and len(first_bbox) >= 4:
        span_width = first_bbox[2] - first_bbox[0]
        if span_width > 20:  # 幅が広すぎる場合はマーカーではない
            return None

    # 箇条書き記号
    if first_text in BULLET_CHARS:
        return ListMarker(
            marker_type="bullet",
            marker_text=first_text,
            number=None,
        )

    # 番号付きリスト (1., 2., など)
    match = NUMBERED_PATTERN.match(first_text)
    if match:
        return ListMarker(
            marker_type="numbered",
            marker_text=first_text,
            number=int(match.group(1)),
        )

    return None
```

**理由:**
- bbox 構造を利用することで物理的な分離を検出
- span 幅チェックで誤検知を防止
- 番号を数値として保持し、連番検出に活用

### 2. Paragraph モデルの拡張

**決定: `list_marker` フィールドを追加**

```python
# models.py
@dataclass
class ListMarker:
    """List marker information extracted from PDF structure."""
    marker_type: Literal["bullet", "numbered"]
    marker_text: str
    number: int | None = None

@dataclass
class Paragraph:
    # ... 既存フィールド ...

    # リストマーカー情報（リスト項目の場合）
    list_marker: ListMarker | None = None
```

**理由:**
- マーカーを本文から分離することで、翻訳時にマーカーが変換されない
- Markdown/PDF 出力時に適切な形式で再構成可能

### 3. 連番リストの検出

**決定: 同一ページ内の連続ブロックで番号パターンを検出**

```python
def _detect_numbered_list_sequence(
    paragraphs: list[Paragraph],
) -> list[list[Paragraph]]:
    """連続した番号付きリストをグループ化"""
    sequences: list[list[Paragraph]] = []
    current_sequence: list[Paragraph] = []
    expected_number = 1

    for para in paragraphs:
        if para.list_marker and para.list_marker.marker_type == "numbered":
            num = para.list_marker.number
            if num == expected_number or num == expected_number + 1:
                current_sequence.append(para)
                expected_number = num + 1
            else:
                # 連番が途切れた
                if len(current_sequence) >= 2:
                    sequences.append(current_sequence)
                current_sequence = [para]
                expected_number = num + 1
        else:
            if len(current_sequence) >= 2:
                sequences.append(current_sequence)
            current_sequence = []
            expected_number = 1

    if len(current_sequence) >= 2:
        sequences.append(current_sequence)

    return sequences
```

**理由:**
- 単独の `1.` は番号付きリストではなく、通常のテキストの可能性
- 2つ以上の連続した番号がある場合のみリストとして扱う
- 番号の飛び（1, 3, 4, 5 など）も許容

### 4. テキスト結合ロジック

**決定: リストマーカー検出時は本文のみを抽出し、改行で結合**

```python
def _process_block(self, block: dict[str, Any], ...) -> Paragraph | None:
    lines_data: list[tuple[ListMarker | None, str]] = []

    for line in block.get("lines", []):
        marker = self._detect_list_marker(line)

        # マーカー以降のspanからテキストを抽出
        spans = line.get("spans", [])
        if marker:
            # 最初のspan（マーカー）とスペースをスキップ
            content_spans = spans[2:] if len(spans) > 2 else []
        else:
            content_spans = spans

        line_text = "".join(span.get("text", "") for span in content_spans)
        line_text = line_text.rstrip()

        if line_text or marker:
            lines_data.append((marker, line_text))

    # リストマーカーがある場合は改行で結合
    has_list = any(m for m, _ in lines_data)
    if has_list:
        merged_text = "\n".join(text for _, text in lines_data if text)
    else:
        merged_text = " ".join(text for _, text in lines_data)

    # 最初のリストマーカーを段落に設定
    first_marker = next((m for m, _ in lines_data if m), None)

    return Paragraph(
        ...,
        text=merged_text,
        list_marker=first_marker,
    )
```

### 5. Markdown 出力

**決定: `list_marker` フィールドに基づいて適切な形式に変換**

```python
def _format_list_item(self, paragraph: Paragraph, text: str) -> str:
    """リスト項目をMarkdown形式にフォーマット"""
    if not paragraph.list_marker:
        return text

    marker = paragraph.list_marker

    if marker.marker_type == "bullet":
        # 箇条書き → "- "
        return f"- {text}"

    elif marker.marker_type == "numbered":
        # 番号付き → "1. ", "2. " など
        return f"{marker.number}. {text}"

    return text
```

**理由:**
- 元の記号（`•`）ではなく、標準的な Markdown 記法を使用
- 番号付きリストは元の番号を保持

### 6. PDF 出力（TextLayoutEngine）

**決定: リスト項目間は段落スペースなし、リスト前後は段落スペースあり**

```python
def wrap_text(self, text: str, ...) -> list[str]:
    all_lines: list[str] = []

    # \n\n で段落分割（段落間スペース）
    paragraphs = text.split("\n\n")

    for para_idx, paragraph in enumerate(paragraphs):
        # \n で行分割（リスト項目間、スペースなし）
        lines = paragraph.split("\n")

        for line in lines:
            line = line.strip()
            if line:
                wrapped = self._wrap_segment(line, ...)
                all_lines.extend(wrapped)

        # 段落間にマーカー挿入
        if para_idx < len(paragraphs) - 1:
            all_lines.append(PARAGRAPH_BREAK_MARKER)

    return all_lines
```

---

## 実装計画

### Phase 1: データモデル拡張

**ファイル:** `src/pdf_translator/core/models.py`

```python
from typing import Literal

@dataclass
class ListMarker:
    """List marker information extracted from PDF structure.

    Attributes:
        marker_type: Type of list marker ("bullet" or "numbered").
        marker_text: Original marker text from PDF ("•", "1.", etc.).
        number: Numeric value for numbered lists, None for bullets.
    """
    marker_type: Literal["bullet", "numbered"]
    marker_text: str
    number: int | None = None


@dataclass
class Paragraph:
    # ... 既存フィールド ...

    # List marker information (None if not a list item)
    list_marker: ListMarker | None = None
```

### Phase 2: TextLayoutEngine 修正

**ファイル:** `src/pdf_translator/core/text_layout.py`

`\n\n` を段落区切り、`\n` を行区切りとして区別:

```python
def wrap_text(
    self,
    text: str,
    max_width: float,
    font_handle: ctypes.c_void_p,
    font_size: float,
) -> list[str]:
    """Wrap text with paragraph and line break handling.

    - \\n\\n: Paragraph break (adds spacing via PARAGRAPH_BREAK_MARKER)
    - \\n: Line break (no extra spacing, for list items)
    """
    all_lines: list[str] = []

    # Split by paragraph breaks
    paragraphs = text.split("\n\n")

    for para_idx, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue

        # Split by line breaks within paragraph
        lines = paragraph.split("\n")

        for line in lines:
            line = line.strip()
            if line:
                wrapped = self._wrap_segment(line, max_width, font_handle, font_size)
                all_lines.extend(wrapped)

        # Add paragraph break marker between paragraphs
        if para_idx < len(paragraphs) - 1:
            all_lines.append(PARAGRAPH_BREAK_MARKER)

    # Remove trailing markers
    while all_lines and all_lines[-1] == PARAGRAPH_BREAK_MARKER:
        all_lines.pop()

    return all_lines
```

### Phase 3: paragraph_merger.py 修正

**ファイル:** `src/pdf_translator/core/paragraph_merger.py`

```python
def _merge_two_paragraphs(para1: Paragraph, para2: Paragraph) -> Paragraph:
    # 段落間は \n\n で結合（段落スペース挿入）
    merged_text = para1.text + "\n\n" + para2.text

    # リストマーカーは最初の段落から継承
    # （通常、リスト項目はマージされないが、念のため）
    merged_marker = para1.list_marker

    return replace(
        para1,
        text=merged_text,
        list_marker=merged_marker,
        # ... other fields ...
    )


def _can_merge(para1: Paragraph, para2: Paragraph, config: MergeConfig) -> bool:
    # ... 既存のチェック ...

    # リスト項目同士はマージしない（構造を保持）
    if para1.list_marker or para2.list_marker:
        return False

    return True
```

### Phase 4: paragraph_extractor.py 修正

**ファイル:** `src/pdf_translator/core/paragraph_extractor.py`

#### 4.1 定数・型定義追加

```python
import re
from pdf_translator.core.models import ListMarker

# Bullet characters for list detection
BULLET_CHARS = frozenset("•◦○●◆◇▸▹‣⁃")

# Numbered list pattern (1., 2., 10., etc.)
NUMBERED_LIST_PATTERN = re.compile(r"^(\d+)\.$")

# Maximum span width for list marker (points)
MAX_MARKER_SPAN_WIDTH = 25.0
```

#### 4.2 リストマーカー検出メソッド追加

```python
@staticmethod
def _detect_list_marker(line: dict[str, Any]) -> ListMarker | None:
    """Detect list marker from first span of a line.

    Analyzes the span structure to identify list markers that are
    physically separated from the content text in the PDF.

    Args:
        line: Line dictionary from pdftext output.

    Returns:
        ListMarker if detected, None otherwise.
    """
    spans = line.get("spans", [])
    if len(spans) < 2:
        return None

    first_span = spans[0]
    first_text = first_span.get("text", "").strip()

    if not first_text:
        return None

    # Check span width (markers are narrow)
    first_bbox = first_span.get("bbox", [])
    if first_bbox and len(first_bbox) >= 4:
        span_width = first_bbox[2] - first_bbox[0]
        if span_width > MAX_MARKER_SPAN_WIDTH:
            return None

    # Bullet detection
    if first_text in BULLET_CHARS:
        return ListMarker(
            marker_type="bullet",
            marker_text=first_text,
            number=None,
        )

    # Numbered list detection (1., 2., etc.)
    match = NUMBERED_LIST_PATTERN.match(first_text)
    if match:
        return ListMarker(
            marker_type="numbered",
            marker_text=first_text,
            number=int(match.group(1)),
        )

    return None

@staticmethod
def _extract_content_after_marker(
    spans: list[dict[str, Any]],
    has_marker: bool,
) -> str:
    """Extract text content, skipping marker span if present.

    Args:
        spans: List of span dictionaries.
        has_marker: Whether first span is a list marker.

    Returns:
        Concatenated text content.
    """
    if has_marker and len(spans) > 1:
        # Skip marker span and optional space span
        start_idx = 1
        if len(spans) > 2:
            second_text = spans[1].get("text", "").strip()
            if not second_text:  # Second span is just whitespace
                start_idx = 2
        content_spans = spans[start_idx:]
    else:
        content_spans = spans

    return "".join(span.get("text", "") for span in content_spans)
```

#### 4.3 `_process_block` 修正

```python
def _process_block(
    self,
    block: dict[str, Any],
    page_idx: int,
    block_idx: int,
    page_height: float,
) -> Paragraph | None:
    """Convert a single block to Paragraph with list marker detection."""
    lines_data: list[tuple[ListMarker | None, str]] = []

    for line in block.get("lines", []):
        spans = line.get("spans", [])

        # Detect list marker from span structure
        marker = self._detect_list_marker(line)

        # Extract content text (skipping marker if present)
        line_text = self._extract_content_after_marker(spans, marker is not None)
        line_text = line_text.rstrip()

        if line_text or marker:
            lines_data.append((marker, line_text))

    if not lines_data:
        return None

    # Determine if this block contains list items
    has_list_marker = any(m for m, _ in lines_data)

    # Join lines: newline for lists, space for regular text
    if has_list_marker:
        # Normalize each line individually, preserve newlines
        normalized_lines = []
        for _, text in lines_data:
            if text:
                normalized = re.sub(r"[ \t]+", " ", text).strip()
                normalized_lines.append(normalized)
        merged_text = "\n".join(normalized_lines)
    else:
        merged_text = " ".join(text for _, text in lines_data if text)
        merged_text = re.sub(r"\s+", " ", merged_text).strip()

    if not merged_text:
        return None

    # Get first list marker for the paragraph
    first_marker = next((m for m, _ in lines_data if m), None)

    # ... bbox, font estimation (existing code) ...

    return Paragraph(
        id=f"para_p{page_idx}_b{block_idx}",
        page_number=page_idx,
        text=merged_text,
        block_bbox=BBox(x0=float(x0), y0=pdf_y0, x1=float(x1), y1=pdf_y1),
        line_count=len(lines_data),
        list_marker=first_marker,
        # ... other fields ...
    )
```

### Phase 5: markdown_writer.py 修正

**ファイル:** `src/pdf_translator/output/markdown_writer.py`

#### 5.1 インポート・定数追加

```python
from pdf_translator.core.models import ListMarker

# Element types where list conversion should be skipped
SKIP_LIST_CONVERSION = frozenset({"code", "code_block"})
```

#### 5.2 リスト変換メソッド追加

```python
def _format_as_list_item(
    self,
    text: str,
    list_marker: ListMarker | None,
) -> str:
    """Format text as a Markdown list item if applicable.

    Args:
        text: Text content to format.
        list_marker: List marker info, or None for regular text.

    Returns:
        Formatted text with Markdown list syntax if applicable.
    """
    if not list_marker:
        return text

    # Handle multi-line text (list item with wrapped content)
    lines = text.split("\n")

    if list_marker.marker_type == "bullet":
        # Bullet list: "- item"
        first_line = f"- {lines[0]}"
        # Continuation lines indented
        rest_lines = ["  " + line for line in lines[1:]]
        return "\n".join([first_line] + rest_lines)

    elif list_marker.marker_type == "numbered":
        # Numbered list: "1. item"
        first_line = f"{list_marker.number}. {lines[0]}"
        # Continuation lines indented
        rest_lines = ["   " + line for line in lines[1:]]
        return "\n".join([first_line] + rest_lines)

    return text
```

#### 5.3 `_paragraph_to_markdown` 修正

```python
def _paragraph_to_markdown(
    self,
    paragraph: Paragraph,
    image_map: dict[str, ExtractedImage] | None = None,
    table_map: dict[str, ExtractedTable] | None = None,
    emitted_keys: set[str] | None = None,
) -> str:
    # ... 既存のスキップ/画像/テーブル処理 ...

    element_type = self._get_element_type(paragraph.category)
    text = self._get_display_text(paragraph)

    if not text:
        return ""

    # Apply list formatting if applicable (skip for code elements)
    if element_type not in SKIP_LIST_CONVERSION:
        text = self._format_as_list_item(text, paragraph.list_marker)

    # For list items, return directly without additional formatting
    if paragraph.list_marker:
        return text + "\n"

    # Regular paragraph formatting
    return self._format_text(text, element_type, paragraph)
```

#### 5.4 parallel モード対応

```python
def _format_text(
    self,
    text: str,
    element_type: str,
    paragraph: Paragraph | None = None,
) -> str:
    # ... 既存の処理 ...

    # Parallel mode: add translated text
    if (
        self._config.output_mode == MarkdownOutputMode.PARALLEL
        and paragraph
        and paragraph.translated_text
    ):
        translated = paragraph.translated_text

        # Apply list formatting to translation too
        if paragraph.list_marker and element_type not in SKIP_LIST_CONVERSION:
            translated = self._format_as_list_item(
                translated,
                paragraph.list_marker,
            )

        # ... rest of parallel mode handling ...
```

### Phase 6: テスト

#### 6.1 models テスト

**ファイル:** `tests/test_models.py`

```python
class TestListMarker:
    """Tests for ListMarker dataclass."""

    def test_bullet_marker(self) -> None:
        marker = ListMarker(marker_type="bullet", marker_text="•")
        assert marker.marker_type == "bullet"
        assert marker.number is None

    def test_numbered_marker(self) -> None:
        marker = ListMarker(marker_type="numbered", marker_text="1.", number=1)
        assert marker.marker_type == "numbered"
        assert marker.number == 1
```

#### 6.2 paragraph_extractor テスト

**ファイル:** `tests/test_paragraph_extractor.py`

```python
class TestListMarkerDetection:
    """Tests for list marker detection from span structure."""

    def test_detect_bullet_marker(self) -> None:
        """Bullet in separate span should be detected."""
        line = {
            "spans": [
                {"text": "•", "bbox": [100, 200, 105, 212]},
                {"text": " ", "bbox": [105, 200, 108, 212]},
                {"text": "Item content", "bbox": [108, 200, 200, 212]},
            ]
        }
        marker = ParagraphExtractor._detect_list_marker(line)
        assert marker is not None
        assert marker.marker_type == "bullet"
        assert marker.marker_text == "•"

    def test_detect_numbered_marker(self) -> None:
        """Number with period should be detected."""
        line = {
            "spans": [
                {"text": "1.", "bbox": [100, 200, 112, 212]},
                {"text": " ", "bbox": [112, 200, 115, 212]},
                {"text": "First item", "bbox": [115, 200, 200, 212]},
            ]
        }
        marker = ParagraphExtractor._detect_list_marker(line)
        assert marker is not None
        assert marker.marker_type == "numbered"
        assert marker.number == 1

    def test_wide_span_not_marker(self) -> None:
        """Wide first span should not be detected as marker."""
        line = {
            "spans": [
                {"text": "This is normal text", "bbox": [100, 200, 250, 212]},
            ]
        }
        marker = ParagraphExtractor._detect_list_marker(line)
        assert marker is None

    def test_single_span_not_marker(self) -> None:
        """Single span line should not have marker."""
        line = {
            "spans": [
                {"text": "• Item without separation", "bbox": [100, 200, 300, 212]},
            ]
        }
        marker = ParagraphExtractor._detect_list_marker(line)
        assert marker is None


class TestContentExtraction:
    """Tests for content extraction after marker."""

    def test_extract_after_marker(self) -> None:
        spans = [
            {"text": "•"},
            {"text": " "},
            {"text": "Content here"},
        ]
        content = ParagraphExtractor._extract_content_after_marker(spans, True)
        assert content == "Content here"

    def test_extract_without_marker(self) -> None:
        spans = [
            {"text": "Regular "},
            {"text": "text"},
        ]
        content = ParagraphExtractor._extract_content_after_marker(spans, False)
        assert content == "Regular text"
```

#### 6.3 markdown_writer テスト

**ファイル:** `tests/test_markdown_writer.py`

```python
class TestListFormatting:
    """Tests for Markdown list formatting."""

    def test_bullet_list_format(self) -> None:
        writer = MarkdownWriter()
        marker = ListMarker(marker_type="bullet", marker_text="•")
        result = writer._format_as_list_item("Item content", marker)
        assert result == "- Item content"

    def test_numbered_list_format(self) -> None:
        writer = MarkdownWriter()
        marker = ListMarker(marker_type="numbered", marker_text="3.", number=3)
        result = writer._format_as_list_item("Third item", marker)
        assert result == "3. Third item"

    def test_multiline_bullet(self) -> None:
        writer = MarkdownWriter()
        marker = ListMarker(marker_type="bullet", marker_text="•")
        text = "First line\nSecond line\nThird line"
        result = writer._format_as_list_item(text, marker)
        assert result == "- First line\n  Second line\n  Third line"

    def test_no_marker(self) -> None:
        writer = MarkdownWriter()
        result = writer._format_as_list_item("Regular text", None)
        assert result == "Regular text"
```

#### 6.4 TextLayoutEngine テスト

**ファイル:** `tests/test_text_layout.py`

```python
class TestParagraphAndLineBreaks:
    """Tests for paragraph vs line break handling."""

    def test_double_newline_paragraph_break(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Double newline creates paragraph break marker."""
        text = "Para 1.\n\nPara 2."
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        assert PARAGRAPH_BREAK_MARKER in lines
        text_lines = [l for l in lines if l != PARAGRAPH_BREAK_MARKER]
        assert len(text_lines) == 2

    def test_single_newline_no_break(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Single newline does NOT create paragraph break."""
        text = "Item 1\nItem 2\nItem 3"
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        assert PARAGRAPH_BREAK_MARKER not in lines
        assert len(lines) == 3

    def test_mixed_breaks(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Mixed paragraph and line breaks."""
        text = "Intro.\n\nItem 1\nItem 2\n\nConclusion."
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        markers = [l for l in lines if l == PARAGRAPH_BREAK_MARKER]
        assert len(markers) == 2
```

#### 6.5 E2E テスト

**ファイル:** `tests/test_list_e2e.py`

```python
import pytest
from pathlib import Path

@pytest.mark.skipif(
    not Path("tests/fixtures/sample_autogen_paper.pdf").exists(),
    reason="Test PDF not available"
)
class TestListExtractionE2E:
    """E2E tests for list extraction from sample PDF."""

    def test_bullet_list_page_10(self) -> None:
        """Bullet list on page 10 should be extracted with markers."""
        from pdf_translator.core.paragraph_extractor import ParagraphExtractor

        paragraphs = ParagraphExtractor.extract_from_pdf(
            "tests/fixtures/sample_autogen_paper.pdf",
            page_range=[9],  # 0-indexed
        )

        # Find bullet list paragraph
        bullet_paras = [
            p for p in paragraphs
            if p.list_marker and p.list_marker.marker_type == "bullet"
        ]

        assert len(bullet_paras) > 0

        # Check content
        privacy_para = next(
            (p for p in bullet_paras if "Privacy" in p.text),
            None
        )
        assert privacy_para is not None
        assert privacy_para.list_marker.marker_text == "•"

    def test_numbered_list_page_21(self) -> None:
        """Numbered list on page 21 should be extracted."""
        from pdf_translator.core.paragraph_extractor import ParagraphExtractor

        paragraphs = ParagraphExtractor.extract_from_pdf(
            "tests/fixtures/sample_autogen_paper.pdf",
            page_range=[20],  # 0-indexed
        )

        # Find numbered list paragraphs
        numbered_paras = [
            p for p in paragraphs
            if p.list_marker and p.list_marker.marker_type == "numbered"
        ]

        assert len(numbered_paras) >= 3

        # Check sequential numbers
        numbers = [p.list_marker.number for p in numbered_paras]
        assert 1 in numbers
```

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/pdf_translator/core/models.py` | `ListMarker` dataclass 追加、`Paragraph.list_marker` フィールド追加 |
| `src/pdf_translator/core/text_layout.py` | `\n\n`/`\n` 区別対応 |
| `src/pdf_translator/core/paragraph_merger.py` | `\n` → `\n\n`、リスト項目マージ防止 |
| `src/pdf_translator/core/paragraph_extractor.py` | span構造からのリストマーカー検出 |
| `src/pdf_translator/output/markdown_writer.py` | `list_marker` に基づくMarkdown変換 |
| `tests/test_models.py` | `ListMarker` テスト追加 |
| `tests/test_text_layout.py` | 段落/行区切りテスト追加 |
| `tests/test_paragraph_extractor.py` | **新規** - リストマーカー検出テスト |
| `tests/test_markdown_writer.py` | リストフォーマットテスト追加 |
| `tests/test_list_e2e.py` | **新規** - E2Eテスト |

---

## リスクと対策

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|-------|------|
| マーカー検出漏れ | 低 | 中 | span幅閾値の調整、テストでの検証 |
| 誤検知（通常テキストをマーカー判定） | 低 | 中 | span幅チェック、複数span必須条件 |
| 番号の連続性誤判定 | 低 | 低 | 2項目以上の連続を条件に |
| 既存テスト破壊 | 中 | 中 | `Paragraph` モデル変更に伴うテスト更新 |
| 翻訳時のリスト構造変化 | 中 | 低 | マーカーを本文から分離して保持 |

**後方互換性:**
- `list_marker` フィールドはオプショナル（`None` がデフォルト）
- 既存の段落処理は影響なし

---

## 将来対応（別 Issue）

1. **ネストリスト対応**
   - bbox x0 の差分からインデントレベルを推定
   - Markdown のネストリスト記法に対応

2. **追加のリストパターン**
   - `1)`, `a)`, `A.` などの番号形式
   - `①`, `②` などの丸数字
   - ローマ数字 (i, ii, iii)

3. **リストグループ化**
   - 連続したリスト項目を単一の構造として管理
   - リスト全体の前後に適切なスペースを挿入

---

## 参考

- [Issue #57](https://github.com/Mega-Gorilla/pdf-translator/issues/57)
- [PR #58](https://github.com/Mega-Gorilla/pdf-translator/pull/58) - paragraph_merger 改行対応
- `tests/fixtures/sample_autogen_paper.pdf`
  - ページ 10: 箇条書きリスト
  - ページ 21: 番号付きリスト
