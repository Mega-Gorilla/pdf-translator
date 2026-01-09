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

### 2. リスト項目の行単位 Paragraph 分割（重要）

**設計変更**: リストマーカーを持つ行は、それぞれ独立した Paragraph として抽出する。

**変更前（旧設計）:**
```
Block (3 list items)
    ↓
1 Paragraph: text="Item1\nItem2\nItem3", list_marker=bullet
    ↓
Markdown: "- Item1\n  Item2\n  Item3"  ← 継続行扱い（誤り）
```

**変更後（新設計）:**
```
Block (3 list items)
    ↓
3 Paragraphs:
  - Paragraph 1: text="Item1", list_marker=bullet
  - Paragraph 2: text="Item2", list_marker=bullet
  - Paragraph 3: text="Item3", list_marker=bullet
    ↓
Markdown: "- Item1\n- Item2\n- Item3"  ← 各行が独立（正しい）
```

**理由:**
- 各リスト項目は意味的に独立した単位
- 翻訳時もリスト項目単位の方が品質が高い
- Markdown 出力がシンプルになる
- `paragraph_merger` でのマージ防止ロジックがシンプル

### 3. 段落区切りの扱い

行単位分割により、リスト項目間の `\n` 結合は不要になる:

| ケース | 処理 | 例 |
|--------|------|-----|
| 通常テキスト | スペース結合 | `"Line 1 Line 2"` |
| リスト項目 | 行ごとに別Paragraph | `Paragraph("Item1")`, `Paragraph("Item2")` |
| 段落マージ | `\n\n` で結合 | `"Para1.\n\nPara2."` |

### 4. リストマーカーの分離保持

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

### 0. 機能の有効化/無効化

**決定: リスト検出機能はデフォルト有効、設定で無効化可能**

```python
@dataclass
class ExtractorConfig:
    """Configuration for paragraph extraction.

    Attributes:
        detect_lists: Enable list marker detection and line-level splitting.
            When True (default), list items are extracted as separate paragraphs.
            When False, all lines are merged with spaces (legacy behavior).
    """
    detect_lists: bool = True
```

**理由:**
- デフォルト有効: 新機能の恩恵を自動的に受けられる
- 無効化オプション: 既存ワークフローとの互換性、特殊なPDFでの問題回避
- 段階的導入: ユーザーが機能を評価してから本格採用できる

**CLI オプション:**

```bash
# デフォルト（リスト検出有効）
uv run translate-pdf paper.pdf

# リスト検出を無効化
uv run translate-pdf paper.pdf --no-detect-lists
```

**影響範囲:**

| 設定 | 動作 |
|------|------|
| `detect_lists=True` | リスト項目を行単位で分割、`list_marker` を設定 |
| `detect_lists=False` | 従来通り全行をスペース結合、`list_marker` は常に `None` |

### 1. リストマーカー検出ロジック

**決定: 最初の span を分析してリストマーカーを検出（相対閾値使用）**

```python
import re
from typing import Any, Literal
from dataclasses import dataclass

# 箇条書き記号
BULLET_CHARS = frozenset("•◦○●◆◇▸▹‣⁃")

# 番号付きリストのパターン（1-999のみ、年号2019.等を除外）
NUMBERED_PATTERN = re.compile(r"^([1-9]\d{0,2})\.$")

# マーカーspan幅の閾値
MAX_MARKER_SPAN_WIDTH_ABSOLUTE = 30.0  # 絶対上限（pt）
MAX_MARKER_SPAN_WIDTH_RATIO = 2.5      # font_size * ratio


def _is_marker_span_width(span_width: float, font_size: float) -> bool:
    """Check if span width is consistent with a list marker.

    Uses both absolute and relative thresholds for robustness
    across different font sizes and resolutions.
    """
    if font_size <= 0:
        font_size = 12.0  # fallback

    absolute_ok = span_width <= MAX_MARKER_SPAN_WIDTH_ABSOLUTE
    relative_ok = span_width <= font_size * MAX_MARKER_SPAN_WIDTH_RATIO
    return absolute_ok and relative_ok


def _detect_list_marker(
    line: dict[str, Any],
    font_size: float = 12.0,
) -> ListMarker | None:
    """行の最初のspanからリストマーカーを検出

    Args:
        line: pdftext出力のline辞書
        font_size: 現在のフォントサイズ（相対閾値計算用）

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

    # span幅チェック（絶対値 + 相対値）
    first_bbox = first_span.get("bbox", [])
    if first_bbox and len(first_bbox) >= 4:
        span_width = first_bbox[2] - first_bbox[0]
        if not _is_marker_span_width(span_width, font_size):
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
- 絶対値 + 相対値の閾値で様々なフォントサイズに対応
- 番号を数値として保持し、連番検出に活用
- 番号パターンを1-999に制限し、年号（2019., 2020.等）の誤検出を防止

### 2. Paragraph モデルの拡張

**決定: `list_marker` フィールドを追加（シリアライズ対応含む）**

```python
# models.py
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "marker_type": self.marker_type,
            "marker_text": self.marker_text,
            "number": self.number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ListMarker":
        """Create from dictionary."""
        return cls(
            marker_type=data["marker_type"],
            marker_text=data["marker_text"],
            number=data.get("number"),
        )


@dataclass
class Paragraph:
    # ... 既存フィールド ...

    # リストマーカー情報（リスト項目の場合）
    list_marker: ListMarker | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            # ... 既存フィールド ...
        }
        # list_marker のシリアライズ
        if self.list_marker is not None:
            result["list_marker"] = self.list_marker.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paragraph":
        """Create from dictionary."""
        # list_marker のデシリアライズ
        list_marker = None
        if "list_marker" in data:
            list_marker = ListMarker.from_dict(data["list_marker"])

        return cls(
            # ... 既存フィールド ...
            list_marker=list_marker,
        )
```

**理由:**
- マーカーを本文から分離することで、翻訳時にマーカーが変換されない
- Markdown/PDF 出力時に適切な形式で再構成可能
- シリアライズ/デシリアライズ対応で中間JSON保存が可能

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

### 4. テキスト抽出ロジック（行単位分割）

**決定: リストマーカー検出時は行ごとに別 Paragraph を生成**

```python
def _process_block(
    self,
    block: dict[str, Any],
    page_idx: int,
    block_idx: int,
    page_height: float,
) -> list[Paragraph]:
    """Convert a single block to Paragraph(s) with list marker detection.

    Returns:
        List of Paragraphs. For list blocks, each line becomes a separate
        Paragraph. For regular text, all lines are merged into one Paragraph.
    """
    lines = block.get("lines", [])
    if not lines:
        return []

    # First pass: detect markers and extract content for each line
    line_data: list[tuple[ListMarker | None, str, dict]] = []

    for line in lines:
        spans = line.get("spans", [])
        font_size = self._estimate_font_size(spans)
        marker = self._detect_list_marker(line, font_size)

        # Extract content text (skipping marker if present)
        content = self._extract_content_after_marker(spans, marker is not None)
        content = content.rstrip()  # 末尾空白のみ除去、先頭は保持

        if content or marker:
            line_data.append((marker, content, line))

    if not line_data:
        return []

    # Check if any line has a list marker
    has_list = any(m for m, _, _ in line_data)

    if has_list:
        # List block: create separate Paragraph for each line
        paragraphs: list[Paragraph] = []
        for idx, (marker, content, line) in enumerate(line_data):
            if not content:
                continue

            para = self._create_paragraph(
                text=content,
                line=line,
                page_idx=page_idx,
                block_idx=block_idx,
                line_idx=idx,
                list_marker=marker,
            )
            paragraphs.append(para)
        return paragraphs
    else:
        # Regular block: merge all lines into one Paragraph
        merged_text = " ".join(content for _, content, _ in line_data if content)
        merged_text = re.sub(r"\s+", " ", merged_text).strip()

        if not merged_text:
            return []

        para = self._create_paragraph(
            text=merged_text,
            lines=lines,
            page_idx=page_idx,
            block_idx=block_idx,
            list_marker=None,
        )
        return [para]


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

### 5. Markdown 出力

**決定: `list_marker` フィールドに基づいて適切な形式に変換**

行単位分割により、変換ロジックがシンプルになる:

```python
# Element types where list conversion should be skipped
SKIP_LIST_CONVERSION = frozenset({"code", "code_block"})


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

    if list_marker.marker_type == "bullet":
        # Bullet list: "- item"
        return f"- {text}"

    elif list_marker.marker_type == "numbered":
        # Numbered list: "1. item"
        return f"{list_marker.number}. {text}"

    return text


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
    if paragraph.list_marker and element_type not in SKIP_LIST_CONVERSION:
        text = self._format_as_list_item(text, paragraph.list_marker)
        return text + "\n"

    # Regular paragraph formatting
    return self._format_text(text, element_type, paragraph)
```

**parallel モード対応:**

```python
def _paragraph_to_markdown(self, paragraph: Paragraph, ...) -> str:
    # ... 省略 ...

    # List items in parallel mode
    if paragraph.list_marker and element_type not in SKIP_LIST_CONVERSION:
        original = self._format_as_list_item(
            paragraph.text,
            paragraph.list_marker,
        )

        if (
            self._config.output_mode == MarkdownOutputMode.PARALLEL
            and paragraph.translated_text
        ):
            translated = self._format_as_list_item(
                paragraph.translated_text,
                paragraph.list_marker,
            )
            return f"{original}\n{translated}\n"

        return original + "\n"
```

**理由:**
- 元の記号（`•`）ではなく、標準的な Markdown 記法を使用
- 番号付きリストは元の番号を保持
- 行単位分割によりマルチライン処理が不要

### 6. PDF 出力（TextLayoutEngine）

**決定: `\n\n` を段落区切りとして扱う（現行 `\n` から変更）**

```python
def wrap_text(
    self,
    text: str,
    max_width: float,
    font_handle: ctypes.c_void_p,
    font_size: float,
) -> list[str]:
    """Wrap text with paragraph break handling.

    - \\n\\n: Paragraph break (adds spacing via PARAGRAPH_BREAK_MARKER)
    - \\n: Line break within paragraph (no extra spacing)
    """
    if not text:
        return []

    all_lines: list[str] = []

    # Split by paragraph breaks (\n\n)
    paragraphs = text.split("\n\n")

    for para_idx, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue

        # Split by line breaks within paragraph (\n)
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

**リスト項目の場合:**

行単位分割により、各リスト項目は独立した Paragraph として処理される。
そのため、リスト項目間では段落スペースは入らず、自然な間隔で配置される。

### 7. paragraph_merger の変更

**決定: リスト項目はマージしない、通常段落間は `\n\n` で結合**

```python
def _merge_two_paragraphs(para1: Paragraph, para2: Paragraph) -> Paragraph:
    # 段落間は \n\n で結合（段落スペース挿入）
    merged_text = para1.text + "\n\n" + para2.text

    return replace(
        para1,
        text=merged_text,
        # list_marker は None のまま（リスト項目はマージされないため）
        # ... other fields ...
    )


def _can_merge(para1: Paragraph, para2: Paragraph, config: MergeConfig) -> bool:
    # ... 既存のチェック ...

    # リスト項目同士はマージしない（構造を保持）
    if para1.list_marker is not None or para2.list_marker is not None:
        return False

    return True
```

---

## 実装計画

### Phase 1: データモデル拡張

**ファイル:** `src/pdf_translator/core/models.py`

1. `ListMarker` dataclass を追加
   - `marker_type`, `marker_text`, `number` フィールド
   - `to_dict()`, `from_dict()` メソッド

2. `Paragraph` に `list_marker` フィールドを追加
   - オプショナル（`None` がデフォルト）
   - `to_dict()` でシリアライズ
   - `from_dict()` でデシリアライズ

### Phase 2: paragraph_extractor.py 修正

**ファイル:** `src/pdf_translator/core/paragraph_extractor.py`

1. `ExtractorConfig` dataclass 追加
   ```python
   @dataclass
   class ExtractorConfig:
       detect_lists: bool = True  # リスト検出の有効/無効
   ```

2. 定数追加
   - `BULLET_CHARS`, `NUMBERED_PATTERN`
   - `MAX_MARKER_SPAN_WIDTH_ABSOLUTE`, `MAX_MARKER_SPAN_WIDTH_RATIO`

3. 検出メソッド追加
   - `_is_marker_span_width()`: 相対閾値チェック
   - `_detect_list_marker()`: span構造からマーカー検出
   - `_extract_content_after_marker()`: マーカー以降のテキスト抽出

4. `_process_block()` 修正
   - 戻り値を `Paragraph | None` → `list[Paragraph]` に変更
   - `config.detect_lists=True`: リストブロックは行ごとに別Paragraphを生成
   - `config.detect_lists=False`: 従来通り全行をスペース結合（レガシー動作）

5. 呼び出し元の修正
   - `extract_paragraphs()` で `_process_block()` の戻り値を展開
   - `config` パラメータを追加（デフォルト: `ExtractorConfig()`）

### Phase 3: paragraph_merger.py 修正

**ファイル:** `src/pdf_translator/core/paragraph_merger.py`

1. `_can_merge()` に条件追加
   - `list_marker` を持つ段落はマージ対象外

2. `_merge_two_paragraphs()` 修正
   - 結合を `\n` → `\n\n` に変更

### Phase 4: text_layout.py 修正

**ファイル:** `src/pdf_translator/core/text_layout.py`

1. `wrap_text()` 修正
   - `\n\n` を段落区切り、`\n` を行区切りとして扱う
   - 現行の `\n` 段落区切りから変更

### Phase 5: markdown_writer.py 修正

**ファイル:** `src/pdf_translator/output/markdown_writer.py`

1. `SKIP_LIST_CONVERSION` 定数追加

2. `_format_as_list_item()` メソッド追加
   - 箇条書き → `- item`
   - 番号付き → `N. item`

3. `_paragraph_to_markdown()` 修正
   - `list_marker` がある場合は専用フォーマット
   - parallel モード対応

### Phase 6: CLI オプション追加

**ファイル:** `src/pdf_translator/cli.py`

1. `--no-detect-lists` オプション追加
   ```python
   @click.option(
       "--no-detect-lists",
       is_flag=True,
       default=False,
       help="Disable list detection (legacy behavior)",
   )
   ```

2. オプションを `ExtractorConfig` に渡す
   ```python
   config = ExtractorConfig(detect_lists=not no_detect_lists)
   paragraphs = extractor.extract_paragraphs(config=config)
   ```

### Phase 7: テスト

#### 7.1 models テスト

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

    def test_to_dict(self) -> None:
        marker = ListMarker(marker_type="bullet", marker_text="•")
        d = marker.to_dict()
        assert d["marker_type"] == "bullet"
        assert d["marker_text"] == "•"
        assert d["number"] is None

    def test_from_dict(self) -> None:
        d = {"marker_type": "numbered", "marker_text": "3.", "number": 3}
        marker = ListMarker.from_dict(d)
        assert marker.marker_type == "numbered"
        assert marker.number == 3


class TestParagraphListMarker:
    """Tests for Paragraph.list_marker serialization."""

    def test_paragraph_with_list_marker_to_dict(self) -> None:
        marker = ListMarker(marker_type="bullet", marker_text="•")
        para = Paragraph(
            id="test",
            page_number=0,
            text="Item content",
            block_bbox=BBox(0, 0, 100, 20),
            line_count=1,
            list_marker=marker,
        )
        d = para.to_dict()
        assert "list_marker" in d
        assert d["list_marker"]["marker_type"] == "bullet"

    def test_paragraph_without_list_marker_to_dict(self) -> None:
        para = Paragraph(
            id="test",
            page_number=0,
            text="Regular text",
            block_bbox=BBox(0, 0, 100, 20),
            line_count=1,
        )
        d = para.to_dict()
        assert "list_marker" not in d

    def test_paragraph_from_dict_with_list_marker(self) -> None:
        d = {
            "id": "test",
            "page_number": 0,
            "text": "Item",
            "block_bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 20},
            "line_count": 1,
            "list_marker": {"marker_type": "numbered", "marker_text": "1.", "number": 1},
        }
        para = Paragraph.from_dict(d)
        assert para.list_marker is not None
        assert para.list_marker.number == 1
```

#### 7.2 paragraph_extractor テスト

**ファイル:** `tests/test_paragraph_extractor.py`

```python
class TestListMarkerDetection:
    """Tests for list marker detection from span structure."""

    def test_detect_bullet_marker(self) -> None:
        """Bullet in separate span should be detected."""
        line = {
            "spans": [
                {"text": "•", "bbox": [100, 200, 108, 212]},
                {"text": " ", "bbox": [108, 200, 111, 212]},
                {"text": "Item content", "bbox": [111, 200, 200, 212]},
            ]
        }
        marker = ParagraphExtractor._detect_list_marker(line, font_size=12.0)
        assert marker is not None
        assert marker.marker_type == "bullet"
        assert marker.marker_text == "•"

    def test_detect_numbered_marker(self) -> None:
        """Number with period should be detected."""
        line = {
            "spans": [
                {"text": "1.", "bbox": [100, 200, 115, 212]},
                {"text": " ", "bbox": [115, 200, 118, 212]},
                {"text": "First item", "bbox": [118, 200, 200, 212]},
            ]
        }
        marker = ParagraphExtractor._detect_list_marker(line, font_size=12.0)
        assert marker is not None
        assert marker.marker_type == "numbered"
        assert marker.number == 1

    def test_wide_span_not_marker(self) -> None:
        """Wide first span should not be detected as marker."""
        line = {
            "spans": [
                {"text": "This is normal", "bbox": [100, 200, 250, 212]},
            ]
        }
        marker = ParagraphExtractor._detect_list_marker(line, font_size=12.0)
        assert marker is None

    def test_single_span_not_marker(self) -> None:
        """Single span line should not have marker."""
        line = {
            "spans": [
                {"text": "• Item without separation", "bbox": [100, 200, 300, 212]},
            ]
        }
        marker = ParagraphExtractor._detect_list_marker(line, font_size=12.0)
        assert marker is None

    def test_relative_threshold(self) -> None:
        """Marker detection should scale with font size."""
        # Large font (24pt) - wider marker span is acceptable
        line_large = {
            "spans": [
                {"text": "•", "bbox": [100, 200, 125, 224]},  # 25pt wide
                {"text": " ", "bbox": [125, 200, 130, 224]},
                {"text": "Item", "bbox": [130, 200, 200, 224]},
            ]
        }
        marker = ParagraphExtractor._detect_list_marker(line_large, font_size=24.0)
        assert marker is not None  # 25pt < 24 * 2.5 = 60pt

        # Small font (8pt) - same width should fail
        marker_small = ParagraphExtractor._detect_list_marker(line_large, font_size=8.0)
        assert marker_small is None  # 25pt > 8 * 2.5 = 20pt


class TestBlockToMultipleParagraphs:
    """Tests for list block splitting into multiple paragraphs."""

    def test_list_block_creates_multiple_paragraphs(self) -> None:
        """Block with list items should create separate paragraphs."""
        block = {
            "lines": [
                {
                    "spans": [
                        {"text": "•", "bbox": [100, 200, 108, 212]},
                        {"text": " ", "bbox": [108, 200, 111, 212]},
                        {"text": "First item", "bbox": [111, 200, 200, 212]},
                    ]
                },
                {
                    "spans": [
                        {"text": "•", "bbox": [100, 180, 108, 192]},
                        {"text": " ", "bbox": [108, 180, 111, 192]},
                        {"text": "Second item", "bbox": [111, 180, 200, 192]},
                    ]
                },
            ]
        }
        extractor = ParagraphExtractor()
        paragraphs = extractor._process_block(block, page_idx=0, block_idx=0, page_height=800)

        assert len(paragraphs) == 2
        assert paragraphs[0].text == "First item"
        assert paragraphs[0].list_marker is not None
        assert paragraphs[1].text == "Second item"
        assert paragraphs[1].list_marker is not None

    def test_regular_block_creates_single_paragraph(self) -> None:
        """Block without list markers should create one paragraph."""
        block = {
            "lines": [
                {"spans": [{"text": "Line one.", "bbox": [100, 200, 200, 212]}]},
                {"spans": [{"text": "Line two.", "bbox": [100, 180, 200, 192]}]},
            ]
        }
        extractor = ParagraphExtractor()
        paragraphs = extractor._process_block(block, page_idx=0, block_idx=0, page_height=800)

        assert len(paragraphs) == 1
        assert paragraphs[0].text == "Line one. Line two."
        assert paragraphs[0].list_marker is None


class TestExtractorConfig:
    """Tests for ExtractorConfig feature toggle."""

    def test_detect_lists_disabled(self) -> None:
        """When detect_lists=False, list markers should not be detected."""
        block = {
            "lines": [
                {
                    "spans": [
                        {"text": "•", "bbox": [100, 200, 108, 212]},
                        {"text": " ", "bbox": [108, 200, 111, 212]},
                        {"text": "Item", "bbox": [111, 200, 150, 212]},
                    ]
                },
            ]
        }
        config = ExtractorConfig(detect_lists=False)
        extractor = ParagraphExtractor()
        paragraphs = extractor._process_block(
            block, page_idx=0, block_idx=0, page_height=800, config=config
        )

        # Should merge with space, not split by lines
        assert len(paragraphs) == 1
        assert paragraphs[0].list_marker is None  # No marker when disabled

    def test_detect_lists_enabled_default(self) -> None:
        """Default config should have detect_lists=True."""
        config = ExtractorConfig()
        assert config.detect_lists is True
```

#### 7.3 markdown_writer テスト

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

    def test_no_marker(self) -> None:
        writer = MarkdownWriter()
        result = writer._format_as_list_item("Regular text", None)
        assert result == "Regular text"

    def test_paragraph_with_list_marker_output(self) -> None:
        """Paragraph with list_marker should output as list item."""
        marker = ListMarker(marker_type="bullet", marker_text="•")
        para = Paragraph(
            id="test",
            page_number=0,
            text="List item content",
            block_bbox=BBox(0, 0, 100, 20),
            line_count=1,
            category="text",
            list_marker=marker,
        )
        writer = MarkdownWriter()
        result = writer._paragraph_to_markdown(para)
        assert result == "- List item content\n"

    def test_code_element_no_list_conversion(self) -> None:
        """Code elements should not be converted to list format."""
        marker = ListMarker(marker_type="bullet", marker_text="•")
        para = Paragraph(
            id="test",
            page_number=0,
            text="• some code",
            block_bbox=BBox(0, 0, 100, 20),
            line_count=1,
            category="inline_formula",
            list_marker=marker,
        )
        writer = MarkdownWriter()
        result = writer._paragraph_to_markdown(para)
        # Should be formatted as code, not as list
        assert result.startswith("`")
```

#### 7.4 TextLayoutEngine テスト

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
        assert text_lines[0] == "Para 1."
        assert text_lines[1] == "Para 2."

    def test_single_newline_no_paragraph_break(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Single newline does NOT create paragraph break marker."""
        text = "Line 1\nLine 2\nLine 3"
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        assert PARAGRAPH_BREAK_MARKER not in lines
        assert len(lines) == 3
        assert lines[0] == "Line 1"
        assert lines[1] == "Line 2"
        assert lines[2] == "Line 3"

    def test_mixed_breaks(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Mixed paragraph and line breaks."""
        text = "Intro.\n\nItem 1\nItem 2\n\nConclusion."
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        markers = [l for l in lines if l == PARAGRAPH_BREAK_MARKER]
        assert len(markers) == 2  # Two paragraph breaks

        text_lines = [l for l in lines if l != PARAGRAPH_BREAK_MARKER]
        assert len(text_lines) == 4  # Intro, Item1, Item2, Conclusion
```

#### 7.5 paragraph_merger テスト

**ファイル:** `tests/test_paragraph_merger.py`

```python
class TestListItemMerging:
    """Tests for list item merge prevention."""

    def test_list_items_not_merged(self) -> None:
        """Paragraphs with list_marker should not be merged."""
        marker = ListMarker(marker_type="bullet", marker_text="•")
        para1 = Paragraph(
            id="p1",
            page_number=0,
            text="Item 1",
            block_bbox=BBox(100, 100, 300, 120),
            line_count=1,
            category="text",
            list_marker=marker,
        )
        para2 = Paragraph(
            id="p2",
            page_number=0,
            text="Item 2",
            block_bbox=BBox(100, 80, 300, 100),
            line_count=1,
            category="text",
            list_marker=marker,
        )

        config = MergeConfig()
        result = merge_adjacent_paragraphs([para1, para2], config)

        # Should remain as 2 separate paragraphs
        assert len(result) == 2

    def test_regular_paragraphs_merged_with_double_newline(self) -> None:
        """Merged paragraphs should be joined with \\n\\n."""
        para1 = Paragraph(
            id="p1",
            page_number=0,
            text="First paragraph.",
            block_bbox=BBox(100, 100, 300, 120),
            line_count=1,
            category="text",
            original_font_size=12.0,
        )
        para2 = Paragraph(
            id="p2",
            page_number=0,
            text="Second paragraph.",
            block_bbox=BBox(100, 94, 300, 100),  # Close gap
            line_count=1,
            category="text",
            original_font_size=12.0,
        )

        config = MergeConfig(gap_tolerance=1.0)
        result = merge_adjacent_paragraphs([para1, para2], config)

        assert len(result) == 1
        assert "\n\n" in result[0].text
        assert "First paragraph.\n\nSecond paragraph." == result[0].text
```

#### 7.6 E2E テスト

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
        """Bullet list on page 10 should create separate paragraphs."""
        from pdf_translator.core.paragraph_extractor import ParagraphExtractor

        paragraphs = ParagraphExtractor.extract_from_pdf(
            "tests/fixtures/sample_autogen_paper.pdf",
            page_range=[9],  # 0-indexed
        )

        # Find bullet list paragraphs
        bullet_paras = [
            p for p in paragraphs
            if p.list_marker and p.list_marker.marker_type == "bullet"
        ]

        # Should have multiple separate paragraphs (not one merged)
        assert len(bullet_paras) >= 2

        # Each should have its own content
        for para in bullet_paras:
            assert para.list_marker.marker_text == "•"
            assert para.text  # Non-empty

    def test_numbered_list_page_21(self) -> None:
        """Numbered list on page 21 should create separate paragraphs."""
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

        # Check numbers are present
        numbers = [p.list_marker.number for p in numbered_paras]
        assert 1 in numbers


    def test_markdown_output_format(self) -> None:
        """Markdown output should have proper list syntax."""
        from pdf_translator.core.paragraph_extractor import ParagraphExtractor
        from pdf_translator.output.markdown_writer import MarkdownWriter

        paragraphs = ParagraphExtractor.extract_from_pdf(
            "tests/fixtures/sample_autogen_paper.pdf",
            page_range=[9],
        )

        writer = MarkdownWriter()
        markdown = writer.write(paragraphs)

        # Should contain proper Markdown list syntax
        assert "- " in markdown  # Bullet list items
```

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/pdf_translator/core/models.py` | `ListMarker` dataclass 追加（`to_dict/from_dict`含む）、`Paragraph.list_marker` フィールド追加 |
| `src/pdf_translator/core/paragraph_extractor.py` | `ExtractorConfig` 追加、span構造からのリストマーカー検出、行単位Paragraph分割 |
| `src/pdf_translator/core/paragraph_merger.py` | リスト項目マージ防止、`\n` → `\n\n` 結合変更 |
| `src/pdf_translator/core/text_layout.py` | `\n\n`=段落区切り、`\n`=行区切りに変更 |
| `src/pdf_translator/output/markdown_writer.py` | `list_marker` に基づくMarkdown変換 |
| `src/pdf_translator/cli.py` | `--no-detect-lists` オプション追加 |
| `tests/test_models.py` | `ListMarker` テスト、シリアライズテスト追加 |
| `tests/test_paragraph_extractor.py` | **新規** - リストマーカー検出、行単位分割、`ExtractorConfig`テスト |
| `tests/test_paragraph_merger.py` | リスト項目マージ防止テスト追加 |
| `tests/test_text_layout.py` | `\n\n`/`\n` 区別テスト追加 |
| `tests/test_markdown_writer.py` | リストフォーマットテスト追加 |
| `tests/test_list_e2e.py` | **新規** - E2Eテスト |

---

## リスクと対策

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|-------|------|
| マーカー検出漏れ | 低 | 中 | 相対閾値の導入、テストでの検証 |
| 誤検知（通常テキストをマーカー判定） | 低 | 中 | 複数span必須条件、幅閾値チェック |
| Paragraph数増加によるパフォーマンス影響 | 低 | 低 | リストブロックのみ分割、影響は限定的 |
| 既存テスト破壊 | 中 | 中 | `_process_block` 戻り値変更に伴う呼び出し元修正 |
| 翻訳時のリスト構造変化 | 低 | 低 | マーカーを本文から分離して保持 |

**後方互換性:**
- `list_marker` フィールドはオプショナル（`None` がデフォルト）
- 既存の段落処理は影響なし
- シリアライズ済み中間JSON は `list_marker` なしでも読み込み可能

---

## 検証結果

`tests/fixtures/sample_autogen_paper.pdf` を使用して検出ロジックを検証:

### テスト対象

| ページ | 対象 | 内容 |
|--------|------|------|
| 16 | 箇条書き | "Ease of use", "Modularity" 等 |
| 5, 16 | 番号付きリスト | "1. Unified interfaces...", "2. Control by fusion..." 等 |
| 11, 12 | 年号参照 | "2019.", "2020." 等（誤検出対象） |

### 結果

```
✅ Bullet list detection: 31 items correctly detected
   - Span structure: • (3.5pt) + ' ' + content

✅ Numbered list detection: 24 items correctly detected
   - Span structure: 1. (7.5pt) + ' ' + content

✅ Year false positive prevention: 7 year references correctly rejected
   - Pattern r'^([1-9]\d{0,2})\.$' limits to 1-999
```

### 検証されたspan構造

```
=== 箇条書き (Page 16) ===
Span 0: x0=108.0, width=3.5pt, text='•'
Span 1: x0=111.5, width=0.0pt, text=' '
Span 2: x0=116.5, width=47.6pt, text='Ease of use'
→ ✅ DETECTED: bullet '•'

=== 番号付きリスト (Page 5) ===
Span 0: x0=108.0, width=7.5pt, text='1.'
Span 1: x0=115.5, width=0.0pt, text=' '
Span 2: x0=120.5, width=330.1pt, text='Unified interfaces...'
→ ✅ DETECTED: numbered '1.' (num=1)

=== 年号参照 (Page 11) - 正しく除外 ===
Span 0: x0=118.0, width=22.4pt, text='2019.'
Span 1: x0=140.4, width=0.0pt, text='\n'
→ ❌ Correctly rejected (4-digit year pattern)
```

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
  - ページ 16: 箇条書きリスト（"Ease of use", "Modularity" 等）
  - ページ 5, 16: 番号付きリスト（"1. Unified interfaces..." 等）
  - ページ 11, 12: 年号参照（誤検出テスト用）

---

## 変更履歴

| 日付 | 変更内容 |
|------|---------|
| 2026-01-09 | 初版作成（テキストパース → span構造分析に変更） |
| 2026-01-09 | レビューFB対応: 行単位Paragraph分割、シリアライズ追加、相対閾値導入 |
| 2026-01-09 | 実PDF検証: 年号誤検出防止のためパターンを1-999に制限、検証結果セクション追加 |
| 2026-01-09 | 機能トグル追加: `ExtractorConfig.detect_lists`、`--no-detect-lists` CLIオプション |
