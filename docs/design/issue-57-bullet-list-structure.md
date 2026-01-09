# Issue #57: 箇条書き構造の保持と Markdown 変換

## 概要

PDF から抽出された箇条書き（ビュレットリスト）の構造が段落抽出時に失われる問題を修正し、
PDF 出力および Markdown 出力の両方で正しく表示されるようにする。

**関連 Issue**: [#57](https://github.com/Mega-Gorilla/pdf-translator/issues/57)

---

## 問題の分析

### 現象

**元の PDF 構造:**
```
• Privacy and Data Protection: The framework allows...
• Bias and Fairness: LLMs have been shown to exhibit...
• Accountability and Transparency: As discussed...
```

**現在の出力（PDF/Markdown 両方）:**
```
• Privacy and Data Protection: The framework allows... • Bias and Fairness: LLMs have been shown to exhibit...
```

### 根本原因

`paragraph_extractor.py` の `_process_block` メソッドで行がスペース結合される:

```python
# paragraph_extractor.py:125
merged_text = " ".join(lines)  # ← 箇条書き構造が失われる
```

### 関連する既存実装

PR #58 で `paragraph_merger.py` は改行結合 (`\n`) に修正済み:

```python
# paragraph_merger.py:384
merged_text = para1.text + "\n" + para2.text
```

しかし、`TextLayoutEngine` は `\n` を段落区切りとして扱い、段落間スペースを挿入する。
箇条書きの行区切りにも `\n` を使うと、各項目間に余計なスペースが入る問題がある。

---

## 設計方針

### 段落区切りと行区切りの区別

| 用途 | マーカー | PDF 出力 | Markdown 出力 |
|------|----------|----------|---------------|
| 段落間 | `\n\n` | 段落スペース挿入 | 空行で段落分離 |
| 行区切り（箇条書き等） | `\n` | 単純な改行 | 改行 |

この設計により:
- マージされた段落間には視覚的なスペースが入る
- 箇条書き項目間は詰まって表示される（リストとして自然）

---

## 技術的決定事項

### 1. 箇条書き検出パターン

**決定: 記号＋スペース必須**

誤検知を防ぐため、記号の直後にスペースまたはタブがあることを条件とする。

```python
# 検出対象
BULLET_CHARS = frozenset("•◦○●◆◇▸▹‣⁃")

def _is_bullet_line(line: str) -> bool:
    """行が箇条書きで始まるか判定"""
    stripped = line.lstrip()
    if len(stripped) < 2:
        return False

    # 記号 + スペース/タブ
    if stripped[0] in BULLET_CHARS and stripped[1] in " \t":
        return True

    # ダッシュ/アスタリスク + スペース
    if stripped[0] in "-*" and stripped[1] == " ":
        return True

    # 番号付きリスト (1. 2. など)
    if re.match(r"^\d+\.\s", stripped):
        return True

    return False
```

**理由:**
- 単独の `•` や `-` は他の用途（数式、ハイフン）と混同する可能性
- 実際の箇条書きは必ずスペースが続く

**将来拡張:**
- `1)`, `a)`, `①` などのパターンは必要に応じて追加

### 2. インデント保持

**決定: `strip()` → `rstrip()` に変更**

```python
# Before
line_text = line_text.strip()

# After
line_text = line_text.rstrip()  # 先頭空白を保持
```

**理由:**
- 先頭の空白はインデントレベルを示す可能性がある
- 末尾の空白は不要なので削除

**制限:**
- ネストされたリストの完全なインデント復元には bbox x0 からの推定が必要
- これは将来の Issue として分離

### 3. Markdown 変換の適用条件

**決定: element_type に基づいて適用/スキップを判断**

```python
# 変換をスキップする要素タイプ
SKIP_BULLET_CONVERSION = {"code", "code_block"}

def _format_text(self, text: str, element_type: str, ...) -> str:
    if element_type not in SKIP_BULLET_CONVERSION:
        text = self._convert_bullets_to_markdown(text)
    # ...
```

**理由:**
- コードブロック内の `•` や `-` は変換すべきでない
- `parallel` モードでは原文・訳文の両方に変換を適用

### 4. TextLayoutEngine の改行処理

**決定: `\n\n` を段落区切り、`\n` を行区切りとして処理**

```python
def wrap_text(self, text: str, max_width: float, ...) -> list[str]:
    all_lines: list[str] = []

    # \n\n で段落分割
    paragraphs = text.split("\n\n")

    for i, paragraph in enumerate(paragraphs):
        # 各段落内を \n で行分割
        lines = paragraph.split("\n")

        for line in lines:
            line = line.strip()
            if line:
                # 行を折り返し処理
                wrapped = self._wrap_segment(line, max_width, font_handle, font_size)
                all_lines.extend(wrapped)

        # 段落間にマーカーを挿入（最後の段落以外）
        if i < len(paragraphs) - 1:
            all_lines.append(PARAGRAPH_BREAK_MARKER)

    return all_lines
```

**理由:**
- 既存の `PARAGRAPH_BREAK_MARKER` 機構を活用
- 単一 `\n` は段落スペースなしの改行として扱う

---

## 実装計画

### Phase 1: TextLayoutEngine 修正

**ファイル:** `src/pdf_translator/core/text_layout.py`

`wrap_text()` メソッドを修正し、`\n\n` と `\n` を区別:

```python
def wrap_text(
    self,
    text: str,
    max_width: float,
    font_handle: ctypes.c_void_p,
    font_size: float,
) -> list[str]:
    """Wrap text to fit within max_width, handling paragraph and line breaks.

    - \\n\\n: Paragraph break (adds spacing via PARAGRAPH_BREAK_MARKER)
    - \\n: Line break (no extra spacing)
    """
    all_lines: list[str] = []

    # Split by paragraph breaks (\n\n)
    paragraphs = text.split("\n\n")

    for para_idx, paragraph in enumerate(paragraphs):
        # Split by line breaks (\n) within paragraph
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

### Phase 2: paragraph_merger.py 修正

**ファイル:** `src/pdf_translator/core/paragraph_merger.py`

段落マージ時の区切りを `\n` → `\n\n` に変更:

```python
def _merge_two_paragraphs(para1: Paragraph, para2: Paragraph) -> Paragraph:
    # Before (PR #58)
    # merged_text = para1.text + "\n" + para2.text

    # After
    merged_text = para1.text + "\n\n" + para2.text
    # ...
```

### Phase 3: paragraph_extractor.py 修正

**ファイル:** `src/pdf_translator/core/paragraph_extractor.py`

#### 3.1 定数追加

```python
# Bullet characters for list detection
BULLET_CHARS = frozenset("•◦○●◆◇▸▹‣⁃")
```

#### 3.2 箇条書き検出メソッド追加

```python
@staticmethod
def _is_bullet_line(line: str) -> bool:
    """Check if line starts with a bullet pattern.

    Requires symbol + space to avoid false positives.
    """
    stripped = line.lstrip()
    if len(stripped) < 2:
        return False

    # Symbol bullets (•, ◦, etc.) + space/tab
    if stripped[0] in BULLET_CHARS and stripped[1] in " \t":
        return True

    # ASCII bullets (-, *) + space
    if stripped[0] in "-*" and stripped[1] == " ":
        return True

    # Numbered lists (1., 2., etc.)
    if re.match(r"^\d+\.\s", stripped):
        return True

    return False

@staticmethod
def _has_bullet_structure(lines: list[str]) -> bool:
    """Check if block contains bullet list structure."""
    return any(ParagraphExtractor._is_bullet_line(line) for line in lines)
```

#### 3.3 `_process_block` 修正

```python
def _process_block(self, block: dict[str, Any], ...) -> Paragraph | None:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        line_text = "".join(span.get("text", "") for span in spans)
        line_text = line_text.rstrip()  # strip() → rstrip() で先頭空白を保持
        if line_text:
            lines.append(line_text)

    if not lines:
        return None

    # 箇条書き構造がある場合は改行で結合、それ以外はスペースで結合
    if self._has_bullet_structure(lines):
        merged_text = "\n".join(lines)
    else:
        merged_text = " ".join(lines)

    merged_text = re.sub(r"\s+", " ", merged_text).strip()  # 箇条書き時はスキップが必要
    # ...
```

**注意:** 箇条書き時の正規化処理の調整が必要:

```python
if self._has_bullet_structure(lines):
    merged_text = "\n".join(lines)
    # 各行の余分な空白のみ正規化（改行は保持）
    merged_text = "\n".join(
        re.sub(r"[ \t]+", " ", line).strip()
        for line in merged_text.split("\n")
    )
else:
    merged_text = " ".join(lines)
    merged_text = re.sub(r"\s+", " ", merged_text).strip()
```

### Phase 4: markdown_writer.py 修正

**ファイル:** `src/pdf_translator/output/markdown_writer.py`

#### 4.1 定数追加

```python
# Bullet characters to convert to Markdown list syntax
BULLET_CHARS = frozenset("•◦○●◆◇▸▹‣⁃")

# Element types where bullet conversion should be skipped
SKIP_BULLET_CONVERSION = frozenset({"code", "code_block"})
```

#### 4.2 変換メソッド追加

```python
def _convert_bullets_to_markdown(self, text: str) -> str:
    """Convert bullet characters to Markdown list syntax.

    Converts lines starting with bullet characters (•, ◦, etc.)
    to Markdown list format using '-'.

    Args:
        text: Text potentially containing bullet lists.

    Returns:
        Text with bullet characters converted to Markdown syntax.
    """
    if "\n" not in text:
        # Single line - check and convert if needed
        return self._convert_single_bullet_line(text)

    lines = text.split("\n")
    result = []
    for line in lines:
        result.append(self._convert_single_bullet_line(line))
    return "\n".join(result)

def _convert_single_bullet_line(self, line: str) -> str:
    """Convert a single line's bullet to Markdown if applicable."""
    stripped = line.lstrip()
    if len(stripped) >= 2 and stripped[0] in BULLET_CHARS and stripped[1] in " \t":
        indent = len(line) - len(stripped)
        return " " * indent + "- " + stripped[1:].lstrip()
    return line
```

#### 4.3 `_format_text` 修正

```python
def _format_text(
    self,
    text: str,
    element_type: str,
    paragraph: Paragraph | None = None,
) -> str:
    # 箇条書き変換（code系要素以外）
    if element_type not in SKIP_BULLET_CONVERSION:
        text = self._convert_bullets_to_markdown(text)

    # Apply style formatting
    formatted = self._apply_style(text, paragraph)

    # ... 以降は既存のフォーマット処理
```

#### 4.4 parallel モード対応

`_format_text` 内の parallel モード処理でも変換を適用:

```python
if (
    self._config.output_mode == MarkdownOutputMode.PARALLEL
    and paragraph
    and paragraph.translated_text
):
    translated = paragraph.translated_text
    # 訳文にも箇条書き変換を適用
    if element_type not in SKIP_BULLET_CONVERSION:
        translated = self._convert_bullets_to_markdown(translated)
    translated = self._apply_style(translated, paragraph)
    # ...
```

### Phase 5: テスト

#### 5.1 TextLayoutEngine テスト

**ファイル:** `tests/test_text_layout.py`

```python
class TestParagraphAndLineBreaks:
    """Tests for paragraph (\\n\\n) vs line (\\n) break handling."""

    def test_double_newline_creates_paragraph_break(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Double newline should create paragraph break marker."""
        text = "First paragraph.\n\nSecond paragraph."
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        assert PARAGRAPH_BREAK_MARKER in lines
        text_lines = [l for l in lines if l != PARAGRAPH_BREAK_MARKER]
        assert len(text_lines) == 2

    def test_single_newline_no_paragraph_break(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Single newline should NOT create paragraph break marker."""
        text = "• Item 1\n• Item 2"
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        assert PARAGRAPH_BREAK_MARKER not in lines
        assert len(lines) == 2

    def test_mixed_breaks(
        self,
        layout_engine: TextLayoutEngine,
        helvetica_font: ctypes.c_void_p,
    ) -> None:
        """Mixed paragraph and line breaks."""
        text = "Intro.\n\n• Item 1\n• Item 2\n\nConclusion."
        lines = layout_engine.wrap_text(text, 200.0, helvetica_font, 12.0)

        markers = [l for l in lines if l == PARAGRAPH_BREAK_MARKER]
        assert len(markers) == 2  # Two paragraph breaks
```

#### 5.2 paragraph_extractor テスト

**ファイル:** `tests/test_paragraph_extractor.py`

```python
class TestBulletDetection:
    """Tests for bullet list detection."""

    def test_is_bullet_line_symbol_with_space(self) -> None:
        """Symbol bullet with space should be detected."""
        assert ParagraphExtractor._is_bullet_line("• Item")
        assert ParagraphExtractor._is_bullet_line("◦ Nested")
        assert ParagraphExtractor._is_bullet_line("● Point")

    def test_is_bullet_line_symbol_without_space(self) -> None:
        """Symbol without space should NOT be detected."""
        assert not ParagraphExtractor._is_bullet_line("•Item")  # No space
        assert not ParagraphExtractor._is_bullet_line("•")  # Too short

    def test_is_bullet_line_ascii_dash(self) -> None:
        """ASCII dash with space should be detected."""
        assert ParagraphExtractor._is_bullet_line("- Item")
        assert ParagraphExtractor._is_bullet_line("* Point")

    def test_is_bullet_line_numbered(self) -> None:
        """Numbered list should be detected."""
        assert ParagraphExtractor._is_bullet_line("1. First")
        assert ParagraphExtractor._is_bullet_line("10. Tenth")
        assert not ParagraphExtractor._is_bullet_line("1.First")  # No space

    def test_is_bullet_line_indented(self) -> None:
        """Indented bullets should be detected."""
        assert ParagraphExtractor._is_bullet_line("  • Indented")
        assert ParagraphExtractor._is_bullet_line("    - Deep")


class TestBulletBlockProcessing:
    """Tests for bullet block text merging."""

    def test_bullet_block_joins_with_newline(self) -> None:
        """Block with bullets should join lines with newline."""
        # This test requires mocking block structure
        pass

    def test_regular_block_joins_with_space(self) -> None:
        """Block without bullets should join lines with space."""
        pass
```

#### 5.3 markdown_writer テスト

**ファイル:** `tests/test_markdown_writer.py`

```python
class TestBulletConversion:
    """Tests for bullet to Markdown conversion."""

    def test_convert_bullet_symbol(self) -> None:
        """Bullet symbol should convert to dash."""
        writer = MarkdownWriter()
        result = writer._convert_bullets_to_markdown("• Item")
        assert result == "- Item"

    def test_convert_multiple_bullets(self) -> None:
        """Multiple bullet lines should all convert."""
        writer = MarkdownWriter()
        text = "• First\n• Second\n• Third"
        result = writer._convert_bullets_to_markdown(text)
        assert result == "- First\n- Second\n- Third"

    def test_preserve_indentation(self) -> None:
        """Indentation should be preserved."""
        writer = MarkdownWriter()
        text = "  • Indented item"
        result = writer._convert_bullets_to_markdown(text)
        assert result == "  - Indented item"

    def test_skip_in_code_block(self) -> None:
        """Bullets in code blocks should not be converted."""
        # Test via _format_text with element_type="code_block"
        pass

    def test_parallel_mode_converts_both(self) -> None:
        """Parallel mode should convert both original and translated."""
        pass
```

#### 5.4 E2E テスト

**ファイル:** `tests/test_bullet_e2e.py`

```python
@pytest.mark.skipif(
    not Path("tests/fixtures/sample_autogen_paper.pdf").exists(),
    reason="Test PDF not available"
)
def test_bullet_extraction_page_10() -> None:
    """Test bullet extraction from sample PDF page 10."""
    from pdf_translator.core.paragraph_extractor import ParagraphExtractor

    paragraphs = ParagraphExtractor.extract_from_pdf(
        "tests/fixtures/sample_autogen_paper.pdf",
        page_range=[9],  # 0-indexed
    )

    # Find paragraph with bullet content
    bullet_para = None
    for para in paragraphs:
        if "Privacy and Data Protection" in para.text:
            bullet_para = para
            break

    assert bullet_para is not None
    # Should contain newlines between bullet items
    assert "\n" in bullet_para.text
    assert "• Privacy" in bullet_para.text
    assert "• Bias" in bullet_para.text
```

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/pdf_translator/core/text_layout.py` | `\n\n`/`\n` 区別対応 |
| `src/pdf_translator/core/paragraph_merger.py` | `\n` → `\n\n` 変更 |
| `src/pdf_translator/core/paragraph_extractor.py` | 箇条書き検出・改行結合 |
| `src/pdf_translator/output/markdown_writer.py` | 箇条書き → Markdown 変換 |
| `tests/test_text_layout.py` | 段落/行区切りテスト追加 |
| `tests/test_paragraph_extractor.py` | **新規** - 箇条書き検出テスト |
| `tests/test_markdown_writer.py` | 箇条書き変換テスト追加 |

---

## リスクと対策

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|-------|------|
| 箇条書き誤検知 | 低 | 中 | 記号＋スペース条件で保守的に検出 |
| 通常段落の改行混入 | 低 | 中 | 箇条書きブロックのみ改行結合 |
| 翻訳品質への影響 | 低 | 低 | 箇条書き構造が翻訳エンジンに伝わりやすくなる |
| 既存テスト破壊 | 中 | 中 | `\n` → `\n\n` 変更に伴うテスト更新 |

**後方互換性:**
- 通常の段落処理は変更なし（スペース結合）
- 箇条書き検出がない限り既存の動作を維持

---

## 将来対応（別 Issue）

1. **ネストリスト対応**
   - bbox x0 からインデントレベルを推定
   - Markdown のネストリスト記法に対応

2. **追加の箇条書きパターン**
   - `1)`, `a)`, `A.` などの番号形式
   - `①`, `②` などの丸数字

3. **翻訳後の箇条書き検出**
   - 翻訳エンジンが箇条書き構造を変換してしまう場合の対応

---

## 参考

- [Issue #57](https://github.com/Mega-Gorilla/pdf-translator/issues/57)
- [PR #58](https://github.com/Mega-Gorilla/pdf-translator/pull/58) - paragraph_merger 改行対応
- `tests/fixtures/sample_autogen_paper.pdf` - ページ 10 に箇条書きあり
