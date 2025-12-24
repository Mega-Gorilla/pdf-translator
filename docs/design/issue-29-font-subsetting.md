# Issue #29: フォントサブセット化による PDF ファイルサイズ最適化

## 概要

翻訳後の PDF ファイルサイズが元の PDF と比較して大幅に増加する問題を解決するため、
fonttools を使用したフォントサブセット化機能を実装する。

**関連 Issue**: [#29](https://github.com/Mega-Gorilla/pdf-translator/issues/29)

---

## 問題の分析

### 現状

| PDF | ページ数 | 元サイズ | 翻訳後 | 倍率 |
|-----|---------|---------|--------|------|
| sample_llama.pdf | 1 | 111 KB | 31 MB | **280x** |
| sample_autogen_paper.pdf | 43 | 3.5 MB | 34 MB | 10x |
| side-by-side (43p) | - | 3.5 MB | 632 MB | **180x** |

### 根本原因

pypdfium2 の `FPDFText_LoadFont()` は**フォント全体を埋め込む**ため、
19MB の CJK フォントがそのまま PDF に含まれる。

```
埋め込まれたフォント分析:
- NotoSansCJKjp-Regular: 18.6 MB  ← 問題の原因
- その他のフォント (サブセット済み): 各 7-22 KB
```

PDFium の `fpdf_edit.h` によると、`FPDFText_LoadFont` は
「data is copied by the font object」とあり、サブセット化はサポートされていない。

---

## 解決策: fonttools によるフォントサブセット化

### 検証結果

CID フォント互換性の検証に成功。

| 項目 | 結果 |
|------|------|
| fonttools でサブセット生成 | ✅ 成功 |
| pypdfium2 で CID フォントとしてロード | ✅ 成功 |
| 日本語テキストを挿入して PDF 保存 | ✅ 成功 |
| 生成された PDF の検証 | ✅ 成功 |

### サイズ削減効果

| 項目 | サイズ |
|------|--------|
| 元の CJK フォント | 18.6 MB |
| サブセットフォント (74文字) | 41.5 KB |
| **削減率** | **99.8%** |

| PDF | サイズ |
|-----|--------|
| フルフォント埋め込み | 15.2 MB |
| サブセット使用 | 147.5 KB |
| **削減率** | **99.0%** |

---

## 技術的決定事項

### 1. fonttools の依存関係

**決定: 必須依存として追加**

```toml
dependencies = [
    ...
    "fonttools>=4.50.0",
]
```

**理由**:
- デフォルトで 31MB の PDF を出力するのは許容できない
- ユーザーが追加インストールなしで最適な体験を得るべき
- fonttools は軽量 (5MB) で外部依存なし
- 翻訳ツールとして「まともなサイズの PDF を出力する」のは基本機能

### 2. ライセンス互換性

| パッケージ | ライセンス | Apache-2.0 互換性 |
|-----------|-----------|------------------|
| fonttools | MIT | ✅ 互換 |

Apache Software Foundation ポリシーにより MIT は "Category A" として互換性確認済み。

### 3. キャッシュ戦略

| 項目 | 決定 |
|------|------|
| 保存場所 | 一時ディレクトリ (デフォルト) + ユーザー指定可能 |
| 有効期間 | セッション限り |
| キー | フォントフルパス + サイズ + mtime + 文字セットハッシュ |

**キャッシュキーの設計**:
同名フォントが異なるディレクトリに存在するケースに対応するため、
`font_path.stem` ではなく以下の情報を組み合わせてキーを生成:

```python
# フォントの一意性を保証するキー
font_identity = f"{font_path.resolve()}:{font_path.stat().st_size}:{font_path.stat().st_mtime}"
cache_key = hashlib.sha256(f"{font_identity}:{sorted_chars}".encode()).hexdigest()[:16]
```

**クリーンアップ方針**:
- 一時ディレクトリ使用時: OS の一時ファイルクリーンアップに委ねる
- `FontSubsetter` インスタンス破棄時に `cleanup()` メソッドで明示的削除可能
- ユーザー指定ディレクトリ使用時: ユーザー責任で管理

### 4. パイプライン統合位置

**決定: `_stage_apply` 内に統合**

サブセット化はフォントロードの最適化であり、独立した処理ステージではない。
`FontSubsetter` クラスに処理を委譲することで責務分離を維持。

### 5. Bold/Italic フォントの扱い

**決定: 初期実装で対応**

Issue #29 の主目的は PDF ファイルサイズの削減であり、
Regular のみサブセット化して Bold/Italic をそのままにすると目標を達成できない。

```
Regular テキスト → サブセット (25KB) ✓
Bold テキスト → フルフォント埋め込み (18.6MB) ✗  ← 問題
```

**NotoSansCJK のファイル構造**:

| スタイル | ファイル |
|---------|---------|
| Regular | `NotoSansCJK-Regular.ttc` |
| Bold | `NotoSansCJK-Bold.ttc` |
| Light | `NotoSansCJK-Light.ttc` |

**実装アプローチ**:

1. `is_bold`/`is_italic` 属性（既存の Paragraph モデル）を活用
2. フォントファイル名のウェイト部分を置換して対応ファイルを探索
3. バリアントが見つからない場合は Regular にフォールバック

```python
def _find_font_variant(
    base_font_path: Path,
    is_bold: bool,
    is_italic: bool,
) -> Path | None:
    """Find font file for the given style variant.

    NotoSansCJK-Regular.ttc → NotoSansCJK-Bold.ttc (is_bold=True)
    """
    stem = base_font_path.stem  # "NotoSansCJK-Regular"
    parent = base_font_path.parent

    # Determine target weight
    if is_bold:
        target_weight = "Bold"
    else:
        target_weight = "Regular"

    # Replace weight in filename
    import re
    new_stem = re.sub(
        r"-(Regular|Bold|Light|Medium|Thin|Black)",
        f"-{target_weight}",
        stem,
    )

    variant_path = parent / f"{new_stem}{base_font_path.suffix}"
    return variant_path if variant_path.exists() else None
```

**キャッシュへの影響**:

キャッシュキーは `font_path` を含むため、Bold/Italic 用の別ファイルは
自動的に別キャッシュエントリになる:

```
Regular: sha256("NotoSansCJK-Regular.ttc:...:chars") → subset_abc123.ttf
Bold:    sha256("NotoSansCJK-Bold.ttc:...:chars")    → subset_def456.ttf
```

**フォールバック戦略**:

```python
def get_subset_font(
    font_path: Path,
    chars: set[str],
    is_bold: bool = False,
    is_italic: bool = False,
) -> Path:
    """Get subset font, with fallback to Regular if variant not found."""
    if is_bold or is_italic:
        variant_path = _find_font_variant(font_path, is_bold, is_italic)
        if variant_path:
            return self._create_subset(variant_path, chars)
        else:
            logger.warning(
                "Font variant not found for bold=%s, italic=%s. Using %s",
                is_bold, is_italic, font_path.name
            )

    return self._create_subset(font_path, chars)
```

**Italic の注意点**:

NotoSansCJK は Italic バリアントを持たない。
Italic 指定時は以下の動作とする:

1. `_find_font_variant()` で Italic ファイルを探索
2. 見つからない場合は Regular/Bold にフォールバック
3. **PDF transform による斜体変換を適用** (新規実装)

**Italic transform 実装**:

フォントファイルがない場合、skew transform を使用して斜体を表現する。

**統合位置**:

翻訳適用の主経路は `apply_paragraphs()` → `insert_laid_out_text()` であるため、
Italic skew transform は以下の両方に統合する:

1. **`insert_laid_out_text()`** (主経路): 回転と skew を合成した変換行列を適用
2. **`insert_text_object()`** (フォールバック): 単独の skew 変換を適用

**行列合成**:

PDF の affine transform `[a, b, c, d, e, f]` は以下の行列を表す:
```
| a  b  0 |
| c  d  0 |
| e  f  1 |
```

現在の `insert_laid_out_text()` の回転変換:
```
| cos(r)   sin(r)  0 |
| -sin(r)  cos(r)  0 |
| x        y       1 |
```

Italic skew 変換 (s = tan(12°) ≈ 0.21):
```
| 1  s  0 |
| 0  1  0 |
| 0  0  1 |
```

**合成順序**: Skew を先に適用し、その後回転 (オブジェクト座標系での斜体)

`Rotation × Skew` の結果:
```
| cos(r)   s*cos(r) + sin(r)   0 |
| -sin(r)  -s*sin(r) + cos(r)  0 |
| x        y                   1 |
```

**実装コード** (`insert_laid_out_text()` 内):

```python
# Constants
ITALIC_SKEW = 0.21  # tan(12°) ≈ 0.21

# In insert_laid_out_text(), when is_italic=True and no italic font:
if needs_italic_skew:
    # Combined rotation + skew transform
    a = cos_r
    b = ITALIC_SKEW * cos_r + sin_r
    c = -sin_r
    d = -ITALIC_SKEW * sin_r + cos_r
else:
    # Rotation only (current behavior)
    a = cos_r
    b = sin_r
    c = -sin_r
    d = cos_r

pdfium.raw.FPDFPageObj_Transform(
    text_obj,
    ctypes.c_double(a),
    ctypes.c_double(b),
    ctypes.c_double(c),
    ctypes.c_double(d),
    ctypes.c_double(x_pos),
    ctypes.c_double(y_pos),
)
```

**`insert_text_object()` フォールバック用** (回転なしの単純ケース):

```python
def _apply_italic_transform(text_obj_handle: int, skew: float = 0.21) -> None:
    """Apply italic skew transform to text object (no rotation)."""
    pdfium.raw.FPDFPageObj_Transform(
        text_obj_handle,
        ctypes.c_double(1.0),   # a
        ctypes.c_double(skew),  # b: horizontal skew
        ctypes.c_double(0.0),   # c
        ctypes.c_double(1.0),   # d
        ctypes.c_double(0.0),   # e
        ctypes.c_double(0.0),   # f
    )
```

### 6. TTC 対応

**決定: `fontNumber` パラメータで明示指定可能**

```python
# TTC の場合は fontNumber で特定のフォントを選択
font = TTFont(font_path, fontNumber=font_number)
```

`PipelineConfig.cjk_font_number` でユーザーが指定可能。
デフォルトは 0。

**fontNumber の特定方法**:

TTC ファイル内のフォント一覧は以下のコマンドで確認可能:

```bash
# fonttools の ttx コマンドで確認
python -c "from fontTools.ttLib import TTCollection; tc = TTCollection('/path/to/font.ttc'); print([f['name'].getDebugName(4) for f in tc.fonts])"
```

NotoSansCJK-Regular.ttc の例:
| fontNumber | フォント名 |
|------------|-----------|
| 0 | Noto Sans CJK JP (日本語) |
| 1 | Noto Sans CJK KR (韓国語) |
| 2 | Noto Sans CJK SC (簡体中国語) |
| 3 | Noto Sans CJK TC (繁体中国語) |
| 4 | Noto Sans CJK HK (香港) |

日本語翻訳の場合はデフォルト (0) で問題なし。

### 7. side-by-side モード

**決定: 追加対応不要**

デバッグラベルは Helvetica (標準フォント) を使用するため、
CJK フォントのサブセットに含める必要なし。

---

## 実装計画

### Phase 1: FontSubsetter クラスの実装

**新規ファイル**: `src/pdf_translator/core/font_subsetter.py`

```python
"""Font subsetting using fonttools."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Common punctuation and digits to include in subset
SAFETY_MARGIN_CHARS = "。、！？「」『』（）…―　0123456789"

# Font weight patterns for variant detection
WEIGHT_PATTERN = re.compile(r"-(Regular|Bold|Light|Medium|Thin|Black)")


@dataclass
class SubsetConfig:
    """Font subsetting configuration."""

    include_common_punctuation: bool = True
    cache_dir: Path | None = None  # None = use temp directory


def _find_font_variant(
    base_font_path: Path,
    is_bold: bool,
    is_italic: bool,
) -> Path | None:
    """Find font file for the given style variant.

    NotoSansCJK-Regular.ttc → NotoSansCJK-Bold.ttc (is_bold=True)

    Note: NotoSansCJK does not have Italic variants.
    When Italic is requested but not available, falls back to Regular/Bold.

    Args:
        base_font_path: Path to the base font file (typically Regular).
        is_bold: Whether Bold variant is requested.
        is_italic: Ignored for NotoSansCJK (no italic files).

    Returns:
        Path to variant font file, or None if not found.
    """
    stem = base_font_path.stem
    parent = base_font_path.parent

    # Determine target weight
    target_weight = "Bold" if is_bold else "Regular"

    # Replace weight in filename
    if WEIGHT_PATTERN.search(stem):
        new_stem = WEIGHT_PATTERN.sub(f"-{target_weight}", stem)
    else:
        # No weight pattern found, return None
        return None

    variant_path = parent / f"{new_stem}{base_font_path.suffix}"
    return variant_path if variant_path.exists() else None


class FontSubsetter:
    """Font subsetter using fonttools."""

    def __init__(self, config: SubsetConfig | None = None) -> None:
        self._config = config or SubsetConfig()
        self._cache: dict[str, Path] = {}

    def subset_for_texts(
        self,
        font_path: Path,
        texts: list[str],
        font_number: int = 0,
        is_bold: bool = False,
        is_italic: bool = False,
    ) -> Path | None:
        """Create subset font containing only characters used in texts.

        Args:
            font_path: Path to the base font file (TTF or TTC).
            texts: List of texts to extract characters from.
            font_number: Font index for TTC files (default: 0).
            is_bold: Use Bold variant if available.
            is_italic: Use Italic variant if available (fallback to base if not).

        Returns:
            Path to the subset font file, or None if subsetting failed.
        """
        from fontTools.ttLib import TTFont
        from fontTools.subset import Options, Subsetter

        # Collect unique characters
        chars = set()
        for text in texts:
            if text:
                chars.update(text)

        if not chars:
            return None

        # Add safety margin characters
        if self._config.include_common_punctuation:
            chars.update(SAFETY_MARGIN_CHARS)

        # Resolve font variant
        actual_font_path = font_path
        if is_bold or is_italic:
            variant_path = _find_font_variant(font_path, is_bold, is_italic)
            if variant_path:
                actual_font_path = variant_path
                logger.debug(
                    "Using font variant: %s (bold=%s, italic=%s)",
                    variant_path.name, is_bold, is_italic
                )
            else:
                logger.warning(
                    "Font variant not found for bold=%s, italic=%s. Using %s",
                    is_bold, is_italic, font_path.name
                )

        # Check cache
        cache_key = self._get_cache_key(actual_font_path, chars, font_number)
        if cache_key in self._cache:
            cached_path = self._cache[cache_key]
            if cached_path.exists():
                return cached_path

        try:
            # Load font
            if actual_font_path.suffix.lower() == ".ttc":
                font = TTFont(actual_font_path, fontNumber=font_number)
            else:
                font = TTFont(actual_font_path)

            # Create subset
            options = Options()
            options.layout_features = ["*"]
            options.name_IDs = ["*"]
            options.notdef_glyph = True
            options.notdef_outline = True

            subsetter = Subsetter(options=options)
            subsetter.populate(text="".join(chars))
            subsetter.subset(font)

            # Save to cache directory
            subset_path = self._get_subset_path(cache_key)
            font.save(subset_path)
            font.close()

            self._cache[cache_key] = subset_path
            logger.info(
                "Created subset font: %d chars, %s (bold=%s)",
                len(chars),
                subset_path.name,
                is_bold,
            )

            return subset_path

        except Exception as e:
            logger.warning("Font subsetting failed: %s", e)
            return None

    def _get_cache_key(
        self,
        font_path: Path,
        chars: set[str],
        font_number: int,
    ) -> str:
        """Generate cache key for subset.

        Uses full path, size, and mtime to avoid collisions
        with same-named fonts in different directories.
        """
        stat = font_path.stat()
        font_identity = f"{font_path.resolve()}:{stat.st_size}:{stat.st_mtime}"
        chars_str = "".join(sorted(chars))
        full_key = f"{font_identity}:{font_number}:{chars_str}"
        return hashlib.sha256(full_key.encode()).hexdigest()[:16]

    def _get_subset_path(self, cache_key: str) -> Path:
        """Get path for subset file.

        Uses mkstemp for safety instead of mktemp.
        """
        if self._config.cache_dir:
            self._config.cache_dir.mkdir(parents=True, exist_ok=True)
            return self._config.cache_dir / f"{cache_key}.ttf"
        else:
            import tempfile as tmp
            fd, path = tmp.mkstemp(suffix=f"_{cache_key}.ttf")
            os.close(fd)  # Close fd, we'll write via fontTools
            return Path(path)

    def cleanup(self) -> None:
        """Clean up cached subset files."""
        for path in self._cache.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._cache.clear()
```

### Phase 2: PipelineConfig の拡張

**ファイル**: `src/pdf_translator/pipeline/translation_pipeline.py`

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...

    # Font optimization
    optimize_fonts: bool = True  # デフォルト ON
    font_subset_cache_dir: Path | None = None

    # TTC font settings
    cjk_font_number: int = 0  # TTC 内のフォント番号
```

### Phase 3: パイプライン統合

**ファイル**: `src/pdf_translator/pipeline/translation_pipeline.py`

```python
class TranslationPipeline:
    def __init__(self, ...):
        # ...
        self._font_subsetter = FontSubsetter(
            SubsetConfig(cache_dir=config.font_subset_cache_dir)
            if config else None
        )

    def _stage_apply(self, pdf_path: Path, paragraphs: list[Paragraph]) -> bytes:
        # ...

        # Font subsetting: スタイル別にサブセット生成
        if self._config.optimize_fonts and font_path:
            # 1. スタイル別に段落をグループ化
            regular_texts: list[str] = []
            bold_texts: list[str] = []

            for para in paragraphs:
                if para.translated_text:
                    if para.is_bold:
                        bold_texts.append(para.translated_text)
                    else:
                        regular_texts.append(para.translated_text)

            # 2. Regular サブセット生成
            regular_subset = None
            if regular_texts:
                regular_subset = self._font_subsetter.subset_for_texts(
                    font_path,
                    regular_texts,
                    font_number=self._config.cjk_font_number,
                    is_bold=False,
                )

            # 3. Bold サブセット生成
            bold_subset = None
            if bold_texts:
                bold_subset = self._font_subsetter.subset_for_texts(
                    font_path,
                    bold_texts,
                    font_number=self._config.cjk_font_number,
                    is_bold=True,
                )

        # 4. 各段落に適切なサブセットを適用
        for para in paragraphs:
            if para.translated_text:
                subset_path = bold_subset if para.is_bold else regular_subset
                # Use subset_path (or original font_path if None) for text insertion
                # ...
```

**実装のポイント**:

1. **スタイル別文字収集**: Regular と Bold で使用される文字を別々に収集
2. **独立したサブセット生成**: `is_bold=True/False` で適切なフォントファイルからサブセット
3. **段落ごとの適用**: 各段落の `is_bold` フラグに基づいて適切なサブセットを使用
4. **キャッシュ効率**: Regular/Bold は別ファイルのため自動的に別キャッシュエントリ
5. **Italic 処理**: NotoSansCJK は Italic ファイルを持たないため、Italic 段落は
   Regular/Bold と同じサブセットを使用し、`insert_laid_out_text()` 内で
   skew transform を適用して斜体を表現する (セクション 5 参照)

### Phase 4: pyproject.toml の更新

```toml
dependencies = [
    "pypdfium2>=4.30.0",
    "pdftext>=0.6.0",
    "spacy>=3.7.0",
    "deep-translator>=1.11.0",
    "numpy>=1.24.0",
    "fonttools>=4.50.0",  # NEW
]
```

### Phase 5: テスト

**新規ファイル**: `tests/test_font_subsetter.py`

テストケース:

**基本機能**:
- TTC からのサブセット生成
- TTF からのサブセット生成
- キャッシュ動作
- 空テキストリストの処理
- 無効なフォントパスの処理
- cleanup() メソッドの動作

**Bold/Italic 対応**:
- `test_bold_variant_selection`: Bold 指定時に Bold ファイルを使用
- `test_bold_fallback_to_regular`: Bold ファイルが存在しない場合 Regular にフォールバック
- `test_italic_fallback`: Italic 指定時のフォールバック動作
- `test_bold_separate_cache`: Regular と Bold は別キャッシュエントリ
- `test_find_font_variant_pattern`: ウェイトパターンの正規表現テスト
- `test_italic_skew_transform`: Italic フォントがない場合の skew transform 適用

**フォント fixture 方針**:

| 環境 | フォント | 対応 |
|------|---------|------|
| ローカル (Linux) | `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` | 存在すれば使用 |
| ローカル (macOS) | `/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc` | 存在すれば使用 |
| CI (GitHub Actions) | システムフォントなし | **skip** |

```python
import pytest
from pathlib import Path

# System font paths for testing
NOTO_CJK_LINUX = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
HIRAGINO_MACOS = Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")

def get_test_font() -> Path | None:
    """Get available CJK font for testing."""
    for path in [NOTO_CJK_LINUX, HIRAGINO_MACOS]:
        if path.exists():
            return path
    return None

# Skip decorator for font-dependent tests
requires_cjk_font = pytest.mark.skipif(
    get_test_font() is None,
    reason="No CJK font available for testing"
)
```

**CI での対応**:
- フォント依存テストは `@requires_cjk_font` で skip
- フォント非依存のロジックテスト（空リスト処理、無効パス等）は常に実行
- 将来的にはテスト用の小さな TTF フォントを `tests/fixtures/` に追加検討

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/pdf_translator/core/font_subsetter.py` | **新規** - FontSubsetter クラス, `_find_font_variant()` |
| `src/pdf_translator/pipeline/translation_pipeline.py` | PipelineConfig 拡張 + 統合 |
| `tests/test_font_subsetter.py` | **新規** - ユニットテスト (Bold/Italic 含む) |
| `pyproject.toml` | fonttools 依存追加 |

---

## リスクと対策

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|-------|------|
| TTC 抽出失敗 | 低 | 高 | `fontNumber` 明示指定 + フォールバック |
| CID 互換性問題 | 低 | 高 | 検証済み (Step 2 PASS) |
| パフォーマンス劣化 | 中 | 低 | キャッシュ機構 |
| 欠落グリフ | 低 | 中 | 安全マージン文字追加 |
| Bold ファイル欠損 | 低 | 中 | Regular へのフォールバック + 警告ログ |
| Italic ファイル欠損 | 低 | 低 | PDF skew transform で斜体表現 (新規実装) |

**フォールバック戦略**:
- サブセット化失敗時は元のフォントを使用
- Bold/Italic バリアント未検出時は Regular を使用
- 警告ログを出力

---

## 残課題

### side-by-side XObject 効率化

フォントサブセット化実装後も side-by-side PDF が巨大な場合:

1. `page_as_xobject()` によるリソース重複の可能性を調査
2. 必要に応じて二次最適化を検討

**優先度**: サブセット化実装後に効果測定してから判断

---

## 参考リンク

- [pypdfium2 Documentation](https://pypdfium2.readthedocs.io/en/v4/)
- [PDFium fpdf_edit.h](https://pdfium.googlesource.com/pdfium/+/refs/heads/main/public/fpdf_edit.h)
- [fonttools subset](https://fonttools.readthedocs.io/en/stable/subset/)
- [Apache Software Foundation 3rd Party License Policy](https://www.apache.org/legal/resolved.html)
