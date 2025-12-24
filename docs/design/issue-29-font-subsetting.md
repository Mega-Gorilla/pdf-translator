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

**決定: 初期実装は Regular のみ、後で拡張**

現状の `_find_font_variant()` はファイル名ベースの探索であり、
TTC 運用時は実質ベースフォントのみ。

将来的には `PipelineConfig` で明示マッピング可能に:
- `cjk_bold_font_path`
- `cjk_italic_font_path`

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
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Common punctuation and digits to include in subset
SAFETY_MARGIN_CHARS = "。、！？「」『』（）…―　0123456789"


@dataclass
class SubsetConfig:
    """Font subsetting configuration."""

    include_common_punctuation: bool = True
    cache_dir: Path | None = None  # None = use temp directory


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
    ) -> Path | None:
        """Create subset font containing only characters used in texts.

        Args:
            font_path: Path to the font file (TTF or TTC).
            texts: List of texts to extract characters from.
            font_number: Font index for TTC files (default: 0).

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

        # Check cache
        cache_key = self._get_cache_key(font_path, chars, font_number)
        if cache_key in self._cache:
            cached_path = self._cache[cache_key]
            if cached_path.exists():
                return cached_path

        try:
            # Load font
            if font_path.suffix.lower() == ".ttc":
                font = TTFont(font_path, fontNumber=font_number)
            else:
                font = TTFont(font_path)

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
                "Created subset font: %d chars, %s",
                len(chars),
                subset_path.name,
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

        Uses NamedTemporaryFile for safety instead of mktemp.
        """
        if self._config.cache_dir:
            self._config.cache_dir.mkdir(parents=True, exist_ok=True)
            return self._config.cache_dir / f"{cache_key}.ttf"
        else:
            # Use NamedTemporaryFile to avoid race conditions
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

        # Font subsetting
        actual_font_path = font_path
        if self._config.optimize_fonts and font_path:
            texts = [p.translated_text for p in paragraphs if p.translated_text]
            subset_path = self._font_subsetter.subset_for_texts(
                font_path,
                texts,
                font_number=self._config.cjk_font_number,
            )
            if subset_path:
                actual_font_path = subset_path

        # Use actual_font_path for text insertion
        # ...
```

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
- TTC からのサブセット生成
- TTF からのサブセット生成
- キャッシュ動作
- 空テキストリストの処理
- 無効なフォントパスの処理
- cleanup() メソッドの動作

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
| `src/pdf_translator/core/font_subsetter.py` | **新規** - FontSubsetter クラス |
| `src/pdf_translator/pipeline/translation_pipeline.py` | PipelineConfig 拡張 + 統合 |
| `tests/test_font_subsetter.py` | **新規** - ユニットテスト |
| `pyproject.toml` | fonttools 依存追加 |

---

## リスクと対策

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|-------|------|
| TTC 抽出失敗 | 低 | 高 | `fontNumber` 明示指定 + フォールバック |
| CID 互換性問題 | 低 | 高 | 検証済み (Step 2 PASS) |
| パフォーマンス劣化 | 中 | 低 | キャッシュ機構 |
| 欠落グリフ | 低 | 中 | 安全マージン文字追加 |

**フォールバック戦略**:
- サブセット化失敗時は元のフォントを使用
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
