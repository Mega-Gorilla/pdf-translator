# 翻訳バックエンドモジュール設計書

## 1. 概要

本ドキュメントは Issue #3「翻訳バックエンドモジュールの移植」の実装計画を定義する。

### 1.1 目的

PDF翻訳パイプラインで使用する翻訳バックエンド（Google Translate / DeepL / OpenAI GPT）を実装する。

### 1.2 スコープ

- 翻訳バックエンドの共通インターフェース定義
- 3種類のバックエンド実装（Google / DeepL / OpenAI）
- ユニットテストおよび統合テスト

### 1.3 依存関係

| バックエンド | ライブラリ | 依存タイプ |
|-------------|-----------|-----------|
| Google | `deep-translator` | 必須 |
| DeepL | `aiohttp` | オプション (`[deepl]`) |
| OpenAI | `openai` (pydantic含む) | オプション (`[openai]`) |

すべての依存関係は `pyproject.toml` に設定済み。

---

## 2. アーキテクチャ

### 2.1 ファイル構成

```
src/pdf_translator/translators/
├── __init__.py      # エクスポート、ファクトリ関数
├── base.py          # Protocol、例外クラス
├── google.py        # Google Translate バックエンド
├── deepl.py         # DeepL バックエンド
└── openai.py        # OpenAI GPT バックエンド

tests/
└── test_translators.py  # テスト
```

### 2.2 クラス図

```
┌─────────────────────────────────────┐
│     TranslatorBackend (Protocol)    │
├─────────────────────────────────────┤
│ + name: str                         │
│ + translate(text, target_lang): str │
└─────────────────────────────────────┘
              △
              │ implements
    ┌─────────┼─────────┐
    │         │         │
┌───┴───┐ ┌───┴───┐ ┌───┴───┐
│Google │ │ DeepL │ │OpenAI │
│Trans. │ │Trans. │ │Trans. │
└───────┘ └───────┘ └───────┘
```

---

## 3. インターフェース定義

### 3.1 TranslatorBackend Protocol

```python
@runtime_checkable
class TranslatorBackend(Protocol):
    """翻訳バックエンドのプロトコル定義"""

    @property
    def name(self) -> str:
        """バックエンド名 ("google", "deepl", "openai")"""
        ...

    async def translate(self, text: str, target_lang: str) -> str:
        """
        テキストを翻訳する。

        Args:
            text: 翻訳対象テキスト（改行や[[[BR]]]を含む場合あり）
            target_lang: 対象言語コード ("en", "ja")

        Returns:
            翻訳されたテキスト

        Raises:
            TranslationError: 翻訳失敗時
        """
        ...
```

### 3.2 TranslationError

```python
class TranslationError(Exception):
    """翻訳処理中のエラー"""
    pass
```

---

## 4. バックエンド実装

### 4.1 GoogleTranslator

**特徴:**
- APIキー不要（デフォルトバックエンド）
- `deep-translator` ライブラリを使用
- 言語コードはそのまま使用 (`"ja"`, `"en"`)

**実装ポイント:**
- `asyncio.to_thread()` で同期APIを非同期化
- 空文字・空白のみの場合は早期リターン
- `TranslationNotFound` を `TranslationError` に変換

```python
class GoogleTranslator:
    @property
    def name(self) -> str:
        return "google"

    async def translate(self, text: str, target_lang: str) -> str:
        if not text.strip():
            return text
        # deep-translator を asyncio.to_thread で呼び出し
        ...
```

### 4.2 DeepLTranslator

**特徴:**
- 高品質翻訳（APIキー必須）
- Free API / Pro API 両対応
- 言語コードは大文字変換 (`"ja"` → `"JA"`)

**実装ポイント:**
- `aiohttp` で非同期HTTP通信
- APIキー未設定時は `ValueError`
- オプション依存（`ImportError` でヒント表示）

```python
class DeepLTranslator:
    DEFAULT_API_URL = "https://api-free.deepl.com/v2/translate"

    def __init__(self, api_key: str, api_url: str | None = None):
        if not api_key:
            raise ValueError("DeepL API key is required")
        ...

    async def translate(self, text: str, target_lang: str) -> str:
        # aiohttp で POST リクエスト
        ...
```

### 4.3 OpenAITranslator

**特徴:**
- Structured Outputs で配列構造を保証
- カスタムプロンプト対応（3段階優先度）
- セパレータトークン `[[[BR]]]` 対応

**実装ポイント:**
- `openai` ライブラリの `beta.chat.completions.parse` 使用
- Pydantic モデルで `response_format` を定義
- 空文字列の位置を保持

```python
class OpenAITranslator:
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        source_lang: str = "en",
        system_prompt: str | None = None,
    ):
        ...

    async def translate(self, text: str, target_lang: str) -> str:
        # [[[BR]]] で分割し、translate_texts() で翻訳
        ...

    async def translate_texts(self, texts: list[str], target_lang: str) -> list[str]:
        # Structured Outputs で配列翻訳
        ...
```

**プロンプト優先度:**
1. `translate()` / `translate_texts()` の引数
2. コンストラクタの `system_prompt`
3. `DEFAULT_SYSTEM_PROMPT`

---

## 5. セパレータトークン方式

### 5.1 概要

複数のテキストブロックを一度の API 呼び出しで翻訳するため、セパレータトークン `[[[BR]]]` を使用する。

```
Block1[[[BR]]]Block2[[[BR]]]Block3
  ↓ 翻訳
翻訳1[[[BR]]]翻訳2[[[BR]]]翻訳3
```

### 5.2 各バックエンドの対応

| バックエンド | 方式 |
|-------------|------|
| Google | セパレータを含むテキストをそのまま翻訳（API側で保持） |
| DeepL | 同上 |
| OpenAI | `[[[BR]]]` で分割 → 配列翻訳 → 再結合 |

OpenAI は Structured Outputs で配列を保証するため、セパレータの消失リスクがない。

---

## 6. エラーハンドリング

### 6.1 例外の統一

すべてのバックエンドは `TranslationError` を送出する。

```python
# Google
except TranslationNotFound as e:
    raise TranslationError(f"Translation failed: {e}")
except Exception as e:
    raise TranslationError(f"Google Translate error: {e}")

# DeepL
if response.status != 200:
    raise TranslationError(f"DeepL API error (status {response.status}): ...")

# OpenAI
except Exception as e:
    raise TranslationError(f"OpenAI API error: {e}")
```

### 6.2 オプション依存のエラー

```python
# deepl.py
try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for DeepL backend. "
        "Install with: pip install pdf-translator[deepl]"
    )
```

---

## 7. エクスポート設計

### 7.1 `__init__.py`

```python
# 直接インポート可能
from pdf_translator.translators import (
    GoogleTranslator,
    TranslationError,
    TranslatorBackend,
)

# 遅延インポート（オプション依存）
from pdf_translator.translators import get_deepl_translator, get_openai_translator

DeepLTranslator = get_deepl_translator()
OpenAITranslator = get_openai_translator()
```

### 7.2 遅延インポートの理由

- DeepL / OpenAI はオプション依存
- インストールされていない場合でも `GoogleTranslator` は使用可能
- 必要時のみ `ImportError` を発生

---

## 8. テスト計画

### 8.1 ユニットテスト（モック使用）

| テスト | 内容 |
|--------|------|
| `test_name` | バックエンド名の確認 |
| `test_translate_empty_string` | 空文字の早期リターン |
| `test_translate_whitespace_only` | 空白のみの早期リターン |
| `test_translate_simple_mocked` | 基本翻訳（モック） |
| `test_translate_error_handling` | エラーハンドリング |
| `test_requires_api_key` | APIキー必須の検証 |
| `test_custom_model` | カスタム設定の検証 |

### 8.2 統合テスト（実API）

`@pytest.mark.integration` でマーク、CI ではスキップ。

| テスト | 条件 |
|--------|------|
| Google | 常に実行可能 |
| DeepL | `DEEPL_API_KEY` 環境変数が必要 |
| OpenAI | `OPENAI_API_KEY` 環境変数が必要 |

```python
@pytest.mark.integration
@pytest.mark.skipif("DEEPL_API_KEY" not in os.environ, reason="...")
class TestDeepLTranslatorIntegration:
    ...
```

---

## 9. 実装フェーズ

### Phase 1: 基盤 (`base.py`)
- [ ] `TranslationError` 例外クラス
- [ ] `TranslatorBackend` Protocol

### Phase 2: Google Translate (`google.py`)
- [ ] `GoogleTranslator` クラス
- [ ] `asyncio.to_thread()` による非同期化
- [ ] エラーハンドリング

### Phase 3: DeepL (`deepl.py`)
- [ ] `DeepLTranslator` クラス
- [ ] `aiohttp` による非同期API呼び出し
- [ ] Free/Pro API URL 対応

### Phase 4: OpenAI GPT (`openai.py`)
- [ ] `OpenAITranslator` クラス
- [ ] Structured Outputs 対応
- [ ] `translate_texts()` バッチ翻訳
- [ ] カスタムプロンプト対応

### Phase 5: エクスポート (`__init__.py`)
- [ ] 直接エクスポート設定
- [ ] 遅延インポート関数

### Phase 6: テスト (`test_translators.py`)
- [ ] ユニットテスト（モック）
- [ ] 統合テスト（実API）

---

## 10. 成功基準

| 項目 | 基準 |
|------|------|
| pytest | 全テスト通過 |
| ruff | All checks passed |
| mypy | Success: no issues found |
| Google統合テスト | 動作確認済み |

---

## 11. 注意事項

### 11.1 ライセンス

- 本プロジェクト: Apache-2.0
- 旧実装 (`_archive/`): AGPL-3.0
- **コードのコピーは不可**、設計の参考のみ

### 11.2 言語コード

| 内部形式 | Google | DeepL | OpenAI |
|----------|--------|-------|--------|
| `"en"` | `"en"` | `"EN"` | `"English"` (プロンプト内) |
| `"ja"` | `"ja"` | `"JA"` | `"Japanese"` (プロンプト内) |

---

## 12. 参照

- Issue #3: https://github.com/Mega-Gorilla/pdf-translator/issues/3
- pyproject.toml: 依存関係定義
- 旧実装（参考のみ）: `_archive/Index_PDF_Translation/src/index_pdf_translation/translators/`
