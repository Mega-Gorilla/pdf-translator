# 翻訳バックエンドモジュール設計書

## 1. 概要

本ドキュメントは Issue #3「翻訳バックエンドモジュールの移植」の実装計画を定義する。

### 1.1 目的

PDF翻訳パイプラインで使用する翻訳バックエンド（Google Translate / DeepL / OpenAI GPT）を実装する。

### 1.2 スコープ

- 翻訳バックエンドの共通インターフェース定義
- 3種類のバックエンド実装（Google / DeepL / OpenAI）
- 共通ユーティリティ（リトライ、チャンク分割）
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
┌──────────────────────────────────────────────────────┐
│           TranslatorBackend (Protocol)               │
├──────────────────────────────────────────────────────┤
│ + name: str                                          │
│ + translate(text, source_lang, target_lang): str     │
│ + translate_batch(texts, source_lang, target_lang):  │
│                                          list[str]   │
└──────────────────────────────────────────────────────┘
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

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        単一テキストを翻訳する。

        Args:
            text: 翻訳対象テキスト
            source_lang: ソース言語コード ("en", "ja", "auto")
            target_lang: 対象言語コード ("en", "ja")

        Returns:
            翻訳されたテキスト

        Raises:
            TranslationError: 翻訳失敗時
        """
        ...

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        """
        複数テキストをバッチ翻訳する。

        Args:
            texts: 翻訳対象テキストのリスト
            source_lang: ソース言語コード
            target_lang: 対象言語コード

        Returns:
            翻訳されたテキストのリスト（入力と同じ順序・長さ）

        Raises:
            TranslationError: 翻訳失敗時
        """
        ...
```

### 3.2 例外クラス階層

```python
class TranslatorError(Exception):
    """翻訳モジュールの基底例外"""
    pass

class TranslationError(TranslatorError):
    """翻訳処理中のエラー（API呼び出し失敗、レート制限等）"""
    pass

class ConfigurationError(TranslatorError):
    """設定エラー（APIキー未設定、無効なパラメータ等）"""
    pass
```

**設計理由:**
- `TranslationError`: 実行時のAPI呼び出しエラー（リトライ可能な場合あり）
- `ConfigurationError`: 設定ミス（リトライしても解決しない）
- パイプライン側で例外タイプに応じた処理が可能

---

## 4. バックエンド実装

### 4.1 GoogleTranslator

**特徴:**
- APIキー不要（デフォルトバックエンド）
- `deep-translator` ライブラリを使用
- 言語コードはそのまま使用 (`"ja"`, `"en"`, `"auto"`)

**実装ポイント:**
- `asyncio.to_thread()` で同期APIを非同期化
- 空文字・空白のみの場合は早期リターン
- `translate_batch()` は並列実行（`asyncio.gather` + Semaphore）

```python
class GoogleTranslator:
    def __init__(self, max_concurrent: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def name(self) -> str:
        return "google"

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        if not text.strip():
            return text
        async with self._semaphore:
            return await asyncio.to_thread(self._translate_sync, text, source_lang, target_lang)

    async def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        tasks = [self.translate(t, source_lang, target_lang) for t in texts]
        return await asyncio.gather(*tasks)
```

### 4.2 DeepLTranslator

**特徴:**
- 高品質翻訳（APIキー必須）
- Free API / Pro API 両対応
- **配列翻訳対応**（1リクエストで最大50テキスト）
- 言語コードは大文字変換 (`"ja"` → `"JA"`)

**実装ポイント:**
- `aiohttp.ClientSession` を再利用（コンテキストマネージャ対応）
- `translate_batch()` は複数 `text` パラメータで一括送信
- リクエストサイズ制限（128KB）を考慮したチャンク分割

```python
class DeepLTranslator:
    DEFAULT_API_URL = "https://api-free.deepl.com/v2/translate"
    MAX_TEXTS_PER_REQUEST = 50
    MAX_REQUEST_SIZE = 128 * 1024  # 128KB

    def __init__(
        self,
        api_key: str,
        api_url: str | None = None,
    ):
        if not api_key:
            raise ConfigurationError("DeepL API key is required")
        self._api_key = api_key
        self._api_url = api_url or self.DEFAULT_API_URL
        self._session: aiohttp.ClientSession | None = None

    async def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        # 複数textパラメータで一括送信
        params = [("text", t) for t in texts]
        params.extend([
            ("auth_key", self._api_key),
            ("source_lang", source_lang.upper()),
            ("target_lang", target_lang.upper()),
        ])
        # aiohttp POST...
```

### 4.3 OpenAITranslator

**特徴:**
- Structured Outputs で配列構造を保証
- カスタムプロンプト対応（3段階優先度）
- バッチ翻訳がネイティブ（配列で送受信）

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
        system_prompt: str | None = None,
    ):
        if not api_key:
            raise ConfigurationError("OpenAI API key is required")
        ...

    async def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        # Structured Outputs で配列翻訳
        ...
```

**プロンプト優先度:**
1. `translate()` / `translate_batch()` の引数 `system_prompt`
2. コンストラクタの `system_prompt`
3. `DEFAULT_SYSTEM_PROMPT`

---

## 5. バッチ翻訳方式（セパレータトークン廃止）

### 5.1 設計変更の理由

従来のセパレータトークン方式 (`[[[BR]]]`) は以下のリスクがある：
- Google/DeepL でセパレータが変形・欠落する可能性
- エスケープ処理が複雑化

**新方式: 配列ベースのバッチ翻訳**

```
# 旧方式（廃止）
"Block1[[[BR]]]Block2[[[BR]]]Block3" → 翻訳 → 分割

# 新方式（採用）
["Block1", "Block2", "Block3"] → translate_batch() → ["翻訳1", "翻訳2", "翻訳3"]
```

### 5.2 各バックエンドの実装

| バックエンド | バッチ翻訳方式 |
|-------------|---------------|
| Google | `asyncio.gather` + Semaphore で並列実行 |
| DeepL | 複数 `text` パラメータで1リクエスト（最大50件） |
| OpenAI | Structured Outputs で配列送受信 |

### 5.3 互換性

パイプライン（#4）からは `translate_batch()` を呼び出すため、
セパレータトークンの知識は不要。

---

## 6. 共通機能

### 6.1 リトライ機構

```python
async def with_retry(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_errors: tuple[type[Exception], ...] = (TranslationError,),
    **kwargs,
) -> T:
    """指数バックオフでリトライ"""
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_errors as e:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)
```

### 6.2 文字数制限とチャンク分割

| バックエンド | 制限 | 対応 |
|-------------|------|------|
| Google | なし（実質5000文字程度推奨） | 長文は分割 |
| DeepL | 128KB/リクエスト, 50テキスト/リクエスト | 自動チャンク分割 |
| OpenAI | モデル依存（128K tokens等） | 長文は分割 |

```python
def chunk_texts(
    texts: list[str],
    max_texts: int = 50,
    max_total_chars: int = 100000,
) -> list[list[str]]:
    """テキストリストをチャンクに分割"""
    ...
```

### 6.3 aiohttp セッション再利用

```python
class DeepLTranslator:
    async def __aenter__(self) -> "DeepLTranslator":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
```

---

## 7. エラーハンドリング

### 7.1 例外の使い分け

| 状況 | 例外 |
|------|------|
| APIキー未設定 | `ConfigurationError` |
| 無効なパラメータ | `ConfigurationError` |
| API呼び出し失敗（ネットワーク） | `TranslationError` |
| レート制限（429） | `TranslationError`（リトライ可能） |
| サーバーエラー（5xx） | `TranslationError`（リトライ可能） |

### 7.2 実装例

```python
# deepl.py
if response.status == 429:
    raise TranslationError("Rate limit exceeded, please retry later")
elif response.status >= 500:
    raise TranslationError(f"DeepL server error (status {response.status})")
elif response.status == 403:
    raise ConfigurationError("Invalid API key")
```

### 7.3 オプション依存のエラー

```python
# deepl.py（モジュールレベルではなく、クラス内で遅延import）
class DeepLTranslator:
    def __init__(self, ...):
        try:
            import aiohttp
            self._aiohttp = aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for DeepL backend. "
                "Install with: pip install pdf-translator[deepl]"
            )
```

---

## 8. エクスポート設計

### 8.1 `__init__.py`

```python
# 常に利用可能
from pdf_translator.translators.base import (
    TranslatorBackend,
    TranslatorError,
    TranslationError,
    ConfigurationError,
)
from pdf_translator.translators.google import GoogleTranslator

__all__ = [
    "TranslatorBackend",
    "TranslatorError",
    "TranslationError",
    "ConfigurationError",
    "GoogleTranslator",
    "get_deepl_translator",
    "get_openai_translator",
]

def get_deepl_translator() -> type:
    """DeepLTranslator クラスを遅延インポートして返す"""
    from pdf_translator.translators.deepl import DeepLTranslator
    return DeepLTranslator

def get_openai_translator() -> type:
    """OpenAITranslator クラスを遅延インポートして返す"""
    from pdf_translator.translators.openai import OpenAITranslator
    return OpenAITranslator
```

### 8.2 使用例

```python
# Google（常に利用可能）
from pdf_translator.translators import GoogleTranslator
translator = GoogleTranslator()

# DeepL（オプション依存）
from pdf_translator.translators import get_deepl_translator
DeepLTranslator = get_deepl_translator()  # ここでimport
translator = DeepLTranslator(api_key="...")

# 型ヒント用（TYPE_CHECKING）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pdf_translator.translators.deepl import DeepLTranslator
```

---

## 9. テスト計画

### 9.1 ユニットテスト（モック使用）

| テスト | 内容 |
|--------|------|
| `test_name` | バックエンド名の確認 |
| `test_translate_empty_string` | 空文字の早期リターン |
| `test_translate_whitespace_only` | 空白のみの早期リターン |
| `test_translate_batch_mocked` | バッチ翻訳（モック） |
| `test_translate_error_handling` | エラーハンドリング |
| `test_requires_api_key` | APIキー必須の検証（ConfigurationError） |
| `test_retry_on_error` | リトライ動作の検証 |

### 9.2 統合テスト（実API）

**CI方針:** 統合テストはCIでスキップ、ローカルでopt-in実行。

```python
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Integration tests disabled (set RUN_INTEGRATION=1)"
)
class TestGoogleTranslatorIntegration:
    ...

@pytest.mark.integration
@pytest.mark.skipif(
    "DEEPL_API_KEY" not in os.environ,
    reason="DEEPL_API_KEY not set"
)
class TestDeepLTranslatorIntegration:
    ...
```

| テスト | 条件 |
|--------|------|
| Google | `RUN_INTEGRATION=1` |
| DeepL | `DEEPL_API_KEY` 環境変数 |
| OpenAI | `OPENAI_API_KEY` 環境変数 |

---

## 10. 実装フェーズ

### Phase 1: 基盤 (`base.py`)
- [ ] `TranslatorError` / `TranslationError` / `ConfigurationError`
- [ ] `TranslatorBackend` Protocol（source_lang対応）

### Phase 2: Google Translate (`google.py`)
- [ ] `GoogleTranslator` クラス
- [ ] `translate()` / `translate_batch()`
- [ ] Semaphore による並列制限

### Phase 3: DeepL (`deepl.py`)
- [ ] `DeepLTranslator` クラス
- [ ] 複数textパラメータによるバッチ翻訳
- [ ] セッション再利用（async context manager）

### Phase 4: OpenAI GPT (`openai.py`)
- [ ] `OpenAITranslator` クラス
- [ ] Structured Outputs 対応
- [ ] カスタムプロンプト対応

### Phase 5: エクスポート (`__init__.py`)
- [ ] 直接エクスポート設定
- [ ] 遅延インポート関数（関数内import）

### Phase 6: テスト (`test_translators.py`)
- [ ] ユニットテスト（モック）
- [ ] 統合テスト（opt-in）

---

## 11. 成功基準

| 項目 | 基準 |
|------|------|
| pytest | 全テスト通過 |
| ruff | All checks passed |
| mypy | Success: no issues found |
| 統合テスト | ローカルで動作確認済み |

---

## 12. 注意事項

### 12.1 ライセンス

- 本プロジェクト: Apache-2.0
- 旧実装 (`_archive/`): AGPL-3.0
- **コードのコピーは不可**、設計の参考のみ

### 12.2 言語コード

| 内部形式 | Google | DeepL | OpenAI |
|----------|--------|-------|--------|
| `"en"` | `"en"` | `"EN"` | `"English"` (プロンプト内) |
| `"ja"` | `"ja"` | `"JA"` | `"Japanese"` (プロンプト内) |
| `"auto"` | `"auto"` | 非対応（省略で自動） | プロンプトで対応 |

---

## 13. 参照

- Issue #3: https://github.com/Mega-Gorilla/pdf-translator/issues/3
- pyproject.toml: 依存関係定義
- DeepL API: https://developers.deepl.com/docs/api-reference/translate
- 旧実装（参考のみ）: `_archive/Index_PDF_Translation/src/index_pdf_translation/translators/`
