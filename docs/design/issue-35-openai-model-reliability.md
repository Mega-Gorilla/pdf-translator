# Issue #35: OpenAI Structured Outputs の信頼性問題への対応

## 概要

OpenAI の GPT-4o-mini モデルを使用した翻訳時に、Structured Outputs が要求した配列長と異なる長さの配列を返すことがあり、翻訳パイプラインが失敗する問題への対応。

## 問題の詳細

### 現象

```
TranslationError: OpenAI returned 9 translations for 10 texts
```

- 10テキストを要求したのに9翻訳しか返されない
- 同じ入力でリトライしても同じ結果になる
- 5回リトライ後に `PipelineError: Translation failed after 5 retries` で失敗

### 再現手順

1. `sample_autogen_paper.pdf` (43ページ、682段落) を OpenAI 翻訳で実行
2. レイアウト解析を有効化
3. バッチサイズ 10 で翻訳

### 根本原因

GPT-4o-mini の Structured Outputs には既知の信頼性問題がある。

OpenAI コミュニティフォーラムで同様の問題が複数報告されている：
- [Structured Outputs not reliable with GPT-4o-mini and GPT-4o](https://community.openai.com/t/structured-outputs-not-reliable-with-gpt-4o-mini-and-gpt-4o/918735)
- [Structured Output Issue in GPT-4o API – Response Truncation](https://community.openai.com/t/structured-output-issue-in-gpt-4o-api-response-truncation-at-specific-index/1146715)

---

## 調査結果

### 最新モデル価格比較（2025年12月時点）

| モデル | Input (per 1M) | Output (per 1M) | コンテキスト | Structured Outputs | リリース日 |
|--------|----------------|-----------------|-------------|-------------------|-----------|
| **GPT-5-nano** | $0.05 | $0.40 | 128K | ✅ | 2025-08 |
| GPT-5-mini | $0.25 | $2.00 | 256K | ✅ | 2025-08 |
| GPT-5 | $1.25 | $10.00 | 272K | ✅ | 2025-08 |
| GPT-5.2 | $1.75 | $14.00 | 400K | ✅ | 2025-12 |
| GPT-4.1-mini | $0.40 | $1.60 | 1M | ✅ | 2025-04 |
| GPT-4o-mini | $0.15 | $0.60 | 128K | ⚠️ 不安定 | 2024-07 |
| GPT-4o | $2.50 | $10.00 | 128K | ✅ 100% | 2024-05 |

### コスト比較（43ページPDF翻訳想定：入力500K tokens、出力300K tokens）

| モデル | 推定コスト | 現行比 |
|--------|-----------|--------|
| GPT-4o-mini (現行) | $0.26 | 1.0x |
| **GPT-5-nano** | **$0.15** | **0.6x (40%安い)** |
| GPT-5-mini | $0.73 | 2.8x |
| GPT-4.1-mini | $0.68 | 2.6x |

### 推奨モデル

**GPT-5-nano** を推奨：
1. **最も安価**: GPT-4o-mini より40%安い
2. **新しいモデル**: 2025年8月リリース、Structured Outputs の信頼性が改善
3. **翻訳に十分**: 単純な抽出・変換タスクに最適化

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/pdf_translator/translators/base.py` | `ArrayLengthMismatchError` 追加 |
| `src/pdf_translator/translators/openai.py` | デフォルトモデル変更 + 環境変数サポート |
| `src/pdf_translator/pipeline/translation_pipeline.py` | バッチ分割フォールバック実装 |
| `tests/test_openai_translator.py` | 新規テスト追加 |

---

## 実装詳細

### 1. `ArrayLengthMismatchError` の追加

**ファイル**: `src/pdf_translator/translators/base.py`

```python
class ArrayLengthMismatchError(TranslationError):
    """Raised when API returns wrong number of translations.

    This error occurs with some models (e.g., GPT-4o-mini) due to
    Structured Outputs reliability issues.

    This error should NOT be retried with the same input.
    Instead, the batch should be split into smaller chunks.
    """

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"Expected {expected} translations but got {actual}")
        self.expected = expected
        self.actual = actual
```

### 2. OpenAI Translator の変更

**ファイル**: `src/pdf_translator/translators/openai.py`

#### 2.1 デフォルトモデルの変更

```python
class OpenAITranslator:
    DEFAULT_MODEL = "gpt-5-nano"  # Changed from "gpt-4o-mini"
```

#### 2.2 環境変数サポートの追加

```python
def __init__(
    self,
    api_key: str,
    model: str | None = None,
    system_prompt: str | None = None,
) -> None:
    # ...
    import os
    env_model = os.environ.get("OPENAI_MODEL")
    self._model = model or env_model or self.DEFAULT_MODEL
```

#### 2.3 ArrayLengthMismatchError の使用

```python
from pdf_translator.translators.base import (
    ArrayLengthMismatchError,
    ConfigurationError,
    TranslationError,
)

# _translate_with_structured_output 内
if len(translations) != len(texts):
    raise ArrayLengthMismatchError(
        expected=len(texts),
        actual=len(translations),
    )
```

### 3. バッチ分割フォールバックの実装

**ファイル**: `src/pdf_translator/pipeline/translation_pipeline.py`

#### 3.1 import の追加

```python
from pdf_translator.translators.base import (
    ArrayLengthMismatchError,
    ConfigurationError,
    TranslationError,
)
```

#### 3.2 `_translate_with_retry` の変更

```python
async def _translate_with_retry(self, texts: list[str]) -> list[str]:
    """Translate texts with retry and fallback logic."""
    last_error: TranslationError | None = None

    for attempt in range(self._config.max_retries + 1):
        try:
            return await self._translator.translate_batch(
                texts,
                self._config.source_lang,
                self._config.target_lang,
            )
        except ConfigurationError:
            raise
        except ArrayLengthMismatchError:
            # バッチ分割フォールバック（リトライしない）
            return await self._translate_with_split(texts)
        except TranslationError as exc:
            last_error = exc
            if attempt < self._config.max_retries:
                delay = self._config.retry_delay * (2**attempt)
                await asyncio.sleep(delay)

    raise PipelineError(
        f"Translation failed after {self._config.max_retries} retries",
        stage="translate",
        cause=last_error,
    )
```

#### 3.3 `_translate_with_split` の追加

```python
async def _translate_with_split(self, texts: list[str]) -> list[str]:
    """Translate texts by splitting batch recursively.

    When ArrayLengthMismatchError occurs, split the batch in half
    and retry. If batch size is 1, fall back to individual translation.
    """
    if len(texts) == 0:
        return []

    if len(texts) == 1:
        # 個別翻訳にフォールバック
        try:
            result = await self._translator.translate(
                texts[0],
                self._config.source_lang,
                self._config.target_lang,
            )
            return [result]
        except TranslationError:
            # 個別翻訳も失敗した場合は元のテキストを返す（ログ出力）
            return [texts[0]]

    # バッチを半分に分割
    mid = len(texts) // 2
    left = await self._translate_with_retry(texts[:mid])
    right = await self._translate_with_retry(texts[mid:])
    return left + right
```

---

## モデル設定の優先順位

```
1. コンストラクタ引数 model=
2. 環境変数 OPENAI_MODEL
3. デフォルト値 gpt-5-nano
```

### 使用例

```python
# デフォルト (gpt-5-nano)
translator = OpenAITranslator(api_key=key)

# コンストラクタで指定
translator = OpenAITranslator(api_key=key, model="gpt-5-mini")

# 環境変数で指定
# export OPENAI_MODEL=gpt-5
translator = OpenAITranslator(api_key=key)  # gpt-5 が使用される
```

---

## テスト計画

### ユニットテスト

| テストケース | 説明 |
|-------------|------|
| `test_default_model_is_gpt5_nano` | デフォルトモデルが gpt-5-nano |
| `test_model_from_env_variable` | 環境変数からモデル読み込み |
| `test_model_constructor_overrides_env` | コンストラクタが環境変数より優先 |
| `test_array_length_mismatch_triggers_split` | 配列長不一致でバッチ分割 |
| `test_split_to_individual_translation` | バッチサイズ1で個別翻訳 |
| `test_split_recursive` | 再帰的なバッチ分割 |

### 統合テスト

- [ ] GPT-5-nano での翻訳テスト（sample_llama.pdf）
- [ ] sample_autogen_paper.pdf の完全翻訳テスト
- [ ] バッチ分割が正常に動作することを確認

---

## 後方互換性

| 項目 | 影響 |
|------|------|
| デフォルトモデル変更 | ⚠️ `gpt-4o-mini` → `gpt-5-nano` |
| 環境変数追加 | ✅ 既存コードに影響なし |
| 新エラー型 | ✅ 既存の `TranslationError` を継承 |

### マイグレーション

既存のコードで `gpt-4o-mini` を明示的に使用したい場合：
```python
translator = OpenAITranslator(api_key=key, model="gpt-4o-mini")
```

または環境変数で：
```bash
export OPENAI_MODEL=gpt-4o-mini
```

---

## ロールバック計画

問題が発生した場合：
1. 環境変数 `OPENAI_MODEL=gpt-4o-mini` で旧モデルに戻す
2. または git revert でコミットを取り消す

---

## 参考リンク

- [Issue #35](https://github.com/Mega-Gorilla/pdf-translator/issues/35)
- [GPT-5 nano Model | OpenAI API](https://platform.openai.com/docs/models/gpt-5-nano)
- [GPT-5 mini Model | OpenAI API](https://platform.openai.com/docs/models/gpt-5-mini)
- [Structured Outputs not reliable with GPT-4o-mini](https://community.openai.com/t/structured-outputs-not-reliable-with-gpt-4o-mini-and-gpt-4o/918735)
