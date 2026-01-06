# Examples

pdf-translator の使用例です。

## translate_pdf.py

PDFを翻訳するサンプルスクリプトです。

### 使い方

```bash
cd examples
python translate_pdf.py
```

### 設定変数

スクリプト先頭の設定変数を変更して動作をカスタマイズできます。

| 変数 | デフォルト | 説明 |
|-----|-----------|------|
| `TRANSLATOR` | `"google"` | 翻訳サービス: `"google"`, `"openai"`, `"deepl"` |
| `SOURCE_LANG` | `"en"` | 原文の言語 |
| `TARGET_LANG` | `"ja"` | 翻訳先の言語 |
| `DEBUG_DRAW_BBOX` | `False` | レイアウト解析結果のbbox描画 |
| `SIDE_BY_SIDE` | `True` | 見開きPDF生成（左右に原文と翻訳を並べる） |
| `SIDE_BY_SIDE_ORDER` | `"original_translated"` | 見開きの配置順序 |
| `SIDE_BY_SIDE_GAP` | `10.0` | 見開きの間隔（ポイント） |
| `STRICT_MODE` | `False` | 翻訳失敗時にエラーを発生させるか |

### 翻訳サービス

#### Google Translate（デフォルト）

APIキー不要で無料で利用できます。レート制限があります。

```python
TRANSLATOR = "google"
```

#### OpenAI

環境変数 `OPENAI_API_KEY` が必要です。

```bash
export OPENAI_API_KEY='your-api-key'
```

```python
TRANSLATOR = "openai"
```

モデルを指定する場合は `OPENAI_MODEL` 環境変数を設定：

```bash
export OPENAI_MODEL='gpt-4o'
```

##### サポートされているモデルとトークン制限

各モデルに最適化されたトークン制限が自動設定されます：

| モデル | コンテキスト | テキスト制限 | バッチ制限 |
|--------|-------------|-------------|-----------|
| gpt-5-nano (デフォルト) | 400K | 100K tokens | 200K tokens |
| gpt-5 / gpt-5-mini | 400K | 100K tokens | 200K tokens |
| gpt-5-chat | 128K | 32K tokens | 64K tokens |
| gpt-4.1 / gpt-4.1-mini | 1M | 250K tokens | 500K tokens |
| gpt-4o / gpt-4o-mini | 128K | 32K tokens | 64K tokens |

未知のモデルは保守的なデフォルト（8K tokens）が適用されます。

#### DeepL

環境変数 `DEEPL_API_KEY` が必要です。

```bash
export DEEPL_API_KEY='your-api-key'
```

```python
TRANSLATOR = "deepl"
```

### デバッグ: bbox表示

レイアウト解析で検出された段落の境界ボックスをPDFに描画します。

```python
DEBUG_DRAW_BBOX = True
```

### 見開きPDF

原文と翻訳文を左右に並べて表示する見開きPDFを生成します。

```python
SIDE_BY_SIDE = True
SIDE_BY_SIDE_ORDER = "original_translated"  # 左=原文、右=翻訳
SIDE_BY_SIDE_GAP = 10.0  # 間隔（ポイント）
```

### 出力ファイル

出力は `examples/outputs/` ディレクトリに保存されます。

- `{input}_google.pdf` - 翻訳済みPDF
- `{input}_google_side_by_side.pdf` - 見開きPDF（`SIDE_BY_SIDE=True` の場合）

## 出力ディレクトリ

`outputs/` ディレクトリは `.gitignore` に含まれており、生成されたPDFはリポジトリにコミットされません。
