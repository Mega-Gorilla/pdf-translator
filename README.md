# PDF Translator

レイアウトを保持したままPDFを翻訳するツールです。Markdown、PDF、JSONを出力します。

## 機能

- **レイアウト保持翻訳**: 元のフォーマットを維持しながらPDFコンテンツを翻訳
- **複数の出力形式**: 翻訳済みPDF、Markdown、構造化JSONを生成
- **ドキュメントレイアウト解析**: ML基盤のレイアウト検出（PP-DocLayout）による正確なテキストブロック識別
- **複数の翻訳バックエンド**: Google翻訳（デフォルト）、DeepL、OpenAI GPT
- **LLM要約生成**: Gemini、OpenAI、Anthropicを使用したドキュメント要約生成
- **サムネイル生成**: PDFの1ページ目からサムネイル画像を作成
- **画像・テーブル抽出**: PDFから画像とテーブルを抽出してMarkdownに埋め込み
- **多言語JSON出力**: 効率的な多言語対応のためのベースドキュメントと翻訳ファイルの分離

## インストール

```bash
uv sync
```

### 環境設定

サンプル環境ファイルをコピーしてAPIキーを設定します：

```bash
cp .env.example .env
```

`.env`を編集してAPIキーを追加：

```bash
# 翻訳バックエンド
DEEPL_API_KEY=your-deepl-api-key      # --backend deepl 用
OPENAI_API_KEY=your-openai-api-key    # --backend openai 用

# LLM機能 (--llm-summary, --llm-fallback)
GEMINI_API_KEY=your-gemini-api-key    # デフォルトLLMプロバイダー
ANTHROPIC_API_KEY=your-anthropic-key  # --llm-provider anthropic 用
```

### GPUアクセラレーション（推奨）

PDF TranslatorはレイアウトンのためにPP-DocLayoutを使用しており、GPUアクセラレーションにより大幅に高速化されます。GPUを使用すると、レイアウト解析が**4倍高速**になります（ベンチマーク: 9.5秒 → 2.3秒）。

> **重要**: `uv sync`はデフォルトでCPU版のPaddlePaddleをインストールします。GPUアクセラレーションを使用するには、以下の手順でGPU版に置き換える必要があります。この置き換えは`uv sync`を実行するたびに必要です。

**CUDAバージョンの確認：**

```bash
nvcc --version
# または
nvidia-smi
```

**PaddlePaddle GPU版のインストール：**

```bash
# まずCPU版をアンインストール
uv pip uninstall paddlepaddle

# pip をインストール（GPUパッケージのインストールに必要）
uv pip install pip

# CUDAバージョンに合わせてGPU版をインストール：

# CUDA 11.8
uv run pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CUDA 12.3 / 12.6
uv run pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

> **注意**: 依存関係エラー（例: `nvidia-cuda-cccl-cu12`）が発生した場合は、まずPyPIから不足パッケージをインストールしてください：
> ```bash
> uv run pip install nvidia-cuda-cccl-cu12==12.3.52
> ```

その他のCUDAバージョンについては、[PaddlePaddle インストールガイド](https://www.paddlepaddle.org.cn/install/quick)を参照してください。

**GPUが有効か確認：**

```bash
uv run python -c "import paddle; print('CUDA:', paddle.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count())"
```

GPU有効時の期待される出力：
```
CUDA: True
GPU count: 1
```

## 使い方

### 基本的な翻訳

```bash
# 基本的な翻訳（Google翻訳、英語 → 日本語）
uv run translate-pdf paper.pdf

# 出力ファイルを指定
uv run translate-pdf paper.pdf -o ./output/paper.ja.pdf

# 別のバックエンドを使用
uv run translate-pdf paper.pdf --backend deepl
uv run translate-pdf paper.pdf --backend openai

# 言語を指定
uv run translate-pdf paper.pdf -s en -t zh  # 英語から中国語
```

### ドキュメント要約とサムネイル

ドキュメントメタデータ、サムネイル、LLM要約を生成：

```bash
# 1ページ目からサムネイルを生成
uv run translate-pdf paper.pdf --thumbnail

# LLMベースのドキュメント要約を生成（APIキーが必要）
uv run translate-pdf paper.pdf --llm-summary

# レイアウト解析が失敗した場合のLLMフォールバックを有効化
uv run translate-pdf paper.pdf --llm-fallback

# すべての要約機能を組み合わせ
uv run translate-pdf paper.pdf --thumbnail --llm-summary --llm-fallback

# LLMプロバイダーとモデルを指定
uv run translate-pdf paper.pdf --llm-summary --llm-provider openai --llm-model gpt-4o
```

**対応LLMプロバイダー**: `gemini`（デフォルト）、`openai`、`anthropic`

### Markdown出力

```bash
# PDFと一緒にMarkdownを生成
uv run translate-pdf paper.pdf --markdown

# 原文と翻訳を並べて表示（parallelモード）
uv run translate-pdf paper.pdf -m --markdown-mode parallel

# すべてのカテゴリをMarkdownに含める（ヘッダー、フッターなど）
uv run translate-pdf paper.pdf -m --markdown-include-all

# YAMLフロントマターを無効化
uv run translate-pdf paper.pdf -m --markdown-no-metadata
```

#### Markdown出力モード

| モード | 原文 | 翻訳 | 用途 |
|-------|------|------|------|
| `translated_only` | フォールバックのみ | あり | 翻訳コンテンツの読解（デフォルト） |
| `original_only` | あり | なし | ソースの構造化Markdown |
| `parallel` | あり | あり | 比較、レビュー、学習 |

**`translated_only`**（デフォルト）: 翻訳テキストのみを出力。翻訳がない場合は原文にフォールバック。

```markdown
## Abstract
AutoGenは、開発者が相互に対話してタスクを実行できる複数のエージェントを介して
LLMアプリケーションを構築できるオープンソースフレームワークです。
```

**`parallel`**: 各段落の原文と翻訳を両方出力。比較や品質レビューに最適。

```markdown
## Abstract
AutoGen is an open-source framework that allows developers to build LLM applications
via multiple agents that can converse with each other to accomplish tasks.

AutoGenは、開発者が相互に対話してタスクを実行できる複数のエージェントを介して
LLMアプリケーションを構築できるオープンソースフレームワークです。
```

### 画像・テーブル抽出

`--markdown`を使用すると、画像とテーブルが自動的に抽出されます：

```bash
# 画像とテーブルを抽出（--markdownでデフォルト有効）
uv run translate-pdf paper.pdf --markdown

# 画像抽出を無効化
uv run translate-pdf paper.pdf --markdown --no-extract-images

# テーブル抽出を無効化
uv run translate-pdf paper.pdf --markdown --no-extract-tables

# 画像出力をカスタマイズ
uv run translate-pdf paper.pdf -m --image-format jpeg --image-quality 90 --image-dpi 200
```

### 翻訳カテゴリ制御

デフォルトでは、本文カテゴリ（`text`、`abstract`など）のみが翻訳されます。
タイトル、数式、図はオリジナル言語のまま保持されます。

```bash
# すべてのカテゴリを翻訳（タイトル、数式を含む）
uv run translate-pdf paper.pdf --translate-all

# 翻訳するカスタムカテゴリを指定
uv run translate-pdf paper.pdf --translate-categories "text,abstract,doc_title"
```

### JSON出力（中間ファイル）

デバッグ、再生成、またはWebサービス統合のための構造化JSONファイルを保存：

```bash
# 中間JSONファイルを保存
uv run translate-pdf paper.pdf --save-intermediate

# すべてのメタデータを含む完全な例
uv run translate-pdf paper.pdf -t ja --save-intermediate --thumbnail --llm-summary --markdown
```

これにより2つのJSONファイルが生成されます：
- `paper.json` - ベースドキュメント（原文コンテンツ、メタデータ、原文言語の要約）
- `paper.ja.json` - 翻訳ドキュメント（翻訳コンテンツ、翻訳言語の要約）

### 高度なオプション

```bash
# デバッグモード（バウンディングボックスを描画）
uv run translate-pdf paper.pdf --debug

# 見開き比較PDF
uv run translate-pdf paper.pdf --side-by-side

# 詳細出力
uv run translate-pdf paper.pdf -v
```

すべてのオプションは`uv run translate-pdf --help`を参照してください。

## 出力ファイル

すべての機能を有効にして実行した場合：

```bash
uv run translate-pdf paper.pdf -t ja --markdown --thumbnail --llm-summary --save-intermediate
```

以下のファイルが生成されます：

```
output/
├── paper.ja.pdf              # 翻訳済みPDF
├── paper.ja.md               # 翻訳済みMarkdown
├── paper.md                  # 原文Markdown
├── paper.json                # ベースドキュメント（スキーマ v2.0.0）
├── paper.ja.json             # 翻訳ドキュメント
├── paper_thumbnail.png       # サムネイル画像
└── images/                   # 抽出された画像（存在する場合）
    ├── paper_p0_img0.png
    └── ...
```

### JSONスキーマ（v2.0.0）

**ベースドキュメント（`paper.json`）**：
```json
{
  "schema_version": "2.0.0",
  "metadata": {
    "source_file": "paper.pdf",
    "source_lang": "en",
    "page_count": 10,
    "paragraph_count": 145
  },
  "summary": {
    "title": "ドキュメントタイトル",
    "abstract": "要約テキスト...",
    "summary": "LLMで生成された原文言語の要約..."
  },
  "paragraphs": [...]
}
```

**翻訳ドキュメント（`paper.ja.json`）**：
```json
{
  "schema_version": "2.0.0",
  "target_lang": "ja",
  "base_file": "paper.json",
  "translated_at": "2026-01-15T12:00:00",
  "translator_backend": "google",
  "translated_count": 120,
  "summary": {
    "title": "翻訳されたタイトル",
    "abstract": "翻訳された要約...",
    "summary": "LLMで生成された翻訳言語の要約..."
  },
  "paragraphs": {
    "para_p0_b1": "翻訳文1",
    "para_p0_b2": "翻訳文2"
  }
}
```

## サンプル

`examples/`ディレクトリにサンプル出力があります：

- `examples/summary_output/` - サムネイル、JSON、Markdown出力を含む完全な例

## ライセンス

Apache-2.0
