# PDF Translator

PDF translation tool with layout preservation - outputs Markdown and PDF.

## Features

- **Layout-preserving translation**: Translates PDF content while maintaining original formatting
- **Multiple output formats**: Generates translated PDF, Markdown, and structured JSON
- **Document layout analysis**: Uses ML-based layout detection (PP-DocLayout) for accurate text block identification
- **Multiple translation backends**: Google Translate (default), DeepL, OpenAI GPT
- **LLM-powered summaries**: Generate document summaries using Gemini, OpenAI, or Anthropic
- **Thumbnail generation**: Create thumbnail images from PDF first pages
- **Image/Table extraction**: Extract images and tables from PDFs into Markdown
- **Multilingual JSON output**: Separate base document and translation files for efficient multi-language support

## Installation

```bash
uv sync
```

### Environment Setup

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` to add your API keys:

```bash
# Translation backends
DEEPL_API_KEY=your-deepl-api-key      # For --backend deepl
OPENAI_API_KEY=your-openai-api-key    # For --backend openai

# LLM features (--llm-summary, --llm-fallback)
GEMINI_API_KEY=your-gemini-api-key    # Default LLM provider
ANTHROPIC_API_KEY=your-anthropic-key  # For --llm-provider anthropic
```

### GPU Acceleration (Recommended)

PDF Translator uses PP-DocLayout for layout analysis, which benefits significantly from GPU acceleration. With GPU, layout analysis is **4x faster** (benchmark: 9.5s → 2.3s).

> **Important**: `uv sync` installs the CPU version of PaddlePaddle by default. To use GPU acceleration, you must replace it with the GPU version using the steps below. This replacement is required after every `uv sync`.

**Check your CUDA version:**

```bash
nvcc --version
# or
nvidia-smi
```

**Install PaddlePaddle GPU version:**

```bash
# First, uninstall CPU version
uv pip uninstall paddlepaddle

# Install pip (required for GPU package installation)
uv pip install pip

# Install GPU version for your CUDA version:

# CUDA 11.8
uv run pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CUDA 12.3 / 12.6
uv run pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

> **Note**: If you encounter dependency errors (e.g., `nvidia-cuda-cccl-cu12`), install the missing package from PyPI first:
> ```bash
> uv run pip install nvidia-cuda-cccl-cu12==12.3.52
> ```

For other CUDA versions, see [PaddlePaddle Installation Guide](https://www.paddlepaddle.org.cn/install/quick).

**Verify GPU is enabled:**

```bash
uv run python -c "import paddle; print('CUDA:', paddle.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count())"
```

Expected output with GPU:
```
CUDA: True
GPU count: 1
```

## Usage

### Basic Translation

```bash
# Basic translation (Google Translate, EN → JA)
uv run translate-pdf paper.pdf

# Specify output file
uv run translate-pdf paper.pdf -o ./output/paper.ja.pdf

# Use different backend
uv run translate-pdf paper.pdf --backend deepl
uv run translate-pdf paper.pdf --backend openai

# Specify languages
uv run translate-pdf paper.pdf -s en -t zh  # English to Chinese
```

### Document Summary & Thumbnail

Generate document metadata, thumbnails, and LLM-powered summaries:

```bash
# Generate thumbnail from first page
uv run translate-pdf paper.pdf --thumbnail

# Generate LLM-based document summary (requires API key)
uv run translate-pdf paper.pdf --llm-summary

# Use LLM fallback for metadata extraction when layout analysis fails
uv run translate-pdf paper.pdf --llm-fallback

# Combine all summary features
uv run translate-pdf paper.pdf --thumbnail --llm-summary --llm-fallback

# Specify LLM provider and model
uv run translate-pdf paper.pdf --llm-summary --llm-provider openai --llm-model gpt-4o
```

**Supported LLM providers**: `gemini` (default), `openai`, `anthropic`

### Markdown Output

```bash
# Generate Markdown alongside PDF
uv run translate-pdf paper.pdf --markdown

# Markdown with original + translation (parallel mode)
uv run translate-pdf paper.pdf -m --markdown-mode parallel

# Include all categories in Markdown (headers, footers, etc.)
uv run translate-pdf paper.pdf -m --markdown-include-all

# Disable YAML frontmatter
uv run translate-pdf paper.pdf -m --markdown-no-metadata
```

#### Markdown Output Modes

| Mode | Original | Translation | Use Case |
|------|----------|-------------|----------|
| `translated_only` | Fallback only | Yes | Read translated content (default) |
| `original_only` | Yes | No | Structured Markdown of source |
| `parallel` | Yes | Yes | Compare, review, or learn |

**`translated_only`** (default): Outputs translated text only. Falls back to original if translation is unavailable.

```markdown
## Abstract
AutoGenは、開発者が相互に対話してタスクを実行できる複数のエージェントを介して
LLMアプリケーションを構築できるオープンソースフレームワークです。
```

**`parallel`**: Outputs both original and translation for each paragraph. Ideal for comparison or quality review.

```markdown
## Abstract
AutoGen is an open-source framework that allows developers to build LLM applications
via multiple agents that can converse with each other to accomplish tasks.

AutoGenは、開発者が相互に対話してタスクを実行できる複数のエージェントを介して
LLMアプリケーションを構築できるオープンソースフレームワークです。
```

### Image & Table Extraction

When using `--markdown`, images and tables are automatically extracted:

```bash
# Extract images and tables (default with --markdown)
uv run translate-pdf paper.pdf --markdown

# Disable image extraction
uv run translate-pdf paper.pdf --markdown --no-extract-images

# Disable table extraction
uv run translate-pdf paper.pdf --markdown --no-extract-tables

# Customize image output
uv run translate-pdf paper.pdf -m --image-format jpeg --image-quality 90 --image-dpi 200
```

### Translation Category Control

By default, only body text categories are translated (`text`, `abstract`, etc.).
Titles, formulas, and figures are kept in the original language.

```bash
# Translate all categories (including titles, formulas)
uv run translate-pdf paper.pdf --translate-all

# Specify custom categories to translate
uv run translate-pdf paper.pdf --translate-categories "text,abstract,doc_title"
```

### JSON Output (Intermediate Files)

Save structured JSON files for debugging, regeneration, or web service integration:

```bash
# Save intermediate JSON files
uv run translate-pdf paper.pdf --save-intermediate

# Full example with all metadata
uv run translate-pdf paper.pdf -t ja --save-intermediate --thumbnail --llm-summary --markdown
```

This generates two JSON files:
- `paper.json` - Base document (original content, metadata, source-language summary)
- `paper.ja.json` - Translation document (translated content, target-language summary)

### Advanced Options

```bash
# Debug mode (draw bounding boxes)
uv run translate-pdf paper.pdf --debug

# Side-by-side comparison PDF
uv run translate-pdf paper.pdf --side-by-side

# Verbose output
uv run translate-pdf paper.pdf -v
```

See `uv run translate-pdf --help` for all options.

## Output Files

When running with all features enabled:

```bash
uv run translate-pdf paper.pdf -t ja --markdown --thumbnail --llm-summary --save-intermediate
```

The following files are generated:

```
output/
├── paper.ja.pdf              # Translated PDF
├── paper.ja.md               # Translated Markdown
├── paper.md                  # Original Markdown
├── paper.json                # Base document (schema v2.0.0)
├── paper.ja.json             # Translation document
├── paper_thumbnail.png       # Thumbnail image
└── images/                   # Extracted images (if any)
    ├── paper_p0_img0.png
    └── ...
```

### JSON Schema (v2.0.0)

**Base Document (`paper.json`)**:
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
    "title": "Document Title",
    "abstract": "Abstract text...",
    "summary": "LLM-generated summary in source language..."
  },
  "paragraphs": [...]
}
```

**Translation Document (`paper.ja.json`)**:
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
    "summary": "LLMで生成された要約（翻訳言語）..."
  },
  "paragraphs": {
    "para_p0_b1": "翻訳文1",
    "para_p0_b2": "翻訳文2"
  }
}
```

## Examples

See the `examples/` directory for sample outputs:

- `examples/summary_output/` - Full example with thumbnail, JSON, and Markdown output

## License

Apache-2.0
