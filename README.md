# PDF Translator

PDF translation tool with layout preservation - outputs Markdown and PDF.

## Features

- **Layout-preserving translation**: Translates PDF content while maintaining original formatting
- **Multiple output formats**: Generates both translated PDF and Markdown
- **Document layout analysis**: Uses ML-based layout detection for accurate text block identification
- **Multiple translation backends**: Google Translate (default), DeepL, OpenAI GPT

## Installation

```bash
uv sync
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
translate-pdf paper.pdf

# Specify output file
translate-pdf paper.pdf -o ./output/translated.pdf

# Use different backend
translate-pdf paper.pdf --backend deepl
translate-pdf paper.pdf --backend openai
```

### Markdown Output

```bash
# Generate Markdown alongside PDF
translate-pdf paper.pdf --markdown

# Markdown with original + translation (parallel mode)
translate-pdf paper.pdf -m --markdown-mode parallel

# Include all categories in Markdown (headers, footers, etc.)
translate-pdf paper.pdf -m --markdown-include-all
```

### Translation Category Control

By default, only body text categories are translated (`text`, `abstract`, etc.).
Titles, formulas, and figures are kept in the original language.

```bash
# Translate all categories (including titles, formulas)
translate-pdf paper.pdf --translate-all

# Specify custom categories to translate
translate-pdf paper.pdf --translate-categories "text,abstract,doc_title"
```

### Advanced Options

```bash
# Debug mode (draw bounding boxes)
translate-pdf paper.pdf --debug

# Side-by-side comparison PDF
translate-pdf paper.pdf --side-by-side

# Save intermediate JSON for later regeneration
translate-pdf paper.pdf --save-intermediate
```

See `translate-pdf --help` for all options.

## License

Apache-2.0
