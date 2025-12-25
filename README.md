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

```bash
translate-pdf paper.pdf
```

## License

Apache-2.0
