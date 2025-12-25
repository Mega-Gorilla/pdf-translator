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

PDF Translator uses PP-DocLayout for layout analysis, which benefits significantly from GPU acceleration. With GPU, layout analysis is 5-10x faster.

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

# Then install GPU version for your CUDA version:

# CUDA 11.8
uv pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CUDA 12.3
uv pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

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
