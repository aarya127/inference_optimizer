# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- CUDA Toolkit 11.8+ (for GPU acceleration)
- At least 16GB RAM
- Sufficient disk space for models

## Basic Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aarya127/inference_optimizer.git
cd inference_optimizer
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Core Dependencies

```bash
pip install -e .
```

This installs the core framework with PyTorch baseline support.

## Backend-Specific Installation

### vLLM

```bash
pip install -e ".[vllm]"
# Or directly:
pip install vllm>=0.2.0
```

### TensorRT-LLM

TensorRT-LLM requires additional setup:

```bash
# Install TensorRT
pip install tensorrt>=8.6.0

# Clone and install TensorRT-LLM
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
pip install -r requirements.txt
python setup.py install
```

See [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM) for detailed instructions.

### DeepSpeed-Inference

```bash
pip install -e ".[deepspeed]"
# Or directly:
pip install deepspeed>=0.12.0
```

### Triton Inference Server

Triton requires a separate server setup:

1. Install Triton client:
```bash
pip install -e ".[triton]"
# Or:
pip install tritonclient[all]>=2.40.0
```

2. Start Triton server (using Docker):
```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  nvcr.io/nvidia/tritonserver:24.01-py3
```

See [Triton documentation](https://github.com/triton-inference-server/server) for more details.

## Install All Backends

```bash
pip install -e ".[all]"
```

⚠️ **Note**: This will attempt to install all backends. Some may require manual configuration.

## Development Installation

For development with testing and formatting tools:

```bash
pip install -e ".[dev]"
```

## Verification

Verify installation by listing available backends:

```bash
inference-optimizer list-backends
```

Or in Python:

```python
from src.backends import BACKEND_REGISTRY
print("Available backends:", list(BACKEND_REGISTRY.keys()))
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:

1. Verify CUDA installation:
```bash
nvidia-smi
nvcc --version
```

2. Ensure PyTorch detects CUDA:
```python
import torch
print(torch.cuda.is_available())
```

3. Install CUDA-compatible PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

For large models, you may need:

- Multi-GPU setup with tensor parallelism
- Model quantization (INT8, INT4)
- Gradient checkpointing
- Reduce batch size

### Backend-Specific Issues

**vLLM**:
- Requires GPU with compute capability 7.0+ (Volta or newer)
- May need to increase shared memory: `--shm-size=8g` in Docker

**TensorRT-LLM**:
- Requires NVIDIA GPU
- Models must be pre-compiled to TensorRT engines
- See building guide in TensorRT-LLM docs

**DeepSpeed**:
- May require specific CUDA versions
- Check compatibility matrix in DeepSpeed docs

## Next Steps

After installation, see [USAGE.md](USAGE.md) for usage examples.
