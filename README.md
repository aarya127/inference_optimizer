# Inference Optimizer

[![CI](https://github.com/aarya127/inference_optimizer/workflows/CI/badge.svg)](https://github.com/aarya127/inference_optimizer/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive benchmarking tool for comparing LLM inference engines across multiple performance dimensions.

> **🚀 Compare vLLM, TensorRT-LLM, DeepSpeed, Triton, and more!**  
> Measure latency, throughput, memory usage across quantization levels and batch sizes.

## Overview

This tool provides systematic comparison of popular inference optimization frameworks:

- **vLLM** - High-throughput serving with PagedAttention
- **TensorRT-LLM** - NVIDIA's optimized inference engine
- **DeepSpeed-Inference** - Microsoft's inference optimization
- **Triton Inference Server** - NVIDIA's scalable inference platform
- **Custom CUDA/Triton Kernels** - Hand-optimized implementations

## Benchmark Dimensions

### Performance Metrics
- **Latency** - Time to first token (TTFT) and per-token latency
- **Throughput** - Tokens/second, requests/second
- **Memory Usage** - GPU memory footprint, CPU memory
- **Batch Size Scaling** - Performance across batch sizes (1-128+)

### Quantization Support
- FP16 (Half precision)
- FP8 (8-bit floating point)
- INT8 (8-bit integer)
- INT4 (4-bit integer)

### Model Coverage
- Small models (7B parameters)
- Medium models (13B-34B parameters)
- Large models (70B+ parameters)

## Project Structure

```
inference_optimizer/
├── src/
│   ├── backends/          # Engine-specific implementations
│   ├── benchmarks/        # Benchmark orchestration
│   ├── metrics/           # Metrics collection and analysis
│   ├── models/            # Model management
│   └── utils/             # Helper utilities
├── configs/               # Benchmark configurations
├── results/               # Benchmark outputs
├── notebooks/             # Analysis notebooks
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simple benchmark
python -m src.benchmarks.run --backend vllm --model llama-7b --quantization fp16

# Compare multiple backends
python -m src.benchmarks.compare --backends vllm,tensorrt --model llama-7b

# Generate report
python -m src.benchmarks.report --results-dir results/
```

## Installation

See [INSTALL.md](docs/INSTALL.md) for detailed installation instructions for each backend.

## Usage

See [USAGE.md](docs/USAGE.md) for comprehensive usage examples.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.
