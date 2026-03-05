# Phase 0: Adaptive Multimodal Inference Optimizer (AMIO) - Foundation

## Overview

Phase 0 establishes the technical boundaries and foundational infrastructure for AMIO, specifically designed for **Apple Silicon M3** hardware. This phase validates feasibility, defines simulation methodologies, and establishes performance baselines before controller development.

## Project Structure

```
phase0_amio/
├── config/           # Hardware specs, model configs, SLA definitions
├── models/           # MLX-based multimodal model implementations
├── simulation/       # Single-GPU TP simulation framework
├── metrics/          # Performance metric collectors (TTFT, TBT, Fragmentation)
├── docs/             # Design documents and technical specifications
├── tests/            # Validation and benchmark tests
└── examples/         # Usage examples and demos
```

## Hardware Constraints (M3 Mac)

- **Unified Memory**: 16-36GB shared between CPU/GPU/Neural Engine
- **GPU Cores**: 10-40 cores (M3/Pro/Max/Ultra variants)
- **Memory Bandwidth**: 100-800GB/s depending on variant
- **No Multi-GPU**: Single-node system requiring TP simulation
- **Apple Neural Engine**: 16-core for INT8/INT16 operations

## Key Design Decisions

1. **Framework**: MLX for native Apple Silicon acceleration
2. **Model Scale**: 7B parameter multimodal models (LLaVA-1.5, Qwen-VL)
3. **Quantization**: 4-bit (INT4/W4A8) reducing 14GB → 4GB footprint
4. **Simulation**: 126 All-Reduce sync points with artificial delays
5. **Target SLA**: TTFT < 500ms for 95th percentile

## Success Criteria

- [ ] 7B multimodal model runs with <6GB peak memory
- [ ] 4-bit quantization maintains >90% accuracy on benchmarks
- [ ] TP simulation produces realistic latency estimates (±15% of real multi-GPU)
- [ ] TTFT consistently under 500ms for single image + prompt
- [ ] Memory fragmentation stays below 20%

## Quick Start

See [INSTALL.md](docs/INSTALL.md) for setup instructions and [DESIGN.md](docs/DESIGN.md) for the complete Phase 0 design document.

## Next Phase

Phase 1 will build the adaptive controller using the cost models and metrics established in Phase 0.
