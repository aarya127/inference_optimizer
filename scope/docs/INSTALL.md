# AMIO Phase 0 - Installation Guide

## Prerequisites

### Hardware Requirements
- **Apple Silicon Mac** (M1/M2/M3 series)
- **Minimum 16GB unified memory** (32GB+ recommended for batch processing)
- **20GB free disk space** for model weights and cache

### Software Requirements
- **macOS** 12.0 (Monterey) or later
- **Python** 3.9-3.11
- **Xcode Command Line Tools**

## Quick Start

### 1. Install Xcode Command Line Tools
```bash
xcode-select --install
```

### 2. Create Python Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate

# Or using conda
conda create -n amio python=3.10
conda activate amio
```

### 3. Install Core Dependencies
```bash
# Install MLX framework
pip install mlx>=0.10.0 mlx-lm>=0.10.0

# Try to install mlx-vlm (optional, may require building from source)
pip install mlx-vlm || echo "⚠️  mlx-vlm optional, continue without it"

# Install required packages
pip install numpy>=1.24.0 pillow>=10.0.0
pip install transformers>=4.40.0 huggingface-hub>=0.20.0
pip install psutil>=5.9.0

# Install visualization packages
pip install matplotlib>=3.7.0 seaborn>=0.12.0

# Install packaging for version checking
pip install packaging
```

### 4. Verify Installation
```bash
cd phase0_amio
python config/validate_stack.py
```

Expected output:
```
================================================================================
AMIO Phase 0 - Tech Stack Validation
================================================================================

✅ PASS | Platform            | Running on Darwin
✅ PASS | Apple Silicon       | Architecture: arm64, Chip: Apple M3
✅ PASS | MLX Installation    | MLX installed: 0.x.x
✅ PASS | MLX Version         | MLX 0.x.x (required: >=0.10.0)
✅ PASS | Unified Memory      | 16.0 GB available (recommended: ≥16GB)
✅ PASS | GPU/Metal Access    | Metal GPU accessible
✅ PASS | Dependencies        | All required dependencies installed

Summary: 7/7 checks passed
✅ Tech stack is ready for Phase 0!
```

## Detailed Installation

### Option A: Automated Setup (Recommended)
```bash
cd phase0_amio
chmod +x scripts/setup_phase0.sh
./scripts/setup_phase0.sh
```

### Option B: Manual Setup

#### Step 1: Install Python Packages
```bash
# Core MLX packages
pip install mlx>=0.10.0
pip install mlx-lm>=0.10.0

# Optional: MLX Vision-Language (may need manual build)
# See: https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm
```

#### Step 2: Install Supporting Libraries
```bash
# NumPy for numerical operations
pip install numpy>=1.24.0

# Pillow for image processing
pip install pillow>=10.0.0

# HuggingFace for model loading
pip install transformers>=4.40.0
pip install huggingface-hub>=0.20.0

# System monitoring
pip install psutil>=5.9.0

# Visualization
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

#### Step 3: Verify GPU Access
```python
import mlx.core as mx

# Test GPU computation
x = mx.array([1.0, 2.0, 3.0])
y = mx.add(x, x)
mx.eval(y)

print(f"✅ MLX GPU test passed")
print(f"Default device: {mx.default_device()}")
```

## Model Downloads

### LLaVA-1.5-7B (Primary Model)
Models will be automatically downloaded on first use via HuggingFace Hub:

```python
from phase0_amio.models.multimodal_loader import load_model

# Downloads ~14GB (cached at ~/.cache/huggingface/)
model = load_model(
    model_type="llava",
    model_id="llava-hf/llava-1.5-7b-hf",
    quantization_bits=4
)
```

**Manual download** (optional):
```bash
# Install huggingface-cli
pip install huggingface-hub[cli]

# Download model
huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir models/llava-1.5-7b
```

## Troubleshooting

### MLX Installation Issues

**Problem**: `pip install mlx` fails  
**Solution**: Ensure you're on Apple Silicon Mac with compatible Python version
```bash
# Check architecture
uname -m  # Should output: arm64

# Check Python version
python --version  # Should be 3.9-3.11
```

**Problem**: MLX installed but GPU not accessible  
**Solution**: Update macOS to latest version
```bash
softwareupdate -l
softwareupdate -i -a
```

### Memory Issues

**Problem**: Out of memory during model loading  
**Solution**: Close other applications, or use smaller batch size
```python
config = ModelConfig(
    model_id="llava-hf/llava-1.5-7b-hf",
    quantization_bits=4,
    max_batch_size=2  # Reduce from default 4
)
```

**Problem**: Model loading is very slow  
**Solution**: Ensure model is quantized to INT4
```python
# This loads in ~4.5GB instead of ~14GB
model = load_model(quantization_bits=4)  # ✅ Fast
# model = load_model(quantization_bits=16)  # ❌ Slow, uses 3x memory
```

### HuggingFace Hub Issues

**Problem**: Model download fails or times out  
**Solution**: Set up HuggingFace cache and retry
```bash
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub

# Retry download with resume
huggingface-cli download llava-hf/llava-1.5-7b-hf --resume-download
```

**Problem**: Authentication required for gated models  
**Solution**: Login to HuggingFace
```bash
huggingface-cli login
# Enter your HF token from: https://huggingface.co/settings/tokens
```

## Verification Tests

### Test 1: Tech Stack Validation
```bash
python config/validate_stack.py
```

### Test 2: Model Loading
```bash
python models/multimodal_loader.py
```

### Test 3: Quantization Framework
```bash
python models/quantization.py
```

### Test 4: TP Simulation
```bash
python simulation/tp_simulator.py
```

### Test 5: Metrics Collection
```bash
python metrics/collector.py
```

### Test 6: SLA Validation
```bash
python metrics/sla_validator.py
```

All tests should complete with "✅ ... test complete" messages.

## Next Steps

After successful installation:

1. **Review Design Document**: Read `docs/DESIGN.md` for technical details
2. **Run Example**: Execute example inference in `examples/` (Phase 1)
3. **Collect Baselines**: Run baseline benchmarks to establish metrics
4. **Proceed to Phase 1**: Begin adaptive controller development

## System Requirements by M3 Variant

| Variant | GPU Cores | Memory | Batch Size | Max Throughput |
|---------|-----------|--------|------------|----------------|
| M3 Base | 10 | 16GB | 1-2 | ~20 tok/s |
| M3 Pro | 18 | 36GB | 1-4 | ~25 tok/s |
| M3 Max | 40 | 64-128GB | 1-8 | ~30 tok/s |

## Additional Resources

- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **MLX Examples**: https://github.com/ml-explore/mlx-examples
- **LLaVA Paper**: https://arxiv.org/abs/2304.08485
- **HuggingFace Hub**: https://huggingface.co/llava-hf

## Support

For issues specific to AMIO Phase 0, consult:
- Design document: `docs/DESIGN.md`
- Configuration files: `config/*.yaml`
- Test scripts in each module

For upstream issues:
- MLX: https://github.com/ml-explore/mlx/issues
- Transformers: https://github.com/huggingface/transformers/issues
