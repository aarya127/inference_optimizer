# Inference Optimizer - Quick Start

## Installation

```bash
# Clone the repository
git clone https://github.com/aarya127/inference_optimizer.git
cd inference_optimizer

# Install
pip install -e .
```

## Run Your First Benchmark

```bash
# Simple benchmark with PyTorch baseline
inference-optimizer benchmark \
  --model gpt2 \
  --backends pytorch \
  --batch-sizes 1 4 8 \
  --num-requests 50 \
  --output-dir results/my_first_benchmark
```

Results will be saved with visualizations and a detailed report!

## Compare Multiple Backends

```bash
# Install vLLM
pip install vllm

# Compare PyTorch vs vLLM
inference-optimizer benchmark \
  --model gpt2 \
  --backends pytorch vllm \
  --quantizations fp16 \
  --batch-sizes 1 4 8 16 \
  --num-requests 100 \
  --output-dir results/comparison
```

## Python API

```python
from src.benchmarks import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    model_name="gpt2",
    backends=["pytorch", "vllm"],
    quantizations=["fp16"],
    batch_sizes=[1, 4, 8],
    num_requests=100,
)

runner = BenchmarkRunner(config)
results = runner.run()
```

## What Gets Generated

After running a benchmark, you'll find:

- **results_TIMESTAMP.json** - Raw benchmark data
- **results_TIMESTAMP.csv** - Results in CSV format
- **report_TIMESTAMP.md** - Detailed markdown report
- **latency_comparison.png** - Latency visualizations
- **throughput_comparison.png** - Throughput charts
- **memory_comparison.png** - Memory usage plots
- **efficiency_comparison.png** - Efficiency metrics

## Next Steps

- See [USAGE.md](docs/USAGE.md) for detailed usage
- Check [INSTALL.md](docs/INSTALL.md) for backend installation
- Browse [examples/](examples/) for more examples
- Read [CONTRIBUTING.md](CONTRIBUTING.md) to add custom backends

## Key Features

✅ Compare 5+ inference engines  
✅ Test multiple quantization levels (FP16, FP8, INT8, INT4)  
✅ Analyze batch size scaling (1-128+)  
✅ Measure latency, throughput, and memory  
✅ Beautiful visualizations and reports  
✅ Easy to extend with custom backends  

## Common Commands

```bash
# List available backends
inference-optimizer list-backends

# Generate visualizations from existing results
inference-optimizer visualize results/my_results.json

# Generate report only
inference-optimizer report results/my_results.json

# Compare multiple runs
inference-optimizer compare \
  results/run1.json \
  results/run2.json \
  --output-dir results/comparison
```

## Need Help?

- Check the [full documentation](README.md)
- Open an issue on GitHub
- Review the [examples/](examples/) directory
