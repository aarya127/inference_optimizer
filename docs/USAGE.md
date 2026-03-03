# Usage Guide

## Quick Start

### Command Line Interface

Run a basic benchmark:

```bash
inference-optimizer benchmark \
  --model gpt2 \
  --backends pytorch vllm \
  --quantizations fp16 \
  --batch-sizes 1 4 8 \
  --num-requests 100 \
  --output-dir results/my_benchmark
```

### Python API

```python
from src.benchmarks import BenchmarkRunner, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    model_name="gpt2",
    backends=["pytorch", "vllm"],
    quantizations=["fp16"],
    batch_sizes=[1, 4, 8],
    num_requests=100,
    output_dir="results/my_benchmark"
)

# Run benchmark
runner = BenchmarkRunner(config)
results = runner.run()

# Generate visualizations
from src.visualization import BenchmarkVisualizer, ReportGenerator

visualizer = BenchmarkVisualizer(results, "results/my_benchmark")
visualizer.plot_all()

report_gen = ReportGenerator(results, "results/my_benchmark")
report_gen.generate_markdown_report()
```

## CLI Commands

### benchmark

Run benchmarks with specified configuration:

```bash
inference-optimizer benchmark [OPTIONS]
```

**Options**:
- `--model, -m`: Model name (required)
- `--backends, -b`: Backends to test (multiple)
- `--quantizations, -q`: Quantization levels (multiple)
- `--batch-sizes, -bs`: Batch sizes (multiple)
- `--num-requests, -n`: Number of requests
- `--max-tokens`: Maximum tokens to generate
- `--output-dir, -o`: Output directory
- `--no-visualize`: Skip visualization

**Example**:
```bash
inference-optimizer benchmark \
  -m meta-llama/Llama-2-7b-hf \
  -b vllm -b tensorrt \
  -q fp16 -q int8 \
  -bs 1 -bs 4 -bs 8 \
  -n 200 \
  -o results/llama7b
```

### visualize

Generate visualizations from existing results:

```bash
inference-optimizer visualize results/my_results.json
```

**Options**:
- `--output-dir, -o`: Output directory for plots

### report

Generate a report from existing results:

```bash
inference-optimizer report results/my_results.json
```

### compare

Compare multiple benchmark results:

```bash
inference-optimizer compare \
  results/run1.json \
  results/run2.json \
  results/run3.json \
  --output-dir results/comparison
```

### list-backends

List all available backends:

```bash
inference-optimizer list-backends
```

## Configuration Files

You can also use YAML configuration files:

```bash
inference-optimizer benchmark --config configs/production.yaml
```

See `configs/` directory for examples.

## Common Use Cases

### 1. Compare Backends for a Specific Model

```python
from src.benchmarks import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    backends=["pytorch", "vllm", "deepspeed"],
    quantizations=["fp16"],
    batch_sizes=[1],
    num_requests=100,
)

runner = BenchmarkRunner(config)
results = runner.run()
```

### 2. Test Quantization Impact

```python
config = BenchmarkConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    backends=["vllm"],
    quantizations=["fp16", "fp8", "int8", "int4"],
    batch_sizes=[4],
    num_requests=100,
)

runner = BenchmarkRunner(config)
results = runner.run()
```

### 3. Batch Size Scaling Analysis

```python
config = BenchmarkConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    backends=["vllm"],
    quantizations=["fp16"],
    batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
    num_requests=200,
)

runner = BenchmarkRunner(config)
results = runner.run()
```

### 4. Custom Backend Testing

```python
from src.backends import BACKEND_REGISTRY
from examples.custom_backend import CustomBackend

# Register custom backend
BACKEND_REGISTRY["custom"] = CustomBackend

# Use in benchmark
config = BenchmarkConfig(
    model_name="your-model",
    backends=["custom"],
    quantizations=["fp16"],
    batch_sizes=[1, 4],
    num_requests=50,
)

runner = BenchmarkRunner(config)
results = runner.run()
```

## Output Structure

After running a benchmark, the output directory contains:

```
results/
├── results_20260303_123456.json    # Raw results data
├── results_20260303_123456.csv     # Results in CSV format
├── report_20260303_123456.md       # Markdown report
├── latency_comparison.png          # Latency plots
├── throughput_comparison.png       # Throughput plots
├── memory_comparison.png           # Memory plots
└── efficiency_comparison.png       # Efficiency plots
```

## Advanced Usage

### Using Context Manager

```python
from src.backends import get_backend
from src.backends.base import ModelConfig, InferenceRequest

config = ModelConfig(
    model_name="gpt2",
    quantization="fp16",
    max_batch_size=8,
)

# Automatically loads and unloads model
with get_backend("vllm", config) as backend:
    request = InferenceRequest(
        prompt="Write a story about AI",
        max_new_tokens=128,
    )
    result = backend.infer(request)
    print(result.generated_text)
    print(f"Latency: {result.latency_ms:.2f}ms")
    print(f"Throughput: {result.tokens_per_second:.2f} tok/s")
```

### Batch Inference

```python
from src.backends import get_backend
from src.backends.base import ModelConfig, InferenceRequest

config = ModelConfig(model_name="gpt2", quantization="fp16")
backend = get_backend("vllm", config)
backend.load_model()

# Create batch of requests
requests = [
    InferenceRequest(prompt=f"Question {i}: What is AI?", max_new_tokens=64)
    for i in range(8)
]

# Run batch inference
results = backend.batch_infer(requests)

for i, result in enumerate(results):
    print(f"Request {i}: {result.latency_ms:.2f}ms")

backend.unload_model()
```

### Custom Metrics Collection

```python
from src.metrics import MetricsCollector

collector = MetricsCollector()
collector.start()

# Run your inferences...
for i in range(100):
    result = backend.infer(request)
    collector.record_inference(
        latency_ms=result.latency_ms,
        ttft_ms=result.time_to_first_token_ms,
        memory_mb=result.memory_used_mb,
        tokens_generated=result.tokens_generated,
        success=result.success,
    )

collector.stop()

# Get summary
summary = collector.get_summary()
print(summary)
```

## Tips and Best Practices

1. **Start Small**: Test with small models (e.g., GPT-2) before scaling to larger models

2. **Warmup**: Always include warmup runs to ensure fair comparisons

3. **Multiple Runs**: Run benchmarks multiple times and average results for stability

4. **Monitor Resources**: Keep an eye on GPU memory and temperature during benchmarks

5. **Quantization**: Test different quantization levels to find the best speed/quality tradeoff

6. **Batch Sizes**: Larger batch sizes improve throughput but increase latency

7. **Model Selection**: Choose appropriate model sizes for your hardware capabilities

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use more aggressive quantization (INT8, INT4)
- Enable gradient checkpointing
- Use smaller models

### Slow Performance

- Ensure GPU is being used
- Check CUDA is properly installed
- Verify model is loaded correctly
- Try different backends

### Backend Errors

- Check backend-specific requirements
- Verify installation (see INSTALL.md)
- Check model compatibility with backend
- Review backend documentation

## Next Steps

- See [CONTRIBUTING.md](../CONTRIBUTING.md) for adding custom backends
- Check `examples/` directory for more usage patterns
- Review generated reports for optimization insights
