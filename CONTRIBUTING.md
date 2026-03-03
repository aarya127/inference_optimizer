# Contributing to Inference Optimizer

Thank you for your interest in contributing! This guide will help you get started.

## Ways to Contribute

1. **Add New Backends**: Implement support for additional inference engines
2. **Improve Metrics**: Add new performance metrics or improve existing ones
3. **Enhance Visualizations**: Create new plots or improve existing visualizations
4. **Bug Fixes**: Find and fix bugs
5. **Documentation**: Improve documentation and examples
6. **Testing**: Add tests for better coverage

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/inference_optimizer.git
cd inference_optimizer
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs testing, linting, and formatting tools.

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Adding a New Backend

### Step 1: Create Backend Class

Create a new file in `src/backends/your_backend.py`:

```python
from typing import List
from src.backends.base import BaseBackend, ModelConfig

class YourBackend(BaseBackend):
    """Your backend implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Initialize your backend-specific attributes
    
    def load_model(self) -> None:
        """Load model using your engine."""
        # Implement model loading
        self._is_loaded = True
    
    def unload_model(self) -> None:
        """Unload model and free resources."""
        # Implement cleanup
        self._is_loaded = False
    
    def generate(self, prompt: str, max_new_tokens: int = 128, 
                 temperature: float = 0.8, top_p: float = 0.95, 
                 top_k: int = 50, **kwargs) -> str:
        """Generate text from prompt."""
        # Implement generation
        pass
    
    def batch_generate(self, prompts: List[str], 
                      max_new_tokens: int = 128, **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        # Implement batch generation
        pass
```

### Step 2: Register Backend

Add to `src/backends/__init__.py`:

```python
from src.backends.your_backend import YourBackend

BACKEND_REGISTRY = {
    # ... existing backends
    "yourbackend": YourBackend,
}
```

### Step 3: Add Tests

Create `tests/test_your_backend.py`:

```python
import pytest
from src.backends import get_backend
from src.backends.base import ModelConfig

def test_your_backend_load():
    config = ModelConfig(model_name="gpt2", quantization="fp16")
    backend = get_backend("yourbackend", config)
    backend.load_model()
    assert backend.is_loaded
    backend.unload_model()

def test_your_backend_generate():
    config = ModelConfig(model_name="gpt2", quantization="fp16")
    backend = get_backend("yourbackend", config)
    backend.load_model()
    
    result = backend.generate("Hello", max_new_tokens=10)
    assert isinstance(result, str)
    assert len(result) > 0
    
    backend.unload_model()
```

### Step 4: Update Documentation

Add your backend to:
- `README.md` - List in supported backends
- `docs/INSTALL.md` - Installation instructions
- `docs/USAGE.md` - Usage examples

## Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run formatters:

```bash
black src/ tests/
isort src/ tests/
```

Run linters:

```bash
flake8 src/ tests/
mypy src/
```

## Testing

Run tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src --cov-report=html
```

## Pull Request Process

1. **Update Tests**: Ensure all tests pass and add tests for new features

2. **Update Documentation**: Update relevant documentation

3. **Format Code**: Run formatters and linters

4. **Commit Changes**:
   ```bash
   git add .
   git commit -m "feat: add support for YourBackend"
   ```

5. **Push to Fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**: Go to GitHub and create a PR with:
   - Clear title and description
   - Reference any related issues
   - List of changes made
   - Test results

## Commit Message Guidelines

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat: add TensorRT-LLM backend support
fix: correct memory measurement in PyTorch backend
docs: update installation guide for vLLM
test: add tests for batch inference
```

## Adding New Metrics

To add a new metric:

1. Update `BenchmarkMetrics` in `src/metrics/collector.py`
2. Update `MetricsCollector.compute_benchmark_metrics()`
3. Update visualization in `src/visualization/plots.py`
4. Update report generation in `src/visualization/reports.py`

Example:

```python
@dataclass
class BenchmarkMetrics:
    # ... existing fields
    new_metric: float  # Add your metric
```

## Adding New Visualizations

To add a new plot:

1. Add method to `BenchmarkVisualizer` in `src/visualization/plots.py`:

```python
def plot_your_metric(self, save: bool = True) -> Optional[str]:
    """Plot your metric comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create your plot
    data = self.df.groupby('backend')['your_metric'].mean()
    data.plot(kind='bar', ax=ax)
    
    ax.set_title("Your Metric Comparison")
    ax.set_ylabel("Your Metric")
    
    plt.tight_layout()
    
    if save:
        path = self.output_dir / "your_metric.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(path)
    else:
        plt.show()
        return None
```

2. Add to `plot_all()` method

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues and PRs

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow project guidelines

Thank you for contributing! 🚀
