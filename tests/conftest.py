"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_model_config():
    """Fixture for sample model configuration."""
    from src.backends.base import ModelConfig
    
    return ModelConfig(
        model_name="gpt2",
        quantization="fp16",
        max_batch_size=8,
        max_seq_length=1024,
    )


@pytest.fixture
def sample_inference_request():
    """Fixture for sample inference request."""
    from src.backends.base import InferenceRequest
    
    return InferenceRequest(
        prompt="Test prompt for inference",
        max_new_tokens=64,
        temperature=0.8,
    )


@pytest.fixture
def sample_benchmark_config():
    """Fixture for sample benchmark configuration."""
    from src.benchmarks import BenchmarkConfig
    
    return BenchmarkConfig(
        model_name="gpt2",
        backends=["pytorch"],
        quantizations=["fp16"],
        batch_sizes=[1, 4],
        num_requests=10,
        output_dir="tests/tmp",
    )


@pytest.fixture
def metrics_collector():
    """Fixture for metrics collector."""
    from src.metrics import MetricsCollector
    
    collector = MetricsCollector()
    collector.start()
    return collector
