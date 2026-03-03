"""Test suite for benchmark runner."""

import pytest
from src.benchmarks import BenchmarkRunner, BenchmarkConfig
from src.backends.base import ModelConfig


def test_benchmark_config_defaults():
    """Test BenchmarkConfig default values."""
    config = BenchmarkConfig(model_name="gpt2")
    
    assert config.model_name == "gpt2"
    assert "pytorch" in config.backends
    assert "fp16" in config.quantizations
    assert 1 in config.batch_sizes
    assert config.num_requests == 100


def test_benchmark_config_custom():
    """Test BenchmarkConfig with custom values."""
    config = BenchmarkConfig(
        model_name="test-model",
        backends=["pytorch"],
        quantizations=["fp16", "int8"],
        batch_sizes=[1, 4],
        num_requests=50,
    )
    
    assert config.model_name == "test-model"
    assert config.backends == ["pytorch"]
    assert config.quantizations == ["fp16", "int8"]
    assert config.batch_sizes == [1, 4]
    assert config.num_requests == 50


def test_benchmark_runner_initialization():
    """Test BenchmarkRunner initialization."""
    config = BenchmarkConfig(model_name="gpt2")
    runner = BenchmarkRunner(config)
    
    assert runner.config == config
    assert runner.results == []


def test_generate_prompts():
    """Test prompt generation."""
    config = BenchmarkConfig(model_name="gpt2")
    runner = BenchmarkRunner(config)
    
    prompts = runner._generate_prompts(10)
    
    assert len(prompts) == 10
    assert all(isinstance(p, str) for p in prompts)
    assert all(len(p) > 0 for p in prompts)


# Note: Full benchmark tests require actual model loading
# and can be expensive. These should be run separately
# with proper fixtures and mocking.
