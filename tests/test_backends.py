"""Test suite for backend implementations."""

import pytest
from src.backends import get_backend, BACKEND_REGISTRY
from src.backends.base import ModelConfig, InferenceRequest


def test_backend_registry():
    """Test backend registry contains expected backends."""
    expected_backends = ["pytorch", "vllm", "tensorrt", "deepspeed", "triton"]
    
    for backend_name in expected_backends:
        assert backend_name in BACKEND_REGISTRY


def test_get_backend():
    """Test getting backend instance."""
    config = ModelConfig(model_name="gpt2", quantization="fp16")
    backend = get_backend("pytorch", config)
    
    assert backend is not None
    assert backend.config == config
    assert backend.name == "PyTorch"


def test_get_unknown_backend():
    """Test getting unknown backend raises error."""
    config = ModelConfig(model_name="gpt2")
    
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("unknown_backend", config)


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig(model_name="test-model")
    
    assert config.model_name == "test-model"
    assert config.quantization == "fp16"
    assert config.max_batch_size == 32
    assert config.max_seq_length == 2048
    assert config.device == "cuda"


def test_inference_request_defaults():
    """Test InferenceRequest default values."""
    request = InferenceRequest(prompt="Test prompt")
    
    assert request.prompt == "Test prompt"
    assert request.max_new_tokens == 128
    assert request.temperature == 0.8
    assert request.top_p == 0.95


def test_backend_properties():
    """Test backend properties."""
    config = ModelConfig(model_name="gpt2")
    backend = get_backend("pytorch", config)
    
    assert not backend.is_loaded
    assert backend.name == "PyTorch"
    
    info = backend.get_model_info()
    assert info["backend"] == "PyTorch"
    assert info["model_name"] == "gpt2"
    assert not info["is_loaded"]


# Note: Tests requiring actual model loading should be
# run separately with proper fixtures and resources.
