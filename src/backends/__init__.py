"""Backend implementations for different inference engines."""

from src.backends.base import BaseBackend
from src.backends.vllm_backend import VLLMBackend
from src.backends.tensorrt_backend import TensorRTBackend
from src.backends.deepspeed_backend import DeepSpeedBackend
from src.backends.triton_backend import TritonBackend
from src.backends.pytorch_backend import PyTorchBackend

__all__ = [
    "BaseBackend",
    "VLLMBackend",
    "TensorRTBackend",
    "DeepSpeedBackend",
    "TritonBackend",
    "PyTorchBackend",
]

BACKEND_REGISTRY = {
    "vllm": VLLMBackend,
    "tensorrt": TensorRTBackend,
    "deepspeed": DeepSpeedBackend,
    "triton": TritonBackend,
    "pytorch": PyTorchBackend,
}


def get_backend(backend_name: str, config) -> BaseBackend:
    """Get backend instance by name.
    
    Args:
        backend_name: Name of the backend
        config: Model configuration
        
    Returns:
        Backend instance
    """
    backend_class = BACKEND_REGISTRY.get(backend_name.lower())
    if backend_class is None:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available backends: {list(BACKEND_REGISTRY.keys())}"
        )
    return backend_class(config)
