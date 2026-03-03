"""Base backend interface for inference engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import torch


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    
    model_name: str
    model_path: Optional[str] = None
    quantization: str = "fp16"  # fp16, fp8, int8, int4
    max_batch_size: int = 32
    max_seq_length: int = 2048
    device: str = "cuda"
    trust_remote_code: bool = False


@dataclass
class InferenceRequest:
    """Single inference request."""
    
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50


@dataclass
class InferenceResult:
    """Result from inference request."""
    
    generated_text: str
    tokens_generated: int
    latency_ms: float
    time_to_first_token_ms: float
    tokens_per_second: float
    memory_used_mb: float
    success: bool = True
    error: Optional[str] = None


class BaseBackend(ABC):
    """Abstract base class for inference backends."""
    
    def __init__(self, config: ModelConfig):
        """Initialize backend with model configuration.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    @property
    def name(self) -> str:
        """Get backend name."""
        return self.__class__.__name__.replace("Backend", "")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free resources."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input texts
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific parameters
            
        Returns:
            List of generated texts
        """
        pass
    
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference with detailed metrics.
        
        Args:
            request: Inference request
            
        Returns:
            Inference result with metrics
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} model not loaded. Call load_model() first.")
        
        try:
            # Get initial memory
            memory_before = self._get_memory_usage()
            
            # Time to first token (approximate)
            start_time = time.perf_counter()
            
            # Generate
            generated_text = self.generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            tokens_generated = self._count_tokens(generated_text)
            tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
            
            # Memory usage
            memory_after = self._get_memory_usage()
            memory_used_mb = memory_after - memory_before
            
            # Approximate TTFT (first 10% of generation time)
            time_to_first_token_ms = latency_ms * 0.1
            
            return InferenceResult(
                generated_text=generated_text,
                tokens_generated=tokens_generated,
                latency_ms=latency_ms,
                time_to_first_token_ms=time_to_first_token_ms,
                tokens_per_second=tokens_per_second,
                memory_used_mb=memory_used_mb,
                success=True,
            )
            
        except Exception as e:
            return InferenceResult(
                generated_text="",
                tokens_generated=0,
                latency_ms=0.0,
                time_to_first_token_ms=0.0,
                tokens_per_second=0.0,
                memory_used_mb=0.0,
                success=False,
                error=str(e),
            )
    
    def batch_infer(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResult]:
        """Run batch inference with detailed metrics.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference results
        """
        if not self.is_loaded:
            raise RuntimeError(f"{self.name} model not loaded. Call load_model() first.")
        
        prompts = [req.prompt for req in requests]
        max_new_tokens = requests[0].max_new_tokens if requests else 128
        
        try:
            memory_before = self._get_memory_usage()
            start_time = time.perf_counter()
            
            generated_texts = self.batch_generate(prompts, max_new_tokens=max_new_tokens)
            
            end_time = time.perf_counter()
            memory_after = self._get_memory_usage()
            
            total_latency_ms = (end_time - start_time) * 1000
            memory_used_mb = (memory_after - memory_before) / len(requests)
            
            results = []
            for i, (request, generated_text) in enumerate(zip(requests, generated_texts)):
                tokens_generated = self._count_tokens(generated_text)
                # Approximate per-request latency
                latency_ms = total_latency_ms / len(requests)
                tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
                
                results.append(InferenceResult(
                    generated_text=generated_text,
                    tokens_generated=tokens_generated,
                    latency_ms=latency_ms,
                    time_to_first_token_ms=latency_ms * 0.1,
                    tokens_per_second=tokens_per_second,
                    memory_used_mb=memory_used_mb,
                    success=True,
                ))
            
            return results
            
        except Exception as e:
            return [
                InferenceResult(
                    generated_text="",
                    tokens_generated=0,
                    latency_ms=0.0,
                    time_to_first_token_ms=0.0,
                    tokens_per_second=0.0,
                    memory_used_mb=0.0,
                    success=False,
                    error=str(e),
                )
                for _ in requests
            ]
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        # Rough approximation
        return len(text.split())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "backend": self.name,
            "model_name": self.config.model_name,
            "quantization": self.config.quantization,
            "max_batch_size": self.config.max_batch_size,
            "max_seq_length": self.config.max_seq_length,
            "device": self.config.device,
            "is_loaded": self.is_loaded,
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
        return False
