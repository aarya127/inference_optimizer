"""vLLM backend implementation."""

from typing import List, Optional
from src.backends.base import BaseBackend, ModelConfig


class VLLMBackend(BaseBackend):
    """vLLM inference backend with PagedAttention."""
    
    def __init__(self, config: ModelConfig):
        """Initialize vLLM backend.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.llm = None
        self.sampling_params = None
    
    def load_model(self) -> None:
        """Load model using vLLM."""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )
        
        # Map quantization to vLLM format
        quantization_map = {
            "fp16": None,
            "fp8": "fp8",
            "int8": "int8",
            "int4": "awq",  # vLLM uses AWQ for INT4
        }
        
        quantization = quantization_map.get(self.config.quantization)
        
        # Load model
        self.llm = LLM(
            model=self.config.model_name,
            quantization=quantization,
            max_model_len=self.config.max_seq_length,
            tensor_parallel_size=1,
            trust_remote_code=self.config.trust_remote_code,
            dtype="float16" if self.config.quantization == "fp16" else "auto",
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        self._is_loaded = True
    
    def unload_model(self) -> None:
        """Unload model and free resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._is_loaded = False
        
        # Force cleanup
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
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
            **kwargs: Additional parameters
            
        Returns:
            List of generated texts
        """
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 0.8),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 50),
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
