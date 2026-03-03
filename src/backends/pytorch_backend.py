"""PyTorch baseline backend implementation."""

from typing import List, Optional
import torch
from src.backends.base import BaseBackend, ModelConfig


class PyTorchBackend(BaseBackend):
    """Baseline PyTorch backend for comparison."""
    
    def __init__(self, config: ModelConfig):
        """Initialize PyTorch backend.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
    
    def load_model(self) -> None:
        """Load model using standard PyTorch/Transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Map quantization to dtype
        dtype_map = {
            "fp16": torch.float16,
            "fp8": torch.float16,  # PyTorch doesn't natively support FP8
            "int8": torch.float16,  # Load as FP16, quantize later
            "int4": torch.float16,
        }
        
        dtype = dtype_map.get(self.config.quantization, torch.float16)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.config.device if self.config.device != "cuda" else "auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Apply quantization if needed
        if self.config.quantization == "int8":
            self._apply_int8_quantization()
        elif self.config.quantization == "int4":
            self._apply_int4_quantization()
        
        self.model.eval()
        self._is_loaded = True
    
    def _apply_int8_quantization(self):
        """Apply INT8 quantization."""
        try:
            from transformers import BitsAndBytesConfig
            # Note: This requires reloading the model with proper config
            # This is a simplified version
            pass
        except ImportError:
            print("Warning: bitsandbytes not available for INT8 quantization")
    
    def _apply_int4_quantization(self):
        """Apply INT4 quantization."""
        try:
            from transformers import BitsAndBytesConfig
            # Note: This requires reloading the model with proper config
            pass
        except ImportError:
            print("Warning: bitsandbytes not available for INT4 quantization")
    
    def unload_model(self) -> None:
        """Unload model and free resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._is_loaded = False
        
        import gc
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
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode (exclude input prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
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
        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length - max_new_tokens,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 50),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        input_lengths = inputs.input_ids.shape[1]
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(
                output[input_lengths:],
                skip_special_tokens=True
            )
            generated_texts.append(text)
        
        return generated_texts
