"""DeepSpeed-Inference backend implementation."""

from typing import List, Optional
import torch
from src.backends.base import BaseBackend, ModelConfig


class DeepSpeedBackend(BaseBackend):
    """DeepSpeed-Inference backend."""
    
    def __init__(self, config: ModelConfig):
        """Initialize DeepSpeed-Inference backend.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.ds_engine = None
    
    def load_model(self) -> None:
        """Load model using DeepSpeed-Inference."""
        try:
            import deepspeed
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "DeepSpeed not installed. Install with: pip install deepspeed"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        dtype_map = {
            "fp16": torch.float16,
            "fp8": torch.float16,  # DeepSpeed doesn't support FP8 directly
            "int8": torch.int8,
            "int4": torch.int8,  # Approximate
        }
        
        dtype = dtype_map.get(self.config.quantization, torch.float16)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Initialize DeepSpeed inference
        self.ds_engine = deepspeed.init_inference(
            model=self.model,
            mp_size=1,  # Model parallelism size
            dtype=dtype,
            replace_with_kernel_inject=True,
            max_tokens=self.config.max_seq_length,
        )
        
        self._is_loaded = True
    
    def unload_model(self) -> None:
        """Unload model and free resources."""
        if self.ds_engine is not None:
            del self.ds_engine
            self.ds_engine = None
        
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.ds_engine.module.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.ds_engine.module.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
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
        ).to(self.ds_engine.module.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.ds_engine.module.generate(
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
