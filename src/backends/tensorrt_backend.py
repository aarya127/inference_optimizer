"""TensorRT-LLM backend implementation."""

from typing import List, Optional
from src.backends.base import BaseBackend, ModelConfig


class TensorRTBackend(BaseBackend):
    """TensorRT-LLM inference backend."""
    
    def __init__(self, config: ModelConfig):
        """Initialize TensorRT-LLM backend.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.engine = None
        self.runner = None
    
    def load_model(self) -> None:
        """Load model using TensorRT-LLM."""
        try:
            import tensorrt_llm
            from tensorrt_llm.runtime import ModelRunner
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "TensorRT-LLM not installed. Please follow installation instructions at: "
                "https://github.com/NVIDIA/TensorRT-LLM"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Note: TensorRT-LLM requires pre-built engines
        # This is a simplified implementation
        # In practice, you need to build the engine first using trtllm-build
        
        engine_path = self.config.model_path or f"{self.config.model_name}_trt_engine"
        
        # Load pre-built engine
        try:
            self.runner = ModelRunner.from_dir(engine_path)
            self._is_loaded = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TensorRT engine from {engine_path}. "
                f"Please build the engine first using trtllm-build. Error: {e}"
            )
    
    def unload_model(self) -> None:
        """Unload model and free resources."""
        if self.runner is not None:
            del self.runner
            self.runner = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._is_loaded = False
        
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
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate
        outputs = self.runner.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
        )
        
        # Decode output
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):]
        
        return output_text
    
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
        # Tokenize inputs
        input_ids = [
            self.tokenizer.encode(prompt, return_tensors="pt")
            for prompt in prompts
        ]
        
        # Pad to same length
        max_len = max(ids.shape[1] for ids in input_ids)
        input_ids_padded = [
            torch.nn.functional.pad(
                ids,
                (0, max_len - ids.shape[1]),
                value=self.tokenizer.pad_token_id
            )
            for ids in input_ids
        ]
        
        batch_input_ids = torch.cat(input_ids_padded, dim=0)
        
        # Generate
        outputs = self.runner.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 0.8),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 50),
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
        )
        
        # Decode outputs
        generated_texts = []
        for i, (output, prompt) in enumerate(zip(outputs, prompts)):
            output_text = self.tokenizer.decode(output, skip_special_tokens=True)
            if output_text.startswith(prompt):
                output_text = output_text[len(prompt):]
            generated_texts.append(output_text)
        
        return generated_texts
