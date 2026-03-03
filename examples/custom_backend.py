"""
Example of custom backend implementation.
This demonstrates how to add support for a new inference engine.
"""

from typing import List
from src.backends.base import BaseBackend, ModelConfig


class CustomBackend(BaseBackend):
    """Custom backend implementation example."""
    
    def __init__(self, config: ModelConfig):
        """Initialize custom backend.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.custom_engine = None
    
    def load_model(self) -> None:
        """Load model using custom engine.
        
        This is where you would:
        1. Initialize your custom inference engine
        2. Load the model weights
        3. Configure optimizations
        4. Load the tokenizer
        """
        # Example pseudo-code:
        # from my_custom_engine import InferenceEngine
        # 
        # self.custom_engine = InferenceEngine(
        #     model_path=self.config.model_name,
        #     quantization=self.config.quantization,
        #     max_batch_size=self.config.max_batch_size,
        # )
        # 
        # self.tokenizer = load_tokenizer(self.config.model_name)
        # self._is_loaded = True
        
        raise NotImplementedError("Implement model loading for your custom backend")
    
    def unload_model(self) -> None:
        """Unload model and free resources."""
        if self.custom_engine is not None:
            # Clean up your engine
            del self.custom_engine
            self.custom_engine = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._is_loaded = False
        
        # Force cleanup
        import gc
        gc.collect()
    
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
        
        This is where you implement single request generation.
        """
        # Example pseudo-code:
        # input_ids = self.tokenizer.encode(prompt)
        # output_ids = self.custom_engine.generate(
        #     input_ids,
        #     max_tokens=max_new_tokens,
        #     temperature=temperature,
        #     top_p=top_p,
        #     top_k=top_k,
        # )
        # generated_text = self.tokenizer.decode(output_ids)
        # return generated_text
        
        raise NotImplementedError("Implement generation for your custom backend")
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts.
        
        This is where you implement batch generation.
        If your engine doesn't support batching, you can fall back to:
        return [self.generate(p, max_new_tokens, **kwargs) for p in prompts]
        """
        # Example pseudo-code:
        # input_ids_batch = [self.tokenizer.encode(p) for p in prompts]
        # output_ids_batch = self.custom_engine.batch_generate(
        #     input_ids_batch,
        #     max_tokens=max_new_tokens,
        #     **kwargs
        # )
        # return [self.tokenizer.decode(ids) for ids in output_ids_batch]
        
        raise NotImplementedError("Implement batch generation for your custom backend")


def register_custom_backend():
    """Register custom backend with the framework."""
    from src.backends import BACKEND_REGISTRY
    
    BACKEND_REGISTRY["custom"] = CustomBackend
    print("Custom backend registered! Use with: --backend custom")


# Usage example
if __name__ == "__main__":
    # Register the backend
    register_custom_backend()
    
    # Now you can use it in benchmarks
    from src.benchmarks import BenchmarkRunner, BenchmarkConfig
    
    config = BenchmarkConfig(
        model_name="your-model",
        backends=["custom"],
        quantizations=["fp16"],
        batch_sizes=[1, 4],
        num_requests=50,
    )
    
    runner = BenchmarkRunner(config)
    results = runner.run()
