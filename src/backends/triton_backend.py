"""Triton Inference Server backend implementation."""

from typing import List, Optional
import numpy as np
from src.backends.base import BaseBackend, ModelConfig


class TritonBackend(BaseBackend):
    """Triton Inference Server backend."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Triton backend.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.triton_client = None
        self.model_name_triton = None
    
    def load_model(self) -> None:
        """Connect to Triton Inference Server."""
        try:
            import tritonclient.grpc as grpcclient
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "Triton client not installed. Install with: pip install tritonclient[all]"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Connect to Triton server
        # Default Triton server URL
        triton_url = "localhost:8001"
        
        try:
            self.triton_client = grpcclient.InferenceServerClient(url=triton_url)
            
            # Use model name for Triton (may differ from HF model name)
            self.model_name_triton = self.config.model_path or self.config.model_name.split("/")[-1]
            
            # Check if model is loaded
            if not self.triton_client.is_model_ready(self.model_name_triton):
                raise RuntimeError(
                    f"Model {self.model_name_triton} not ready on Triton server. "
                    "Please ensure the model is deployed."
                )
            
            self._is_loaded = True
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Triton server at {triton_url}. "
                f"Please ensure Triton is running. Error: {e}"
            )
    
    def unload_model(self) -> None:
        """Disconnect from Triton server."""
        if self.triton_client is not None:
            self.triton_client.close()
            self.triton_client = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._is_loaded = False
    
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
        import tritonclient.grpc as grpcclient
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        
        # Prepare inputs for Triton
        input_ids_np = np.array([input_ids], dtype=np.int32)
        max_tokens_np = np.array([[max_new_tokens]], dtype=np.int32)
        temperature_np = np.array([[temperature]], dtype=np.float32)
        top_p_np = np.array([[top_p]], dtype=np.float32)
        top_k_np = np.array([[top_k]], dtype=np.int32)
        
        inputs = [
            grpcclient.InferInput("input_ids", input_ids_np.shape, "INT32"),
            grpcclient.InferInput("max_tokens", max_tokens_np.shape, "INT32"),
            grpcclient.InferInput("temperature", temperature_np.shape, "FP32"),
            grpcclient.InferInput("top_p", top_p_np.shape, "FP32"),
            grpcclient.InferInput("top_k", top_k_np.shape, "INT32"),
        ]
        
        inputs[0].set_data_from_numpy(input_ids_np)
        inputs[1].set_data_from_numpy(max_tokens_np)
        inputs[2].set_data_from_numpy(temperature_np)
        inputs[3].set_data_from_numpy(top_p_np)
        inputs[4].set_data_from_numpy(top_k_np)
        
        # Define outputs
        outputs = [grpcclient.InferRequestedOutput("output_ids")]
        
        # Inference
        response = self.triton_client.infer(
            model_name=self.model_name_triton,
            inputs=inputs,
            outputs=outputs,
        )
        
        # Get output
        output_ids = response.as_numpy("output_ids")[0]
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Remove prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
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
        import tritonclient.grpc as grpcclient
        
        # Tokenize all prompts
        input_ids_list = [self.tokenizer.encode(prompt) for prompt in prompts]
        
        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids_padded = [
            ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
            for ids in input_ids_list
        ]
        
        input_ids_np = np.array(input_ids_padded, dtype=np.int32)
        max_tokens_np = np.array([[max_new_tokens]] * len(prompts), dtype=np.int32)
        temperature_np = np.array([[kwargs.get("temperature", 0.8)]] * len(prompts), dtype=np.float32)
        top_p_np = np.array([[kwargs.get("top_p", 0.95)]] * len(prompts), dtype=np.float32)
        top_k_np = np.array([[kwargs.get("top_k", 50)]] * len(prompts), dtype=np.int32)
        
        inputs = [
            grpcclient.InferInput("input_ids", input_ids_np.shape, "INT32"),
            grpcclient.InferInput("max_tokens", max_tokens_np.shape, "INT32"),
            grpcclient.InferInput("temperature", temperature_np.shape, "FP32"),
            grpcclient.InferInput("top_p", top_p_np.shape, "FP32"),
            grpcclient.InferInput("top_k", top_k_np.shape, "INT32"),
        ]
        
        inputs[0].set_data_from_numpy(input_ids_np)
        inputs[1].set_data_from_numpy(max_tokens_np)
        inputs[2].set_data_from_numpy(temperature_np)
        inputs[3].set_data_from_numpy(top_p_np)
        inputs[4].set_data_from_numpy(top_k_np)
        
        outputs = [grpcclient.InferRequestedOutput("output_ids")]
        
        # Batch inference
        response = self.triton_client.infer(
            model_name=self.model_name_triton,
            inputs=inputs,
            outputs=outputs,
        )
        
        # Get outputs
        output_ids_batch = response.as_numpy("output_ids")
        
        # Decode
        generated_texts = []
        for i, (output_ids, prompt) in enumerate(zip(output_ids_batch, prompts)):
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]
            generated_texts.append(text)
        
        return generated_texts
