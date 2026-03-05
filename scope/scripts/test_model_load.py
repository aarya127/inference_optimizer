#!/usr/bin/env python3
"""
Test script to download and load LLaVA model with MLX
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import time

def main():
    print("=" * 80)
    print("AMIO Phase 0 - Model Loading Test")
    print("=" * 80)
    print()
    
    # Model configuration
    model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
    
    print(f"📦 Loading model: {model_id}")
    print(f"   This will download ~14GB (first time only)...")
    print()
    
    try:
        start_time = time.time()
        
        # Load model with 4-bit quantization
        print("⏳ Loading with 4-bit quantization...")
        model, processor = load(
            model_id,
            quantize=True,  # Enable quantization
        )
        
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print()
        
        # Get model size estimate
        print("📊 Model Statistics:")
        print(f"   Model ID: {model_id}")
        print(f"   Quantization: 4-bit (INT4)")
        print(f"   Device: {mx.default_device()}")
        
        # Test a simple prompt
        print()
        print("🧪 Testing inference...")
        
        test_prompt = "What is the capital of France?"
        
        # Generate response
        start_time = time.time()
        output = model.generate(
            processor(test_prompt, return_tensors="mlx"),
            max_new_tokens=50,
            temperature=0.7,
        )
        inference_time = time.time() - start_time
        
        response = processor.decode(output[0], skip_special_tokens=True)
        
        print(f"   Prompt: {test_prompt}")
        print(f"   Response: {response}")
        print(f"   Inference time: {inference_time:.2f}s")
        print()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  - Test with image inputs")
        print("  - Run full benchmark suite")
        print("  - Measure TTFT and TBT metrics")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print()
        print("Troubleshooting:")
        print("  - Ensure you have enough disk space (~14GB)")
        print("  - Check internet connection for download")
        print("  - Verify Hugging Face access (model may require authentication)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
