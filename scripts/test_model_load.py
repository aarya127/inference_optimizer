#!/usr/bin/env python3
"""
Test script to download and load a small VLM model with MLX.

Primary target: Qwen2.5-VL-3B-Instruct-4bit (MLX community checkpoint)
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from mlx_vlm import load, generate


MODEL_CANDIDATES = [
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "mlx-community/SmolVLM-Instruct-4bit",
]

def main():
    print("=" * 80)
    print("AMIO Phase 0 - Model Loading Test")
    print("=" * 80)
    print()
    
    loaded_model_id = None
    model = None
    processor = None
    last_error = None

    for model_id in MODEL_CANDIDATES:
        print(f"📦 Loading model: {model_id}")
        print("   This may download model files on first run...")
        print()
        try:
            start_time = time.time()
            model, processor = load(model_id)
            load_time = time.time() - start_time
            loaded_model_id = model_id
            print(f"✅ Model loaded successfully in {load_time:.2f}s")
            print()
            break
        except Exception as e:
            last_error = e
            print(f"⚠️ Failed to load {model_id}: {e}")
            print("   Trying next candidate...\n")

    if model is None or processor is None:
        print("❌ Could not load any candidate model.")
        print(f"Last error: {last_error}")
        return 1

    print("📊 Model Statistics:")
    print(f"   Model ID: {loaded_model_id}")
    print("   Quantization: pre-quantized checkpoint (4-bit expected)")
    print(f"   Device: {mx.default_device()}")
    print()

    print("🧪 Testing inference...")
    test_prompt = "What is the capital of France? Answer in one short sentence."

    try:
        start_time = time.time()
        result = generate(
            model,
            processor,
            prompt=test_prompt,
            image=None,
            max_tokens=40,
            temperature=0.2,
            verbose=False,
        )
        inference_time = time.time() - start_time

        response = getattr(result, "text", str(result))
        print(f"   Prompt: {test_prompt}")
        print(f"   Response: {response}")
        print(f"   Inference time: {inference_time:.2f}s")
        print()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("Next steps:")
        print("  - Test with an image input")
        print("  - Measure TTFT and TBT in your metrics pipeline")
        return 0
    except Exception as e:
        print(f"❌ Inference failed after successful load: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
