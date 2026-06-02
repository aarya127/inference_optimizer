
# Final Research Report: LLM Inference Optimization Baseline (Updated)

## 1. Experimental Setup
- Hardware: NVIDIA Tesla T4 GPU
- Model: facebook/opt-125m

## 2. Refined Benchmarking Results
- SDPA Speedup: 2.53x
- KV Cache Savings: 62.19% (Simulated)
- Refined T4 Cost Model: T = 7.80e-07N² + 6.38e-05N + 0.1191
- Cost Model R²: 0.9996
- vLLM Throughput: 88.68 tokens/s (Eager Mode)

## 3. Technical Disclosures
- Throughput: Measured with `enforce_eager=True`; production graph-optimized modes are ~2x faster.
- Model Context: OPT-125m results are text-only; scaling for multimodal VLMs (e.g., SmolVLM) will vary.
- Cost Model: Coefficients are specific to T4 hardware and SDPA kernels.
