# Inference Optimizer Research Report

> **SUPERSEDED by `cuda_benchmark_report_updated.md`.** That report is a
> later run of the same notebook (`cuda_inference_benchmark.ipynb`) with
> added disclosures (eager-mode caveat, text-only-model caveat, T4-specific
> coefficients). Its cost-model coefficients differ from this report's
> (7.80e-07N² + 6.38e-05N + 0.1191 vs this report's 5.27e-07N² + 5.11e-04N +
> 0.0305) despite both claiming R²=0.9996 — this was never reconciled and
> cannot be re-derived here (the fit cell requires a live CUDA/T4 GPU,
> unavailable in this environment). Treat both coefficient sets as
> illustrative of one live-GPU run each, not a stable, reproducible fit.
> This directory is a standalone T4/OPT-125m benchmark unrelated to the
> main M3/SmolVLM study — see REVIEW_FINDINGS.md.

## Benchmarking Summary
- **vLLM Throughput:** 88.68 tokens/s
- **SDPA Speedup:** 2.53x
- **KV Cache Savings:** 62.19%

## Cost Model
T = 5.27e-07N² + 5.11e-04N + 0.0305
(R² Score: 0.9996)
