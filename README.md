# AMIO — Adaptive Multimodal Inference Optimizer

AMIO is a research simulation of adaptive multimodal inference for
`mlx-community/SmolVLM-Instruct-4bit` on an Apple M3 with 8 GB unified memory.
It combines a small real MLX measurement campaign with analytical models for
crop selection, token pruning, KV allocation, batching, and stage scheduling.

> **Disclosure:** Most headline comparisons in this repository are SIMULATED
> outputs parameterized by four stage-isolated MLX measurements in
> `baseline/results_v2.json`. The report and service distinguish MEASURED from
> SIMULATED values. The service loads no model and returns simulated telemetry.
> The 500 ms TTFT SLA is infeasible at every real crop setting on this hardware;
> that is a finding, not a hidden or relaxed constraint.

**Target:** Apple M3 base die, 10 real GPU cores, 8 GB unified memory,
100 GB/s bandwidth. The scheduler also uses 38 abstract modeled
compute-scheduling shares; these are priority units, not Apple GPU cores or
hardware partitions.

## Results at a glance

| Metric | Corrected result | Provenance |
|--------|------------------|------------|
| TTFT crop range | 17 crops: **14,395 ms** → 1 crop: **897 ms** (**16.0×**) | MEASURED stage sums; less visual input, not same-work acceleration |
| 500 ms SLA pass rate | Static **0.0%**, AMIO **0.0%** | SIMULATED 24-point comparison; all real settings are infeasible |
| KV allocator waste | Mean **65.8%** contiguous → **3.98%** paged | SIMULATED comparison of two assumed allocator policies |
| Quality proxy | Static **5.25**, AMIO **0.11** | SIMULATED `n_crops × keep_ratio`; not accuracy |
| Prefill fit | R² **0.9996**, MAPE **3.01%** in-sample; LOOCV MAPE **20.7%** | Four measured points only |

There is no VQA, TextVQA, or other task-accuracy benchmark in this repository.
The “quality” value used by the simulator measures retained input compute, not
answer correctness.

The report generator’s standardized prompt scenario produces this computed
latency journey:

| Optimization stage | TTFT | Provenance |
|--------------------|------|------------|
| 17-crop baseline | 13,705 ms | Analytical stage models |
| 5-crop setting | 3,791 ms | Analytical stage models |
| 5 crops + ParVTS, keep=0.75 | 3,605 ms | Analytical; includes 28 ms modeled migration |
| 1-crop configuration | 977 ms | Analytical; still fails the 500 ms SLA |

These values differ slightly from the measured 14,395/897 ms anchors because
the waterfall uses a standardized 32-token prompt and model-derived token
counts.

## What is measured

`baseline/measure_v2.py` directly isolates the vision tower/connector and LM
prefill with `mx.eval` on real outputs. It records five trials at the only four
processor crop settings: 1, 5, 10, and 17 crops.

- Vision: `T_vision(c) = 553.5c + 27.4 ms`.
- LM prefill:
  `T_prefill(N) = 0.001170N² + 1.2073N + 244.6 ms`,
  fit at N = 100, 466, 922, and 1,560.
- Batch-1 decode:
  `TBT(ctx) = 18.33 + 0.0032 × ctx_tokens ms`.
- Measured stage-sum TTFT: 897 ms at 1 crop and 14,395 ms at 17 crops.

Batch scaling beyond one sequence, ParVTS migration, allocator
fragmentation, scheduling behavior, A100 performance, and all three-system
comparisons remain modeled.

## Repository layout

```text
baseline/
  measure_v2.py, results_v2.json   Corrected MLX measurement campaign
  measure_llamacpp.py              Second model+backend: Qwen2.5-0.5B/llama.cpp
  measure_quantization_tradeoff.py Real fp16/Q8_0/Q4_K_M measurement (size,
                                   speed, perplexity)
  measure_llamacpp_concurrency.py  Real concurrent-decode measurement vs
                                   ConcurrencyThroughputTechnique's MODELED
                                   prediction (Tier 6) -- see
                                   core/TIER6_REAL_CONCURRENCY_FINDINGS.md
  measure_llamacpp_native_batching.py  Same check via real native
                                   multi-sequence batching (llama_batch
                                   API) -- see core/TIER6_REAL_CONCURRENCY_FINDINGS.md
  measure_llamacpp_prefix_cache.py Real prefix/prompt-cache reuse savings;
                                   also where a real GPU-sync bug was found
                                   (Tier 7) -- see core/TIER7_PREFIX_CACHE_FINDINGS.md
model_calibration/
  cost_model.py                    Measured fit + modeled extensions
  fit_llamacpp_prefill.py          Cost-model fit for the llama.cpp profile
core/
  profile.py, technique.py         Per-(model,backend) profiles + the
                                   Technique interface (Tier 3) — see
                                   core/README.md
  techniques/                     PrefillBudgetTechnique,
                                   QuantizationLevelTechnique,
                                   KVCacheBudgetTechnique,
                                   ConcurrencyThroughputTechnique (MODELED,
                                   not measured -- see core/README.md),
                                   ThroughputOptimalBatchTechnique,
                                   PrefixCacheTechnique (last two REAL-data-
                                   only, Tiers 6-7)
  executor.py, executors/         Real execution: recommend -> run for
                                   real -> compare predicted vs measured
                                   (Tier 4) -- see core/TIER4_CLOSED_LOOP_FINDINGS.md
  optimize.py                     CLI: closed-loop recommend+execute+verify
  calibrate.py, calibrate_mlx.py,  Automated onboarding for llama.cpp,
  calibrate_mlx_vlm.py             MLX-LM, and MLX-VLM models (Tier 5) --
                                   see core/TIER5_AUTO_ONBOARDING_FINDINGS.md
simulation/
  controller.py                    48-strategy adaptive controller
  resolution_scaler.py             Four real crop settings
  kv_manager.py                    Assumed contiguous/paged allocators
  batching_engine.py               Event-driven continuous batching
  sm_orchestrator.py               Abstract compute-share scheduler
  parallelism_engine.py            Cost-neutral TP/DP annotation on one GPU
  w4a8_quantizer.py                Quantization roofline model
  vision_offloader.py              Offload/scheduling model
integrated_service.py              Simulation service and benchmark matrix
evaluation/
  generate_report.py               Figures 1–4 and FINAL_REPORT.md
  vllm_baseline.py                 Figure 5 and modeled A100 extension
amio_constants.py                  Architecture and latency source of truth
REVIEW_FINDINGS.md                 Original honesty/correctness review
```

## Phase summary

### Phase 1–2: measurement and calibration

The corrected campaign replaces the earlier unexplained 5,991 ms “vision”
residual and synthetic-embedding prefill fit. The four-point quadratic has
excellent in-sample fit but 20.7% leave-one-out MAPE. With only four points and
three fit parameters, LOOCV is a weak generalization estimate, and the model
should not be trusted outside its measured range without more data.

### Phase 3: crop scaling and parallelism

The processor exposes `{384: 1, 768: 5, 1152: 10, 1536: 17}` as
`longest_edge → crops`. There is no 24-crop setting. The controller retains TP
and DP labels, but single-chip tensor parallelism is cost-neutral because the
target has one GPU; the two modes create 48 enumerated strategies but only 24
cost-distinct choices.

### Phase 4: memory models

The LM KV derivation is:

```text
2 × 24 layers × 32 KV heads × 64 head dimension × 2 bytes = 196,608 B/token
```

Modeled W4 KV is 49,152 B/token. The shipped checkpoint already has 4-bit
weights, so the repository does not claim a second measured weight-compression
speedup. Paged-vs-contiguous fragmentation is an assumed-policy comparison,
not a measurement of MLX’s allocator.

### Phase 5–6: batching and adaptive control

The event simulator models continuous batching, SJF scheduling, and Nova-style
stage allocation. “38 shares” means abstract compute priority; Metal exposes no
per-task M3 core partitioning. Under the operational 500 ms SLA, the controller
returns an explicit safe-minimal fallback with `sla_pass=False` rather than
claiming success.

ParVTS migration is also modeled. Its cost increases with migration depth and
pruning magnitude; large prunes can add hundreds of milliseconds.

### Phase 7: simulation API

`integrated_service.py` uses four daemon workers: Vision, Prefill, Decode, and
Collector. It loads no MLX model; stage latency is produced by analytical
formulas plus noise. Responses carry explicit `"simulated": true` fields and
`X-AMIO-Simulated-*` telemetry. The current implementation emits ten
`X-AMIO-*` response headers.

### Phase 8: evaluation

`evaluation/generate_report.py` creates four figures and `FINAL_REPORT.md`.
`evaluation/vllm_baseline.py` adds a fifth figure and an A100/vLLM analytical
projection. The A100 crop-endpoint ratio is preserved by applying the same
hardware scaling factor to both M3 endpoints; this is algebra, not independent
evidence of hardware portability.

## Quick start

Use the repository environment:

```bash
# Generate figures 1–4 and FINAL_REPORT.md
scope/venv_phase0/bin/python evaluation/generate_report.py

# Generate figure 5 and append/replace the A100 projection
scope/venv_phase0/bin/python evaluation/vllm_baseline.py

# Run the 144-row benchmark matrix:
# 4 resolutions × 4 concurrencies × 3 SLA budgets × 3 systems
scope/venv_phase0/bin/python integrated_service.py

# Run regression tests
scope/venv_phase0/bin/python -m pytest tests/test_phase4.py -v

# Start the explicitly simulated API scaffold
scope/venv_phase0/bin/python integrated_service.py --api
```

## Key constants

| Constant | Value | Status/source |
|----------|-------|---------------|
| Real crop settings | `{1, 5, 10, 17}` | Processor + measurement campaign |
| Vision model | `553.5 ms/crop + 27.4 ms` | MEASURED |
| LM prefill model | `0.001170N² + 1.2073N + 244.6 ms` | Fit to four MEASURED points |
| Decode TBT, batch=1 | `18.33 + 0.0032 × ctx_tokens ms` | MEASURED; larger batches modeled |
| KV bytes/token, FP16 | 196,608 B | Checkpoint architecture |
| KV bytes/token, W4 | 49,152 B | MODELED 4× reduction |
| KV pool budget | 4,754 MiB | 8,192 − 1,390 weights − 2,048 reserve |
| M3 GPU | 10 real cores | Hardware |
| Scheduler budget | 38 abstract shares | MODELED, not hardware partitions |
| TTFT SLA | 500 ms | Infeasible at every crop setting |

## Documentation

- [FINAL_REPORT.md](FINAL_REPORT.md) — generated technical report
- [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md) — original detailed review
- [baseline/PREFILL_V3_FINDINGS.md](baseline/PREFILL_V3_FINDINGS.md) — an
  attempt to densify the prefill calibration past 4 points; inconclusive and
  not adopted (likely confounded by concurrent load on the measurement
  machine), documented rather than discarded
- [baseline/TIER2_LLAMACPP_FINDINGS.md](baseline/TIER2_LLAMACPP_FINDINGS.md)
  — a second model (Qwen2.5-0.5B) on a second backend (llama.cpp/Metal)
  validates the same quadratic cost-model form with LOOCV MAPE 1.68%
  (5.72% before the sync-audit re-run; vs SmolVLM/MLX's 20.7%), supporting that the 20.7% gap was a data-density
  issue, not a modeling one; also a real, measured fp16/Q8_0/Q4_K_M
  quantization tradeoff (size, speed, perplexity) replacing this study's
  earlier fabricated quantization claims
- [core/README.md](core/README.md) — the Technique abstraction: one
  interface, run against two real calibrated (model, backend) profiles,
  with genuinely different applicability and recommendations per profile
- [core/TIER4_CLOSED_LOOP_FINDINGS.md](core/TIER4_CLOSED_LOOP_FINDINGS.md)
  — recommend, execute for real, compare predicted vs measured: Qwen/
  llama.cpp validated to 0.5%, but SmolVLM/MLX was off by 41-108% against a
  fresh real run despite R²=0.9996 in-sample -- a genuine, reproducible
  finding, not smoothed over. The safety margin now uses max (not mean)
  LOOCV error after a mean-based version was caught missing its SLA
  reproducibly; re-verified across all four profiles, every one that
  returns a token budget now clears its SLA with real headroom, and
  SmolVLM/MLX's margin is now honest enough to correctly report "no
  feasible token count" at 500ms instead of a lucky guess.
- [core/TIER5_AUTO_ONBOARDING_FINDINGS.md](core/TIER5_AUTO_ONBOARDING_FINDINGS.md)
  — a third model (SmolLM2-360M, llama.cpp), a fourth (the same
  Qwen2.5-0.5B already on llama.cpp, this time on MLX), and a fifth
  (SmolVLM-256M, proving the vision side auto-calibrates too) onboarded
  end-to-end with zero hand-written per-model code. The same-model
  cross-backend pair exposed two real bugs (margin basis, an unphysical
  negative decode slope) -- both now fixed, with before/after numbers kept
- [core/TIER6_REAL_CONCURRENCY_FINDINGS.md](core/TIER6_REAL_CONCURRENCY_FINDINGS.md)
  — checks `ConcurrencyThroughputTechnique`'s MODELED concurrency
  prediction against real decode throughput two ways: independent threads
  (~2.8x worse than predicted at B=4) and real native multi-sequence
  batching via llama.cpp's low-level `llama_batch` API (2.28x worse at
  B=4, 4.28x at B=8; ratios corrected by the sync audit) -- real batching helps over the naive proxy but the
  model still substantially underestimates per-sequence compute cost; not
  silently recalibrated to these results, just honestly flagged
- [core/TIER7_PREFIX_CACHE_FINDINGS.md](core/TIER7_PREFIX_CACHE_FINDINGS.md)
  — real measured prefix/prompt-cache reuse savings (46-68% prefill time
  saved depending on suffix length), and more importantly a real
  GPU-dispatch synchronization bug found while measuring it: `llama_decode()`
  can return before its GPU work finishes, silently corrupting timing that
  interleaves different shapes -- the same class of bug, independently
  rediscovered, that originally motivated this project's MLX honesty
  rewrite.
- [core/SYNC_AUDIT_FINDINGS.md](core/SYNC_AUDIT_FINDINGS.md) — the audit of
  every other llama.cpp timing script for that bug. Prefill: not
  materially affected. Decode: inflated up to ~85% (the first decode step
  absorbed the preceding prefill), which created a fake per-context
  decode slope in both llama.cpp profiles and skewed Tier 6's modeled
  baseline. Also found a separate quantization-timing bug (a ~620 MB
  logits copy inside every timed prefill) whose fix reverses the "quant
  speeds up prefill" result. Recalibration then exposed a real cost-model
  blind spot: prefill has n_batch chunk steps the calibration grid never
  samples, which is why the Qwen closed loop now misses its SLA 1 time in 3.
- [docs/DESIGN.md](docs/DESIGN.md) — historical design document; some original
  targets are retained as superseded context
- [docs/INSTALL.md](docs/INSTALL.md) — historical install guide for the
  abandoned LLaVA-1.5-7B design phase; marked stale, use Quick start above
