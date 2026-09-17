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
                                   ConcurrencyThroughputTechnique
                                   (last one MODELED batch scaling, not
                                   measured -- see core/README.md)
  executor.py, executors/         Real execution: recommend -> run for
                                   real -> compare predicted vs measured
                                   (Tier 4) -- see core/TIER4_CLOSED_LOOP_FINDINGS.md
  optimize.py                     CLI: closed-loop recommend+execute+verify
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
  validates the same quadratic cost-model form with LOOCV MAPE 5.72% (vs
  SmolVLM/MLX's 20.7%), supporting that the 20.7% gap was a data-density
  issue, not a modeling one; also a real, measured fp16/Q8_0/Q4_K_M
  quantization tradeoff (size, speed, perplexity) replacing this study's
  earlier fabricated quantization claims
- [core/README.md](core/README.md) — the Technique abstraction: one
  interface, run against two real calibrated (model, backend) profiles,
  with genuinely different applicability and recommendations per profile
- [core/TIER4_CLOSED_LOOP_FINDINGS.md](core/TIER4_CLOSED_LOOP_FINDINGS.md)
  — recommend, execute for real, compare predicted vs measured: Qwen/
  llama.cpp validated to 0.5%, but SmolVLM/MLX was off by 41-72% against a
  fresh real run despite R²=0.9996 in-sample -- a genuine, reproducible
  finding, not smoothed over
- [docs/DESIGN.md](docs/DESIGN.md) — historical design document; some original
  targets are retained as superseded context
- [docs/INSTALL.md](docs/INSTALL.md) — historical install guide for the
  abandoned LLaVA-1.5-7B design phase; marked stale, use Quick start above
