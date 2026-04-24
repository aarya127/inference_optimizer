# AMIO — Adaptive Multimodal Inference Optimizer

A hardware-aware inference scheduling and optimisation framework for **SmolVLM multimodal inference on Apple M3**, spanning eight engineering phases from baseline profiling through a production-grade OpenAI-compatible API.

> **Hardware target**: Apple M3, 8 GB unified memory, 100 GB/s bandwidth, 38 GPU SMs  
> **Model**: `mlx-community/SmolVLM-Instruct-4bit` (SigLIP 27-layer vision encoder + 500 M-param LM)  
> **Framework**: [MLX](https://github.com/ml-explore/mlx) (native Apple Silicon)

---

## Results at a glance

| Metric | Static Baseline | AMIO Adaptive | Improvement |
|--------|----------------|---------------|-------------|
| TTFT (512 px, idle) | 8,536 ms | **349 ms** | **24.4× faster** |
| SLA pass rate (≤ 500 ms) | 4.7% | **55.6%** | **+50.9 pp** |
| KV fragmentation | 62.8% | **5.3%** | **−57.5 pp** |
| Avg quality score | 10.25 | **0.83** (vs Greedy 0.11) | **7.5× vs fast baseline** |
| Cost model accuracy | — | MAPE **2.55%**, R² **0.9978** | — |

The TTFT reduction follows a four-step latency journey:

| Optimisation stage | TTFT |
|--------------------|------|
| Ph 1 Baseline (24 crops, no opt) | 8,536 ms |
| + Ph 3 Adaptive crop scaling (9 crops) | 3,208 ms |
| + Ph 4 ParVTS token pruning (keep=0.75) | 2,980 ms |
| + Ph 6/7 AMIO full adaptive (1 crop, idle, TP) | **349 ms** |

---

## Project structure

```
inference_optimizer/
├── baseline/                       # Phase 1 — profiling & measurement
│   ├── baseline_bottleneck_analysis.ipynb
│   ├── run_experiments.py
│   └── results.json
│
├── model_calibration/              # Phase 2 — hardware cost model
│   ├── calibrate_cost_model.py
│   ├── cost_model.py               # T_prefill = γN² + βN + α  (R²=0.9978)
│   └── calibration_results.json   # 7 calibration points, γ=2.096e-5
│
├── simulation/                     # Phases 3–6 — analytical engines
│   ├── parallelism_engine.py       # Phase 3: TP/DP mode selector
│   ├── tp_simulator.py             # Phase 3: TP/DP/HYBRID comm cost
│   ├── resolution_scaler.py        # Phase 3: crop-count Pareto curve
│   ├── sm_orchestrator.py          # Phase 3+4: M3 SM partitioning
│   ├── kv_manager.py               # Phase 4: contiguous vs paged KV backends
│   ├── w4a8_quantizer.py           # Phase 4: W4A8 + GAR quantisation model
│   ├── vision_offloader.py         # Phase 4: Nova async layer swapping
│   ├── batching_engine.py          # Phase 5: continuous batching + SJF scheduler
│   └── controller.py               # Phase 6: AdaptiveController + Nova SM reallocator
│
├── integrated_service.py           # Phase 7: production orchestrator + OpenAI API
│
├── evaluation/
│   └── generate_report.py          # Phase 8: figures + technical report generator
│
├── figures/                        # Generated evaluation figures (Phase 8)
│   ├── fig1_latency_waterfall.png
│   ├── fig2_strategy_heatmaps.png
│   ├── fig3_pareto_curves.png
│   └── fig4_nova_convergence.png
│
├── FINAL_REPORT.md                # Full technical report
├── config/                         # YAML configs (model, SLA, TP simulation)
├── metrics/                        # SLA validator, metric collector
├── models/                         # MLX multimodal loader, quantisation helpers
├── scripts/                        # Setup and verification scripts
├── tests/                          # Integration smoke tests
└── docs/                           # Design docs and analysis notes
```

---

## Phase summaries

### Phase 1 — Baseline profiling
Profiled SmolVLM end-to-end on M3 hardware. Identified the **dual bottleneck**: vision encoding (SigLIP, 27 layers) at **5,991 ms** and transformer prefill at **2,499 ms**, totalling **8,489 ms** measured TTFT against a 500 ms SLA target. KV fragmentation at 22.2% under static contiguous allocation.

### Phase 2 — Cost model calibration
Ran 7 controlled prefill timing experiments (N = 128–1,548 tokens) and fitted a quadratic latency model:

```
T_prefill(N) = γN² + βN + α
             = 2.096e-5·N²  +  1.591·N  −  20.08  ms
```

Achieves **R² = 0.9978, MAPE = 2.55%** on held-out points. This model powers every downstream latency prediction in the controller.

### Phase 3 — Parallelism engine
Compared TP (tensor-parallel) and DP (data-parallel) communication modes on M3 unified memory — TP allreduce approaches zero-copy due to shared DRAM. Built a crop-count Pareto resolver (`resolution_scaler.py`) mapping image resolution → SigLIP crop grid {224→1, 448→6, 512→9, 1512→24}. HYBRID mode yields ~49.8% latency gain over pure DP.

### Phase 4 — Memory-bound decode optimisation

| Module | What it models | Key result |
|--------|---------------|------------|
| `kv_manager.py` | Contiguous vs paged KV backends | Fragmentation: **62.8% → 5.3%** |
| `w4a8_quantizer.py` | W4A8 + GAR quantisation roofline | KV bytes/token: **110,592 → 27,648** (4×) |
| `vision_offloader.py` | Nova async layer swapping | 189.8 MB freed, 2,787× BW margin |
| `sm_orchestrator.py` | Batch expansion ceiling | Paged W4 KV: **5.4× more concurrent seqs** |

### Phase 5 — Continuous batching engine
Iteration-level event-driven scheduler in `batching_engine.py` with SJF + anti-starvation:

| Scheduler | Throughput | GPU util | p99 TTFT |
|-----------|-----------|---------|---------|
| **Continuous SJF** | 0.153 req/s | **100%** | 12,884 ms |
| Continuous FCFS | 0.152 req/s | 100% | 12,936 ms |
| Static FCFS (baseline) | 0.094 req/s | 87.6% | 20,999 ms |

**1.63× throughput** and **−39% p99 TTFT** vs static FCFS at λ = 50 req/s.

### Phase 6 — Adaptive controller + Nova SM reallocator
`simulation/controller.py` is the intelligence layer. At each request admission, `AdaptiveController.optimize()` enumerates **96 candidate strategies** (8 crop counts × 6 keep-ratios × 2 parallelism modes) and selects the Pareto-optimal feasible plan subject to SLA and memory constraints.

**Nova SM heuristic** — dynamically returns SMs to the vision encoder as the decode queue grows:

```
SM_decode = max(4,  30 − floor(2.0 × max(0, N_pending − 1)))
SM_vision = 38 − SM_decode
```

At idle (`N_decoding = 0`): all 38 SMs go to vision. During a 15-request burst the allocator drives `sm_vision` down to 8 SMs, then recovers back to 38 once the queue clears.

### Phase 7 — Integrated service + OpenAI API
`integrated_service.py` wires all phases into a production-grade `SystemOrchestrator` with three daemon worker threads (Vision → Prefill → Decode) connected by non-blocking queues, plus an **OpenAI Chat Completions-compatible HTTP API**:

```
POST /v1/multimodal/chat/completions
```

Response includes six `X-AMIO-*` telemetry headers (TTFT, crops, tokens, SM allocation, parallelism mode, SLA pass/fail) and a `_amio_telemetry` field in the JSON body.

### Phase 8 — Evaluation & report
`evaluation/generate_report.py` produces four publication-quality figures and `FINAL_REPORT.md`:

| Figure | Content |
|--------|---------|
| `fig1_latency_waterfall.png` | Four-stage TTFT reduction: 8,536 ms → 349 ms |
| `fig2_strategy_heatmaps.png` | Crop count and parallelism mode across resolution × queue depth |
| `fig3_pareto_curves.png` | Quality × latency Pareto frontier for 3 systems |
| `fig4_nova_convergence.png` | Nova SM reallocation timeline during a 15-request burst |

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full evaluation (generates figures + FINAL_REPORT.md)
python evaluation/generate_report.py

# Run the Phase 7 benchmark matrix (144 scenarios, 3 systems)
python integrated_service.py

# Start the OpenAI-compatible API server
python integrated_service.py --api

# Run individual phase engines
python simulation/batching_engine.py --n 2000 --rate 50.0
python simulation/controller.py       # self-test + optimisation table
python simulation/kv_manager.py
python simulation/w4a8_quantizer.py
python simulation/parallelism_engine.py
```

---

## Key constants

| Constant | Value | Source |
|----------|-------|--------|
| Vision encoder (24 crops, full SMs) | 5,991 ms | Phase 1 measurement |
| KV bytes / token (FP16) | 110,592 B | 2 × 24 layers × 1152 hidden × 2 B |
| KV bytes / token (W4) | 27,648 B | 4× compression |
| KV pool budget | 5,694 MB | 8192 − model weights − OS overhead |
| Cost model γ | 2.096 × 10⁻⁵ ms/tok² | Phase 2 calibration |
| Cost model R² | 0.9978 | Phase 2 calibration |
| Cost model MAPE | 2.55% | Phase 2 validation |
| Nova SM_op | 30 | Phase 6 heuristic |
| Nova α | 2.0 | Phase 6 heuristic |
| TBT model | 83.75 + 3.95·B ms | Phase 5 calibration |
| SLA budget | 500 ms TTFT | System-wide target |

---

## Documentation

- [FINAL_REPORT.md](FINAL_REPORT.md) — full 8-section technical report with cost model derivation, ablation study, and experimental results
- [docs/DESIGN.md](docs/DESIGN.md) — architecture and design decisions
- [docs/INSTALL.md](docs/INSTALL.md) — environment setup
