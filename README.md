# AMIO — Adaptive Multimodal Inference Optimizer

An analytical simulation framework for optimising **SmolVLM multimodal inference on Apple M3**, progressing through five engineering phases: baseline measurement, cost-model calibration, parallelism planning, memory-bound decode optimisation, and continuous batching.

> **Hardware target**: Apple M3, 8 GB unified memory, 100 GB/s bandwidth, 38 GPU SMs  
> **Model**: `mlx-community/SmolVLM-Instruct-4bit` (SigLIP vision encoder + 500 M-param LM)  
> **Framework**: [MLX](https://github.com/ml-explore/mlx) (native Apple Silicon)

---

## Baseline results (Phase 1)

| Metric | Measured | SLA target | Gap |
|---|---|---|---|
| TTFT | 8,489 ms | 500 ms | **17×** |
| Vision encoder | 5,991 ms (71% of TTFT) | — | dominant bottleneck |
| Decode throughput | 0.9 tok/s | — | memory-bandwidth bound |
| KV fragmentation | 22.2% | < 4% | static allocation waste |

---

## Project structure

```
inference_optimizer/
├── baseline/                       # Phase 1 — measurement & profiling
│   ├── baseline_bottleneck_analysis.ipynb
│   ├── run_experiments.py
│   └── results.json                # ground-truth measurements
│
├── model_calibration/              # Phase 2 — cost model
│   ├── calibrate_cost_model.py
│   ├── cost_model.py               # T_prefill = γN² + βN + α  (R²=0.9978)
│   └── calibration_results.json
│
├── simulation/                     # Phases 3–5 — analytical engines
│   ├── tp_simulator.py             # Phase 3: TP / DP / HYBRID comm cost
│   ├── resolution_scaler.py        # Phase 3: crop-count Pareto curve
│   ├── sm_orchestrator.py          # Phase 3+4: M3 SM partitioning + batch expansion
│   ├── parallelism_engine.py       # Phase 3: InferenceExecutionPlan orchestrator
│   ├── kv_manager.py               # Phase 4: contiguous vs paged KV backends
│   ├── w4a8_quantizer.py           # Phase 4: W4A8 + GAR quantisation model
│   ├── vision_offloader.py         # Phase 4: Nova-style async layer swapping
│   └── batching_engine.py          # Phase 5: continuous batching + SJF scheduler
│
├── config/                         # YAML configs (model, SLA, TP simulation)
├── metrics/                        # SLA validator, metric collector
├── models/                         # MLX multimodal loader, quantisation helpers
├── scripts/                        # Setup and verification scripts
├── tests/                          # Integration smoke tests
├── docs/                           # Design docs + analysis notes
│   ├── DESIGN.md
│   ├── INSTALL.md
│   ├── bottlenecks.txt
│   ├── kv_cache.txt
│   ├── hybrid_parallelism_tools.txt
│   └── controller_algorithm_API_control.txt
├── requirements.txt
└── scope/venv_phase0/              # MLX runtime virtual environment
```

---

## Phase summaries

### Phase 1 — Baseline
Profiled SmolVLM end-to-end via `baseline/baseline_bottleneck_analysis.ipynb`.  Identified the vision encoder (SigLIP, 27 layers) as the dominant bottleneck at 5,991 ms and measured KV fragmentation of 22.2% under static contiguous allocation.

### Phase 2 — Cost model calibration
Fitted a two-component latency model from 7 calibration points (128–1548 tokens):

```
T_TTFT = T_vision_encoder + γ·N² + β·N + α
       = 5991 ms  +  2.096e-5·N²  +  1.591·N  −  20.08
```

Achieves R² = 0.9978.  Linear regime dominates for N ≤ 1548 (quadratic term < 2%).

### Phase 3 — Parallelism engine
Compared TP / DP / HYBRID communication modes on M3 unified memory (TP comm ≈ free due to zero-copy).  Built a drain-first SM partitioner (`sm_orchestrator.py`) and a crop-count Pareto resolver (`resolution_scaler.py`).  HYBRID mode recommended: ~49.8% latency gain.

### Phase 4 — Memory-bound decode optimisation
Four deliverables targeting KV cache efficiency and decode throughput:

| Module | What it models | Key result |
|---|---|---|
| `kv_manager.py` | Contiguous vs paged KV backends | Fragmentation: 24.4% → **0.26%** (paged) |
| `w4a8_quantizer.py` | W4A8 + GAR quantisation roofline | TBT: 217.7 ms → **87.7 ms** (2.5× gain) |
| `vision_offloader.py` | Nova async layer swapping | 189.8 MB freed, **2787× BW margin**, zero stall |
| `sm_orchestrator.py` | Batch expansion ceiling | Paged W4 KV: **5.4× more concurrent seqs** |

### Phase 5 — Continuous batching engine
Iteration-level scheduler in `simulation/batching_engine.py` with three modes:

| Scheduler | Throughput | GPU decode util | p99 TTFT |
|---|---|---|---|
| **Continuous SJF** | 0.153 req/s | **100%** | 12,884 s |
| Continuous FCFS | 0.152 req/s | 100% | 12,936 s |
| Static FCFS (baseline) | 0.094 req/s | 87.6% | 20,999 s |

**Continuous SJF vs Static FCFS** (2000-request Poisson burst, λ = 50 req/s):
- **1.63× throughput gain**
- **−39% p99 TTFT**
- **+12.4 pp GPU utilisation**
- Avg TBT: 85.3 ms vs 162.7 ms

---

## Quick start

```bash
# Install dependencies (MLX + numpy + scipy)
pip install -r requirements.txt

# Verify Phase 0 environment
python scripts/verify_phase0.py

# Run the Phase 5 continuous batching simulation
python simulation/batching_engine.py --n 200 --rate 2.0

# Run the 2000-request Poisson burst (takes ~2s)
python simulation/batching_engine.py --n 2000 --rate 50.0

# Run individual phase engines
python simulation/kv_manager.py
python simulation/w4a8_quantizer.py
python simulation/vision_offloader.py
python simulation/sm_orchestrator.py
python simulation/parallelism_engine.py
```

---

## Key numerical constants

| Constant | Value | Source |
|---|---|---|
| KV bytes / token (FP16) | 110,592 B | 2 × 24 layers × 1152 hidden × 2 B |
| KV bytes / token (W4) | 27,648 B | 4× reduction |
| KV pool budget | 5,694 MB | 8192 − 250 − 200 − 2048 MB |
| Max concurrent seqs (paged W4) | 140 | KV budget ÷ avg seq size |
| TBT at batch=1 (W4A8+GAR) | 87.7 ms | Phase 4 roofline |
| Vision encoder (baseline) | 5,991 ms | Phase 1 measurement |
| Cost model R² | 0.9978 | Phase 2 calibration |

---

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — architecture and design decisions
- [docs/INSTALL.md](docs/INSTALL.md) — environment setup
- [ARCHITECTURE.md](ARCHITECTURE.md) — module dependency diagram
- [QUICKREF.md](QUICKREF.md) — command cheat-sheet
