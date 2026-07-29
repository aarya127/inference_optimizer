"""
integrated_service.py  —  Phase 7: Full System Integration (SIMULATION)

*** SIMULATION SERVICE — NO MODEL IS LOADED. ***
No vision encoder, language model, or tokenizer is ever instantiated by this
file.  Every latency it reports is an analytic cost-model formula evaluated
with Gaussian noise; worker threads sleep for their simulated durations so
that wall-clock timing reflects modeled load, but no inference occurs and
all completion text is a placeholder.

Combines the Phase 6 Adaptive Controller (the "brain") with the Phase 3–5
simulated hardware engines (the "muscle") into a single pipelined service.

Architecture
============

  ┌─────────────────────────────────────────────────────────┐
  │                  SystemOrchestrator                      │
  │                                                          │
  │  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
  │  │  Vision  │   │ Prefill  │   │  Decode  │            │
  │  │  Worker  │──▶│  Worker  │──▶│  Worker  │──▶ Done    │
  │  │ (Thread) │   │ (Thread) │   │ (Thread) │   Queue    │
  │  └──────────┘   └──────────┘   └──────────┘            │
  │       ▲                ▲              ▲          ▲       │
  │  Phase 3 shares  Phase 6 ParVTS  Phase 4 Paged  Collector│
  │                                       KV        (Thread) │
  │  Phase 6 AdaptiveController  (called at admission)       │
  └─────────────────────────────────────────────────────────┘
             ▲
  ┌──────────┴──────────────────────────────────────────────┐
  │   OpenAI-format API  POST /v1/multimodal/chat/...       │
  │   X-AMIO-* telemetry headers + _amio_telemetry field    │
  │   (all telemetry labeled "simulated" — nothing measured │
  │    from a real model)                                    │
  └─────────────────────────────────────────────────────────┘

Four worker threads run the pipeline: VisionWorker, PrefillWorker,
DecodeWorker, and the Collector.  The stage queues are standard blocking
queues (get() blocks until an item or sentinel arrives).

Timing definitions
------------------
  simulated_stage_ttft_ms : sum of the simulated vision + prefill +
                            migration stage formulas (excludes queue wait
                            and the first decode step).
  wall_ttft_ms            : wall-clock time from submit() to collection,
                            i.e. queue wait + slept stage durations + one
                            simulated decode step (first token).  SLA gating
                            uses THIS value.
The pipeline simulates time up to the first token in wall-clock terms; the
remainder of decoding is accounted analytically only.

Run modes
---------
  python integrated_service.py              # benchmark matrix + Pareto analysis
  python integrated_service.py --api        # persistent HTTP API server
  python integrated_service.py --api-test   # API demo (server + 5 requests)
"""

from __future__ import annotations

import argparse
import http.server
import json
import math
import queue
import random
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── project root on path ────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import amio_constants as _C

from simulation.controller import (
    AdaptiveController,
    InferenceRequest,
    SystemState,
    ExecutionPlan,
    KV_QUANT_BITS,
)
from simulation.parallelism_engine import ParallelismMode
from simulation.kv_manager import (
    ContiguousBackend,
    PagedBackend,
    KV_POOL_BUDGET_MB,
    kv_cache_size_mb,
)
from model_calibration.cost_model import CostModel

# ── Shared constants (single source of truth: amio_constants) ──────────────
# MEASURED (baseline/results_v2.json, 2026-07-28): vision cost is linear in
# crop count, decode TBT floor is ~18-25 ms (comfortably under the 80ms TBT
# SLA); the binding, INFEASIBLE constraint is the 500ms TTFT SLA — measured
# vision cost alone at 1 crop (~581 ms) already exceeds it.
SLA_TTFT_MS        = _C.TTFT_SLA_MS               # 500 ms (infeasible on this HW)
TBT_SLA_MS         = _C.TBT_SLA_MS                # 80 ms (comfortably met)
DECODE_OVERHEAD_MS = _C.DECODE_OVERHEAD_MS_MEASURED   # 18.33 measured @ batch=1
DECODE_KV_MS_PER_CTX_TOKEN = _C.DECODE_KV_MS_PER_CTX_TOKEN  # 0.00320 measured
VISION_MS_PER_CROP = _C.VISION_MS_PER_CROP        # 553.5 measured, linear fit
VISION_FIXED_MS    = _C.VISION_FIXED_MS           # 27.4 measured
BASELINE_N_CROPS   = _C.MAX_CROPS                 # 17 (processor max; no "24")
TOKENS_PER_CROP    = _C.TOKENS_PER_CROP           # 81 (from config.json)
M3_TOTAL_SMS       = _C.TOTAL_COMPUTE_SHARES      # 38 modeled shares (not HW SMs)
M3_TOTAL_MEMORY_MB = _C.TOTAL_MEMORY_MB           # 8192

# KV bytes per token at W4 (4-bit) quantization (modeled)
_KV_W4_BYTES = _C.KV_BYTES_PER_TOKEN_W4           # 49,152


def _predict_tbt_ms(ctx_tokens: int, batch: int) -> float:
    """MEASURED batch=1 decode TBT + modeled batch-scaling term.

    TBT(ctx, B) = DECODE_OVERHEAD_MS + DECODE_KV_MS_PER_CTX_TOKEN * ctx
                  + (B - 1) * (ctx * KV_BYTES_PER_TOKEN_FP16 / BW_GBps / 1e6)

    The last term (extra concurrent sequences' KV reads) is a modeling
    assumption, not measured; batch=1 terms are direct measurements from
    baseline/results_v2.json.
    """
    kv_read_ms_per_extra_seq = (
        ctx_tokens * _C.KV_BYTES_PER_TOKEN_FP16 / (_C.M3_MEMORY_BW_GBPS * 1e6)
    )
    return (
        DECODE_OVERHEAD_MS
        + DECODE_KV_MS_PER_CTX_TOKEN * ctx_tokens
        + max(0, batch - 1) * kv_read_ms_per_extra_seq
    )

# Resolution → max crops mapping — MEASURED processor settings (there is no
# "24 crops" mode in this pipeline; see amio_constants.CROP_SETTINGS).
_RESOLUTION_TO_CROPS: Dict[int, int] = dict(_C.CROP_SETTINGS)  # {384:1,768:5,1152:10,1536:17}

_SENTINEL = object()   # sentinel for queue shutdown


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted list (q in 0..1)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


# ===========================================================================
# Pipeline data structures
# ===========================================================================

@dataclass
class TelemetryRecord:
    """Per-request telemetry record — all latencies SIMULATED, none measured
    from a real model."""
    req_id: int
    image_resolution: int
    prompt_length: int
    n_crops: int
    token_keep_ratio: float
    parallelism_mode: str
    sm_vision: int
    sm_decode: int
    is_fallback: bool
    predicted_ttft_ms: float
    simulated_stage_ttft_ms: float   # formula sum (vision+prefill+migration)
    wall_ttft_ms: float              # wall clock submit → first token (SLA-gated)
    ttft_error_ms: float             # simulated stage sum − predicted
    simulated_t_vision_ms: float
    simulated_t_prefill_ms: float
    simulated_t_migration_ms: float
    simulated_tbt_ms: float
    tbt_sla_pass: bool
    kv_alloc_mb: float
    sla_pass: bool                   # gated on wall_ttft_ms
    quality_score: float
    timestamp_ms: float


@dataclass
class PipelineRequest:
    """
    Wraps an InferenceRequest with execution state, an attached ExecutionPlan,
    and in-flight simulated latency values.

    The plan is attached at admission time by the SystemOrchestrator and
    propagated through Vision → Prefill → Decode without modification.
    """
    request:       InferenceRequest
    system_state:  SystemState          # snapshot captured at admission
    plan:          Optional[ExecutionPlan] = None

    # Simulated latencies produced by each worker (formula + noise)
    simulated_t_vision_ms:    float = 0.0
    simulated_t_prefill_ms:   float = 0.0
    simulated_t_migration_ms: float = 0.0
    simulated_stage_ttft_ms:  float = 0.0
    wall_ttft_ms:             float = 0.0
    simulated_tbt_ms:         float = 0.0
    tbt_sla_pass:             bool  = False
    kv_alloc_mb:              float = 0.0
    kv_admitted_mb:           float = 0.0   # exact amount added to _kv_used_mb at admission
    kv_seq_id:                int   = -1    # PagedBackend seq_id (from KVAllocationResult)
    sla_pass:                 bool  = False

    admission_time_ms: float = 0.0          # epoch ms (telemetry timestamp)
    submit_perf:       float = 0.0          # perf_counter at submit (for wall TTFT)

    # Completion signal — callers wait on this
    _done: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    def wait(self, timeout_s: float = 30.0) -> bool:
        """Wait for completion.  Returns False on timeout — callers MUST
        check the result (the API layer returns HTTP 504 on timeout)."""
        return self._done.wait(timeout_s)

    def mark_done(self) -> None:
        self._done.set()

    @property
    def telemetry(self) -> TelemetryRecord:
        p = self.plan
        return TelemetryRecord(
            req_id=self.request.req_id,
            image_resolution=self.request.image_resolution,
            prompt_length=self.request.prompt_length,
            n_crops=p.n_crops if p else 0,
            token_keep_ratio=p.token_keep_ratio if p else 0.0,
            parallelism_mode=p.parallelism_mode.value if p else "unknown",
            sm_vision=p.sm_vision if p else 0,
            sm_decode=p.sm_decode if p else 0,
            is_fallback=p.is_fallback if p else True,
            predicted_ttft_ms=p.predicted_ttft_ms if p else 0.0,
            simulated_stage_ttft_ms=self.simulated_stage_ttft_ms,
            wall_ttft_ms=self.wall_ttft_ms,
            ttft_error_ms=self.simulated_stage_ttft_ms - (p.predicted_ttft_ms if p else 0.0),
            simulated_t_vision_ms=self.simulated_t_vision_ms,
            simulated_t_prefill_ms=self.simulated_t_prefill_ms,
            simulated_t_migration_ms=self.simulated_t_migration_ms,
            simulated_tbt_ms=self.simulated_tbt_ms,
            tbt_sla_pass=self.tbt_sla_pass,
            kv_alloc_mb=self.kv_alloc_mb,
            sla_pass=self.sla_pass,
            quality_score=p.quality_score if p else 0.0,
            timestamp_ms=self.admission_time_ms,
        )


# ===========================================================================
# Section 1 — Simulated Stage Workers
# ===========================================================================
#
# Worker protocol (shared by all three stage workers):
#   * Normal item : compute simulated latency, sleep it (so wall-clock time
#     reflects modeled load), put downstream FIRST, then task_done().
#     Put-before-task_done matters: shutdown() may join() a queue, and the
#     old task_done-before-put ordering let a shutdown sentinel overtake an
#     in-flight request, which then hung forever.
#   * Sentinel    : forward the sentinel downstream (cascade), task_done(),
#     exit.  The orchestrator injects a single sentinel into the FIRST queue
#     only; FIFO ordering guarantees it drains behind all in-flight work.

class VisionWorker(threading.Thread):
    """
    SIMULATED vision-encoding worker (no encoder runs).

    Reads from vision_q, applies the compute-share partition from the
    attached ExecutionPlan using the MEASURED Phase 3 linear scaling model:

        T_vision = (VISION_MS_PER_CROP × n_crops + VISION_FIXED_MS) × (38 / sm_vision)

    VISION_MS_PER_CROP/VISION_FIXED_MS are fit to baseline/results_v2.json
    (direct stage-isolated measurement on the real M3 target, 5 trials x 4
    crop settings) — see amio_constants. The worker sleeps the simulated
    duration so downstream wall-clock timing includes this stage.
    """

    def __init__(
        self,
        in_q:  queue.Queue,
        out_q: queue.Queue,
        noise_sigma: float = 0.03,
        daemon: bool = True,
    ):
        super().__init__(daemon=daemon, name="VisionWorker")
        self._in    = in_q
        self._out   = out_q
        self._rng   = random.Random(42)
        self._sigma = noise_sigma

    def run(self) -> None:
        while True:
            item = self._in.get()
            if item is _SENTINEL:
                self._out.put(_SENTINEL)     # cascade shutdown downstream
                self._in.task_done()
                break

            p: PipelineRequest = item
            plan = p.plan
            assert plan is not None, "Plan must be attached before VisionWorker"

            # Share partition from plan (Nova allocation, computed at admission)
            sm_vis = plan.sm_vision if plan.sm_vision > 0 else M3_TOTAL_SMS

            # Compute-bound linear model (Phase 3 SMOrchestrator formula, measured)
            t_vis = (
                (VISION_MS_PER_CROP * plan.n_crops + VISION_FIXED_MS)
                * (M3_TOTAL_SMS / sm_vis)
            )
            # Gaussian timing noise (±σ of predicted)
            t_vis *= max(0.5, 1.0 + self._rng.gauss(0.0, self._sigma))
            p.simulated_t_vision_ms = t_vis

            time.sleep(t_vis / 1000.0)       # occupy simulated wall time

            self._out.put(p)
            self._in.task_done()


class PrefillWorker(threading.Thread):
    """
    SIMULATED LM prefill worker implementing ParVTS scheduling (no LM runs).

    ParVTS logic (Phase 6.3):
      1. Both subject and non-subject token paths enter the LM together.
      2. After `migration_depth` transformer layers the background tokens are
         pruned (predict_migration_cost captures this overhead).
      3. Only n_effective_tokens continue through the remaining layers.

    Latency model: MEASURED Phase 2 quadratic cost model (fit to 4 real
    stage-isolated points; see amio_constants). Single-chip TP is cost-
    neutral — there is one GPU, so parallelism_mode never changes predicted
    latency (the former 25% "TP" discount was a modeling fiction and has
    been removed). The worker sleeps the simulated duration.
    """

    def __init__(
        self,
        in_q:  queue.Queue,
        out_q: queue.Queue,
        cost_model: CostModel,
        noise_sigma: float = 0.03,
        daemon:      bool  = True,
    ):
        super().__init__(daemon=daemon, name="PrefillWorker")
        self._in    = in_q
        self._out   = out_q
        self._cm    = cost_model
        self._rng   = random.Random(43)
        self._sigma = noise_sigma

    def run(self) -> None:
        while True:
            item = self._in.get()
            if item is _SENTINEL:
                self._out.put(_SENTINEL)     # cascade shutdown downstream
                self._in.task_done()
                break

            p: PipelineRequest = item
            plan = p.plan
            assert plan is not None

            # LM prefill cost (measured Phase 2 CostModel). Single-chip TP is
            # cost-neutral (there is one GPU) — no parallelism-mode discount.
            t_pref = self._cm.predict_t_lm_prefill(plan.n_lm_tokens)

            # ParVTS migration overhead
            t_mig = 0.0
            if plan.use_parvts and plan.token_keep_ratio < 1.0:
                t_mig = self._cm.predict_migration_cost(
                    n_full_tokens=plan.n_visual_tokens,
                    n_pruned_tokens=plan.n_effective_tokens,
                    migration_depth=plan.migration_depth,
                )

            noise_p = max(0.5, 1.0 + self._rng.gauss(0.0, self._sigma))
            noise_m = max(0.5, 1.0 + self._rng.gauss(0.0, self._sigma))
            p.simulated_t_prefill_ms   = t_pref * noise_p
            p.simulated_t_migration_ms = t_mig  * noise_m

            time.sleep((p.simulated_t_prefill_ms + p.simulated_t_migration_ms) / 1000.0)

            self._out.put(p)
            self._in.task_done()


class DecodeWorker(threading.Thread):
    """
    SIMULATED auto-regressive decode worker with PagedAttention KV
    bookkeeping (no decoding runs).

    On-demand KV page allocation: blocks are assigned only when a sequence
    enters the decode stage — no pre-allocation at request admission.

    TBT model (Phase 5, UNVALIDATED constants):
        TBT(B) = DECODE_OVERHEAD + DECODE_BW_COST × B
    where B is the effective batch size including this new request.

    The worker sleeps ONE simulated TBT (the first decode step) so that
    wall-clock TTFT includes it; the remaining generation is accounted
    analytically only (sleeping the full generation would take minutes at
    benchmark scale).

    Exception contract: PagedBackend.allocate raises ``MemoryError`` on pool
    exhaustion — that is what this worker catches.  (A previous version
    caught RuntimeError, which killed the thread and wedged the pipeline.)
    """

    def __init__(
        self,
        in_q:       queue.Queue,
        out_q:      queue.Queue,
        kv_backend: PagedBackend,
        noise_sigma: float = 0.03,
        daemon:      bool  = True,
        on_decode_start: Optional[Callable[[], None]] = None,
    ):
        super().__init__(daemon=daemon, name="DecodeWorker")
        self._in    = in_q
        self._out   = out_q
        self._kv    = kv_backend
        self._rng   = random.Random(44)
        self._sigma = noise_sigma
        self._on_decode_start = on_decode_start
        self._batch_size:  int = 0
        self._batch_lock = threading.Lock()

    @property
    def current_batch_size(self) -> int:
        return self._batch_size

    def run(self) -> None:
        while True:
            item = self._in.get()
            if item is _SENTINEL:
                self._out.put(_SENTINEL)     # cascade shutdown to collector
                self._in.task_done()
                break

            p: PipelineRequest = item
            plan = p.plan
            assert plan is not None

            # Mark this request as decoding (orchestrator counter — the
            # matching decrement happens in the Collector).
            if self._on_decode_start is not None:
                self._on_decode_start()

            # On-demand KV page allocation (Phase 4 PagedBackend)
            seq_len = plan.n_lm_tokens + p.request.max_output_tokens
            try:
                alloc = self._kv.allocate(seq_len)
                p.kv_seq_id   = alloc.seq_id     # backend-assigned id
                p.kv_alloc_mb = alloc.kv_used_mb
            except (MemoryError, RuntimeError):
                # Pool exhausted: estimate without storing.  allocate() is
                # transactional, so nothing leaked.
                p.kv_alloc_mb = kv_cache_size_mb(seq_len, quantization_bits=KV_QUANT_BITS)
                p.kv_seq_id   = -1

            # TBT with batch awareness (note: single serial worker → the
            # instantaneous batch here is nearly always 1)
            with self._batch_lock:
                self._batch_size += 1
                batch = self._batch_size

            tbt = _predict_tbt_ms(seq_len, batch)
            p.simulated_tbt_ms = tbt * max(0.5, 1.0 + self._rng.gauss(0.0, self._sigma))

            # Sleep one simulated decode step (first token) — wall TTFT
            # therefore includes the first decode step.
            time.sleep(p.simulated_tbt_ms / 1000.0)

            with self._batch_lock:
                self._batch_size = max(0, self._batch_size - 1)

            if p.kv_seq_id >= 0:
                try:
                    self._kv.free(p.kv_seq_id)
                except Exception:
                    pass

            self._out.put(p)
            self._in.task_done()


# ===========================================================================
# Section 1 (cont.) — System Orchestrator
# ===========================================================================

class SystemOrchestrator:
    """
    Centralized event loop implementing the Nova architectural pattern —
    over SIMULATED stage workers (no model is loaded).

    Pipeline topology
    -----------------
    Admit → [vision_q] → VisionWorker
                      → [prefill_q] → PrefillWorker
                                    → [decode_q] → DecodeWorker
                                                 → [done_q] → Collector

    Four threads total: VisionWorker, PrefillWorker, DecodeWorker, Collector.
    The stage queues are standard blocking queues (get() blocks).

    State propagation
    -----------------
    The ExecutionPlan generated by the Phase 6 AdaptiveController is attached
    to every PipelineRequest at admission and propagated verbatim through all
    stages — workers read the plan but never mutate it.

    Resource synchronisation
    ------------------------
    A single shared lock (_lock) protects the live counters (n_pending,
    n_decoding, kv_used_mb).  n_decoding is incremented when the DecodeWorker
    picks a request up (on_decode_start callback) and decremented by the
    Collector — so the controller's SystemState snapshot actually sees decode
    occupancy.  kv_used_mb is incremented by the plan's predicted KV at
    admission and decremented by exactly that same amount at completion
    (stored on the request), so the counter cannot drift.
    """

    def __init__(
        self,
        sla_budget_ms:   float = SLA_TTFT_MS,
        memory_budget_mb: float = M3_TOTAL_MEMORY_MB,
        log_telemetry:   bool  = True,
    ):
        self.sla_budget_ms    = sla_budget_ms
        self.memory_budget_mb = memory_budget_mb
        self.log_telemetry    = log_telemetry

        # Phase 6 controller and supporting components
        self._controller = AdaptiveController(
            sla_budget_ms=sla_budget_ms,
            memory_budget_mb=memory_budget_mb,
        )
        self._cost_model = CostModel()
        self._kv_backend = PagedBackend(
            kv_bytes_per_token=_KV_W4_BYTES,
            pool_budget_mb=KV_POOL_BUDGET_MB,
        )

        # Pipeline queues (thread-safe; get() blocks until an item arrives)
        self._vision_q  = queue.Queue()
        self._prefill_q = queue.Queue()
        self._decode_q  = queue.Queue()
        self._done_q    = queue.Queue()

        # Live system state (protected by _lock)
        self._lock       = threading.Lock()
        self._n_pending:  int   = 0
        self._n_decoding: int   = 0
        self._kv_used_mb: float = 0.0
        self._n_tbt_violations: int = 0

        # Telemetry log
        self._telemetry: List[TelemetryRecord] = []
        self._tel_lock = threading.Lock()

        # Simulated stage workers (four threads incl. Collector)
        self._vision_w  = VisionWorker(self._vision_q,  self._prefill_q, noise_sigma=0.03)
        self._prefill_w = PrefillWorker(self._prefill_q, self._decode_q,  self._cost_model)
        self._decode_w  = DecodeWorker(
            self._decode_q, self._done_q, self._kv_backend,
            on_decode_start=self._incr_decoding,
        )
        self._collector = threading.Thread(
            target=self._collect_done, daemon=True, name="Collector"
        )

        self._start_lock = threading.Lock()
        self._started  = False
        self._shutdown = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._vision_w.start()
            self._prefill_w.start()
            self._decode_w.start()
            self._collector.start()
            self._started = True

    def shutdown(self, drain_timeout_s: float = 10.0) -> None:
        if not self._started or self._shutdown:
            return

        # A single sentinel is injected into the FIRST queue only; each
        # worker forwards it downstream after finishing all items queued
        # ahead of it (FIFO), so the sentinel can never overtake in-flight
        # work.  Workers put-then-task_done, so join() is also safe.
        self._vision_q.put(_SENTINEL)

        for w in (self._vision_w, self._prefill_w, self._decode_w, self._collector):
            w.join(timeout=drain_timeout_s)

        self._shutdown = True

    # ── Public API ───────────────────────────────────────────────────────────

    def submit(self, request: InferenceRequest) -> PipelineRequest:
        """
        Admit a request into the pipeline.

        Steps:
          1. Capture current system state snapshot.
          2. Call Phase 6 controller → ExecutionPlan.
          3. Attach plan to PipelineRequest.
          4. Enqueue in Vision stage.
        """
        if not self._started:
            self.start()    # start() is internally locked (no double-start race)

        state = self._snapshot_state()
        plan  = self._controller.optimize(request, state)

        p = PipelineRequest(
            request=request,
            system_state=state,
            plan=plan,
            admission_time_ms=time.time() * 1000,
            submit_perf=time.perf_counter(),
        )
        # Remember exactly how much we add to the KV counter so completion
        # can subtract the identical amount (predicted-vs-actual unit mixing
        # previously made the counter drift monotonically).
        p.kv_admitted_mb = plan.predicted_kv_seq_mb

        with self._lock:
            self._n_pending  += 1
            self._kv_used_mb += p.kv_admitted_mb

        self._vision_q.put(p)
        return p

    @property
    def telemetry(self) -> List[TelemetryRecord]:
        with self._tel_lock:
            return list(self._telemetry)

    def print_telemetry_summary(self) -> None:
        recs = self.telemetry
        if not recs:
            print("  No telemetry records.")
            return
        n          = len(recs)
        n_pass     = sum(1 for r in recs if r.sla_pass)
        n_tbt_viol = sum(1 for r in recs if not r.tbt_sla_pass)
        sim_ttfts  = sorted(r.simulated_stage_ttft_ms for r in recs)
        wall_ttfts = sorted(r.wall_ttft_ms for r in recs)
        errors     = [abs(r.ttft_error_ms) for r in recs]
        avg_sim    = sum(sim_ttfts) / n
        avg_wall   = sum(wall_ttfts) / n
        avg_err    = sum(errors) / n
        avg_crops  = sum(r.n_crops for r in recs) / n
        avg_qual   = sum(r.quality_score for r in recs) / n
        n_fb       = sum(1 for r in recs if r.is_fallback)
        print(f"  Requests processed  : {n}   (ALL LATENCIES SIMULATED — no model)")
        print(f"  SLA pass rate       : {100*n_pass/n:.1f}%  (gated on wall TTFT incl. queue wait)")
        print(f"  Avg simulated TTFT  : {avg_sim:.1f} ms  (stage-formula sum)")
        print(f"  Avg wall TTFT       : {avg_wall:.1f} ms  (submit → first token)")
        print(f"  P50 / P99 wall TTFT : {_percentile(wall_ttfts, 0.50):.1f} / "
              f"{_percentile(wall_ttfts, 0.99):.1f} ms")
        print(f"  Avg prediction error: {avg_err:.1f} ms  (|sim − predicted|)")
        print(f"  TBT SLA violations  : {n_tbt_viol}/{n}  "
              f"(measured batch=1 TBT floor ~{DECODE_OVERHEAD_MS:.1f} ms vs "
              f"{TBT_SLA_MS:.0f} ms target)")
        print(f"  Avg crops selected  : {avg_crops:.2f}")
        print(f"  Avg quality score   : {avg_qual:.3f}  (crop×keep input proxy, not accuracy)")
        print(f"  Fallbacks           : {n_fb} ({100*n_fb/n:.1f}%)")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _incr_decoding(self) -> None:
        """Called by DecodeWorker when a request enters the decode stage."""
        with self._lock:
            self._n_decoding += 1

    def _snapshot_state(self) -> SystemState:
        with self._lock:
            return SystemState(
                n_pending_requests=self._n_pending,
                n_decoding_requests=self._n_decoding,
                kv_used_mb=self._kv_used_mb,
                current_decode_batch=max(1, self._n_decoding),
                sim_time_ms=time.time() * 1000,
            )

    def _collect_done(self) -> None:
        """Collect finished PipelineRequests, compute TTFT (both definitions),
        validate SLAs, log telemetry."""
        while True:
            item = self._done_q.get()
            if item is _SENTINEL:
                self._done_q.task_done()
                break

            p: PipelineRequest = item

            # Simulated stage TTFT = formula sum (excludes queue wait + first
            # decode step); wall TTFT = wall time since submit (queue wait +
            # slept stage durations + first decode step).  SLA gates on WALL.
            p.simulated_stage_ttft_ms = (
                p.simulated_t_vision_ms
                + p.simulated_t_prefill_ms
                + p.simulated_t_migration_ms
            )
            p.wall_ttft_ms = (time.perf_counter() - p.submit_perf) * 1000.0
            p.sla_pass     = p.wall_ttft_ms <= self.sla_budget_ms

            # TBT SLA validation (previously never checked)
            p.tbt_sla_pass = p.simulated_tbt_ms <= TBT_SLA_MS

            with self._lock:
                self._n_pending  = max(0, self._n_pending - 1)
                self._n_decoding = max(0, self._n_decoding - 1)
                # Subtract exactly what admission added (no unit mixing)
                self._kv_used_mb = max(0.0, self._kv_used_mb - p.kv_admitted_mb)
                if not p.tbt_sla_pass:
                    self._n_tbt_violations += 1

            if self.log_telemetry:
                rec = p.telemetry
                with self._tel_lock:
                    self._telemetry.append(rec)

            p.mark_done()
            self._done_q.task_done()


# ===========================================================================
# Section 2 — OpenAI-Format API Layer (SIMULATION — placeholder responses)
# ===========================================================================

class _ChatHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTP handler for POST /v1/multimodal/chat/completions.

    Follows the OpenAI Chat Completions response SHAPE, but the content is a
    simulation placeholder: no model is loaded, no tokens are generated, and
    all telemetry values are cost-model formula evaluations.  The response
    carries `"simulated": true` markers and X-AMIO-Simulated-* headers so it
    cannot be mistaken for real inference output.
    """

    # Set by make_api_server() via class-level injection
    orchestrator: SystemOrchestrator = None   # type: ignore[assignment]
    _req_counter: int = 0
    _counter_lock = threading.Lock()

    def log_message(self, fmt: str, *args) -> None:
        pass  # suppress default noisy request logging

    def do_POST(self) -> None:
        if self.path not in ("/v1/multimodal/chat/completions",):
            self._json_error(404, "endpoint not found — use /v1/multimodal/chat/completions")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length) if length > 0 else b"{}"
            data   = json.loads(body)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json_error(400, f"invalid JSON: {exc}")
            return

        # ── Parse request body ───────────────────────────────────────────
        image_resolution  = max(1, int(data.get("image_resolution",  512)))
        prompt_length     = max(1, int(data.get("prompt_length",      32)))
        max_output_tokens = max(1, int(data.get("max_tokens",         60)))

        with self.__class__._counter_lock:
            self.__class__._req_counter += 1
            req_id = self.__class__._req_counter

        req = InferenceRequest(
            req_id=req_id,
            image_resolution=image_resolution,
            prompt_length=prompt_length,
            max_output_tokens=max_output_tokens,
            arrival_time_ms=time.time() * 1000,
        )

        # ── Submit and wait (timeout → HTTP 504, not a fake 200) ─────────
        pr = self.orchestrator.submit(req)
        if not pr.wait(timeout_s=60.0):
            self._json_error(
                504,
                f"request {req_id} timed out in the simulation pipeline "
                f"(60 s) — no result available",
            )
            return

        plan = pr.plan

        # ── Build OpenAI-format response (SIMULATION PLACEHOLDER) ────────
        content = (
            f"[SIMULATION PLACEHOLDER — no model is loaded, no text was "
            f"generated] req={req_id}  crops={plan.n_crops}  "
            f"keep={plan.token_keep_ratio:.2f}  "
            f"simulated_TTFT={pr.simulated_stage_ttft_ms:.1f}ms  "
            f"wall_TTFT={pr.wall_ttft_ms:.1f}ms  "
            f"SLA={'PASS' if pr.sla_pass else 'FAIL'}"
        )
        sm_ratio = plan.sm_vision / M3_TOTAL_SMS

        amio_tel = {
            "simulated":               True,   # nothing here is measured from a model
            "req_id":                  req_id,
            "n_crops":                 plan.n_crops,
            "token_keep_ratio":        round(plan.token_keep_ratio, 4),
            "parallelism_mode":        plan.parallelism_mode.value,
            "sm_partition":            f"{plan.sm_vision}/{plan.sm_decode}",
            "sm_vision_ratio":         round(sm_ratio, 4),
            "is_fallback":             plan.is_fallback,
            "quality_score":           round(plan.quality_score, 4),
            "predicted_ttft_ms":       round(plan.predicted_ttft_ms, 2),
            "simulated_stage_ttft_ms": round(pr.simulated_stage_ttft_ms, 2),
            "wall_ttft_ms":            round(pr.wall_ttft_ms, 2),
            "ttft_prediction_error":   round(pr.simulated_stage_ttft_ms - plan.predicted_ttft_ms, 2),
            "sla_pass":                pr.sla_pass,   # gated on wall TTFT
            "kv_alloc_mb":             round(pr.kv_alloc_mb, 3),
            "simulated_tbt_ms":        round(pr.simulated_tbt_ms, 2),
            "tbt_sla_pass":            pr.tbt_sla_pass,
        }

        # Token counts derived from the SIMULATED plan (planned LM input
        # tokens and the requested completion budget) — no tokenizer ran.
        response = {
            "id":      f"amio-cmpl-{req_id}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   "SmolVLM-Instruct-4bit-AMIO-simulated",
            "choices": [{
                "index": 0,
                "message":       {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "simulated":         True,
                "prompt_tokens":     plan.n_lm_tokens,        # planned, not tokenized
                "completion_tokens": max_output_tokens,       # requested budget
                "total_tokens":      plan.n_lm_tokens + max_output_tokens,
                "_amio_telemetry":   amio_tel,
            },
        }

        body_bytes = json.dumps(response, indent=2).encode()

        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        # Telemetry response headers for low-latency scraping (all simulated)
        self.send_header("X-AMIO-Simulated",           "true")
        self.send_header("X-AMIO-N-Crops",             str(plan.n_crops))
        self.send_header("X-AMIO-SM-Partition",        f"{plan.sm_vision}/{plan.sm_decode}")
        self.send_header("X-AMIO-SM-Vision-Ratio",     f"{sm_ratio:.4f}")
        self.send_header("X-AMIO-Predicted-TTFT-MS",   f"{plan.predicted_ttft_ms:.2f}")
        self.send_header("X-AMIO-Simulated-TTFT-MS",   f"{pr.simulated_stage_ttft_ms:.2f}")
        self.send_header("X-AMIO-Wall-TTFT-MS",        f"{pr.wall_ttft_ms:.2f}")
        self.send_header("X-AMIO-SLA-Pass",            "true" if pr.sla_pass else "false")
        self.send_header("X-AMIO-Quality-Score",       f"{plan.quality_score:.4f}")
        self.send_header("X-AMIO-Is-Fallback",         "true" if plan.is_fallback else "false")
        self.end_headers()
        self.wfile.write(body_bytes)

    def _json_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message, "simulated": True}).encode()
        self.send_response(code)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTP server that spawns a new daemon thread per request."""
    daemon_threads    = True
    allow_reuse_address = True


def make_api_server(
    orchestrator: SystemOrchestrator,
    host: str = "127.0.0.1",
    port: int  = 8080,
) -> _ThreadedHTTPServer:
    """Create a threaded HTTP server bound to the given SystemOrchestrator."""
    # Inject orchestrator at class level (safe — single server instance)
    handler = type(
        "_BoundChatHandler",
        (_ChatHandler,),
        {"orchestrator": orchestrator},
    )
    return _ThreadedHTTPServer((host, port), handler)


def run_api_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the AMIO simulation API server; blocks until Ctrl+C."""
    orch = SystemOrchestrator(log_telemetry=True)
    orch.start()
    server = make_api_server(orch, host, port)
    url = f"http://{host}:{port}/v1/multimodal/chat/completions"
    print("AMIO SIMULATION service — no model is loaded; all latencies are")
    print("analytic cost-model formulas + noise. Responses are placeholders.")
    print(f"API server  →  POST {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        server.shutdown()
        orch.shutdown()
        print("Session telemetry:")
        orch.print_telemetry_summary()


# ===========================================================================
# Section 3 — Orchestrator Pipeline Demo
# ===========================================================================

def _demo_orchestrator_pipeline() -> None:
    """
    Demonstrate the SystemOrchestrator end-to-end with 10 sample requests.
    Workers run in daemon threads and sleep their simulated stage durations;
    we wait for each request to complete and then print per-request telemetry.
    """
    print()
    print("=" * 74)
    print("  Phase 7 — System Orchestrator Pipeline Demo  (SIMULATED — no model)")
    print("=" * 74)
    print("  Starting workers: VisionWorker | PrefillWorker | DecodeWorker | Collector")

    orch = SystemOrchestrator(log_telemetry=True)
    orch.start()

    scenarios = [
        (224,  16), (448, 32), (512, 48), (756,  32),
        (1008, 64), (512, 16), (224, 32), (756,  48),
        (512,  32), (1512, 64),
    ]
    requests = [
        InferenceRequest(req_id=i, image_resolution=res, prompt_length=pl)
        for i, (res, pl) in enumerate(scenarios)
    ]

    print(f"  Submitting {len(requests)} requests ...\n")
    pipeline_reqs = [orch.submit(req) for req in requests]

    for pr in pipeline_reqs:
        pr.wait(timeout_s=60.0)

    hdr = (
        f"  {'ID':>3}  {'Res':>5}  {'Crops':>5}  {'Keep%':>6}  "
        f"{'Mode':<5}  {'SM v/d':>6}  {'T_vis':>7}  {'T_pre':>7}  "
        f"{'T_mig':>7}  {'SimTTFT':>8}  {'WallTTFT':>9}  {'SLA':>5}  {'Quality':>8}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for pr in pipeline_reqs:
        p = pr.plan
        if p is None:
            continue
        print(
            f"  {pr.request.req_id:>3}  {pr.request.image_resolution:>5}  "
            f"{p.n_crops:>5}  {p.token_keep_ratio*100:>5.1f}%  "
            f"{p.parallelism_mode.value[:4]:<5}  "
            f"{p.sm_vision}/{p.sm_decode}  "
            f"{pr.simulated_t_vision_ms:>7.1f}  {pr.simulated_t_prefill_ms:>7.1f}  "
            f"{pr.simulated_t_migration_ms:>7.1f}  {pr.simulated_stage_ttft_ms:>8.1f}  "
            f"{pr.wall_ttft_ms:>9.1f}  "
            f"{'PASS' if pr.sla_pass else 'FAIL':>5}  {p.quality_score:>8.3f}"
        )

    print()
    print("  Orchestrator telemetry summary:")
    orch.print_telemetry_summary()
    orch.shutdown()


# ===========================================================================
# Section 4 — Comparative Test Matrix
# ===========================================================================

def _crops_for_resolution(resolution: int) -> int:
    """Return the SigLIP crop count for the nearest lower resolution."""
    best = 1
    for res_key, crops in sorted(_RESOLUTION_TO_CROPS.items()):
        if res_key <= resolution:
            best = crops
    return best


@dataclass
class BenchmarkResult:
    system_name:   str
    resolution:    int
    concurrency:   int
    sla_budget_ms: float
    n_requests:    int
    sla_pass_rate: float       # fraction 0–1
    avg_ttft_ms:   float       # simulated (formula) TTFT
    p99_ttft_ms:   float
    avg_quality:   float       # crop×keep input proxy — NOT accuracy
    avg_frag_pct:  float       # per-allocation KV waste (successful allocs only)
    throughput:    float       # ACHIEVED req/s from simulated completion times
    n_fallbacks:   int
    n_kv_alloc_failures: int = 0   # KV pool exhaustion events (no fabricated frag)


def _simulate_system(
    system_name:   str,
    reqs_data:     List[Tuple[int, int, float]],   # (resolution, prompt_len, arrival_ms)
    sla_budget_ms: float,
    rng:           random.Random,
) -> BenchmarkResult:
    """
    Simulate one system variant (Static Baseline / Greedy Fast / AMIO Adaptive)
    across a synthetic request stream.

    HONESTY NOTE: all three systems are graded with the same Phase 2 cost
    model that the AMIO controller optimizes against, plus 3% noise — this is
    a model-vs-model comparison (the controller is evaluated by the simulator
    it was built to satisfy), not an external measurement.

    Decode overlap is modeled from arrival times: each request occupies the
    decode stage from (arrival + TTFT) until (arrival + TTFT + 60×TBT), and
    n_decoding at any admission is the number of requests still inside that
    window.  KV allocations are freed when their request finishes; if the
    pool still exhausts, the failure is COUNTED (n_kv_alloc_failures) rather
    than replaced with a hardcoded fragmentation number.

    Throughput is ACHIEVED throughput: n / (last simulated completion − first
    arrival), not offered load.

    Returns aggregate BenchmarkResult.
    """
    noise = 0.03
    max_out = 60   # output token budget per request

    # Per-system initialisation
    if system_name == "AMIO Adaptive":
        ctrl        = AdaptiveController(sla_budget_ms=sla_budget_ms)
        kv_backend  = PagedBackend(
            kv_bytes_per_token=_KV_W4_BYTES,
            pool_budget_mb=KV_POOL_BUDGET_MB,
        )
    else:
        ctrl       = None
        kv_backend = ContiguousBackend(kv_bytes_per_token=_KV_W4_BYTES)

    cm = CostModel()

    ttfts:      List[float] = []
    qualities:  List[float] = []
    frags:      List[float] = []
    n_sla_pass:  int = 0
    n_fallbacks: int = 0
    n_kv_fail:   int = 0
    # In-flight requests: (decode_start_ms, finish_ms, kv_seq_id, kv_mb)
    in_flight: List[Tuple[float, float, int, float]] = []
    completion_times: List[float] = []

    for i, (resolution, prompt_len, arrival_ms) in enumerate(reqs_data):
        # ── Retire requests that finished before this arrival; free their KV
        still_live: List[Tuple[float, float, int, float]] = []
        for dec_start, finish, seq_id, kv_live in in_flight:
            if finish <= arrival_ms:
                if seq_id >= 0:
                    kv_backend.free(seq_id)
            else:
                still_live.append((dec_start, finish, seq_id, kv_live))
        in_flight = still_live

        # Actual overlap given arrival times (not a monotonic counter)
        n_decoding = sum(1 for ds, _, _, _ in in_flight if ds <= arrival_ms)
        n_front    = len(in_flight) - n_decoding   # admitted, still pre-decode
        kv_used_now = sum(k for _, _, _, k in in_flight)

        req = InferenceRequest(
            req_id=i,
            image_resolution=resolution,
            prompt_length=prompt_len,
            max_output_tokens=max_out,
            arrival_time_ms=arrival_ms,
        )
        state = SystemState(
            n_pending_requests=n_front,
            n_decoding_requests=n_decoding,
            kv_used_mb=kv_used_now,
            current_decode_batch=max(1, n_decoding),
        )

        # ── Determine strategy ───────────────────────────────────────────
        if system_name == "AMIO Adaptive":
            assert ctrl is not None
            plan = ctrl.optimize(req, state)
            n_crops    = plan.n_crops
            keep_ratio = plan.token_keep_ratio
            mode       = plan.parallelism_mode
            sm_vis     = plan.sm_vision if plan.sm_vision > 0 else M3_TOTAL_SMS
            use_parvts = plan.use_parvts
            mig_depth  = plan.migration_depth
            quality    = plan.quality_score
            if plan.is_fallback:
                n_fallbacks += 1

        elif system_name == "Static Baseline":
            # Fixed max crops for resolution, no pruning, DP, no SM management
            n_crops    = _crops_for_resolution(resolution)
            keep_ratio = 1.0
            mode       = ParallelismMode.DP
            sm_vis     = M3_TOTAL_SMS if n_decoding == 0 else 8
            use_parvts = False
            mig_depth  = 0
            quality    = float(n_crops)

        else:  # "Greedy Fast"
            # Always 1 crop + maximum pruning
            n_crops    = 1
            keep_ratio = 0.111
            mode       = ParallelismMode.DP
            sm_vis     = M3_TOTAL_SMS if n_decoding == 0 else 8
            use_parvts = True
            mig_depth  = 3
            quality    = 1 * 0.111

        # ── Vision latency (simulated, MEASURED linear model) ────────────
        sm_eff = max(sm_vis, 1)
        t_vis  = (VISION_MS_PER_CROP * n_crops + VISION_FIXED_MS) * (M3_TOTAL_SMS / sm_eff)
        t_vis *= max(0.5, 1.0 + rng.gauss(0.0, noise))

        # ── LM prefill (simulated) ───────────────────────────────────────
        n_vis = max(1, round(n_crops * TOKENS_PER_CROP))
        n_eff = max(1, round(n_vis * keep_ratio))
        n_lm  = n_eff + prompt_len
        t_pre = cm.predict_t_lm_prefill(n_lm)
        # NOTE: single-chip TP is cost-neutral (there is one GPU) — no
        # parallelism-mode discount is applied, matching simulation/controller.py.
        t_pre *= max(0.5, 1.0 + rng.gauss(0.0, noise))

        # ── ParVTS migration cost (simulated) ────────────────────────────
        t_mig = 0.0
        if use_parvts and keep_ratio < 1.0:
            t_mig = cm.predict_migration_cost(n_vis, n_eff, mig_depth)
            t_mig *= max(0.5, 1.0 + rng.gauss(0.0, noise))

        ttft = t_vis + t_pre + t_mig

        # ── Decode duration (analytic) and completion time ───────────────
        tbt = _predict_tbt_ms(n_lm + max_out, n_decoding + 1)
        finish_ms = arrival_ms + ttft + tbt * max_out
        completion_times.append(finish_ms)

        # ── KV allocation (pool state reflects live sequences) ───────────
        seq_len = n_lm + max_out
        kv_mb   = 0.0
        seq_id  = -1
        try:
            alloc  = kv_backend.allocate(min(seq_len, 2048))
            kv_mb  = alloc.kv_used_mb
            seq_id = alloc.seq_id
            frags.append(alloc.fragmentation_pct)
        except MemoryError:
            # Pool genuinely exhausted even after retiring finished requests:
            # count the failure honestly — no hardcoded fragmentation value.
            n_kv_fail += 1
            kv_mb = kv_cache_size_mb(seq_len, quantization_bits=KV_QUANT_BITS)

        in_flight.append((arrival_ms + ttft, finish_ms, seq_id, kv_mb))

        ttfts.append(ttft)
        qualities.append(quality)
        if ttft <= sla_budget_ms:
            n_sla_pass += 1

    n = len(ttfts)
    ttfts_s = sorted(ttfts)
    # Achieved throughput: completions per second of simulated wall time
    span_s = max(1e-9, (max(completion_times) - reqs_data[0][2]) / 1000.0)

    return BenchmarkResult(
        system_name=system_name,
        resolution=reqs_data[0][0],
        concurrency=n,
        sla_budget_ms=sla_budget_ms,
        n_requests=n,
        sla_pass_rate=n_sla_pass / n if n > 0 else 0.0,
        avg_ttft_ms=sum(ttfts) / n,
        p99_ttft_ms=_percentile(ttfts_s, 0.99),
        avg_quality=sum(qualities) / n,
        avg_frag_pct=(sum(frags) / len(frags)) if frags else 0.0,
        throughput=n / span_s,
        n_fallbacks=n_fallbacks,
        n_kv_alloc_failures=n_kv_fail,
    )


def run_benchmark_matrix(seed: int = 42) -> List[BenchmarkResult]:
    """
    Run the comparative test matrix (ALL SIMULATED — no model):

    Variables
    ---------
    - Resolution:   [224, 448, 756, 1024]  (4 levels)
    - Concurrency:  [1, 10, 50, 100]       (4 levels, Poisson arrivals)
    - SLA budget:   [200, 500, 1000] ms    (3 levels)
    - Systems:      Static Baseline / Greedy Fast / AMIO Adaptive

    Total runs: 4 × 4 × 3 × 3 = 144 simulation runs.

    Thrpt column = ACHIEVED req/s derived from simulated completion times.
    KVfail = KV pool exhaustion events (counted, not painted over).
    """
    print()
    print("=" * 74)
    print("  Phase 7 — Comparative Test Matrix  (SIMULATED — no model loaded)")
    print("  All three systems are graded by the same cost model the AMIO")
    print("  controller optimizes against — a model-vs-model comparison.")
    print("=" * 74)

    RESOLUTIONS:   List[int]   = [224, 448, 756, 1024]
    CONCURRENCIES: List[int]   = [1, 10, 50, 100]
    SLA_BUDGETS:   List[float] = [200.0, 500.0, 1000.0]
    SYSTEMS:       List[str]   = ["Static Baseline", "Greedy Fast", "AMIO Adaptive"]

    all_results: List[BenchmarkResult] = []
    rng = random.Random(seed)

    hdr = (
        f"  {'System':>18}  {'Res':>5}  {'N':>4}  {'SLA':>5}  "
        f"{'Pass%':>6}  {'AvgTTFT':>9}  {'P99':>8}  {'Qual':>6}  {'Frag%':>6}  "
        f"{'Thrpt':>6}  {'KVfail':>6}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for sla_ms in SLA_BUDGETS:
        for resolution in RESOLUTIONS:
            for concurrency in CONCURRENCIES:
                # Poisson arrival stream
                mean_iat_ms  = 5000.0 / max(concurrency, 1)
                reqs_data: List[Tuple[int, int, float]] = []
                t = 0.0
                for _ in range(concurrency):
                    t += rng.expovariate(1.0 / mean_iat_ms)
                    pl = rng.randint(16, 64)
                    reqs_data.append((resolution, pl, t))

                for sys_name in SYSTEMS:
                    rng_sys = random.Random(seed)   # same seed → comparable noise
                    res = _simulate_system(sys_name, reqs_data, sla_ms, rng_sys)
                    all_results.append(res)
                    print(
                        f"  {sys_name:>18}  {resolution:>5}  {concurrency:>4}  "
                        f"{sla_ms:>5.0f}  {res.sla_pass_rate*100:>5.1f}%  "
                        f"{res.avg_ttft_ms:>9.1f}  {res.p99_ttft_ms:>8.1f}  "
                        f"{res.avg_quality:>6.2f}  {res.avg_frag_pct:>5.1f}%  "
                        f"{res.throughput:>6.2f}  {res.n_kv_alloc_failures:>6}"
                    )

            print("  " + "─" * (len(hdr) - 2))

    return all_results


# ===========================================================================
# Section 5 — Pareto Frontier Analysis
# ===========================================================================

def run_pareto_analysis(all_results: List[BenchmarkResult]) -> None:
    """
    Text-based Pareto frontier analysis across three dimensions:

    (A) Throughput vs. SLA Pass Rate — across systems and concurrency levels
    (B) KV Memory Fragmentation — Paged (AMIO) vs Contiguous (Static)
        (per-allocation waste under two ASSUMED policies — see kv_manager)
    (C) U-Curve Verification — Nova SM reallocator heatmap
    (D) Head-to-head aggregate comparison
    """
    print()
    print("=" * 74)
    print("  Phase 7 — Pareto Frontier Analysis  (simulation outputs)")
    print("=" * 74)

    systems = ["Static Baseline", "Greedy Fast", "AMIO Adaptive"]

    # ── (A) SLA Pass Rate vs Throughput ─────────────────────────────────────
    print()
    print("  (A)  SLA Pass Rate vs. Achieved Throughput  [SLA budget = 500 ms]")
    print()

    concs = sorted({r.concurrency for r in all_results})
    hdr_a = (
        f"  {'System':>18}  " +
        "  ".join(f"N={c:>3}  Pass%" for c in concs)
    )
    print(hdr_a)
    print("  " + "─" * (len(hdr_a) - 2))

    for sys_name in systems:
        row = f"  {sys_name:>18}  "
        for c in concs:
            recs = [r for r in all_results
                    if r.system_name == sys_name
                    and r.concurrency == c
                    and r.sla_budget_ms == 500.0]
            if recs:
                avg_pass = sum(r.sla_pass_rate for r in recs) / len(recs) * 100
                row += f"N={c:>3}  {avg_pass:>4.1f}%  "
            else:
                row += f"N={c:>3}    —%  "
        print(row)
    print()

    # ── (B) Memory Fragmentation ─────────────────────────────────────────────
    print()
    print("  (B)  KV Memory Fragmentation  —  Paged (AMIO) vs Contiguous (Static)")
    print("       (per-allocation over-reservation under two assumed policies,")
    print("        not measured allocator behavior)")
    print()
    resols = sorted({r.resolution for r in all_results})
    print(f"  {'Resolution':>12}  {'Static Frag%':>14}  {'AMIO Frag%':>12}  {'Reduction':>11}")
    print("  " + "─" * 55)
    for res in resols:
        sf_recs = [r.avg_frag_pct for r in all_results
                   if r.system_name == "Static Baseline" and r.resolution == res]
        af_recs = [r.avg_frag_pct for r in all_results
                   if r.system_name == "AMIO Adaptive" and r.resolution == res]
        if sf_recs and af_recs:
            sf = sum(sf_recs) / len(sf_recs)
            af = sum(af_recs) / len(af_recs)
            # ASCII bar for reduction
            reduction = sf - af
            bar_len   = min(10, max(0, int(reduction / 10)))
            bar       = "█" * bar_len
            print(
                f"  {res:>12}  {sf:>13.1f}%  {af:>11.1f}%  "
                f"  -{reduction:>5.1f}pp {bar}"
            )
    print()

    # ── (C) U-Curve: Nova SM heatmap ────────────────────────────────────────
    print()
    print("  (C)  Nova SM Reallocator Heatmap  (% of modeled compute shares to vision)")
    print("       █ = high vision share (more priority)  ░ = low (decode-dominant)")
    print()

    ctrl         = AdaptiveController()
    front_loads  = [0, 1, 2, 5, 10, 15]
    decode_loads = [0, 5, 10, 20, 40, 70]
    col_labels   = [f"D={d}" for d in decode_loads]
    row_labels   = [f"F={f}" for f in front_loads]
    col_w = 9
    row_w = 6

    # header
    print(f"  {'':>{row_w}}", end="")
    for cl in col_labels:
        print(f"{cl:>{col_w}}", end="")
    print()

    _HDCHARS = [" ", "░", "▒", "▓", "█"]
    all_pcts: List[float] = []
    grid: List[List[float]] = []
    for nf in front_loads:
        row: List[float] = []
        for nd in decode_loads:
            vis, _ = ctrl._nova_sm_allocation(nf, nd)
            row.append(vis / M3_TOTAL_SMS * 100)
        grid.append(row)
        all_pcts.extend(row)

    vmin, vmax = min(all_pcts), max(all_pcts)
    vrange     = vmax - vmin if vmax != vmin else 1.0

    for rl, row_vals in zip(row_labels, grid):
        print(f"  {rl:>{row_w}}", end="")
        for v in row_vals:
            idx  = int((v - vmin) / vrange * (len(_HDCHARS) - 1))
            char = _HDCHARS[min(idx, len(_HDCHARS) - 1)]
            cell = f"{char}{v:.0f}%"
            print(f"{cell:>{col_w}}", end="")
        print()

    print()
    print("  Rows = front-stage (vision/prefill) queue depth  (F=0..15 requests)")
    print("  Cols = decode worker queue depth                 (D=0..70 requests)")
    print("  Reading: as F increases (more vision pressure), Nova allocates more")
    print("  shares to vision. As D increases, decode claims the share budget.")
    print("  The full 100% column at D=0 is the idle-decode state.")

    # ── (D) Head-to-head aggregate summary ──────────────────────────────────
    print()
    print("  (D)  Head-to-Head Aggregate Comparison (all scenarios averaged)")
    print()
    hdr_d = (
        f"  {'System':>18}  {'SLA Pass%':>10}  {'AvgTTFT':>9}  "
        f"{'P99TTFT':>9}  {'Quality':>8}  {'Frag%':>7}  {'Fallbacks':>10}  {'KVfail':>7}"
    )
    print(hdr_d)
    print("  " + "─" * (len(hdr_d) - 2))
    for sys_name in systems:
        recs = [r for r in all_results if r.system_name == sys_name]
        if not recs:
            continue
        n        = len(recs)
        sla_avg  = sum(r.sla_pass_rate for r in recs) / n * 100
        ttft_avg = sum(r.avg_ttft_ms   for r in recs) / n
        p99_avg  = sum(r.p99_ttft_ms   for r in recs) / n
        qual_avg = sum(r.avg_quality   for r in recs) / n
        frag_avg = sum(r.avg_frag_pct  for r in recs) / n
        fb_total = sum(r.n_fallbacks   for r in recs)
        kvf_total = sum(r.n_kv_alloc_failures for r in recs)
        print(
            f"  {sys_name:>18}  {sla_avg:>9.1f}%  {ttft_avg:>9.1f}  "
            f"{p99_avg:>9.1f}  {qual_avg:>8.3f}  {frag_avg:>6.1f}%  {fb_total:>10}  "
            f"{kvf_total:>7}"
        )
    print()

    # ── Key insight annotations ──────────────────────────────────────────────
    amio_recs   = [r for r in all_results if r.system_name == "AMIO Adaptive"]
    static_recs = [r for r in all_results if r.system_name == "Static Baseline"]
    greedy_recs = [r for r in all_results if r.system_name == "Greedy Fast"]
    if amio_recs and static_recs and greedy_recs:
        amio_sla   = sum(r.sla_pass_rate for r in amio_recs)   / len(amio_recs) * 100
        static_sla = sum(r.sla_pass_rate for r in static_recs) / len(static_recs) * 100
        amio_qual  = sum(r.avg_quality   for r in amio_recs)   / len(amio_recs)
        greedy_qual= sum(r.avg_quality   for r in greedy_recs) / len(greedy_recs)
        static_frag= sum(r.avg_frag_pct  for r in static_recs) / len(static_recs)
        amio_frag  = sum(r.avg_frag_pct  for r in amio_recs)   / len(amio_recs)
        print("  Key observations (simulation-internal — see honesty notes above):")
        print(f"    AMIO vs Static SLA advantage   : {amio_sla-static_sla:+.1f} pp")
        print(f"    AMIO vs Greedy quality gain     : {amio_qual-greedy_qual:+.3f} (crop×keep proxy)")
        print(f"    Per-allocation KV waste         : {static_frag:.1f}% → {amio_frag:.1f}%"
              f"  (-{static_frag-amio_frag:.1f} pp; policy comparison)")
    print()


# ===========================================================================
# Section 6 — API Demo (start server, send 5 requests, print telemetry)
# ===========================================================================

def run_api_demo(port: int = 18_080) -> None:
    """
    Start the API server on a random high port, fire 5 sample requests via
    the standard library urllib, print telemetry, then shut down.
    """
    import urllib.request as _urllib

    print()
    print("=" * 74)
    print("  Phase 7 — OpenAI-Format API Demo  (SIMULATED — no model loaded)")
    print("=" * 74)

    orch   = SystemOrchestrator(log_telemetry=True)
    orch.start()
    server = make_api_server(orch, host="127.0.0.1", port=port)
    srv_th = threading.Thread(target=server.serve_forever, daemon=True, name="APIServer")
    srv_th.start()
    time.sleep(0.15)   # allow server to bind

    base_url = f"http://127.0.0.1:{port}/v1/multimodal/chat/completions"
    print(f"  Server : POST {base_url}")
    print("  Sending 5 sample requests ...\n")

    sample_bodies = [
        dict(image_resolution=224,  prompt_length=16, max_tokens=30),
        dict(image_resolution=512,  prompt_length=32, max_tokens=60),
        dict(image_resolution=756,  prompt_length=48, max_tokens=60),
        dict(image_resolution=1008, prompt_length=64, max_tokens=80),
        dict(image_resolution=1512, prompt_length=32, max_tokens=60),
    ]

    print(
        f"  {'#':>2}  {'Res':>5}  {'Crops':>5}  {'SM v/d':>7}  "
        f"{'Pred ms':>8}  {'Sim ms':>7}  {'Wall ms':>8}  {'SLA':>5}  {'Quality':>8}"
    )
    print("  " + "─" * 66)

    for i, body in enumerate(sample_bodies, start=1):
        raw = json.dumps(body).encode()
        http_req = _urllib.Request(
            base_url,
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _urllib.urlopen(http_req, timeout=30) as resp:
                crops   = resp.headers.get("X-AMIO-N-Crops",           "?")
                sm_part = resp.headers.get("X-AMIO-SM-Partition",      "?")
                pred    = resp.headers.get("X-AMIO-Predicted-TTFT-MS", "?")
                sim     = resp.headers.get("X-AMIO-Simulated-TTFT-MS", "?")
                wall    = resp.headers.get("X-AMIO-Wall-TTFT-MS",      "?")
                sla     = resp.headers.get("X-AMIO-SLA-Pass",          "?")
                qual    = resp.headers.get("X-AMIO-Quality-Score",     "?")
                print(
                    f"  {i:>2}  {body['image_resolution']:>5}  {crops:>5}  {sm_part:>7}  "
                    f"{pred:>8}  {sim:>7}  {wall:>8}  {sla:>5}  {qual:>8}"
                )
        except Exception as exc:
            print(f"  {i:>2}  ERROR: {exc}")

    print()
    server.shutdown()
    orch.shutdown()
    print("  Session telemetry:")
    orch.print_telemetry_summary()
    print()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7: AMIO Full System Integration (SIMULATION — no model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python integrated_service.py              # benchmark + Pareto\n"
            "  python integrated_service.py --api        # persistent HTTP server\n"
            "  python integrated_service.py --api-test   # API demo (5 requests)\n"
        ),
    )
    parser.add_argument("--api",      action="store_true",
                        help="Start persistent HTTP API server")
    parser.add_argument("--api-test", action="store_true",
                        help="Run API demo (launch server, send 5 requests, exit)")
    parser.add_argument("--host",     default="127.0.0.1",
                        help="API server bind address (default: 127.0.0.1)")
    parser.add_argument("--port",     type=int, default=8080,
                        help="API server port (default: 8080)")
    parser.add_argument("--no-demo",  action="store_true",
                        help="Skip the orchestrator pipeline demo in benchmark mode")
    args = parser.parse_args()

    if args.api:
        run_api_server(host=args.host, port=args.port)
        return

    if args.api_test:
        # Find a free port if default is busy
        port = args.port if args.port != 8080 else 18080
        run_api_demo(port=port)
        return

    # ── Default: full benchmark run ──────────────────────────────────────
    print()
    print("=" * 74)
    print("  Phase 7: Full System Integration — SIMULATION SERVICE")
    print("  *** No model is loaded. All latencies are analytic formulas ***")
    print("  *** + Gaussian noise; responses are placeholders.          ***")
    print("  SmolVLM-Instruct-4bit (A)daptive (M)ultimodal (I)nference (O)ptimizer")
    print(f"  Modeled hardware: Apple M3 · 8 GB unified · {M3_TOTAL_SMS} compute shares")
    print("  ('shares' are a modeling abstraction — the M3 has 10 GPU cores and")
    print("   Metal exposes no per-task partitioning)")
    print("=" * 74)

    # Section 3: orchestrator pipeline demo
    if not args.no_demo:
        _demo_orchestrator_pipeline()

    # Section 4: comparative test matrix
    all_results = run_benchmark_matrix()

    # Section 5: Pareto frontier analysis
    run_pareto_analysis(all_results)

    print("=" * 74)
    print("  Phase 7 complete.  (simulation — no model was loaded or run)")
    print("  Run with --api to start the persistent HTTP server.")
    print("  Run with --api-test for an interactive API demo.")
    print("=" * 74)


if __name__ == "__main__":
    main()
