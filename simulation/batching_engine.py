"""
Continuous Batching Engine — Phase 5: Iteration-Level Scheduling
=================================================================

Implements the core idea of modern LLM serving frameworks (vLLM,
TensorRT-LLM) adapted for SmolVLM's multimodal pipeline on Apple M3,
as a discrete-event SIMULATION driven by the Phase 2 calibrated cost
model plus documented modeling assumptions (see amio_constants.py).

Five capabilities
-----------------
1. Request State Machine
   WAITING → VISION_ENCODING → PREFILL_QUEUED → PREFILLING → DECODING →
   FINISHED.  Accounts for SmolVLM's heterogeneous stages: the vision
   stage cost is the MEASURED linear model T_vision(c) = 553.5·c + 27.4 ms
   (baseline/results_v2.json), LM prefill is the MEASURED quadratic fit,
   and decode is the MEASURED batch-1 fit plus a modeled per-extra-sequence
   KV-read term.

2. Iteration-Level Scheduler
   Admits / evicts requests between EVERY token generation step.
   - In-flight merging: a newly prefilled request joins the decode batch
     at the very next tick after its prefill completes.
   - Immediate eviction: a request that hits its stop token frees its
     KV blocks at the same tick it finishes.
   - Concurrent pipeline: vision ↔ prefill ↔ decode all run in parallel,
     gated only by compute-share partitioning.

3. SJF + Starvation Prevention
   Phase 2 cost model predicts total service time (T_prefill + max_output ×
   T_decode_base) per request.  The waiting queue is sorted SJF-first.
   Starvation counter tracks how many shorter requests jumped ahead; once
   starvation_count ≥ STARVATION_THRESHOLD the request is boosted to
   highest priority.  The FCFS control arm sorts strictly by arrival time
   at ALL scheduling points (vision, prefill, decode admission).

4. Compute-Share Partitioning Integration (Phase 3 SMOrchestrator)
   At each decode tick the SMOrchestrator splits the 38 abstract compute
   shares (historically mislabeled "SMs"; the M3 base die has 10 GPU cores
   and Metal exposes no per-task partitioning — see amio_constants) between
   the vision/prefill worker and the decode worker.  Vision latency is
   scaled up by the fraction of shares diverted to decode, and decode TBT
   is penalised (≥ 1.0×, capped at 1.15×) when it loses shares.

5. Static vs Continuous Comparison ("Battle of the Schedulers")
   StaticBatchingEngine processes a fixed batch B end-to-end (all vision,
   then all prefill, then decode until the LAST request finishes).  GPU
   decode is idle during the vision/prefill pipeline.  ContinuousBatching-
   Engine keeps the decode batch full throughout.  compare_schedulers()
   runs both across multiple RNG seeds and reports mean ± std.

Hardware target
---------------
  Apple M3, 8 GB unified memory, 100 GB/s bandwidth, 10 GPU cores
  (modeled as 38 abstract compute shares).
  Model: SmolVLM-Instruct-4bit, W4A8 mode (Phase 4 best config).

  Decode TBT model (MEASURED at batch 1, baseline/results_v2.json):
    TBT(B, ctx) = 18.33 + 0.00320·ctx + (B−1) × (ctx × 196,608 B / 100 GB/s)
  → TBT(1, 1560) ≈ 23.3 ms; the per-extra-sequence term (~3.1 ms/seq at
  1560 ctx) is a MODELED ASSUMPTION (batch scaling unmeasured).
  The 80 ms TBT SLA is met up to a batch of ≈19 at full context — a
  computed result of the measured model, not an assertion.  (The old
  83.75 + 3.95·B model with its "SLA unmeetable" conclusion is SUPERSEDED.)
"""

from __future__ import annotations

import heapq
import math
import random
import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Cross-module imports
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import amio_constants as C

try:
    from simulation.sm_orchestrator import SMOrchestrator, M3_TOTAL_SMs
    _PHASE_DEPS_OK = True
except ImportError:
    _PHASE_DEPS_OK = False
    M3_TOTAL_SMs = C.TOTAL_COMPUTE_SHARES

try:
    from simulation.kv_manager import PAGE_SIZE
except ImportError:
    PAGE_SIZE = 16          # tokens per paged KV block (vLLM default)


# ---------------------------------------------------------------------------
# Hardware & model constants (single source of truth: amio_constants)
# ---------------------------------------------------------------------------

# Decode TBT model — MEASURED batch-1 fit (baseline/results_v2.json):
#   TBT(B, ctx) = OVERHEAD + KV_SLOPE·ctx + (B−1) × ctx × FP16_KV_bytes / BW
# The 18.33 ms overhead already includes the per-step weight read; the
# per-extra-sequence KV term is a MODELED ASSUMPTION (batch unmeasured).
DECODE_OVERHEAD_MS:         float = C.DECODE_OVERHEAD_MS_MEASURED     # 18.33
DECODE_KV_MS_PER_CTX_TOKEN: float = C.DECODE_KV_MS_PER_CTX_TOKEN      # 0.00320
# Modeled per-extra-sequence KV read: FP16 KV prefix / bandwidth (ms/token)
DECODE_EXTRA_SEQ_MS_PER_CTX_TOKEN: float = (
    C.KV_BYTES_PER_TOKEN_FP16 / (C.M3_MEMORY_BW_GBPS * 1e9) * 1000.0  # ≈0.00197
)
# Default decode context: the measured 17-crop total input token count.
DEFAULT_DECODE_CTX: int = C.TOKENS_PER_CONFIG[C.MAX_CROPS]            # 1560
DECODE_TBT_BASE_MS: float = (
    DECODE_OVERHEAD_MS + DECODE_KV_MS_PER_CTX_TOKEN * DEFAULT_DECODE_CTX
)                                                                     # ≈23.3 at B=1

# Vision stage cost — MEASURED linear model (full GPU, no contention):
#   T_vision(c) = VISION_MS_PER_CROP·c + VISION_FIXED_MS
# (the old 5991 ms residual / "24 crops" attribution is SUPERSEDED — see
# amio_constants.VISION_BASE_MS_UNVALIDATED provenance note)
VISION_MS_PER_CROP: float = C.VISION_MS_PER_CROP                      # 553.5
VISION_FIXED_MS:    float = C.VISION_FIXED_MS                         # 27.4
MAX_CROPS:          int   = C.MAX_CROPS                               # 17
TOKENS_PER_CONFIG:  dict  = C.TOKENS_PER_CONFIG   # {1:100, 5:466, 10:922, 17:1560}
CROP_OPTIONS:       list  = sorted(TOKENS_PER_CONFIG)                 # [1, 5, 10, 17]

# LM prefill cost model — MEASURED fit (domain N ∈ (100, 1560))
COST_GAMMA: float = C.PREFILL_GAMMA
COST_BETA:  float = C.PREFILL_BETA
COST_ALPHA: float = C.PREFILL_ALPHA

# W4 KV cache (4× smaller than the 196,608 B/token FP16 baseline)
KV_BYTES_PER_TOKEN:     int   = C.KV_BYTES_PER_TOKEN_FP16          # 196,608 B
KV_W4_BYTES_PER_TOKEN:  int   = C.KV_BYTES_PER_TOKEN_W4            # 49,152 B
BYTES_PER_KV_BLOCK_W4:  int   = PAGE_SIZE * KV_W4_BYTES_PER_TOKEN
MB_PER_KV_BLOCK_W4:     float = BYTES_PER_KV_BLOCK_W4 / (1024 ** 2)
KV_POOL_BUDGET_MB:      float = C.KV_POOL_BUDGET_MB                # 4754 MB
MAX_KV_BLOCKS:          int   = int(KV_POOL_BUDGET_MB / MB_PER_KV_BLOCK_W4)

# Scheduler cap on concurrent decode sequences (Phase 4 starvation analysis).
# NOTE: the cap is a KV/scheduling limit, not an SLA guarantee — under the
# measured model the 80 ms TBT SLA holds up to a batch of ≈19 at full
# (1560-token) context, and TBT(70, 1560) ≈ 235 ms.
MAX_DECODE_BATCH:       int   = 70

# SJF starvation prevention
STARVATION_THRESHOLD:   int   = 8          # times bypassed before priority boost

# Compute-share partitioning constants (abstract shares, not hardware units)
SM_OP:          int = 30   # full decode share when vision holds its 8-share floor
SM_MIN_DECODE:  int = 4
SM_MIN_VISION:  int = 8

# SLA targets
TBT_SLA_MS:  float = C.TBT_SLA_MS    # 80 ms — met at moderate batch under the
                                     # measured model (ceiling computed, not asserted)
TTFT_SLA_MS: float = C.TTFT_SLA_MS


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _predict_prefill_ms(n_tokens: int) -> float:
    """MEASURED cost model: T_prefill(N) = γN² + βN + α (ms). Clipped to ≥1ms.

    Valid domain: N ∈ (100, 1560) (amio_constants.PREFILL_DOMAIN); values
    outside are extrapolations.
    """
    return max(1.0, COST_GAMMA * n_tokens ** 2 + COST_BETA * n_tokens + COST_ALPHA)


def _predict_tbt_ms(batch_size: int, ctx_tokens: float = DEFAULT_DECODE_CTX) -> float:
    """
    Decode TBT as a function of batch size and context length.

    Model (MEASURED at batch 1, baseline/results_v2.json):
        TBT(B, ctx) = 18.33 + 0.00320·ctx
                      + (B−1) × ctx × (196,608 B / 100 GB/s)

    The first two terms are the measured batch-1 fit (per-token timestamps
    over 160 tokens/config); the 18.33 ms overhead already includes the
    per-step weight read.  The per-extra-sequence KV-read term (~3.1 ms/seq
    at 1560 ctx) is a MODELED ASSUMPTION — batch scaling is unmeasured.
    At B=1 and 1560 ctx the formula gives ≈23.3 ms, comfortably under the
    80 ms TBT SLA; the SLA batch ceiling at full context is ≈19.
    """
    if batch_size <= 0:
        return 0.0
    return (
        DECODE_OVERHEAD_MS
        + DECODE_KV_MS_PER_CTX_TOKEN * ctx_tokens
        + (batch_size - 1) * DECODE_EXTRA_SEQ_MS_PER_CTX_TOKEN * ctx_tokens
    )


def _kv_blocks_for_seq(n_tokens: int) -> int:
    """Number of W4 KV blocks needed for a sequence of n_tokens."""
    return math.ceil(n_tokens / PAGE_SIZE)


def _infer_crops(n_visual_tokens: int) -> int:
    """Smallest processor crop setting whose total input tokens cover
    n_visual_tokens (defaults to the 17-crop maximum)."""
    for c in CROP_OPTIONS:
        if TOKENS_PER_CONFIG[c] >= n_visual_tokens:
            return c
    return MAX_CROPS


def _vision_ms_for_crops(n_crops: int) -> float:
    """MEASURED vision stage cost at full GPU: 553.5·c + 27.4 ms."""
    return VISION_MS_PER_CROP * n_crops + VISION_FIXED_MS


def _percentile(samples: List[float], q: float) -> float:
    """Nearest-rank percentile (ceil convention) of an UNSORTED sample list."""
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = max(0, min(math.ceil(q * len(s)) - 1, len(s) - 1))
    return s[idx]


# ---------------------------------------------------------------------------
# Request State Machine
# ---------------------------------------------------------------------------

class RequestState(Enum):
    """
    Lifecycle states for one multimodal inference request.

    Transitions
    -----------
    WAITING         → VISION_ENCODING  : dequeued by vision worker
    VISION_ENCODING → PREFILL_QUEUED   : SigLIP forward pass complete
    PREFILL_QUEUED  → PREFILLING       : prefill worker picks up the request
    PREFILLING      → DECODING         : LM KV cache built; joins decode batch
    DECODING        → FINISHED         : stop token OR max_output_tokens reached
    """
    WAITING         = "WAITING"
    VISION_ENCODING = "VISION_ENCODING"
    PREFILL_QUEUED  = "PREFILL_QUEUED"
    PREFILLING      = "PREFILLING"
    DECODING        = "DECODING"
    FINISHED        = "FINISHED"


@dataclass
class Request:
    """One multimodal inference request tracked through the full pipeline."""

    req_id:            int
    arrival_time_ms:   float
    n_visual_tokens:   int          # total LM input tokens (TOKENS_PER_CONFIG)
    max_output_tokens: int          # stop criterion
    n_crops:           int = 0      # processor crop setting ∈ {1,5,10,17};
                                    # 0 → inferred from n_visual_tokens

    # ── Mutable state ─────────────────────────────────────────────────────
    state:             RequestState = field(default=RequestState.WAITING,    compare=False)
    starvation_count:  int          = field(default=0,                       compare=False)
    tokens_generated:  int          = field(default=0,                       compare=False)
    kv_blocks_held:    int          = field(default=0,                       compare=False)

    # ── Timing telemetry ──────────────────────────────────────────────────
    vision_start_ms:   float = field(default=-1.0, compare=False)
    vision_done_ms:    float = field(default=-1.0, compare=False)
    prefill_start_ms:  float = field(default=-1.0, compare=False)
    prefill_done_ms:   float = field(default=-1.0, compare=False)
    first_token_ms:    float = field(default=-1.0, compare=False)
    last_token_ms:     float = field(default=-1.0, compare=False)
    tbt_samples:       List[float] = field(default_factory=list, compare=False)

    def __post_init__(self) -> None:
        if self.n_crops <= 0:
            self.n_crops = _infer_crops(self.n_visual_tokens)

    # ── Derived ───────────────────────────────────────────────────────────
    @property
    def ttft_ms(self) -> float:
        if self.first_token_ms < 0:
            return -1.0
        return self.first_token_ms - self.arrival_time_ms

    @property
    def total_latency_ms(self) -> float:
        if self.last_token_ms < 0:
            return -1.0
        return self.last_token_ms - self.arrival_time_ms

    @property
    def current_kv_tokens(self) -> int:
        return self.n_visual_tokens + self.tokens_generated

    @property
    def blocks_needed(self) -> int:
        return _kv_blocks_for_seq(self.current_kv_tokens)

    @property
    def sjf_key(self) -> float:
        """
        SJF sort key — lower means higher scheduling priority.

        Uses Phase 2 cost model to predict total service time:
          T_service = T_prefill(n_visual_tokens) + max_output_tokens × T_decode_base

        Starvation promotion: once starvation_count ≥ STARVATION_THRESHOLD
        the key collapses to 0.0, placing the request at the absolute front
        regardless of predicted length.
        """
        if self.starvation_count >= STARVATION_THRESHOLD:
            return 0.0
        t_prefill = _predict_prefill_ms(self.n_visual_tokens)
        t_decode  = self.max_output_tokens * DECODE_TBT_BASE_MS
        return t_prefill + t_decode

    @property
    def is_done(self) -> bool:
        return self.state == RequestState.FINISHED


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class SimMetrics:
    """Aggregated performance metrics collected during one simulation run.

    Accounting convention (both engines):
      gpu_active_ms + gpu_idle_ms ≈ sim_duration_ms  (wall-clock split of the
      decode resource).  Padding waste in static batches (slots generating
      tokens for already-finished requests) is counted as idle and also
      tracked separately in wasted_ms.
    """

    scheduler_name:            str   = "unknown"
    n_requests:                int   = 0
    n_finished:                int   = 0
    sim_duration_ms:           float = 0.0
    total_tokens_generated:    int   = 0

    # Raw samples (populated during simulation)
    ttft_samples:              List[float] = field(default_factory=list)
    tbt_samples:               List[float] = field(default_factory=list)   # per-token weighted
    decode_batch_size_samples: List[int]   = field(default_factory=list)
    kv_util_samples:           List[float] = field(default_factory=list)

    # Counters
    gpu_active_ms:             float = 0.0   # wall time with useful decode work
    gpu_idle_ms:               float = 0.0   # wall time decode idle (incl. padding waste)
    wasted_ms:                 float = 0.0   # padding waste inside gpu_idle_ms (static)
    max_waiting_time_ms:       float = 0.0   # max queue wait: first processing start − arrival
    max_tbt_ms:                float = 0.0   # worst single-step TBT observed

    # Derived (populated by finalize())
    throughput_req_s:          float = 0.0
    throughput_tok_s:          float = 0.0
    gpu_utilization_pct:       float = 0.0
    p50_ttft_ms:               float = 0.0
    p99_ttft_ms:               float = 0.0
    avg_tbt_ms:                float = 0.0
    p99_tbt_ms:                float = 0.0
    avg_decode_batch:          float = 0.0
    avg_kv_utilization_pct:    float = 0.0
    batch_saturation_pct:      float = 0.0

    # Across-seed statistics: metric name → (mean, std).  Populated only by
    # compare_schedulers() when n_seeds > 1.
    headline_mean_std:         Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def finalize(self) -> None:
        """Compute derived statistics from raw samples collected during the run."""
        dur_s = self.sim_duration_ms / 1000.0
        self.throughput_req_s = self.n_finished / dur_s if dur_s > 0 else 0.0
        self.throughput_tok_s = self.total_tokens_generated / dur_s if dur_s > 0 else 0.0

        total_gpu_ms = self.gpu_active_ms + self.gpu_idle_ms
        self.gpu_utilization_pct = (
            self.gpu_active_ms / max(total_gpu_ms, 1e-9) * 100.0
        )

        if self.ttft_samples:
            self.p50_ttft_ms = _percentile(self.ttft_samples, 0.50)
            self.p99_ttft_ms = _percentile(self.ttft_samples, 0.99)

        if self.tbt_samples:
            self.avg_tbt_ms = sum(self.tbt_samples) / len(self.tbt_samples)
            self.p99_tbt_ms = _percentile(self.tbt_samples, 0.99)

        if self.decode_batch_size_samples:
            self.avg_decode_batch = (
                sum(self.decode_batch_size_samples) / len(self.decode_batch_size_samples)
            )
            self.batch_saturation_pct = self.avg_decode_batch / MAX_DECODE_BATCH * 100.0

        if self.kv_util_samples:
            self.avg_kv_utilization_pct = (
                sum(self.kv_util_samples) / len(self.kv_util_samples)
            )


# ---------------------------------------------------------------------------
# Lightweight KV block pool (Phase 4 PagedBackend simplified for simulation)
# ---------------------------------------------------------------------------

class _KVBlockPool:
    """
    Tracks W4-quantised KV block availability across the decode batch.

    Uses a simple free-block counter; the full block-table logic lives in
    simulation/kv_manager.py (PagedBackend).  We track block counts only,
    deferring physical address management to the real runtime.
    """

    def __init__(self, total_blocks: int = MAX_KV_BLOCKS):
        self.total_blocks  = total_blocks
        self._used_blocks  = 0
        self._held: Dict[int, int] = {}   # req_id → blocks held

    @property
    def free_blocks(self) -> int:
        return self.total_blocks - self._used_blocks

    @property
    def utilization_pct(self) -> float:
        return self._used_blocks / self.total_blocks * 100.0

    def can_allocate(self, n_blocks: int) -> bool:
        return self.free_blocks >= n_blocks

    def allocate(self, req_id: int, n_blocks: int) -> bool:
        prev = self._held.get(req_id, 0)
        delta = n_blocks - prev
        if delta > self.free_blocks:
            return False
        self._held[req_id] = n_blocks
        self._used_blocks += delta
        return True

    def grow(self, req_id: int, new_blocks: int) -> bool:
        """Grow an existing allocation (one new block per PAGE_SIZE tokens)."""
        return self.allocate(req_id, new_blocks)

    def free(self, req_id: int) -> None:
        blocks = self._held.pop(req_id, 0)
        self._used_blocks = max(0, self._used_blocks - blocks)


# ---------------------------------------------------------------------------
# Internal event infrastructure
# ---------------------------------------------------------------------------

class _EvType(Enum):
    ARRIVAL      = 0
    VISION_DONE  = 1
    PREFILL_DONE = 2
    DECODE_TICK  = 3


_ev_counter = 0   # monotonic tie-breaker so heapq never compares Request objects


def _push(heap: list, time_ms: float, ev_type: _EvType, req_id: int = -1) -> None:
    global _ev_counter
    heapq.heappush(heap, (time_ms, _ev_counter, ev_type, req_id))
    _ev_counter += 1


# ---------------------------------------------------------------------------
# Continuous Batching Engine
# ---------------------------------------------------------------------------

class ContinuousBatchingEngine:
    """
    Iteration-level scheduler for SmolVLM multimodal inference (simulated).

    Architecture
    ------------
    • vision_worker  — SigLIP encoder, one request at a time.
      Latency from the MEASURED per-crop model (553.5·c + 27.4 ms), scaled
      by compute-share allocation.

    • prefill_worker — LM KV prefill, one request at a time (pipelined
      with vision: while R2 does vision, R1 does prefill).

    • decode_batch   — up to MAX_DECODE_BATCH concurrent requests.
      Admits newly prefilled requests and evicts finished ones at every tick.

    Scheduling policy
    -----------------
    "SJF"  : SJF with starvation prevention (see Request.sjf_key) at all
             three scheduling points (vision start, prefill start, decode
             admission).
    "FCFS" : strict arrival-time order at all three scheduling points.

    Compute-share partitioning
    --------------------------
    Uses Phase 3 SMOrchestrator.  When decode is running and vision/prefill
    is also active, the decode worker claims priority shares; the residual
    goes to vision.  Vision latency is scaled by (total_shares / sm_vision).
    Decode TBT receives a contention PENALTY clamp(sqrt(SM_OP / sm_decode),
    1.0, 1.15) — decode never speeds up under contention.

    Idle accounting
    ---------------
    gpu_idle_ms accumulates real simulated-clock gaps during which the decode
    batch is empty (including the initial gap from t=0 until the first decode
    tick), so gpu_active_ms + gpu_idle_ms ≈ sim_duration_ms.
    """

    def __init__(
        self,
        requests:          List[Request],
        max_decode_batch:  int   = MAX_DECODE_BATCH,
        scheduler_policy:  str   = "SJF",     # "SJF" | "FCFS"
        n_vision_workers:  int   = 1,
        random_seed:       int   = 42,        # kept for API compat; the engine
                                              # itself is deterministic — seeds
                                              # live in workload generation
    ):
        self.requests          = {r.req_id: r for r in requests}
        self.max_decode_batch  = max_decode_batch
        self.policy            = scheduler_policy
        self.n_vision_workers  = n_vision_workers
        self.random_seed       = random_seed

        self._sm_orch = SMOrchestrator() if _PHASE_DEPS_OK else None

        # Worker state: each slot stores (req_id, free_at_ms)
        self._vision_slots: List[Tuple[int, float]] = []   # (req_id, free_at)
        self._prefill_slot: Optional[Tuple[int, float]] = None

        # Queues
        self._waiting:       List[Request] = []   # sorted by policy on admission
        self._prefill_ready: List[Request] = []   # finished vision, awaiting prefill
        self._prefill_done:  List[Request] = []   # ready to join decode batch

        # Decode batch
        self._decode_batch: List[Request] = []

        # KV pool
        self._kv = _KVBlockPool()

        # Metrics accumulator
        self._metrics = SimMetrics(
            scheduler_name=f"Continuous-{scheduler_policy}",
            n_requests=len(requests),
        )
        self._clock = 0.0
        # Sim-clock timestamp at which the decode batch last drained to 0.
        # Starts at 0.0: decode is idle from t=0 until the first decode work.
        self._decode_idle_since: Optional[float] = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SimMetrics:
        """Run the full simulation and return consolidated metrics."""
        events: list = []

        # Seed arrival events
        for req in self.requests.values():
            _push(events, req.arrival_time_ms, _EvType.ARRIVAL, req.req_id)

        # Seed initial decode tick (process immediately if batch non-empty)
        _push(events, 0.0, _EvType.DECODE_TICK)

        while events:
            time_ms, _, ev_type, req_id = heapq.heappop(events)
            self._clock = time_ms

            if ev_type == _EvType.ARRIVAL:
                self._on_arrival(self.requests[req_id], events)

            elif ev_type == _EvType.VISION_DONE:
                self._on_vision_done(self.requests[req_id], events)

            elif ev_type == _EvType.PREFILL_DONE:
                self._on_prefill_done(self.requests[req_id], events)

            elif ev_type == _EvType.DECODE_TICK:
                self._on_decode_tick(events)

        # Close any trailing decode-idle window at the final sim clock.
        if self._decode_idle_since is not None:
            self._metrics.gpu_idle_ms += max(0.0, self._clock - self._decode_idle_since)
            self._decode_idle_since = None

        self._metrics.sim_duration_ms = self._clock
        self._metrics.n_finished = sum(
            1 for r in self.requests.values() if r.is_done
        )
        if self._metrics.n_finished < self._metrics.n_requests:
            print(
                f"WARNING [{self._metrics.scheduler_name}]: only "
                f"{self._metrics.n_finished}/{self._metrics.n_requests} requests "
                f"finished — check KV admission feasibility.",
                file=sys.stderr,
            )
        self._metrics.finalize()
        return self._metrics

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_arrival(self, req: Request, events: list) -> None:
        """Handle a new request arriving at the system."""
        self._waiting.append(req)
        # Immediately try to start vision if a worker slot is free
        self._try_start_vision(events)

    def _on_vision_done(self, req: Request, events: list) -> None:
        """Vision encoding completed — move to prefill queue."""
        req.vision_done_ms = self._clock
        req.state = RequestState.PREFILL_QUEUED

        # Free the vision worker slot
        self._vision_slots = [
            (rid, t) for (rid, t) in self._vision_slots if rid != req.req_id
        ]

        # Move to prefill ready queue (prefill_start_ms is stamped when the
        # prefill worker actually picks the request up, not here)
        self._prefill_ready.append(req)

        # Start prefill if worker is free
        self._try_start_prefill(events)

        # Start next vision if waiting queue is non-empty
        self._try_start_vision(events)

    def _on_prefill_done(self, req: Request, events: list) -> None:
        """Prefill completed — request is ready to join decode batch."""
        req.prefill_done_ms = self._clock
        self._prefill_slot = None
        self._prefill_done.append(req)

        # Start next prefill immediately if another request is ready
        self._try_start_prefill(events)

        # The request will be admitted to the decode batch on the next decode tick.
        # If no decode tick is scheduled (decode batch was empty), push one now.
        if not self._decode_batch and not any(
            e[2] == _EvType.DECODE_TICK for e in events
        ):
            _push(events, self._clock, _EvType.DECODE_TICK)

    def _on_decode_tick(self, events: list) -> None:
        """
        Core iteration-level scheduling loop.

        1. Grow KV blocks for requests that crossed a page boundary.
        2. Evict finished requests (stop token / max output tokens).
        3. Admit newly prefilled requests (KV budget permitting).
        4. Generate one token for every request in the decode batch.
        5. Record metrics.
        6. Schedule the next tick if batch is non-empty.
        """
        # ── Phase 1: grow KV cache for active decode requests ─────────────
        for req in self._decode_batch:
            new_blocks = req.blocks_needed
            if new_blocks > req.kv_blocks_held:
                if self._kv.can_allocate(new_blocks - req.kv_blocks_held):
                    self._kv.grow(req.req_id, new_blocks)
                    req.kv_blocks_held = new_blocks
                # (If KV is full, we stall growth — in practice this triggers eviction)

        # ── Phase 2: evict finished requests ──────────────────────────────
        evicted = []
        for req in self._decode_batch:
            if req.tokens_generated >= req.max_output_tokens:
                req.state = RequestState.FINISHED
                req.last_token_ms = self._clock
                self._kv.free(req.req_id)
                self._metrics.total_tokens_generated += req.tokens_generated
                evicted.append(req)

        if evicted:
            evicted_ids = {r.req_id for r in evicted}
            self._decode_batch = [r for r in self._decode_batch if r.req_id not in evicted_ids]

        # ── Phase 3: admit newly prefilled requests ────────────────────────
        # Order by scheduling policy; admit as many as KV + batch budget allow.
        if self.policy == "SJF":
            self._prefill_done.sort(key=lambda r: r.sjf_key)
        else:
            self._prefill_done.sort(key=lambda r: r.arrival_time_ms)
        admitted = []
        for req in list(self._prefill_done):
            if len(self._decode_batch) >= self.max_decode_batch:
                break
            # Allocate initial KV blocks
            init_blocks = _kv_blocks_for_seq(req.n_visual_tokens)
            if self._kv.allocate(req.req_id, init_blocks):
                req.kv_blocks_held = init_blocks
                req.state = RequestState.DECODING
                self._decode_batch.append(req)
                admitted.append(req.req_id)

        self._prefill_done = [r for r in self._prefill_done if r.req_id not in set(admitted)]

        # ── Phase 4: generate one token ───────────────────────────────────
        B = len(self._decode_batch)
        if B == 0:
            # Decode drained — open an idle window (closed at next decode work)
            if self._decode_idle_since is None:
                self._decode_idle_since = self._clock

            if self._prefill_done:
                # Requests are prefilled but could not be admitted.  With an
                # empty decode batch the KV pool is empty, so a blocked request
                # can only mean it needs more blocks than the entire pool:
                # it can never run.  Drop it (warn) instead of deadlocking.
                still_blocked: List[Request] = []
                for r in self._prefill_done:
                    if _kv_blocks_for_seq(r.n_visual_tokens) > self._kv.total_blocks:
                        print(
                            f"WARNING: req {r.req_id} needs "
                            f"{_kv_blocks_for_seq(r.n_visual_tokens)} KV blocks "
                            f"> pool total {self._kv.total_blocks}; dropping "
                            f"(infeasible request).",
                            file=sys.stderr,
                        )
                    else:
                        still_blocked.append(r)
                self._prefill_done = still_blocked
                # Defensive retry: if anything remains blocked, schedule a
                # retry tick so requests are never silently stranded.
                if self._prefill_done and not any(
                    e[2] == _EvType.DECODE_TICK for e in events
                ):
                    _push(events, self._clock + DECODE_TBT_BASE_MS, _EvType.DECODE_TICK)
            return

        # Decode is doing useful work — close any open idle window.
        if self._decode_idle_since is not None:
            self._metrics.gpu_idle_ms += max(0.0, self._clock - self._decode_idle_since)
            self._decode_idle_since = None

        # Compute-share partitioning: if vision/prefill runs concurrently,
        # decode pays a contention penalty (never a bonus).
        tbt = self._compute_tbt(B)

        for req in self._decode_batch:
            req.tokens_generated += 1
            req.tbt_samples.append(tbt)
            if req.first_token_ms < 0:
                # The first token exists only after this step's TBT elapses
                # (same convention as the static engine).
                req.first_token_ms = self._clock + tbt
                self._metrics.ttft_samples.append(req.ttft_ms)

        # ── Phase 5: record metrics ────────────────────────────────────────
        self._metrics.gpu_active_ms += tbt
        self._metrics.tbt_samples.extend([tbt] * B)   # per-token weighting
        self._metrics.max_tbt_ms = max(self._metrics.max_tbt_ms, tbt)
        self._metrics.decode_batch_size_samples.append(B)
        self._metrics.kv_util_samples.append(self._kv.utilization_pct)

        # ── Phase 6: schedule next tick ────────────────────────────────────
        _push(events, self._clock + tbt, _EvType.DECODE_TICK)

    # ------------------------------------------------------------------
    # Worker scheduling helpers
    # ------------------------------------------------------------------

    def _try_start_vision(self, events: list) -> None:
        """Start vision encoding for the highest-priority waiting request."""
        if not self._waiting:
            return
        if len(self._vision_slots) >= self.n_vision_workers:
            return

        if self.policy == "SJF":
            # Sort by SJF key; update starvation counts for all skipped requests
            self._waiting.sort(key=lambda r: r.sjf_key)
            selected = self._waiting[0]
            # Increment starvation for all other waiting requests that arrived
            # earlier than the selected one (they were displaced by a shorter job)
            for r in self._waiting[1:]:
                if r.arrival_time_ms <= selected.arrival_time_ms:
                    r.starvation_count += 1
        else:
            # FCFS — strict arrival-time order
            self._waiting.sort(key=lambda r: r.arrival_time_ms)
            selected = self._waiting[0]

        self._waiting.remove(selected)
        self._start_vision(selected, events)

    def _try_start_prefill(self, events: list) -> None:
        """Start LM prefill for the head of the prefill-ready queue."""
        if self._prefill_slot is not None:
            return
        if not self._prefill_ready:
            return

        if self.policy == "SJF":
            # SJF key for prefill ordering too (shorter prefill = sooner decode)
            self._prefill_ready.sort(key=lambda r: r.sjf_key)
        else:
            # FCFS — strict arrival-time order
            self._prefill_ready.sort(key=lambda r: r.arrival_time_ms)
        req = self._prefill_ready.pop(0)
        self._start_prefill(req, events)

    def _start_vision(self, req: Request, events: list) -> None:
        """Schedule vision encoding, accounting for compute-share partitioning."""
        req.vision_start_ms = self._clock
        req.state = RequestState.VISION_ENCODING

        # Queue wait = first processing start − arrival
        wait_ms = max(0.0, self._clock - req.arrival_time_ms)
        self._metrics.max_waiting_time_ms = max(
            self._metrics.max_waiting_time_ms, wait_ms
        )

        t_vision = self._compute_vision_latency(req)
        done_at  = self._clock + t_vision
        self._vision_slots.append((req.req_id, done_at))
        _push(events, done_at, _EvType.VISION_DONE, req.req_id)

    def _start_prefill(self, req: Request, events: list) -> None:
        """Schedule LM prefill, accounting for compute-share partitioning."""
        req.state = RequestState.PREFILLING
        req.prefill_start_ms = self._clock

        t_prefill = self._compute_prefill_latency(req.n_visual_tokens)
        done_at   = self._clock + t_prefill
        self._prefill_slot = (req.req_id, done_at)
        _push(events, done_at, _EvType.PREFILL_DONE, req.req_id)

    # ------------------------------------------------------------------
    # Share-partitioned latency models
    # ------------------------------------------------------------------

    def _compute_vision_latency(self, req: Request) -> float:
        """
        Vision latency for one request: the MEASURED per-crop model scaled
        by the fraction of compute shares available for vision.

        T_vision = (553.5 × n_crops + 27.4) × (total_shares / sm_vision)
        """
        base = _vision_ms_for_crops(req.n_crops)
        n_decode = len(self._decode_batch)
        if n_decode > 0 and self._sm_orch is not None:
            alloc     = self._sm_orch.allocate(n_pending_decode=n_decode)
            sm_vision = max(alloc.sm_vision, SM_MIN_VISION)
            scale     = M3_TOTAL_SMs / sm_vision
            return base * scale
        return base

    def _compute_prefill_latency(self, n_tokens: int) -> float:
        """
        Scale LM prefill latency by compute-share availability.

        Prefill is compute-bound (unlike decode which is BW-bound).
        Uses the same share allocation as vision — both share the residual
        shares not claimed by the decode worker.
        """
        base = _predict_prefill_ms(n_tokens)
        n_decode = len(self._decode_batch)
        if n_decode > 0 and self._sm_orch is not None:
            alloc      = self._sm_orch.allocate(n_pending_decode=n_decode)
            sm_prefill = max(alloc.sm_vision, SM_MIN_VISION)
            scale      = M3_TOTAL_SMs / sm_prefill
            return base * scale
        return base

    def _compute_tbt(self, batch_size: int) -> float:
        """
        Compute decode TBT for the current batch, with a contention PENALTY.

        Decode is memory-bandwidth bound, not compute-bound, so losing
        compute shares to vision/prefill hurts it only mildly.  We model
        this as a penalty that grows as sm_decode shrinks below the
        full-decode-share baseline SM_OP (=30), capped at +15%:

            penalty = clamp( sqrt(SM_OP / sm_decode), 1.0, 1.15 )

        The penalty is always ≥ 1.0 — concurrent vision/prefill can never
        make decode FASTER (the previous model inverted this and granted a
        speed bonus under contention).
        """
        if self._decode_batch:
            avg_ctx = sum(r.current_kv_tokens for r in self._decode_batch) / len(
                self._decode_batch
            )
        else:
            avg_ctx = DEFAULT_DECODE_CTX
        base_tbt = _predict_tbt_ms(batch_size, avg_ctx)
        # If vision/prefill is running in parallel, decode may lose some shares
        n_preprocessing = (
            len(self._vision_slots)
            + (1 if self._prefill_slot is not None else 0)
        )
        if n_preprocessing > 0 and self._sm_orch is not None:
            alloc     = self._sm_orch.allocate(n_pending_decode=batch_size)
            sm_decode = max(alloc.sm_decode, SM_MIN_DECODE)
            sm_scale  = min(1.15, max(1.0, math.sqrt(SM_OP / sm_decode)))
            return base_tbt * sm_scale
        return base_tbt


# ---------------------------------------------------------------------------
# Static Batching Engine (Baseline)
# ---------------------------------------------------------------------------

class StaticBatchingEngine:
    """
    Baseline static-batch scheduler.

    Processes a fixed batch B all the way through vision → prefill → decode
    before starting any new requests.  Models the wasted decode slots that
    arise when shorter requests finish decode but cannot be evicted until
    the longest request in the batch completes.

    Accounting: the decode phase wall time is max_steps × TBT(B); the useful
    fraction is Σ max_output_tokens / (B × max_steps).  Useful wall time goes
    to gpu_active_ms; the padding remainder goes to gpu_idle_ms and is also
    tracked in wasted_ms, so gpu_active_ms + gpu_idle_ms == sim_duration_ms.

    Policy options
    --------------
    "FCFS"  : fill each batch in first-come-first-served order (default).
    "SJF"   : fill each batch with the globally shortest-predicted jobs first.
              Note: even with SJF batch selection, ALL B requests must finish
              before the next batch starts (this is the fundamental static
              batching inefficiency).
    """

    def __init__(
        self,
        requests:        List[Request],
        batch_size:      int = 20,
        scheduler_policy: str = "FCFS",
    ):
        self.all_requests    = sorted(requests, key=lambda r: r.arrival_time_ms)
        self.batch_size      = batch_size
        self.policy          = scheduler_policy
        self._metrics = SimMetrics(
            scheduler_name=f"Static-{scheduler_policy}",
            n_requests=len(requests),
        )

    def run(self) -> SimMetrics:
        """
        Simulate static batching analytically.

        For each batch of B requests (selected by policy):
          1. Vision phase:  Σ T_vision(n_vis_i)   (sequential, all shares)
          2. Prefill phase: Σ T_prefill(n_i)      (sequential, all shares)
          3. Decode phase:  max_steps × TBT(B)    (until last request done)

        GPU decode is idle during phases 1 and 2.  In phase 3 the padding
        slots (finished requests still occupying the batch) count as idle
        (wasted_ms), the useful slots as active — see class docstring.
        """
        clock            = 0.0
        remaining        = list(self.all_requests)
        finished         = []

        while remaining:
            # Select next batch
            pool = [r for r in remaining if r.arrival_time_ms <= clock]
            if not pool:
                # Advance clock to next arrival
                next_arrival = min(r.arrival_time_ms for r in remaining)
                self._metrics.gpu_idle_ms += next_arrival - clock
                clock = next_arrival
                pool  = [r for r in remaining if r.arrival_time_ms <= clock]

            if self.policy == "SJF":
                pool.sort(key=lambda r: r.sjf_key)
            # else FCFS — pool is already arrival-time sorted

            batch = pool[: self.batch_size]
            batch_ids = {r.req_id for r in batch}
            remaining = [r for r in remaining if r.req_id not in batch_ids]

            B = len(batch)

            # Queue wait: processing of this batch starts now
            for req in batch:
                wait_ms = max(0.0, clock - req.arrival_time_ms)
                self._metrics.max_waiting_time_ms = max(
                    self._metrics.max_waiting_time_ms, wait_ms
                )

            # ── Vision phase (GPU decode idle; sequential per request) ─────
            for req in batch:
                t_vis = _vision_ms_for_crops(req.n_crops)
                req.vision_start_ms = clock
                req.vision_done_ms  = clock + t_vis
                self._metrics.gpu_idle_ms += t_vis
                clock += t_vis

            # ── Prefill phase (GPU decode idle) ────────────────────────────
            t_prefill_start = clock
            for req in batch:
                t_pf              = _predict_prefill_ms(req.n_visual_tokens)
                req.prefill_start_ms = clock
                req.prefill_done_ms  = clock + t_pf
                clock += t_pf            # sequential prefill
            self._metrics.gpu_idle_ms += clock - t_prefill_start

            # ── Decode phase ───────────────────────────────────────────────
            decode_start = clock
            avg_ctx      = sum(
                r.n_visual_tokens + r.max_output_tokens / 2 for r in batch
            ) / B
            tbt          = _predict_tbt_ms(B, avg_ctx)
            max_steps    = max(r.max_output_tokens for r in batch)
            useful_tokens = sum(r.max_output_tokens for r in batch)

            for req in batch:
                req.tokens_generated = req.max_output_tokens
                req.first_token_ms   = decode_start + tbt    # first token after first step
                req.last_token_ms    = decode_start + req.max_output_tokens * tbt
                self._metrics.ttft_samples.append(req.ttft_ms)

            # All requests wait until the LAST one finishes — static batch drain.
            decode_duration = max_steps * tbt
            # Useful vs wasted wall time (see class docstring): the fraction
            # of batch slots doing useful work is useful_tokens/(B·max_steps).
            active_ms = decode_duration * useful_tokens / (B * max_steps)
            wasted_ms = decode_duration - active_ms
            self._metrics.gpu_active_ms += active_ms
            self._metrics.gpu_idle_ms   += wasted_ms
            self._metrics.wasted_ms     += wasted_ms

            self._metrics.total_tokens_generated += useful_tokens
            self._metrics.tbt_samples.extend([tbt] * useful_tokens)  # per-token
            self._metrics.decode_batch_size_samples.append(B)
            self._metrics.max_tbt_ms = max(self._metrics.max_tbt_ms, tbt)

            clock += decode_duration

            # KV utilisation at peak (all B requests at max tokens)
            kv_tokens_peak = sum(
                r.n_visual_tokens + r.max_output_tokens for r in batch
            )
            kv_blocks_peak = _kv_blocks_for_seq(kv_tokens_peak)
            kv_util = min(100.0, kv_blocks_peak / MAX_KV_BLOCKS * 100.0)
            self._metrics.kv_util_samples.append(kv_util)

            # Mark finished
            for req in batch:
                req.state = RequestState.FINISHED
                finished.append(req)

        self._metrics.n_finished   = len(finished)
        self._metrics.sim_duration_ms = clock
        self._metrics.finalize()
        return self._metrics


# ---------------------------------------------------------------------------
# Request generator (Poisson arrivals)
# ---------------------------------------------------------------------------

def generate_poisson_requests(
    n_requests:        int   = 500,
    arrival_rate_rps:  float = 5.0,      # requests per second
    seed:              int   = 42,
    min_output_tokens: int   = 32,
    max_output_tokens: int   = 512,
) -> List[Request]:
    """
    Generate a Poisson-arrival request workload.

    Each request draws one of the processor's REAL crop settings uniformly
    from {1, 5, 10, 17} (size.longest_edge {384, 768, 1152, 1536}); its LM
    input token count is the measured total for that setting
    (TOKENS_PER_CONFIG: {1:100, 5:466, 10:922, 17:1560}).

    Output token counts follow a Pareto(shape=1.2, scale=32) distribution
    clipped to [min_output, max_output]; the clipped mean is ≈ 100 tokens
    (the unclipped Pareto mean would be 192, but the 512-token cap pulls it
    down).  Most responses are short, with a long tail.

    Parameters
    ----------
    n_requests        : total number of requests in the burst
    arrival_rate_rps  : mean arrivals per second (λ for the Poisson process)
    seed              : RNG seed for reproducibility
    """
    rng   = random.Random(seed)
    reqs  = []
    clock = 0.0

    for i in range(n_requests):
        # Poisson inter-arrival: Exponential(1/λ) distribution
        inter_ms  = rng.expovariate(arrival_rate_rps) * 1000.0
        clock    += inter_ms

        # Crop setting ~ Uniform over the real processor configs
        n_crops = rng.choice(CROP_OPTIONS)
        n_vis   = TOKENS_PER_CONFIG[n_crops]

        # Output tokens ~ Pareto(shape=1.2, scale=32) clipped to [min, max]
        raw_out = int(min_output_tokens * (1.0 - rng.random()) ** (-1 / 1.2))
        n_out   = max(min_output_tokens, min(raw_out, max_output_tokens))

        reqs.append(Request(
            req_id            = i,
            arrival_time_ms   = clock,
            n_visual_tokens   = n_vis,
            max_output_tokens = n_out,
            n_crops           = n_crops,
        ))

    return reqs


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

_HEADLINE_METRICS = (
    "throughput_req_s",
    "throughput_tok_s",
    "gpu_utilization_pct",
    "p50_ttft_ms",
    "p99_ttft_ms",
    "avg_tbt_ms",
    "p99_tbt_ms",
)


def _aggregate_metrics(per_seed: List[SimMetrics]) -> SimMetrics:
    """
    Pool per-seed SimMetrics into one aggregate.

    Raw samples and counters are pooled/summed (so percentiles and ratios are
    computed over all seeds' data); headline_mean_std records the across-seed
    mean ± std of each headline metric.
    """
    if len(per_seed) == 1:
        return per_seed[0]

    agg = SimMetrics(scheduler_name=per_seed[0].scheduler_name)
    for m in per_seed:
        agg.n_requests             += m.n_requests
        agg.n_finished             += m.n_finished
        agg.sim_duration_ms        += m.sim_duration_ms
        agg.total_tokens_generated += m.total_tokens_generated
        agg.gpu_active_ms          += m.gpu_active_ms
        agg.gpu_idle_ms            += m.gpu_idle_ms
        agg.wasted_ms              += m.wasted_ms
        agg.max_waiting_time_ms     = max(agg.max_waiting_time_ms, m.max_waiting_time_ms)
        agg.max_tbt_ms              = max(agg.max_tbt_ms, m.max_tbt_ms)
        agg.ttft_samples.extend(m.ttft_samples)
        agg.tbt_samples.extend(m.tbt_samples)
        agg.decode_batch_size_samples.extend(m.decode_batch_size_samples)
        agg.kv_util_samples.extend(m.kv_util_samples)
    agg.finalize()

    for name in _HEADLINE_METRICS:
        vals = [getattr(m, name) for m in per_seed]
        mean = sum(vals) / len(vals)
        var  = sum((v - mean) ** 2 for v in vals) / len(vals)
        agg.headline_mean_std[name] = (mean, math.sqrt(var))
    return agg


def compare_schedulers(
    n_requests:         int   = 500,
    arrival_rate_rps:   float = 5.0,
    static_batch_size:  int   = 20,
    seed:               int   = 42,
    n_seeds:            int   = 5,
) -> Tuple[SimMetrics, SimMetrics, SimMetrics]:
    """
    Run all three scheduler configurations across n_seeds independent
    workloads and return (continuous_SJF, continuous_FCFS, static_FCFS)
    aggregated metrics.

    Each seed s in [seed, seed + n_seeds) generates its own Poisson workload
    (identical across the three schedulers within a seed).  The returned
    SimMetrics pool raw samples across seeds; SimMetrics.headline_mean_std
    holds mean ± std of the headline metrics across seeds (when n_seeds > 1).

    Parameters
    ----------
    n_requests        : number of Poisson-arrival requests per seed
    arrival_rate_rps  : λ for Poisson inter-arrival process
    static_batch_size : B for the static baseline
    seed              : base RNG seed
    n_seeds           : number of independent seeds (default 5)
    """
    def _clone(reqs: List[Request]) -> List[Request]:
        """Deep-copy request list so each scheduler starts from a clean state."""
        return [
            Request(
                req_id            = r.req_id,
                arrival_time_ms   = r.arrival_time_ms,
                n_visual_tokens   = r.n_visual_tokens,
                max_output_tokens = r.max_output_tokens,
                n_crops           = r.n_crops,
            )
            for r in reqs
        ]

    sjf_runs:  List[SimMetrics] = []
    fcfs_runs: List[SimMetrics] = []
    stat_runs: List[SimMetrics] = []

    for s in range(max(1, n_seeds)):
        run_seed = seed + s
        base_reqs = generate_poisson_requests(
            n_requests=n_requests,
            arrival_rate_rps=arrival_rate_rps,
            seed=run_seed,
        )

        sjf_runs.append(ContinuousBatchingEngine(
            _clone(base_reqs), scheduler_policy="SJF", random_seed=run_seed
        ).run())

        fcfs_runs.append(ContinuousBatchingEngine(
            _clone(base_reqs), scheduler_policy="FCFS", random_seed=run_seed
        ).run())

        stat_runs.append(StaticBatchingEngine(
            _clone(base_reqs),
            batch_size=static_batch_size,
            scheduler_policy="FCFS",
        ).run())

    return (
        _aggregate_metrics(sjf_runs),
        _aggregate_metrics(fcfs_runs),
        _aggregate_metrics(stat_runs),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(m_sjf: SimMetrics, m_fcfs: SimMetrics, m_stat: SimMetrics) -> None:
    """Print a formatted side-by-side comparison of all three schedulers."""

    def _fmt(x: float, unit: str = "", decimals: int = 1) -> str:
        return f"{x:.{decimals}f}{unit}"

    def _gain(continuous: float, static: float, higher_is_better: bool = True) -> str:
        if static == 0:
            return "N/A"
        ratio = continuous / static
        label = f"{ratio:.2f}×"
        if higher_is_better:
            return label + (" PASS" if ratio > 1.0 else " FAIL")
        else:
            return label + (" PASS" if ratio < 1.0 else " FAIL")

    header = (
        f"{'Metric':<38} {'Continuous SJF':>16} {'Continuous FCFS':>16} "
        f"{'Static FCFS':>13} {'SJF vs Static':>14}"
    )
    sep = "─" * len(header)

    print()
    print("╔" + "═" * (len(header) - 2) + "╗")
    print("║  Phase 5 — Static vs Continuous Batching: Battle of the Schedulers" + " " * (len(header) - 72) + "║")
    print("╚" + "═" * (len(header) - 2) + "╝")
    print(header)
    print(sep)

    rows = [
        ("Requests finished",
            _fmt(m_sjf.n_finished),
            _fmt(m_fcfs.n_finished),
            _fmt(m_stat.n_finished),
            ""),
        ("Throughput (req/s)",
            _fmt(m_sjf.throughput_req_s, decimals=3),
            _fmt(m_fcfs.throughput_req_s, decimals=3),
            _fmt(m_stat.throughput_req_s, decimals=3),
            _gain(m_sjf.throughput_req_s, m_stat.throughput_req_s)),
        ("Token throughput (tok/s)",
            _fmt(m_sjf.throughput_tok_s, decimals=1),
            _fmt(m_fcfs.throughput_tok_s, decimals=1),
            _fmt(m_stat.throughput_tok_s, decimals=1),
            _gain(m_sjf.throughput_tok_s, m_stat.throughput_tok_s)),
        ("GPU decode utilisation (%)",
            _fmt(m_sjf.gpu_utilization_pct, "%"),
            _fmt(m_fcfs.gpu_utilization_pct, "%"),
            _fmt(m_stat.gpu_utilization_pct, "%"),
            _gain(m_sjf.gpu_utilization_pct, m_stat.gpu_utilization_pct)),
        ("Avg decode batch size",
            _fmt(m_sjf.avg_decode_batch),
            _fmt(m_fcfs.avg_decode_batch),
            _fmt(m_stat.avg_decode_batch),
            "—"),          # lower avg is fine; better throughput is the goal
        ("Batch saturation (% of max)",
            _fmt(m_sjf.batch_saturation_pct, "%"),
            _fmt(m_fcfs.batch_saturation_pct, "%"),
            _fmt(m_stat.batch_saturation_pct, "%"),
            ""),
        ("p50 TTFT (ms)",
            _fmt(m_sjf.p50_ttft_ms, " ms"),
            _fmt(m_fcfs.p50_ttft_ms, " ms"),
            _fmt(m_stat.p50_ttft_ms, " ms"),
            _gain(m_sjf.p50_ttft_ms, m_stat.p50_ttft_ms, higher_is_better=False)),
        ("p99 TTFT (ms)",
            _fmt(m_sjf.p99_ttft_ms, " ms"),
            _fmt(m_fcfs.p99_ttft_ms, " ms"),
            _fmt(m_stat.p99_ttft_ms, " ms"),
            _gain(m_sjf.p99_ttft_ms, m_stat.p99_ttft_ms, higher_is_better=False)),
        ("Avg TBT (ms, per token)",
            _fmt(m_sjf.avg_tbt_ms, " ms"),
            _fmt(m_fcfs.avg_tbt_ms, " ms"),
            _fmt(m_stat.avg_tbt_ms, " ms"),
            _gain(m_sjf.avg_tbt_ms, m_stat.avg_tbt_ms, higher_is_better=False)),
        ("p99 TBT (ms, per token)",
            _fmt(m_sjf.p99_tbt_ms, " ms"),
            _fmt(m_fcfs.p99_tbt_ms, " ms"),
            _fmt(m_stat.p99_tbt_ms, " ms"),
            _gain(m_sjf.p99_tbt_ms, m_stat.p99_tbt_ms, higher_is_better=False)),
        ("Max queue wait (ms)",
            _fmt(m_sjf.max_waiting_time_ms, " ms"),
            _fmt(m_fcfs.max_waiting_time_ms, " ms"),
            _fmt(m_stat.max_waiting_time_ms, " ms"),
            _gain(m_sjf.max_waiting_time_ms, m_stat.max_waiting_time_ms, higher_is_better=False)),
        ("Max TBT (ms)",
            _fmt(m_sjf.max_tbt_ms, " ms"),
            _fmt(m_fcfs.max_tbt_ms, " ms"),
            _fmt(m_stat.max_tbt_ms, " ms"),
            ""),
        ("Wasted decode time (s)",
            _fmt(m_sjf.wasted_ms / 1000, " s"),
            _fmt(m_fcfs.wasted_ms / 1000, " s"),
            _fmt(m_stat.wasted_ms / 1000, " s"),
            ""),
        ("KV utilisation (%)",
            _fmt(m_sjf.avg_kv_utilization_pct, "%"),
            _fmt(m_fcfs.avg_kv_utilization_pct, "%"),
            _fmt(m_stat.avg_kv_utilization_pct, "%"),
            ""),
        ("Total sim time (s)",
            _fmt(m_sjf.sim_duration_ms / 1000, " s"),
            _fmt(m_fcfs.sim_duration_ms / 1000, " s"),
            _fmt(m_stat.sim_duration_ms / 1000, " s"),
            ""),
    ]

    for label, v_sjf, v_fcfs, v_stat, gain in rows:
        print(f"  {label:<36} {v_sjf:>16} {v_fcfs:>16} {v_stat:>13} {gain:>14}")

    print(sep)

    # Across-seed variability (populated when n_seeds > 1)
    if m_sjf.headline_mean_std:
        print()
        print("  Across-seed variability (mean ± std over independent workload seeds):")
        for name, label in (
            ("throughput_req_s",    "Throughput (req/s)   "),
            ("p99_ttft_ms",         "p99 TTFT (ms)        "),
            ("gpu_utilization_pct", "GPU utilisation (%)  "),
        ):
            parts = []
            for m, tag in ((m_sjf, "SJF"), (m_fcfs, "FCFS"), (m_stat, "Static")):
                mean, std = m.headline_mean_std.get(name, (0.0, 0.0))
                parts.append(f"{tag}: {mean:.3f} ± {std:.3f}")
            print(f"    {label}  " + "   ".join(parts))

    print()
    print("  Key findings:")
    thp_gain = (
        m_sjf.throughput_req_s / m_stat.throughput_req_s
        if m_stat.throughput_req_s > 0 else float("inf")
    )
    ttft_reduction = (
        (m_stat.p99_ttft_ms - m_sjf.p99_ttft_ms) / m_stat.p99_ttft_ms * 100
        if m_stat.p99_ttft_ms > 0 else 0.0
    )
    gpu_gain = m_sjf.gpu_utilization_pct - m_stat.gpu_utilization_pct

    print(f"  • Throughput gain (Continuous SJF vs Static FCFS): {thp_gain:.2f}×")
    print(f"  • p99 TTFT change:  {ttft_reduction:.1f}%  "
          f"({m_stat.p99_ttft_ms/1000:.1f}s → {m_sjf.p99_ttft_ms/1000:.1f}s)")
    print(f"  • GPU utilisation delta: {gpu_gain:+.1f} pp  "
          f"({m_stat.gpu_utilization_pct:.1f}% → {m_sjf.gpu_utilization_pct:.1f}%)")
    print(f"  • SJF vs FCFS p99 TTFT: "
          f"{m_fcfs.p99_ttft_ms/1000:.1f}s → {m_sjf.p99_ttft_ms/1000:.1f}s  "
          f"({(m_fcfs.p99_ttft_ms - m_sjf.p99_ttft_ms)/max(m_fcfs.p99_ttft_ms,1e-9)*100:.1f}% reduction)")
    print()
    tbt_sjf = m_sjf.avg_tbt_ms
    if tbt_sjf <= TBT_SLA_MS:
        print(f"  [PASS] Continuous SJF avg TBT = {tbt_sjf:.1f} ms ≤ {TBT_SLA_MS:.0f} ms SLA "
              f"(measured model floor TBT(1, {DEFAULT_DECODE_CTX}) = "
              f"{DECODE_TBT_BASE_MS:.1f} ms — computed, not asserted)")
    else:
        print(f"  [FAIL] Continuous SJF avg TBT = {tbt_sjf:.1f} ms > {TBT_SLA_MS:.0f} ms SLA "
              f"(measured model floor TBT(1, {DEFAULT_DECODE_CTX}) = "
              f"{DECODE_TBT_BASE_MS:.1f} ms; the violation comes from batch/"
              f"contention effects — reported honestly).")


def print_starvation_analysis(m_sjf: SimMetrics, m_fcfs: SimMetrics) -> None:
    """Show how SJF starvation prevention affects request fairness."""
    print("  SJF fairness analysis:")
    p99_ratio = (
        m_sjf.p99_ttft_ms / m_sjf.p50_ttft_ms
        if m_sjf.p50_ttft_ms > 0 else 0.0
    )
    p99_ratio_fcfs = (
        m_fcfs.p99_ttft_ms / m_fcfs.p50_ttft_ms
        if m_fcfs.p50_ttft_ms > 0 else 0.0
    )
    print(f"    p99/p50 TTFT ratio (SJF):   {p99_ratio:.2f}  "
          f"(closer to 1.0 = more uniform service)")
    print(f"    p99/p50 TTFT ratio (FCFS):  {p99_ratio_fcfs:.2f}")
    if p99_ratio < p99_ratio_fcfs:
        print(f"    [PASS] SJF reduces tail unfairness by "
              f"{(p99_ratio_fcfs - p99_ratio)/p99_ratio_fcfs*100:.1f}%")
    else:
        print(f"    [INFO] Starvation threshold ({STARVATION_THRESHOLD} bypasses) "
              f"prevents runaway unfairness for long requests.")
    print()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    print("Running Phase 5 self-tests …")
    random.seed(42)

    # ── Test 1: TBT model (measured batch-1 fit + modeled batch term) ────────
    assert abs(_predict_tbt_ms(1) - DECODE_TBT_BASE_MS) < 0.01, (
        f"TBT(1) should equal {DECODE_TBT_BASE_MS} ms"
    )
    assert _predict_tbt_ms(10) > _predict_tbt_ms(1), "TBT should grow with batch size"
    assert _predict_tbt_ms(0) == 0.0
    # MEASURED reality: the 80 ms TBT SLA is comfortably met at batch 1
    # (TBT(1, 1560) ≈ 23.3 ms) — assert the honest computed behaviour.
    assert _predict_tbt_ms(1) <= TBT_SLA_MS, (
        f"Measured model gives TBT(1) = {_predict_tbt_ms(1):.1f} ms — must "
        f"be under the {TBT_SLA_MS:.0f} ms SLA"
    )
    sla_batch_ceiling = max(
        b for b in range(1, MAX_DECODE_BATCH + 1)
        if _predict_tbt_ms(b) <= TBT_SLA_MS
    )
    assert 10 <= sla_batch_ceiling <= 30, (
        f"SLA batch ceiling at full context should be ≈19, got {sla_batch_ceiling}"
    )
    print(f"  [OK] TBT model: TBT(1, {DEFAULT_DECODE_CTX}) = "
          f"{_predict_tbt_ms(1):.1f} ms ≤ {TBT_SLA_MS:.0f} ms SLA; "
          f"SLA batch ceiling ≈ {sla_batch_ceiling}")

    # ── Test 2: KV pool ──────────────────────────────────────────────────────
    pool = _KVBlockPool(total_blocks=1000)
    assert pool.free_blocks == 1000
    pool.allocate(req_id=0, n_blocks=100)
    assert pool.free_blocks == 900
    pool.free(req_id=0)
    assert pool.free_blocks == 1000
    assert not pool.allocate(req_id=1, n_blocks=1001), "over-allocation must fail"
    print("  [OK] KV pool accounting")

    # ── Test 3: Request SJF key (real crop configs: 1 vs 17 crops) ──────────
    short_req = Request(req_id=0, arrival_time_ms=0, n_visual_tokens=100,  max_output_tokens=32,  n_crops=1)
    long_req  = Request(req_id=1, arrival_time_ms=0, n_visual_tokens=1560, max_output_tokens=512, n_crops=17)
    assert short_req.sjf_key < long_req.sjf_key, "Short request should have lower SJF key"
    # Starvation promotion
    long_req.starvation_count = STARVATION_THRESHOLD
    assert long_req.sjf_key == 0.0, "Starvation promotion should set key = 0"
    print("  [OK] Request SJF key + starvation promotion")

    # ── Test 4: Small simulation smoke test ──────────────────────────────────
    reqs   = generate_poisson_requests(n_requests=30, arrival_rate_rps=1.0, seed=0)
    engine = ContinuousBatchingEngine(reqs, scheduler_policy="SJF")
    m      = engine.run()
    assert m.n_finished == m.n_requests, "All requests should complete"
    assert m.gpu_utilization_pct >= 0.0
    assert m.throughput_req_s > 0.0
    # Idle accounting sanity: active + idle ≈ sim duration
    acct = m.gpu_active_ms + m.gpu_idle_ms
    assert abs(acct - m.sim_duration_ms) < 1.0, (
        f"active+idle ({acct:.1f}) must ≈ duration ({m.sim_duration_ms:.1f})"
    )
    print(f"  [OK] Continuous-SJF smoke test: {m.n_finished}/{m.n_requests} done, "
          f"util={m.gpu_utilization_pct:.1f}%, "
          f"p99_TTFT={m.p99_ttft_ms/1000:.1f}s, "
          f"active+idle≈duration OK")

    # ── Test 5: Static smoke test ────────────────────────────────────────────
    reqs2  = generate_poisson_requests(n_requests=30, arrival_rate_rps=1.0, seed=0)
    engine2 = StaticBatchingEngine(reqs2, batch_size=10)
    m2     = engine2.run()
    assert m2.n_finished == m2.n_requests
    acct2 = m2.gpu_active_ms + m2.gpu_idle_ms
    assert abs(acct2 - m2.sim_duration_ms) < 1.0, (
        f"static active+idle ({acct2:.1f}) must ≈ duration ({m2.sim_duration_ms:.1f})"
    )
    print(f"  [OK] Static-FCFS smoke test:  {m2.n_finished}/{m2.n_requests} done, "
          f"util={m2.gpu_utilization_pct:.1f}%, "
          f"p99_TTFT={m2.p99_ttft_ms/1000:.1f}s, wasted={m2.wasted_ms/1000:.1f}s")

    # ── Test 6: FCFS arm really is FCFS ──────────────────────────────────────
    # A long job arriving first must start vision before a short job arriving
    # later under FCFS (SJF would reorder them).
    early_long  = Request(req_id=0, arrival_time_ms=0.0,  n_visual_tokens=1560, max_output_tokens=512, n_crops=17)
    late_short  = Request(req_id=1, arrival_time_ms=10.0, n_visual_tokens=100,  max_output_tokens=32,  n_crops=1)
    fcfs_engine = ContinuousBatchingEngine([early_long, late_short], scheduler_policy="FCFS")
    fcfs_engine.run()
    assert early_long.vision_start_ms <= late_short.vision_start_ms, (
        "FCFS must start the earlier-arriving request first"
    )
    print("  [OK] FCFS ordering respected at vision admission")

    # ── Test 7: Continuous vs static comparison (informational) ─────────────
    reqs3a = generate_poisson_requests(n_requests=40, arrival_rate_rps=1.0, seed=7)
    reqs3b = generate_poisson_requests(n_requests=40, arrival_rate_rps=1.0, seed=7)
    c_m = ContinuousBatchingEngine(reqs3a, scheduler_policy="SJF").run()
    s_m = StaticBatchingEngine(reqs3b, batch_size=10).run()
    # With honest idle accounting the direction of the comparison is a
    # RESULT, not a precondition — report it rather than asserting it.
    print(f"  [INFO] Continuous GPU util {c_m.gpu_utilization_pct:.1f}% vs "
          f"Static {s_m.gpu_utilization_pct:.1f}%  "
          f"| p99 TTFT: {c_m.p99_ttft_ms/1000:.1f}s vs {s_m.p99_ttft_ms/1000:.1f}s")

    print()
    print("All Phase 5 self-tests PASSED")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5: Continuous Batching Engine")
    parser.add_argument("--test",       action="store_true", help="Run self-tests only")
    parser.add_argument("--n",          type=int,   default=200,  help="Number of requests")
    parser.add_argument("--rate",       type=float, default=2.0,  help="Arrival rate (req/s)")
    parser.add_argument("--batch-size", type=int,   default=20,   help="Static batch size")
    parser.add_argument("--seed",       type=int,   default=42,   help="Base RNG seed")
    parser.add_argument("--seeds",      type=int,   default=5,    help="Number of workload seeds")
    args = parser.parse_args()

    if args.test:
        _run_self_test()
        sys.exit(0)

    _run_self_test()

    print("=" * 78)
    print("  Phase 5: Continuous Batching Engine — Battle of the Schedulers")
    print("=" * 78)
    print(f"  Workload    : {args.n} requests, Poisson λ={args.rate} req/s, "
          f"{args.seeds} seed(s)")
    print(f"  Static batch: B={args.batch_size}")
    _sla_ceiling = max(
        (b for b in range(1, MAX_DECODE_BATCH + 1)
         if _predict_tbt_ms(b) <= TBT_SLA_MS),
        default=0,
    )
    print(f"  Decode mode : W4A8  (TBT model MEASURED at B=1: "
          f"{DECODE_OVERHEAD_MS} + {DECODE_KV_MS_PER_CTX_TOKEN}·ctx ms, "
          f"+(B−1)·KV-read modeled; TBT(1, {DEFAULT_DECODE_CTX}) = "
          f"{_predict_tbt_ms(1):.1f} ms, SLA batch ceiling ≈ {_sla_ceiling}, "
          f"max_batch = {MAX_DECODE_BATCH}, TBT(70) = "
          f"{_predict_tbt_ms(MAX_DECODE_BATCH):.1f} ms)")
    print(f"  KV pool     : {KV_POOL_BUDGET_MB:.0f} MB  →  {MAX_KV_BLOCKS:,} W4 blocks "
          f"({MB_PER_KV_BLOCK_W4:.4f} MB/block)")
    print(f"  Shares      : {M3_TOTAL_SMs} abstract compute shares "
          f"(M3 has 10 GPU cores; see amio_constants), "
          f"partitioned by Phase 3 SMOrchestrator")
    print()

    print("Running simulations …")
    import time as _t
    t0 = _t.time()
    m_sjf, m_fcfs, m_stat = compare_schedulers(
        n_requests=args.n,
        arrival_rate_rps=args.rate,
        static_batch_size=args.batch_size,
        seed=args.seed,
        n_seeds=args.seeds,
    )
    elapsed = _t.time() - t0
    print(f"  (simulation completed in {elapsed:.2f}s wall-clock time)")

    print_comparison(m_sjf, m_fcfs, m_stat)
    print_starvation_analysis(m_sjf, m_fcfs)

    # ── Poisson burst scenario (high arrival rate) ────────────────────────────
    print("─" * 78)
    print("  Poisson burst scenario: 2000 requests, high arrival rate (λ=50 req/s)")
    print("  (Models a traffic spike overwhelming the queue)")
    print()
    t1 = _t.time()
    b_sjf, b_fcfs, b_stat = compare_schedulers(
        n_requests=2000,
        arrival_rate_rps=50.0,
        static_batch_size=args.batch_size,
        seed=99,
        n_seeds=args.seeds,
    )
    burst_elapsed = _t.time() - t1
    print(f"  (burst simulation completed in {burst_elapsed:.2f}s wall-clock time)")
    print_comparison(b_sjf, b_fcfs, b_stat)
    print_starvation_analysis(b_sjf, b_fcfs)
