"""
Continuous Batching Engine — Phase 5: Iteration-Level Scheduling
=================================================================

Implements the core innovation of modern LLM serving frameworks (vLLM,
TensorRT-LLM) adapted for SmolVLM's multimodal pipeline on Apple M3.

Five capabilities
-----------------
1. Request State Machine
   WAITING → VISION_ENCODING → PREFILLING → DECODING → FINISHED.
   Accounts for SmolVLM's heterogeneous stages: SigLIP vision is the
   dominant bottleneck (5991 ms baseline), LM prefill is quadratic,
   and decode is memory-bandwidth bound.

2. Iteration-Level Scheduler
   Admits / evicts requests between EVERY token generation step.
   - In-flight merging: a newly prefilled request joins the decode batch
     at the very next tick after its prefill completes.
   - Immediate eviction: a request that hits its stop token frees its
     KV blocks at the same tick it finishes.
   - Concurrent pipeline: vision ↔ prefill ↔ decode all run in parallel,
     gated only by SM partitioning.

3. SJF + Starvation Prevention
   Phase 2 cost model predicts total service time (T_prefill + max_output ×
   T_decode_base) per request.  The waiting queue is sorted SJF-first.
   Starvation counter tracks how many shorter requests jumped ahead; once
   starvation_count ≥ STARVATION_THRESHOLD the request is boosted to
   highest priority.

4. SM Partitioning Integration (Phase 3 SMOrchestrator)
   At each decode tick the SMOrchestrator splits M3's 38 SMs between the
   vision/prefill worker and the decode worker.  Decode always receives
   priority.  Vision latency is scaled up by the fraction of SMs diverted
   to decode (fewer SMs → longer vision, but decode stays fast).

5. Static vs Continuous Comparison ("Battle of the Schedulers")
   StaticBatchingEngine processes a fixed batch B end-to-end (all vision,
   then all prefill, then decode until the LAST request finishes).  GPU
   decode is idle during the vision/prefill pipeline.  ContinuousBatching-
   Engine keeps the decode batch full throughout.

Hardware target
---------------
  Apple M3, 8 GB unified memory, 100 GB/s bandwidth, 38 SMs.
  Model: SmolVLM-Instruct-4bit, W4A8+GAR mode (Phase 4 best config).
  Phase 4 results used here:
    DECODE_TBT_BASE_MS = 87.7 ms   (W4A8+GAR at batch=1)
    KV_BYTES_PER_TOKEN_W4 = 27,648 B/token  (4× smaller than FP16)
    MAX_DECODE_BATCH = 70           (TBT ≤ 80 ms KV-budget ceiling)
"""

from __future__ import annotations

import heapq
import math
import random
import statistics
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

try:
    from simulation.sm_orchestrator import SMOrchestrator, M3_TOTAL_SMs
    from simulation.kv_manager import KV_BYTES_PER_TOKEN, PAGE_SIZE, KV_POOL_BUDGET_MB
    _PHASE_DEPS_OK = True
except ImportError:
    _PHASE_DEPS_OK = False
    M3_TOTAL_SMs     = 38
    KV_BYTES_PER_TOKEN = 110_592        # 2 × 24 layers × 1152 hidden × 2 B
    PAGE_SIZE        = 16
    KV_POOL_BUDGET_MB = 5694.0


# ---------------------------------------------------------------------------
# Hardware & model constants
# ---------------------------------------------------------------------------

# Phase 4 W4A8+GAR decode configuration
DECODE_TBT_BASE_MS:     float = 87.7       # measured TBT at batch=1 (W4A8+GAR)
DECODE_OVERHEAD_MS:     float = 83.75      # fixed per-step overhead (framework, Python)
DECODE_BW_COST_MS:      float = 3.95       # bandwidth cost per request per step
#   TBT(B) = DECODE_OVERHEAD_MS + DECODE_BW_COST_MS × B   ← weight-sharing model
#   At B=1: 83.75 + 3.95 = 87.70 ms  ✓
#   Implies weight bytes loaded once, KV traffic scales linearly with batch.

VISION_BASE_MS:         float = 5991.0     # SigLIP full-encoder (Phase 1 baseline)

# Phase 2 LM prefill cost model  (R² = 0.9978)
COST_GAMMA: float = 2.0957e-05             # ms / token²
COST_BETA:  float = 1.5905                 # ms / token
COST_ALPHA: float = -20.08                 # ms (constant)

# W4 KV cache (4× smaller than FP16 baseline)
KV_W4_BYTES_PER_TOKEN:  int   = KV_BYTES_PER_TOKEN // 4        # 27,648 B/token
BYTES_PER_KV_BLOCK_W4:  int   = PAGE_SIZE * KV_W4_BYTES_PER_TOKEN  # 442,368 B
MB_PER_KV_BLOCK_W4:     float = BYTES_PER_KV_BLOCK_W4 / (1024 ** 2)  # ≈ 0.4219 MB
MAX_KV_BLOCKS:          int   = int(KV_POOL_BUDGET_MB / MB_PER_KV_BLOCK_W4)  # ≈ 13,493

# Effective maximum concurrent decode sequences
MAX_DECODE_BATCH:       int   = 70         # TBT ceiling from Phase 4 starvation analysis

# SJF starvation prevention
STARVATION_THRESHOLD:   int   = 8          # times bypassed before priority boost

# SM partitioning constants
SM_MIN_DECODE:  int = 4
SM_MIN_VISION:  int = 8

# Human-perceivable TBT threshold
TBT_SLA_MS: float = 80.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _predict_prefill_ms(n_tokens: int) -> float:
    """Phase 2 cost model: T_prefill(N) = γN² + βN + α (ms). Clipped to ≥1ms."""
    return max(1.0, COST_GAMMA * n_tokens ** 2 + COST_BETA * n_tokens + COST_ALPHA)


def _predict_tbt_ms(batch_size: int) -> float:
    """
    W4A8+GAR TBT as a function of decode batch size.

    Model: TBT(B) = overhead_fixed + bw_cost × B
      overhead_fixed = 83.75 ms   (Python/MLX framework overhead per step)
      bw_cost        =  3.95 ms   (KV-cache bandwidth per request per step)

    This captures the key property of batched inference: model weights are
    loaded ONCE per step (shared by the batch), so only KV-cache traffic
    scales with B.  At B=1 the formula recovers DECODE_TBT_BASE_MS exactly.

    Token throughput = B / TBT(B) → grows super-linearly with batch size,
    which is exactly the efficiency gain we are trying to capture.
    """
    if batch_size <= 0:
        return 0.0
    return DECODE_OVERHEAD_MS + DECODE_BW_COST_MS * batch_size


def _kv_blocks_for_seq(n_tokens: int) -> int:
    """Number of W4 KV blocks needed for a sequence of n_tokens."""
    return math.ceil(n_tokens / PAGE_SIZE)


# ---------------------------------------------------------------------------
# Request State Machine
# ---------------------------------------------------------------------------

class RequestState(Enum):
    """
    Lifecycle states for one multimodal inference request.

    Transitions
    -----------
    WAITING         → VISION_ENCODING  : dequeued by vision worker
    VISION_ENCODING → PREFILLING       : SigLIP forward pass complete
    PREFILLING      → DECODING         : LM KV cache built; joins decode batch
    DECODING        → FINISHED         : stop token OR max_output_tokens reached
    """
    WAITING         = "WAITING"
    VISION_ENCODING = "VISION_ENCODING"
    PREFILLING      = "PREFILLING"
    DECODING        = "DECODING"
    FINISHED        = "FINISHED"


@dataclass
class Request:
    """One multimodal inference request tracked through the full pipeline."""

    req_id:            int
    arrival_time_ms:   float
    n_visual_tokens:   int          # tokens from vision encoder → LM input
    max_output_tokens: int          # stop criterion

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
    """Aggregated performance metrics collected during one simulation run."""

    scheduler_name:            str   = "unknown"
    n_requests:                int   = 0
    n_finished:                int   = 0
    sim_duration_ms:           float = 0.0
    total_tokens_generated:    int   = 0

    # Raw samples (populated during simulation)
    ttft_samples:              List[float] = field(default_factory=list)
    tbt_samples:               List[float] = field(default_factory=list)
    decode_batch_size_samples: List[int]   = field(default_factory=list)
    kv_util_samples:           List[float] = field(default_factory=list)

    # Counters
    gpu_active_ms:             float = 0.0   # decode batch non-empty
    gpu_idle_ms:               float = 0.0   # decode batch empty
    max_waiting_time_ms:       float = 0.0   # worst-case TBT across all ticks

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
            s = sorted(self.ttft_samples)
            n = len(s)
            self.p50_ttft_ms = s[min(int(n * 0.50), n - 1)]
            self.p99_ttft_ms = s[min(int(n * 0.99), n - 1)]

        if self.tbt_samples:
            self.avg_tbt_ms = sum(self.tbt_samples) / len(self.tbt_samples)
            s = sorted(self.tbt_samples)
            self.p99_tbt_ms = s[min(int(len(s) * 0.99), len(s) - 1)]

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
        if not self.can_allocate(n_blocks):
            return False
        prev = self._held.get(req_id, 0)
        delta = n_blocks - prev
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
    Iteration-level scheduler for SmolVLM multimodal inference.

    Architecture
    ------------
    • vision_worker  — SigLIP encoder, one request at a time.
      Latency scaled by SM allocation: fewer SMs → longer TBT for vision,
      but decode is protected.

    • prefill_worker — LM KV prefill, one request at a time (pipelined
      with vision: while R2 does vision, R1 does prefill).

    • decode_batch   — up to MAX_DECODE_BATCH concurrent requests.
      Admits newly prefilled requests and evicts finished ones at every tick.

    Scheduling policy
    -----------------
    SJF with starvation prevention (see Request.sjf_key).  The waiting queue
    is sorted before each vision-worker admission.  Requests that are bypassed
    more than STARVATION_THRESHOLD times get starvation_count ≥ thresh, which
    sets sjf_key = 0 (immediate promotion to front of queue).

    SM partitioning
    ---------------
    Uses Phase 3 SMOrchestrator.  When decode is running and vision/prefill is
    also active, the decode worker claims priority SMs; the residual goes to
    vision.  Vision latency is scaled by (total_sms / sm_vision_allocated).
    """

    def __init__(
        self,
        requests:          List[Request],
        max_decode_batch:  int   = MAX_DECODE_BATCH,
        scheduler_policy:  str   = "SJF",     # "SJF" | "FCFS"
        n_vision_workers:  int   = 1,
        random_seed:       int   = 42,
    ):
        self.requests          = {r.req_id: r for r in requests}
        self.max_decode_batch  = max_decode_batch
        self.policy            = scheduler_policy
        self.n_vision_workers  = n_vision_workers
        self._rng              = random.Random(random_seed)

        self._sm_orch = SMOrchestrator() if _PHASE_DEPS_OK else None

        # Worker state: each slot stores (req_id, free_at_ms)
        self._vision_slots: List[Tuple[int, float]] = []   # (req_id, free_at)
        self._prefill_slot: Optional[Tuple[int, float]] = None

        # Queues
        self._waiting:       List[Request] = []   # sorted by sjf_key on admission
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

        self._metrics.sim_duration_ms = self._clock
        self._metrics.n_finished = sum(
            1 for r in self.requests.values() if r.is_done
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
        req.state = RequestState.PREFILLING

        # Free the vision worker slot
        self._vision_slots = [
            (rid, t) for (rid, t) in self._vision_slots if rid != req.req_id
        ]

        # Move to prefill ready queue
        self._prefill_ready.append(req)
        req.prefill_start_ms = self._clock

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
        # Sort decode_done by SJF key; admit as many as KV + batch budget allow.
        self._prefill_done.sort(key=lambda r: r.sjf_key)
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
            # Decode is idle; record idle time and wait for next prefill
            self._metrics.gpu_idle_ms += 0.0   # no tick scheduled; idle gap handled below
            return

        # SM partitioning: if vision/prefill is running concurrently, split SMs
        tbt = self._compute_tbt(B)

        for req in self._decode_batch:
            req.tokens_generated += 1
            req.tbt_samples.append(tbt)
            if req.first_token_ms < 0:
                req.first_token_ms = self._clock
                self._metrics.ttft_samples.append(req.ttft_ms)

        # ── Phase 5: record metrics ────────────────────────────────────────
        self._metrics.gpu_active_ms              += tbt
        self._metrics.tbt_samples.append(tbt)
        self._metrics.max_waiting_time_ms         = max(
            self._metrics.max_waiting_time_ms, tbt
        )
        self._metrics.decode_batch_size_samples.append(B)
        self._metrics.kv_util_samples.append(self._kv.utilization_pct)

        # ── Phase 6: schedule next tick ────────────────────────────────────
        if self._decode_batch or self._prefill_done:
            _push(events, self._clock + tbt, _EvType.DECODE_TICK)
        else:
            # Batch drained — accumulate idle time until next prefill_done event
            self._metrics.gpu_idle_ms += self._estimate_idle_until_next_prefill()

    # ------------------------------------------------------------------
    # Worker scheduling helpers
    # ------------------------------------------------------------------

    def _try_start_vision(self, events: list) -> None:
        """Start vision encoding for the highest-priority waiting request."""
        if not self._waiting:
            return
        if len(self._vision_slots) >= self.n_vision_workers:
            return

        # Sort by SJF key; update starvation counts for all skipped requests
        if self.policy == "SJF":
            self._waiting.sort(key=lambda r: r.sjf_key)
            selected = self._waiting[0]
            # Increment starvation for all other waiting requests that arrived
            # earlier than the selected one (they were displaced by a shorter job)
            for r in self._waiting[1:]:
                if r.arrival_time_ms <= selected.arrival_time_ms:
                    r.starvation_count += 1
        else:
            # FCFS — sort by arrival time
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

        # Use SJF key for prefill ordering too (shorter prefill = sooner decode)
        self._prefill_ready.sort(key=lambda r: r.sjf_key)
        req = self._prefill_ready.pop(0)
        self._start_prefill(req, events)

    def _start_vision(self, req: Request, events: list) -> None:
        """Schedule vision encoding, accounting for SM partitioning."""
        req.vision_start_ms = self._clock
        req.state = RequestState.VISION_ENCODING

        t_vision = self._compute_vision_latency()
        done_at  = self._clock + t_vision
        self._vision_slots.append((req.req_id, done_at))
        _push(events, done_at, _EvType.VISION_DONE, req.req_id)

    def _start_prefill(self, req: Request, events: list) -> None:
        """Schedule LM prefill, accounting for SM partitioning."""
        req.state = RequestState.PREFILLING
        req.prefill_start_ms = self._clock

        t_prefill = self._compute_prefill_latency(req.n_visual_tokens)
        done_at   = self._clock + t_prefill
        self._prefill_slot = (req.req_id, done_at)
        _push(events, done_at, _EvType.PREFILL_DONE, req.req_id)

    # ------------------------------------------------------------------
    # SM-partitioned latency models
    # ------------------------------------------------------------------

    def _compute_vision_latency(self) -> float:
        """
        Scale SigLIP latency by the fraction of SMs available for vision.

        When decode is running, SMOrchestrator diverts SMs to decode;
        the vision worker receives the remainder.  Fewer SMs → longer vision.
        T_vision_scaled = T_vision_base × (total_sms / sm_vision_allocated)
        """
        n_decode = len(self._decode_batch)
        if n_decode > 0 and self._sm_orch is not None:
            alloc     = self._sm_orch.allocate(n_pending_decode=n_decode)
            sm_vision = max(alloc.sm_vision, SM_MIN_VISION)
            scale     = M3_TOTAL_SMs / sm_vision
            return VISION_BASE_MS * scale
        return VISION_BASE_MS

    def _compute_prefill_latency(self, n_tokens: int) -> float:
        """
        Scale LM prefill latency by SM availability.

        Prefill is compute-bound (unlike decode which is BW-bound).
        Uses the same SM allocation as vision — both share the residual
        SMs not claimed by the decode worker.
        """
        base = _predict_prefill_ms(n_tokens)
        n_decode = len(self._decode_batch)
        if n_decode > 0 and self._sm_orch is not None:
            alloc     = self._sm_orch.allocate(n_pending_decode=n_decode)
            sm_prefill = max(alloc.sm_vision, SM_MIN_VISION)
            scale      = M3_TOTAL_SMs / sm_prefill
            return base * scale
        return base

    def _compute_tbt(self, batch_size: int) -> float:
        """
        Compute decode TBT for the current batch, with optional SM scaling.

        Decode is memory-bandwidth bound, not compute-bound; giving it more
        SMs only helps marginally.  We apply a small (sqrt) correction when
        decode loses SMs to vision/prefill.
        """
        base_tbt = _predict_tbt_ms(batch_size)
        # If vision/prefill is running in parallel, decode may lose some SMs
        n_preprocessing = (
            len(self._vision_slots)
            + (1 if self._prefill_slot is not None else 0)
        )
        if n_preprocessing > 0 and self._sm_orch is not None:
            alloc      = self._sm_orch.allocate(n_pending_decode=batch_size)
            sm_decode  = max(alloc.sm_decode, SM_MIN_DECODE)
            # BW-bound: sqrt degradation model (mild effect)
            sm_scale   = math.sqrt(max(SM_MIN_DECODE, 1) / sm_decode)
            return base_tbt * max(sm_scale, 0.85)   # cap at 15% overhead
        return base_tbt

    def _estimate_idle_until_next_prefill(self) -> float:
        """
        Estimate GPU decode idle time while waiting for the next request to
        finish prefill.  Used only when the decode batch has completely drained.
        """
        if self._prefill_slot is not None:
            _, done_at = self._prefill_slot
            return max(0.0, done_at - self._clock)
        if self._prefill_ready:
            # Start prefill immediately; estimate its duration from the first ready req
            req = min(self._prefill_ready, key=lambda r: r.sjf_key)
            return _predict_prefill_ms(req.n_visual_tokens)
        # No prefill in progress; larger idle gap until vision completes
        if self._vision_slots:
            _, done_at = min(self._vision_slots, key=lambda x: x[1])
            base_prefill = _predict_prefill_ms(512)   # estimate for average request
            return max(0.0, done_at - self._clock) + base_prefill
        return 0.0


# ---------------------------------------------------------------------------
# Static Batching Engine (Baseline)
# ---------------------------------------------------------------------------

class StaticBatchingEngine:
    """
    Baseline static-batch scheduler.

    Processes a fixed batch B all the way through vision → prefill → decode
    before starting any new requests.  Models the GPU idle time that arises
    when shorter requests finish decode but cannot be evicted until the
    longest request in the batch completes.

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
          1. Vision phase:  B × T_vision_base   (sequential, all on full SMs)
          2. Prefill phase: Σ T_prefill(n_i)    (sequential, all on full SMs)
          3. Decode phase:  max_steps × TBT(B)  (batch decode until last request done)

        GPU is "idle" (no decode) during phases 1 and 2.
        GPU "wasted" time in phase 3 = (max_steps - steps_i) × TBT(B) per request
        that finished early but could not be evicted.
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

            # ── Vision phase (GPU decode idle) ─────────────────────────────
            t_vision_phase = B * VISION_BASE_MS
            for req in batch:
                req.vision_start_ms = clock
                req.vision_done_ms  = clock + VISION_BASE_MS
            vision_end = clock + t_vision_phase
            self._metrics.gpu_idle_ms += t_vision_phase
            clock = vision_end

            # ── Prefill phase (GPU decode idle) ────────────────────────────
            t_prefill_start = clock
            for req in batch:
                t_pf              = _predict_prefill_ms(req.n_visual_tokens)
                req.prefill_start_ms = clock
                req.prefill_done_ms  = clock + t_pf
                clock += t_pf            # sequential prefill
            t_prefill_phase = clock - t_prefill_start
            self._metrics.gpu_idle_ms += t_prefill_phase

            # ── Decode phase (GPU decode active) ───────────────────────────
            decode_start = clock
            tbt          = _predict_tbt_ms(B)
            max_steps    = max(r.max_output_tokens for r in batch)

            for req in batch:
                req.tokens_generated = req.max_output_tokens
                req.first_token_ms   = decode_start + tbt    # first token after first step
                req.last_token_ms    = decode_start + req.max_output_tokens * tbt
                self._metrics.ttft_samples.append(req.ttft_ms)

            # All requests wait until the LAST one finishes — static batch drain
            decode_duration = max_steps * tbt
            self._metrics.gpu_active_ms += decode_duration
            self._metrics.total_tokens_generated += sum(r.max_output_tokens for r in batch)
            self._metrics.tbt_samples.extend([tbt] * sum(r.max_output_tokens for r in batch))
            self._metrics.decode_batch_size_samples.append(B)
            self._metrics.max_waiting_time_ms = max(self._metrics.max_waiting_time_ms, tbt)

            # Wasted decode time (shorter requests done but batch not released)
            for req in batch:
                wasted = (max_steps - req.max_output_tokens) * tbt
                self._metrics.gpu_active_ms += wasted  # GPU active but "wasted"

            clock += decode_duration

            # KV utilisation at peak (all B requests at max tokens)
            kv_tokens_peak = sum(
                r.n_visual_tokens + r.max_output_tokens for r in batch
            )
            kv_blocks_peak = _kv_blocks_for_seq(kv_tokens_peak)
            kv_util = kv_blocks_peak / MAX_KV_BLOCKS * 100.0
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
    min_visual_tokens: int   = 256,
    max_visual_tokens: int   = 1548,
    min_output_tokens: int   = 32,
    max_output_tokens: int   = 512,
) -> List[Request]:
    """
    Generate a Poisson-arrival request workload.

    Visual token counts are drawn uniformly over [min_visual, max_visual],
    modelling varying image resolutions in SmolVLM.

    Output token counts follow a Pareto distribution (mean ≈ 128 tokens)
    reflecting real LLM generation: most responses are short, with a long tail.

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

        # Visual tokens ~ Uniform[min, max]
        n_vis = rng.randint(min_visual_tokens, max_visual_tokens)

        # Output tokens ~ Pareto(shape=1.2, scale=32) clipped to [min, max]
        raw_out = int(min_output_tokens * (1.0 - rng.random()) ** (-1 / 1.2))
        n_out   = max(min_output_tokens, min(raw_out, max_output_tokens))

        reqs.append(Request(
            req_id            = i,
            arrival_time_ms   = clock,
            n_visual_tokens   = n_vis,
            max_output_tokens = n_out,
        ))

    return reqs


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def compare_schedulers(
    n_requests:         int   = 500,
    arrival_rate_rps:   float = 5.0,
    static_batch_size:  int   = 20,
    seed:               int   = 42,
) -> Tuple[SimMetrics, SimMetrics, SimMetrics]:
    """
    Run all three scheduler configurations on the same request set and return
    (continuous_SJF, continuous_FCFS, static_FCFS) metrics tuples.

    Parameters
    ----------
    n_requests        : number of Poisson-arrival requests in the burst
    arrival_rate_rps  : λ for Poisson inter-arrival process
    static_batch_size : B for the static baseline
    seed              : RNG seed (same for all three runs)
    """
    # Generate the same request workload for all schedulers
    base_reqs = generate_poisson_requests(
        n_requests=n_requests,
        arrival_rate_rps=arrival_rate_rps,
        seed=seed,
    )

    def _clone(reqs: List[Request]) -> List[Request]:
        """Deep-copy request list so each scheduler starts from a clean state."""
        return [
            Request(
                req_id            = r.req_id,
                arrival_time_ms   = r.arrival_time_ms,
                n_visual_tokens   = r.n_visual_tokens,
                max_output_tokens = r.max_output_tokens,
            )
            for r in reqs
        ]

    # Run continuous SJF (best case)
    m_sjf  = ContinuousBatchingEngine(
        _clone(base_reqs), scheduler_policy="SJF"
    ).run()

    # Run continuous FCFS (shows value of SJF alone)
    m_fcfs = ContinuousBatchingEngine(
        _clone(base_reqs), scheduler_policy="FCFS"
    ).run()

    # Run static FCFS (worst case baseline)
    m_stat = StaticBatchingEngine(
        _clone(base_reqs),
        batch_size=static_batch_size,
        scheduler_policy="FCFS",
    ).run()

    return m_sjf, m_fcfs, m_stat


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
            return label + (" ✅" if ratio > 1.0 else " ❌")
        else:
            return label + (" ✅" if ratio < 1.0 else " ❌")

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
        ("Avg TBT (ms)",
            _fmt(m_sjf.avg_tbt_ms, " ms"),
            _fmt(m_fcfs.avg_tbt_ms, " ms"),
            _fmt(m_stat.avg_tbt_ms, " ms"),
            _gain(m_sjf.avg_tbt_ms, m_stat.avg_tbt_ms, higher_is_better=False)),
        ("p99 TBT (ms)",
            _fmt(m_sjf.p99_tbt_ms, " ms"),
            _fmt(m_fcfs.p99_tbt_ms, " ms"),
            _fmt(m_stat.p99_tbt_ms, " ms"),
            _gain(m_sjf.p99_tbt_ms, m_stat.p99_tbt_ms, higher_is_better=False)),
        ("Max waiting time (ms)",
            _fmt(m_sjf.max_waiting_time_ms, " ms"),
            _fmt(m_fcfs.max_waiting_time_ms, " ms"),
            _fmt(m_stat.max_waiting_time_ms, " ms"),
            _gain(m_sjf.max_waiting_time_ms, m_stat.max_waiting_time_ms, higher_is_better=False)),
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
    print(f"  • p99 TTFT reduction:  {ttft_reduction:.1f}%  "
          f"({m_stat.p99_ttft_ms/1000:.1f}s → {m_sjf.p99_ttft_ms/1000:.1f}s)")
    print(f"  • GPU utilisation delta: {gpu_gain:+.1f} pp  "
          f"({m_stat.gpu_utilization_pct:.1f}% → {m_sjf.gpu_utilization_pct:.1f}%)")
    print(f"  • SJF vs FCFS p99 TTFT: "
          f"{m_fcfs.p99_ttft_ms/1000:.1f}s → {m_sjf.p99_ttft_ms/1000:.1f}s  "
          f"({(m_fcfs.p99_ttft_ms - m_sjf.p99_ttft_ms)/m_fcfs.p99_ttft_ms*100:.1f}% reduction)")
    print()
    tbt_sjf  = m_sjf.avg_tbt_ms
    tbt_stat = m_stat.avg_tbt_ms
    if tbt_sjf <= TBT_SLA_MS:
        print(f"  ✅ Continuous SJF avg TBT = {tbt_sjf:.1f} ms ≤ {TBT_SLA_MS:.0f} ms SLA")
    else:
        print(f"  ⚠  Continuous SJF avg TBT = {tbt_sjf:.1f} ms > {TBT_SLA_MS:.0f} ms SLA  "
              f"(Phase 5 continuous batching reduces tail latency; "
              f"batched decode TBT is a known trade-off vs single-request TBT)")


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
        print(f"    ✅ SJF reduces tail unfairness by "
              f"{(p99_ratio_fcfs - p99_ratio)/p99_ratio_fcfs*100:.1f}%")
    else:
        print(f"    ℹ  Starvation threshold ({STARVATION_THRESHOLD} bypasses) "
              f"prevents runaway unfairness for long requests.")
    print()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    print("Running Phase 5 self-tests …")
    random.seed(42)

    # ── Test 1: TBT model ────────────────────────────────────────────────────
    assert abs(_predict_tbt_ms(1) - DECODE_TBT_BASE_MS) < 0.01, (
        f"TBT(1) should equal {DECODE_TBT_BASE_MS} ms"
    )
    assert _predict_tbt_ms(10) > _predict_tbt_ms(1), "TBT should grow with batch size"
    assert _predict_tbt_ms(0) == 0.0

    # ── Test 2: KV pool ──────────────────────────────────────────────────────
    pool = _KVBlockPool(total_blocks=1000)
    assert pool.free_blocks == 1000
    pool.allocate(req_id=0, n_blocks=100)
    assert pool.free_blocks == 900
    pool.free(req_id=0)
    assert pool.free_blocks == 1000
    print("  ✓ KV pool accounting")

    # ── Test 3: Request SJF key ──────────────────────────────────────────────
    short_req = Request(req_id=0, arrival_time_ms=0, n_visual_tokens=256,  max_output_tokens=32)
    long_req  = Request(req_id=1, arrival_time_ms=0, n_visual_tokens=1548, max_output_tokens=512)
    assert short_req.sjf_key < long_req.sjf_key, "Short request should have lower SJF key"
    # Starvation promotion
    long_req.starvation_count = STARVATION_THRESHOLD
    assert long_req.sjf_key == 0.0, "Starvation promotion should set key = 0"
    print("  ✓ Request SJF key + starvation promotion")

    # ── Test 4: Small simulation smoke test ──────────────────────────────────
    reqs   = generate_poisson_requests(n_requests=30, arrival_rate_rps=1.0, seed=0)
    engine = ContinuousBatchingEngine(reqs, scheduler_policy="SJF")
    m      = engine.run()
    assert m.n_finished > 0, "Some requests should complete"
    assert m.gpu_utilization_pct >= 0.0
    assert m.throughput_req_s > 0.0
    print(f"  ✓ Continuous-SJF smoke test: {m.n_finished}/{m.n_requests} done, "
          f"util={m.gpu_utilization_pct:.1f}%, "
          f"p99_TTFT={m.p99_ttft_ms/1000:.1f}s")

    # ── Test 5: Static smoke test ────────────────────────────────────────────
    reqs2  = generate_poisson_requests(n_requests=30, arrival_rate_rps=1.0, seed=0)
    engine2 = StaticBatchingEngine(reqs2, batch_size=10)
    m2     = engine2.run()
    assert m2.n_finished > 0
    print(f"  ✓ Static-FCFS smoke test:  {m2.n_finished}/{m2.n_requests} done, "
          f"util={m2.gpu_utilization_pct:.1f}%, "
          f"p99_TTFT={m2.p99_ttft_ms/1000:.1f}s")

    # ── Test 6: Continuous beats static on throughput ────────────────────────
    reqs3a = generate_poisson_requests(n_requests=40, arrival_rate_rps=1.0, seed=7)
    reqs3b = generate_poisson_requests(n_requests=40, arrival_rate_rps=1.0, seed=7)
    c_m = ContinuousBatchingEngine(reqs3a, scheduler_policy="SJF").run()
    s_m = StaticBatchingEngine(reqs3b, batch_size=10).run()
    assert c_m.gpu_utilization_pct >= s_m.gpu_utilization_pct, (
        f"Continuous GPU util {c_m.gpu_utilization_pct:.1f}% should ≥ "
        f"Static {s_m.gpu_utilization_pct:.1f}%"
    )
    assert c_m.p99_ttft_ms <= s_m.p99_ttft_ms, (
        f"Continuous p99 TTFT {c_m.p99_ttft_ms:.0f} ms should ≤ "
        f"Static {s_m.p99_ttft_ms:.0f} ms"
    )
    print(f"  ✓ Continuous GPU util {c_m.gpu_utilization_pct:.1f}% ≥ "
          f"Static {s_m.gpu_utilization_pct:.1f}%  "
          f"| p99 TTFT: {c_m.p99_ttft_ms/1000:.1f}s vs {s_m.p99_ttft_ms/1000:.1f}s")

    print()
    print("All Phase 5 self-tests PASSED ✅")
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
    parser.add_argument("--seed",       type=int,   default=42,   help="RNG seed")
    args = parser.parse_args()

    if args.test:
        _run_self_test()
        sys.exit(0)

    _run_self_test()

    print("=" * 78)
    print("  Phase 5: Continuous Batching Engine — Battle of the Schedulers")
    print("=" * 78)
    print(f"  Workload    : {args.n} requests, Poisson λ={args.rate} req/s")
    print(f"  Static batch: B={args.batch_size}")
    print(f"  Decode mode : W4A8+GAR  (TBT_base = {DECODE_TBT_BASE_MS} ms, "
          f"max_batch = {MAX_DECODE_BATCH})")
    print(f"  KV pool     : {KV_POOL_BUDGET_MB:.0f} MB  →  {MAX_KV_BLOCKS:,} W4 blocks "
          f"({MB_PER_KV_BLOCK_W4:.4f} MB/block)")
    print(f"  SMs         : {M3_TOTAL_SMs} (M3 GPU), partitioned by Phase 3 SMOrchestrator")
    print()

    print("Running simulations …")
    import time as _t
    t0 = _t.time()
    m_sjf, m_fcfs, m_stat = compare_schedulers(
        n_requests=args.n,
        arrival_rate_rps=args.rate,
        static_batch_size=args.batch_size,
        seed=args.seed,
    )
    elapsed = _t.time() - t0
    print(f"  (simulation completed in {elapsed:.2f}s wall-clock time)")

    print_comparison(m_sjf, m_fcfs, m_stat)
    print_starvation_analysis(m_sjf, m_fcfs)

    # ── Poisson burst scenario (all requests arrive at once) ─────────────────
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
    )
    burst_elapsed = _t.time() - t1
    print(f"  (burst simulation completed in {burst_elapsed:.2f}s wall-clock time)")
    print_comparison(b_sjf, b_fcfs, b_stat)
    print_starvation_analysis(b_sjf, b_fcfs)
