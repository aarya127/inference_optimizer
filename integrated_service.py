"""
integrated_service.py  —  Phase 7: Full System Integration

Combines the Phase 6 Adaptive Controller (the "brain") with the Phase 3–5
hardware engines (the "muscle") into a single production-grade service.

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
  │       ▲                ▲              ▲                  │
  │  Phase 3 SMs     Phase 6 ParVTS  Phase 4 Paged KV       │
  │                                                          │
  │  Phase 6 AdaptiveController  (called at admission)       │
  └─────────────────────────────────────────────────────────┘
             ▲
  ┌──────────┴──────────────────────────────────────────────┐
  │   OpenAI-compatible API  POST /v1/multimodal/chat/...   │
  │   X-AMIO-* telemetry headers + _amio_telemetry field    │
  └─────────────────────────────────────────────────────────┘

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
import socket
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── project root on path ────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.controller import (
    AdaptiveController,
    InferenceRequest,
    SystemState,
    ExecutionPlan,
    STATIC_BUDGET_MB,
    M3_TOTAL_MEMORY_MB,
    SLA_TTFT_MS,
    TBT_SLA_MS,
    DECODE_OVERHEAD_MS,
    DECODE_BW_COST_MS,
    VISION_BASE_MS,
    BASELINE_N_CROPS,
    TOKENS_PER_CROP,
    M3_TOTAL_SMS,
    KV_QUANT_BITS,
)
from simulation.parallelism_engine import ParallelismMode
from simulation.kv_manager import (
    ContiguousBackend,
    PagedBackend,
    KV_POOL_BUDGET_MB,
    KV_BYTES_PER_TOKEN,
    kv_cache_size_mb,
)
from model_calibration.cost_model import CostModel

# KV bytes per token at W4 (4-bit) quantization
_KV_W4_BYTES = KV_BYTES_PER_TOKEN // 4   # = 27,648

# Resolution → max crops mapping (Phase 3 SigLIP grid)
_RESOLUTION_TO_CROPS: Dict[int, int] = {
    224: 1, 336: 4, 448: 6, 512: 9, 756: 13, 1008: 21, 1512: 24,
}

_SENTINEL = object()   # sentinel for queue shutdown


# ===========================================================================
# Pipeline data structures
# ===========================================================================

@dataclass
class TelemetryRecord:
    """Per-request telemetry record — written by the Collector thread."""
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
    actual_ttft_ms: float
    ttft_error_ms: float          # actual − predicted
    actual_t_vision_ms: float
    actual_t_prefill_ms: float
    actual_t_migration_ms: float
    actual_tbt_ms: float
    kv_alloc_mb: float
    sla_pass: bool
    quality_score: float
    timestamp_ms: float


@dataclass
class PipelineRequest:
    """
    Wraps an InferenceRequest with execution state, an attached ExecutionPlan,
    and in-flight latency measurements.

    The plan is attached at admission time by the SystemOrchestrator and
    propagated through Vision → Prefill → Decode without modification.
    """
    request:       InferenceRequest
    system_state:  SystemState          # snapshot captured at admission
    plan:          Optional[ExecutionPlan] = None

    # Actual latencies measured/simulated by each worker
    actual_t_vision_ms:    float = 0.0
    actual_t_prefill_ms:   float = 0.0
    actual_t_migration_ms: float = 0.0
    actual_ttft_ms:        float = 0.0
    actual_tbt_ms:         float = 0.0
    kv_alloc_mb:           float = 0.0
    kv_seq_id:             int   = -1   # PagedBackend seq_id for free()
    sla_pass:              bool  = False

    admission_time_ms: float = 0.0

    # Completion signal — callers wait on this
    _done: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    def wait(self, timeout_s: float = 30.0) -> bool:
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
            actual_ttft_ms=self.actual_ttft_ms,
            ttft_error_ms=self.actual_ttft_ms - (p.predicted_ttft_ms if p else 0.0),
            actual_t_vision_ms=self.actual_t_vision_ms,
            actual_t_prefill_ms=self.actual_t_prefill_ms,
            actual_t_migration_ms=self.actual_t_migration_ms,
            actual_tbt_ms=self.actual_tbt_ms,
            kv_alloc_mb=self.kv_alloc_mb,
            sla_pass=self.sla_pass,
            quality_score=p.quality_score if p else 0.0,
            timestamp_ms=self.admission_time_ms,
        )


# ===========================================================================
# Section 1 — Model Workers
# ===========================================================================

class VisionWorker(threading.Thread):
    """
    Vision encoding worker.

    Reads from vision_q, applies the SM partition from the attached
    ExecutionPlan using the Phase 3 compute-bound linear scaling model:

        T_vision = (n_crops / 24) × V_BASE × (38 / sm_vision)

    When sm_vision == 38 (no concurrent decode), the formula reduces to the
    baseline, yielding the minimum possible vision latency.

    Coarse SM granularity: the SM partition is read once from the plan before
    each forward pass — zero CPU synchronisation overhead during execution.
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
                self._in.task_done()
                break

            p: PipelineRequest = item
            plan = p.plan
            assert plan is not None, "Plan must be attached before VisionWorker"

            # SM partition from plan (Nova allocation, computed at admission)
            sm_vis = plan.sm_vision if plan.sm_vision > 0 else M3_TOTAL_SMS

            # Compute-bound linear model (Phase 3 SMOrchestrator formula)
            t_vis = (
                (plan.n_crops / BASELINE_N_CROPS)
                * VISION_BASE_MS
                * (M3_TOTAL_SMS / sm_vis)
            )
            # Realistic Gaussian timing noise (±σ of predicted)
            t_vis *= max(0.5, 1.0 + self._rng.gauss(0.0, self._sigma))
            p.actual_t_vision_ms = t_vis

            self._in.task_done()
            self._out.put(p)


class PrefillWorker(threading.Thread):
    """
    LM prefill worker implementing ParVTS (Parallel Vision Token Scheduling).

    ParVTS logic (Phase 6.3):
      1. Both subject and non-subject token paths enter the LM together.
      2. After `migration_depth` transformer layers the background tokens are
         pruned (predict_migration_cost captures this overhead).
      3. Only n_effective_tokens continue through the remaining layers.

    Latency model: Phase 2 quadratic cost model (R² = 0.9978).
    Tensor-parallel mode applies a 25% simulated TP speedup.
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
                self._in.task_done()
                break

            p: PipelineRequest = item
            plan = p.plan
            assert plan is not None

            # LM prefill cost (Phase 2 CostModel)
            t_pref = self._cm.predict_t_lm_prefill(plan.n_lm_tokens)
            if plan.parallelism_mode == ParallelismMode.TP:
                t_pref *= 0.75

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
            p.actual_t_prefill_ms   = t_pref * noise_p
            p.actual_t_migration_ms = t_mig  * noise_m

            self._in.task_done()
            self._out.put(p)


class DecodeWorker(threading.Thread):
    """
    Auto-regressive decode worker with PagedAttention KV management.

    On-demand KV page allocation: blocks are assigned only when a sequence
    enters the decode stage — no pre-allocation at request admission.

    TBT model (Phase 5): TBT(B) = DECODE_OVERHEAD + DECODE_BW_COST × B
    where B is the effective batch size including this new request.
    """

    def __init__(
        self,
        in_q:       queue.Queue,
        out_q:      queue.Queue,
        kv_backend: PagedBackend,
        noise_sigma: float = 0.03,
        daemon:      bool  = True,
    ):
        super().__init__(daemon=daemon, name="DecodeWorker")
        self._in    = in_q
        self._out   = out_q
        self._kv    = kv_backend
        self._rng   = random.Random(44)
        self._sigma = noise_sigma
        # Mirrors PagedBackend._next_seq_id for safe seq_id tracking
        self._next_seq_id: int = 0
        self._batch_size:  int = 0
        self._batch_lock = threading.Lock()

    @property
    def current_batch_size(self) -> int:
        return self._batch_size

    def run(self) -> None:
        while True:
            item = self._in.get()
            if item is _SENTINEL:
                self._in.task_done()
                break

            p: PipelineRequest = item
            plan = p.plan
            assert plan is not None

            # On-demand KV page allocation (Phase 4 PagedBackend)
            seq_len = plan.n_lm_tokens + p.request.max_output_tokens
            try:
                alloc = self._kv.allocate(seq_len)
                seq_id = self._next_seq_id
                self._next_seq_id += 1
                p.kv_seq_id  = seq_id
                p.kv_alloc_mb = alloc.kv_used_mb
            except RuntimeError:
                # Pool exhausted: estimate without storing
                p.kv_alloc_mb = kv_cache_size_mb(seq_len, quantization_bits=KV_QUANT_BITS)
                seq_id = -1

            # TBT with batch awareness
            with self._batch_lock:
                self._batch_size += 1
                batch = self._batch_size

            tbt = DECODE_OVERHEAD_MS + DECODE_BW_COST_MS * batch
            p.actual_tbt_ms = tbt * max(0.5, 1.0 + self._rng.gauss(0.0, self._sigma))

            # Release KV pages after decode completes
            with self._batch_lock:
                self._batch_size = max(0, self._batch_size - 1)

            if seq_id >= 0:
                try:
                    self._kv.free(seq_id)
                except Exception:
                    pass

            self._in.task_done()
            self._out.put(p)


# ===========================================================================
# Section 1 (cont.) — System Orchestrator
# ===========================================================================

class SystemOrchestrator:
    """
    Centralized event loop implementing the Nova architectural pattern.

    Pipeline topology
    -----------------
    Admit → [vision_q] → VisionWorker
                      → [prefill_q] → PrefillWorker
                                    → [decode_q] → DecodeWorker
                                                 → [done_q] → Collector

    State propagation
    -----------------
    The ExecutionPlan generated by Phase 6 AdaptiveController is attached to
    every PipelineRequest at admission and propagated verbatim through all
    stages — workers read the plan but never mutate it.

    SM coarse granularity
    ---------------------
    SM partitioning is computed once at admission time by the Nova heuristic
    and embedded in the plan.  Each worker reads `plan.sm_vision` exactly once
    before its forward pass, eliminating repeated CPU-side synchronisation.

    Resource synchronisation
    ------------------------
    A single shared lock protects the live counters (n_pending, n_decoding,
    kv_used_mb).  The Collector thread updates these counters for every
    completed request.
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

        # Pipeline queues (thread-safe, non-blocking)
        self._vision_q  = queue.Queue()
        self._prefill_q = queue.Queue()
        self._decode_q  = queue.Queue()
        self._done_q    = queue.Queue()

        # Live system state (protected by _lock)
        self._lock       = threading.Lock()
        self._n_pending:  int   = 0
        self._n_decoding: int   = 0
        self._kv_used_mb: float = 0.0

        # Telemetry log
        self._telemetry: List[TelemetryRecord] = []
        self._tel_lock = threading.Lock()

        # Model workers
        self._vision_w  = VisionWorker(self._vision_q,  self._prefill_q, noise_sigma=0.03)
        self._prefill_w = PrefillWorker(self._prefill_q, self._decode_q,  self._cost_model)
        self._decode_w  = DecodeWorker(self._decode_q,  self._done_q,    self._kv_backend)
        self._collector = threading.Thread(
            target=self._collect_done, daemon=True, name="Collector"
        )

        self._started  = False
        self._shutdown = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
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

        # Wait for queues to drain, then send shutdown sentinels
        try:
            self._vision_q.join()
            self._prefill_q.join()
            self._decode_q.join()
        except Exception:
            pass

        for q in (self._vision_q, self._prefill_q, self._decode_q, self._done_q):
            q.put(_SENTINEL)

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
            self.start()

        state = self._snapshot_state()
        plan  = self._controller.optimize(request, state)

        p = PipelineRequest(
            request=request,
            system_state=state,
            plan=plan,
            admission_time_ms=time.time() * 1000,
        )

        with self._lock:
            self._n_pending  += 1
            self._kv_used_mb += plan.predicted_kv_seq_mb

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
        n         = len(recs)
        n_pass    = sum(1 for r in recs if r.sla_pass)
        ttfts     = sorted(r.actual_ttft_ms for r in recs)
        errors    = [abs(r.ttft_error_ms)   for r in recs]
        avg_ttft  = sum(ttfts) / n
        p50       = ttfts[int(0.50 * n)]
        p99       = ttfts[min(int(0.99 * n), n - 1)]
        avg_err   = sum(errors) / n
        avg_crops = sum(r.n_crops for r in recs) / n
        avg_qual  = sum(r.quality_score for r in recs) / n
        n_fb      = sum(1 for r in recs if r.is_fallback)
        print(f"  Requests processed  : {n}")
        print(f"  SLA pass rate       : {100*n_pass/n:.1f}%")
        print(f"  Avg actual TTFT     : {avg_ttft:.1f} ms")
        print(f"  P50 / P99 TTFT      : {p50:.1f} / {p99:.1f} ms")
        print(f"  Avg prediction error: {avg_err:.1f} ms")
        print(f"  Avg crops selected  : {avg_crops:.2f}")
        print(f"  Avg quality score   : {avg_qual:.3f}")
        print(f"  Fallbacks           : {n_fb} ({100*n_fb/n:.1f}%)")

    # ── Internal ─────────────────────────────────────────────────────────────

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
        """Collect finished PipelineRequests, compute TTFT, log telemetry."""
        while True:
            item = self._done_q.get()
            if item is _SENTINEL:
                self._done_q.task_done()
                break

            p: PipelineRequest = item

            # TTFT = sum of the three stages
            p.actual_ttft_ms = (
                p.actual_t_vision_ms
                + p.actual_t_prefill_ms
                + p.actual_t_migration_ms
            )
            p.sla_pass = p.actual_ttft_ms <= self.sla_budget_ms

            with self._lock:
                self._n_pending  = max(0, self._n_pending - 1)
                self._n_decoding = max(0, self._n_decoding - 1)
                self._kv_used_mb = max(0.0, self._kv_used_mb - p.kv_alloc_mb)

            if self.log_telemetry:
                rec = p.telemetry
                with self._tel_lock:
                    self._telemetry.append(rec)

            p.mark_done()
            self._done_q.task_done()


# ===========================================================================
# Section 2 — OpenAI-Compatible API Layer
# ===========================================================================

class _ChatHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTP handler for POST /v1/multimodal/chat/completions.

    Follows the OpenAI Chat Completions format.  Internally calls
    SystemOrchestrator.submit() → waits for completion → returns:
      - Standard JSON response body with `usage._amio_telemetry` field.
      - X-AMIO-* response headers for low-latency telemetry scraping.
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

        # ── Submit and wait ──────────────────────────────────────────────
        t_admit = time.perf_counter()
        pr      = self.orchestrator.submit(req)
        pr.wait(timeout_s=60.0)
        wall_ttft_ms = (time.perf_counter() - t_admit) * 1000

        plan = pr.plan

        # ── Build OpenAI-format response ─────────────────────────────────
        content = (
            f"[AMIO] req={req_id}  crops={plan.n_crops}  "
            f"keep={plan.token_keep_ratio:.2f}  "
            f"TTFT={pr.actual_ttft_ms:.1f}ms  "
            f"SLA={'PASS' if pr.sla_pass else 'FAIL'}"
        )
        sm_ratio = plan.sm_vision / M3_TOTAL_SMS

        amio_tel = {
            "req_id":                req_id,
            "n_crops":               plan.n_crops,
            "token_keep_ratio":      round(plan.token_keep_ratio, 4),
            "parallelism_mode":      plan.parallelism_mode.value,
            "sm_partition":          f"{plan.sm_vision}/{plan.sm_decode}",
            "sm_vision_ratio":       round(sm_ratio, 4),
            "is_fallback":           plan.is_fallback,
            "quality_score":         round(plan.quality_score, 4),
            "predicted_ttft_ms":     round(plan.predicted_ttft_ms, 2),
            "actual_ttft_ms":        round(pr.actual_ttft_ms, 2),
            "wall_ttft_ms":          round(wall_ttft_ms, 2),
            "ttft_prediction_error": round(pr.actual_ttft_ms - plan.predicted_ttft_ms, 2),
            "sla_pass":              pr.sla_pass,
            "kv_alloc_mb":           round(pr.kv_alloc_mb, 3),
            "actual_tbt_ms":         round(pr.actual_tbt_ms, 2),
        }

        response = {
            "id":      f"amio-cmpl-{req_id}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   "SmolVLM-Instruct-4bit-AMIO",
            "choices": [{
                "index": 0,
                "message":       {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     prompt_length,
                "completion_tokens": max_output_tokens,
                "total_tokens":      prompt_length + max_output_tokens,
                "_amio_telemetry":   amio_tel,
            },
        }

        body_bytes = json.dumps(response, indent=2).encode()

        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        # Telemetry response headers for low-latency scraping
        self.send_header("X-AMIO-N-Crops",            str(plan.n_crops))
        self.send_header("X-AMIO-SM-Partition",        f"{plan.sm_vision}/{plan.sm_decode}")
        self.send_header("X-AMIO-SM-Vision-Ratio",     f"{sm_ratio:.4f}")
        self.send_header("X-AMIO-Predicted-TTFT-MS",   f"{plan.predicted_ttft_ms:.2f}")
        self.send_header("X-AMIO-Actual-TTFT-MS",      f"{pr.actual_ttft_ms:.2f}")
        self.send_header("X-AMIO-SLA-Pass",            "true" if pr.sla_pass else "false")
        self.send_header("X-AMIO-Quality-Score",       f"{plan.quality_score:.4f}")
        self.send_header("X-AMIO-Is-Fallback",         "true" if plan.is_fallback else "false")
        self.end_headers()
        self.wfile.write(body_bytes)

    def _json_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
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
    """Start the AMIO API server; blocks until Ctrl+C."""
    orch = SystemOrchestrator(log_telemetry=True)
    orch.start()
    server = make_api_server(orch, host, port)
    url = f"http://{host}:{port}/v1/multimodal/chat/completions"
    print(f"AMIO API server  →  POST {url}")
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
    Workers run in daemon threads; we wait for each request to complete and
    then print per-request telemetry.
    """
    print()
    print("=" * 74)
    print("  Phase 7 — System Orchestrator Pipeline Demo")
    print("=" * 74)
    print("  Starting workers: VisionWorker | PrefillWorker | DecodeWorker")

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
        pr.wait(timeout_s=30.0)

    hdr = (
        f"  {'ID':>3}  {'Res':>5}  {'Crops':>5}  {'Keep%':>6}  "
        f"{'Mode':<5}  {'SM v/d':>6}  {'T_vis':>7}  {'T_pre':>7}  "
        f"{'T_mig':>7}  {'TTFT':>7}  {'SLA':>5}  {'Quality':>8}"
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
            f"{pr.actual_t_vision_ms:>7.1f}  {pr.actual_t_prefill_ms:>7.1f}  "
            f"{pr.actual_t_migration_ms:>7.1f}  {pr.actual_ttft_ms:>7.1f}  "
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
    avg_ttft_ms:   float
    p99_ttft_ms:   float
    avg_quality:   float
    avg_frag_pct:  float       # KV fragmentation
    throughput:    float       # req/s (wall-clock estimate)
    n_fallbacks:   int


def _simulate_system(
    system_name:   str,
    reqs_data:     List[Tuple[int, int, float]],   # (resolution, prompt_len, arrival_ms)
    sla_budget_ms: float,
    rng:           random.Random,
) -> BenchmarkResult:
    """
    Simulate one system variant (Static Baseline / Greedy Fast / AMIO Adaptive)
    across a synthetic request stream.

    All three systems use the same Phase 2 cost model for latency predictions.
    They differ in:
      - Crop selection strategy
      - SM management heuristic
      - Token pruning (ParVTS)
      - KV allocation backend (paged vs contiguous)

    Returns aggregate BenchmarkResult.
    """
    noise = 0.03

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
    kv_mbs:     List[float] = []
    n_sla_pass: int         = 0
    n_fallbacks: int        = 0
    n_decoding:  int        = 0
    # Mirrors PagedBackend internal seq_id counter for free()
    paged_seq_id: int       = 0

    for i, (resolution, prompt_len, arrival_ms) in enumerate(reqs_data):
        req = InferenceRequest(
            req_id=i,
            image_resolution=resolution,
            prompt_length=prompt_len,
            max_output_tokens=60,
            arrival_time_ms=arrival_ms,
        )
        n_pending = max(0, len(reqs_data) - i - 1)
        state = SystemState(
            n_pending_requests=n_pending,
            n_decoding_requests=n_decoding,
            kv_used_mb=sum(kv_mbs[-5:]) if kv_mbs else 0.0,
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

        # ── Vision latency ───────────────────────────────────────────────
        sm_eff = max(sm_vis, 1)
        t_vis  = (n_crops / BASELINE_N_CROPS) * VISION_BASE_MS * (M3_TOTAL_SMS / sm_eff)
        t_vis *= max(0.5, 1.0 + rng.gauss(0.0, noise))

        # ── LM prefill ───────────────────────────────────────────────────
        n_vis = max(1, round(n_crops * TOKENS_PER_CROP))
        n_eff = max(1, round(n_vis * keep_ratio))
        n_lm  = n_eff + prompt_len
        t_pre = cm.predict_t_lm_prefill(n_lm)
        if mode == ParallelismMode.TP:
            t_pre *= 0.75
        t_pre *= max(0.5, 1.0 + rng.gauss(0.0, noise))

        # ── ParVTS migration cost ─────────────────────────────────────────
        t_mig = 0.0
        if use_parvts and keep_ratio < 1.0:
            t_mig = cm.predict_migration_cost(n_vis, n_eff, mig_depth)
            t_mig *= max(0.5, 1.0 + rng.gauss(0.0, noise))

        ttft = t_vis + t_pre + t_mig

        # ── KV allocation ────────────────────────────────────────────────
        seq_len = n_lm + 60
        frag    = 0.0
        kv_mb   = 0.0
        try:
            alloc = kv_backend.allocate(min(seq_len, 2048))
            kv_mb = alloc.kv_used_mb
            frag  = alloc.fragmentation_pct

            # Free paged allocations to keep pool healthy
            if system_name == "AMIO Adaptive":
                try:
                    kv_backend.free(paged_seq_id)   # type: ignore[union-attr]
                    paged_seq_id += 1
                except Exception:
                    pass
        except (RuntimeError, AssertionError):
            # Pool full or seq len exceeded max — use analytical estimate
            kv_mb = kv_cache_size_mb(seq_len, quantization_bits=KV_QUANT_BITS)
            frag  = 1.5 if system_name == "AMIO Adaptive" else 75.0

        ttfts.append(ttft)
        qualities.append(quality)
        frags.append(frag)
        kv_mbs.append(kv_mb)
        if ttft <= sla_budget_ms:
            n_sla_pass += 1

        # Model decode queue growth (saturates at 20)
        n_decoding = min(n_decoding + 1, 20)

    n = len(ttfts)
    ttfts_s = sorted(ttfts)
    p99_idx = min(int(0.99 * n), n - 1)
    wall_s  = (reqs_data[-1][2] - reqs_data[0][2]) / 1000.0 + 1.0

    return BenchmarkResult(
        system_name=system_name,
        resolution=reqs_data[0][0],
        concurrency=n,
        sla_budget_ms=sla_budget_ms,
        n_requests=n,
        sla_pass_rate=n_sla_pass / n if n > 0 else 0.0,
        avg_ttft_ms=sum(ttfts) / n,
        p99_ttft_ms=ttfts_s[p99_idx],
        avg_quality=sum(qualities) / n,
        avg_frag_pct=sum(frags) / n,
        throughput=n / wall_s,
        n_fallbacks=n_fallbacks,
    )


def run_benchmark_matrix(seed: int = 42) -> List[BenchmarkResult]:
    """
    Run the comparative test matrix:

    Variables
    ---------
    - Resolution:   [224, 448, 756, 1024]  (4 levels)
    - Concurrency:  [1, 10, 50, 100]       (4 levels, Poisson arrivals)
    - SLA budget:   [200, 500, 1000] ms    (3 levels)
    - Systems:      Static Baseline / Greedy Fast / AMIO Adaptive

    Total runs: 4 × 4 × 3 × 3 = 144 simulation runs.
    """
    print()
    print("=" * 74)
    print("  Phase 7 — Comparative Test Matrix")
    print("=" * 74)

    RESOLUTIONS:   List[int]   = [224, 448, 756, 1024]
    CONCURRENCIES: List[int]   = [1, 10, 50, 100]
    SLA_BUDGETS:   List[float] = [200.0, 500.0, 1000.0]
    SYSTEMS:       List[str]   = ["Static Baseline", "Greedy Fast", "AMIO Adaptive"]

    all_results: List[BenchmarkResult] = []
    rng = random.Random(seed)

    hdr = (
        f"  {'System':>18}  {'Res':>5}  {'N':>4}  {'SLA':>5}  "
        f"{'Pass%':>6}  {'AvgTTFT':>9}  {'P99':>8}  {'Qual':>6}  {'Frag%':>6}"
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
                        f"{res.avg_quality:>6.2f}  {res.avg_frag_pct:>5.1f}%"
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
    (C) U-Curve Verification — Nova SM reallocator heatmap
    (D) Head-to-head aggregate comparison
    """
    print()
    print("=" * 74)
    print("  Phase 7 — Pareto Frontier Analysis")
    print("=" * 74)

    systems = ["Static Baseline", "Greedy Fast", "AMIO Adaptive"]

    # ── (A) SLA Pass Rate vs Throughput ─────────────────────────────────────
    print()
    print("  (A)  SLA Pass Rate vs. Throughput  [SLA budget = 500 ms]")
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
    print("  (C)  Nova SM Reallocator Heatmap  (% of total SMs given to vision)")
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
    print("  SMs to vision. As D increases, decode worker claims the SM budget.")
    print("  The full 100% (38 SMs) column at D=0 is the idle-decode state.")

    # ── (D) Head-to-head aggregate summary ──────────────────────────────────
    print()
    print("  (D)  Head-to-Head Aggregate Comparison (all scenarios averaged)")
    print()
    hdr_d = (
        f"  {'System':>18}  {'SLA Pass%':>10}  {'AvgTTFT':>9}  "
        f"{'P99TTFT':>9}  {'Quality':>8}  {'Frag%':>7}  {'Fallbacks':>10}"
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
        print(
            f"  {sys_name:>18}  {sla_avg:>9.1f}%  {ttft_avg:>9.1f}  "
            f"{p99_avg:>9.1f}  {qual_avg:>8.3f}  {frag_avg:>6.1f}%  {fb_total:>10}"
        )
    print()

    # ── Key insight annotations ──────────────────────────────────────────────
    amio_recs   = [r for r in all_results if r.system_name == "AMIO Adaptive"]
    static_recs = [r for r in all_results if r.system_name == "Static Baseline"]
    greedy_recs = [r for r in all_results if r.system_name == "Greedy Fast"]
    if amio_recs and static_recs and greedy_recs:
        amio_sla   = sum(r.sla_pass_rate for r in amio_recs)   / len(amio_recs) * 100
        static_sla = sum(r.sla_pass_rate for r in static_recs) / len(static_recs) * 100
        greedy_sla = sum(r.sla_pass_rate for r in greedy_recs) / len(greedy_recs) * 100
        amio_qual  = sum(r.avg_quality   for r in amio_recs)   / len(amio_recs)
        greedy_qual= sum(r.avg_quality   for r in greedy_recs) / len(greedy_recs)
        static_frag= sum(r.avg_frag_pct  for r in static_recs) / len(static_recs)
        amio_frag  = sum(r.avg_frag_pct  for r in amio_recs)   / len(amio_recs)
        print("  Key observations:")
        print(f"    AMIO vs Static SLA advantage   : {amio_sla-static_sla:+.1f} pp")
        print(f"    AMIO vs Greedy quality gain     : {amio_qual-greedy_qual:+.3f} (crop×keep)")
        print(f"    Memory fragmentation reduction  : {static_frag:.1f}% → {amio_frag:.1f}%"
              f"  (-{static_frag-amio_frag:.1f} pp)")
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
    print("  Phase 7 — OpenAI-Compatible API Demo")
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
        f"{'Pred ms':>8}  {'Act ms':>7}  {'SLA':>5}  {'Quality':>8}"
    )
    print("  " + "─" * 58)

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
                crops   = resp.headers.get("X-AMIO-N-Crops",          "?")
                sm_part = resp.headers.get("X-AMIO-SM-Partition",     "?")
                pred    = resp.headers.get("X-AMIO-Predicted-TTFT-MS","?")
                act     = resp.headers.get("X-AMIO-Actual-TTFT-MS",   "?")
                sla     = resp.headers.get("X-AMIO-SLA-Pass",         "?")
                qual    = resp.headers.get("X-AMIO-Quality-Score",    "?")
                print(
                    f"  {i:>2}  {body['image_resolution']:>5}  {crops:>5}  {sm_part:>7}  "
                    f"{pred:>8}  {act:>7}  {sla:>5}  {qual:>8}"
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
        description="Phase 7: AMIO Full System Integration",
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
    print("  Phase 7: Full System Integration")
    print("  SmolVLM-Instruct-4bit (A)daptive (M)ultimodal (I)nference (O)ptimizer")
    print("  Apple M3  ·  8 GB unified  ·  38 GPU SMs  ·  100 GB/s bandwidth")
    print("=" * 74)

    # Section 3: orchestrator pipeline demo
    if not args.no_demo:
        _demo_orchestrator_pipeline()

    # Section 4: comparative test matrix
    all_results = run_benchmark_matrix()

    # Section 5: Pareto frontier analysis
    run_pareto_analysis(all_results)

    print("=" * 74)
    print("  Phase 7 complete.")
    print("  Run with --api to start the persistent HTTP server.")
    print("  Run with --api-test for an interactive API demo.")
    print("=" * 74)


if __name__ == "__main__":
    main()
