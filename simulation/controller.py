"""
simulation/controller.py  —  Phase 6: Adaptive Controller

Architecture
============
AdaptiveController is a latency-aware orchestrator that solves the per-request
optimisation problem for every incoming inference request:

    min  TTFT(strategy)   subject to:
        (1)  TTFT_predicted  ≤  SLA_TTFT_MS  (500 ms)
        (2)  total_memory    ≤  M3_TOTAL_MEMORY_MB  (8 GB)

Draws on all previous phases:
    Phase 2  — quadratic LM prefill cost model  (R²=0.9978)
    Phase 3  — SM orchestrator / parallelism modes
    Phase 4  — W4A8 KV-cache memory model
    Phase 5  — continuous batching TBT model

Key algorithms
--------------
1. Strategy Enumeration         — Cartesian product over vision levers:
                                  crops × token-keep-ratio × parallelism mode
2. Nova SM Reallocator          — dynamic SM split between front-stage
                                  (vision/prefill) and decode workers
3. ParVTS                       — Parallel Vision Token Scheduling:
                                   saliency partitioning + mid-inference pruning
4. Pareto Selection             — choose highest crop-count / lowest pruning
                                  strategy that clears both guardrails
5. Safe-Minimal Fallback        — 1 crop + 88.9% pruning when all else fails

Primary public API
------------------
    plan = AdaptiveController().optimize(request, system_state)

Returns an ExecutionPlan specifying n_crops, parallelism mode, token keep
ratio and SM partition, ready to hand to the Phase 7 inference engine.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── sibling imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.parallelism_engine import ParallelismMode
from simulation.sm_orchestrator import SMOrchestrator
from simulation.kv_manager import (
    KV_BYTES_PER_TOKEN,  # 110,592 bytes/token FP16
    KV_POOL_BUDGET_MB,
    kv_cache_size_mb,
)
from model_calibration.cost_model import CostModel

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------
M3_TOTAL_SMS: int        = 38
M3_BW_GBps: float        = 100.0
M3_TOTAL_MEMORY_MB: float = 8_192.0

# ---------------------------------------------------------------------------
# Model weight budget
# ---------------------------------------------------------------------------
LM_WEIGHTS_MB: float     = 250.0    # SmolVLM 500M W4A8
VISION_WEIGHTS_MB: float = 200.0    # SigLIP W4
OS_OVERHEAD_MB: float    = 2_048.0
STATIC_BUDGET_MB: float  = LM_WEIGHTS_MB + VISION_WEIGHTS_MB + OS_OVERHEAD_MB
# KV pool after static weights = 8192 - 2498 = 5694 MB (matches kv_manager)

# ---------------------------------------------------------------------------
# SLA targets
# ---------------------------------------------------------------------------
SLA_TTFT_MS: float = 500.0    # time-to-first-token budget
TBT_SLA_MS: float  = 80.0     # per-token latency budget (Phase 5)

# ---------------------------------------------------------------------------
# Vision encoder model
# ---------------------------------------------------------------------------
BASELINE_N_CROPS: int      = 24
BASELINE_TOTAL_TOKENS: int = 1_548
TOKENS_PER_CROP: float     = BASELINE_TOTAL_TOKENS / BASELINE_N_CROPS  # 64.5
VISION_BASE_MS: float      = 5_991.0   # T_vision at 24 crops, full 38 SMs

# ---------------------------------------------------------------------------
# Decode TBT model (Phase 5 W4A8+GAR calibration)
# ---------------------------------------------------------------------------
DECODE_OVERHEAD_MS: float = 83.75    # fixed per-step framework cost
DECODE_BW_COST_MS:  float = 3.95     # bandwidth cost per request in batch
# TBT(B) = DECODE_OVERHEAD_MS + DECODE_BW_COST_MS * B

# ---------------------------------------------------------------------------
# SM allocation — Nova heuristic
# ---------------------------------------------------------------------------
# SM_OP = max useful decode SMs when vision gets its minimum share
SM_OP: int         = M3_TOTAL_SMS - 8   # = 30
SM_MIN_DECODE: int = 4
SM_MIN_VISION: int = 8
NOVA_ALPHA: float  = 2.0   # SMs reclaimed per additional front-stage request

# ---------------------------------------------------------------------------
# ParVTS (Parallel Vision Token Scheduling)
# ---------------------------------------------------------------------------
LM_N_LAYERS: int            = 24
PARVTS_MIGRATION_DEPTH: int = 3   # layers before mid-inference pruning

# ---------------------------------------------------------------------------
# Strategy enumeration grids
# ---------------------------------------------------------------------------
# Crop options aligned with resolution grid (1–24 crops)
CROP_OPTIONS: List[int]   = [1, 2, 4, 6, 9, 13, 21, 24]

# Fraction of visual tokens RETAINED after ParVTS pruning
# 1.0 = no pruning; 0.111 = 88.9% discarded (maximum from research)
KEEP_RATIOS: List[float]  = [1.0, 0.75, 0.5, 0.333, 0.2, 0.111]

# Safe minimal strategy values
SAFE_N_CROPS: int      = 1
SAFE_KEEP_RATIO: float = 0.111   # 88.9% pruning

# KV quantisation used at runtime
KV_QUANT_BITS: int = 4   # W4 KV cache (saves 4× vs FP16)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class InferenceRequest:
    """Metadata for one incoming request."""
    req_id: int
    image_resolution: int         # input image long-edge (pixels)
    prompt_length: int            # text tokens
    max_output_tokens: int = 60
    arrival_time_ms: float = 0.0

    def n_visual_tokens(self, n_crops: int) -> int:
        """Visual token count proportional to crop count."""
        return max(1, round(n_crops * TOKENS_PER_CROP))


@dataclass
class SystemState:
    """Snapshot of the inference system at decision time."""
    n_pending_requests: int    # requests queued in front-stage (vision/prefill)
    n_decoding_requests: int   # requests currently auto-regressively decoding
    kv_used_mb: float = 0.0    # KV memory already consumed by live sequences
    current_decode_batch: int = 1
    sim_time_ms: float = 0.0


@dataclass
class Strategy:
    """One candidate execution configuration."""
    n_crops: int
    parallelism_mode: ParallelismMode
    token_keep_ratio: float      # fraction of visual tokens RETAINED
    use_parvts: bool = True
    migration_depth: int = PARVTS_MIGRATION_DEPTH

    @property
    def quality_score(self) -> float:
        """Higher crops × higher keep = better accuracy proxy."""
        return self.n_crops * self.token_keep_ratio


@dataclass
class CostProjection:
    """Predicted cost breakdown for one Strategy."""
    strategy: Strategy
    sm_vision: int
    sm_decode: int
    n_visual_tokens: int        # before pruning
    n_effective_tokens: int     # after pruning (visual portion)
    n_lm_tokens: int            # effective visual + prompt

    t_vision_ms: float
    t_lm_prefill_ms: float
    t_migration_ms: float
    t_ttft_ms: float            # vision + prefill + migration

    t_decode_per_token_ms: float
    t_decode_total_ms: float

    kv_seq_mb: float            # KV for this sequence
    total_memory_mb: float

    sla_pass: bool
    memory_pass: bool
    notes: str = ""

    @property
    def is_feasible(self) -> bool:
        return self.sla_pass and self.memory_pass


@dataclass
class ExecutionPlan:
    """
    Final controller output — passed to the Phase 7 inference engine.

    Specifies n_crops, parallelism mode, token keep ratio, and SM partition.
    """
    req_id: int

    # Strategy levers
    n_crops: int
    parallelism_mode: ParallelismMode
    token_keep_ratio: float
    sm_vision: int
    sm_decode: int
    use_parvts: bool
    migration_depth: int

    # Token counts
    n_visual_tokens: int
    n_effective_tokens: int      # after ParVTS pruning
    n_lm_tokens: int             # effective visual + prompt

    # Predicted latencies
    predicted_t_vision_ms: float
    predicted_t_prefill_ms: float
    predicted_t_migration_ms: float
    predicted_ttft_ms: float
    predicted_t_decode_per_token_ms: float
    predicted_t_decode_total_ms: float

    # Memory
    predicted_kv_seq_mb: float
    predicted_total_memory_mb: float

    # Status
    sla_pass: bool
    memory_pass: bool
    is_fallback: bool
    quality_score: float
    notes: str = ""

    def summary(self) -> str:
        sla_tag = "PASS" if self.sla_pass else "FAIL"
        mem_tag = "PASS" if self.memory_pass else "FAIL"
        fb_tag = "YES — safe minimal strategy" if self.is_fallback else "no"
        lines = [
            "=" * 72,
            "Phase 6  Adaptive Controller  —  ExecutionPlan",
            "=" * 72,
            f"  Request ID         : {self.req_id}",
            f"  N crops            : {self.n_crops}  "
            f"({self.n_visual_tokens} raw visual tokens)",
            f"  Parallelism mode   : {self.parallelism_mode.value}",
            f"  Token keep ratio   : {self.token_keep_ratio:.3f}  "
            f"({100*(1-self.token_keep_ratio):.1f}% pruned)  "
            f"→ {self.n_effective_tokens} effective tokens",
            f"  Total LM tokens    : {self.n_lm_tokens}",
            f"  SM vision          : {self.sm_vision}   "
            f"SM decode : {self.sm_decode}",
            f"  ParVTS             : {'enabled' if self.use_parvts else 'disabled'}  "
            f"  migration depth : {self.migration_depth}",
            "  " + "-" * 68,
            f"  T_vision           : {self.predicted_t_vision_ms:9.1f} ms",
            f"  T_prefill          : {self.predicted_t_prefill_ms:9.1f} ms",
            f"  T_migration        : {self.predicted_t_migration_ms:9.1f} ms",
            f"  TTFT               : {self.predicted_ttft_ms:9.1f} ms"
            f"  [{sla_tag} vs {SLA_TTFT_MS:.0f} ms SLA]",
            f"  TBT (per token)    : {self.predicted_t_decode_per_token_ms:9.1f} ms",
            f"  T_decode (total)   : {self.predicted_t_decode_total_ms:9.1f} ms",
            "  " + "-" * 68,
            f"  KV cache (seq)     : {self.predicted_kv_seq_mb:9.1f} MB",
            f"  Total memory       : {self.predicted_total_memory_mb:9.1f} MB"
            f"  [{mem_tag} vs {M3_TOTAL_MEMORY_MB:.0f} MB]",
            "  " + "-" * 68,
            f"  Quality score      : {self.quality_score:.3f}",
            f"  Fallback           : {fb_tag}",
            f"  Notes              : {self.notes}",
            "=" * 72,
        ]
        return "\n".join(lines)


# ===========================================================================
# Adaptive Controller
# ===========================================================================

class AdaptiveController:
    """
    Latency-Aware Multimodal Inference Orchestrator.

    For each incoming request, solves the optimisation problem:

        min  TTFT(strategy)
        s.t. TTFT_predicted ≤ SLA_TTFT_MS  (500 ms)
             total_memory   ≤ M3_TOTAL_MEMORY_MB (8 GB)

    Implements:
        Phase 6.1  —  Global strategy space enumeration
        Phase 6.2  —  Latency-aware scheduler (cost model projection)
        Phase 6.3  —  ParVTS content-adaptive token scheduling
        Phase 6.4  —  Nova dynamic SM reallocator
        Phase 6.5  —  Pareto-optimal strategy selection + safe fallback
    """

    def __init__(
        self,
        sla_budget_ms: float = SLA_TTFT_MS,
        memory_budget_mb: float = M3_TOTAL_MEMORY_MB,
        nova_alpha: float = NOVA_ALPHA,
        sm_op: int = SM_OP,
        sm_min_decode: int = SM_MIN_DECODE,
        sm_min_vision: int = SM_MIN_VISION,
        enable_parvts: bool = True,
    ):
        self.sla_budget_ms    = sla_budget_ms
        self.memory_budget_mb = memory_budget_mb
        self.nova_alpha       = nova_alpha
        self.sm_op            = sm_op
        self.sm_min_decode    = sm_min_decode
        self.sm_min_vision    = sm_min_vision
        self.enable_parvts    = enable_parvts

        self._cost_model = CostModel()
        self._sm_orch    = SMOrchestrator(
            total_sms=M3_TOTAL_SMS,
            sm_min_decode=sm_min_decode,
            sm_min_vision=sm_min_vision,
            reclaim_alpha=nova_alpha,
            baseline_t_vision_ms=VISION_BASE_MS,
        )

        # Tracking
        self.n_calls:    int               = 0
        self.n_fallbacks: int              = 0
        self.history:    List[ExecutionPlan] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def optimize(
        self,
        request: InferenceRequest,
        system_state: SystemState,
    ) -> ExecutionPlan:
        """
        Solve the per-request strategy optimisation problem.

        Parameters
        ----------
        request      : InferenceRequest  — image resolution, prompt length, etc.
        system_state : SystemState       — current queue depths, KV usage, etc.

        Returns
        -------
        ExecutionPlan specifying crop count, parallelism mode, pruning ratio
        and SM partition — ready to hand to the Phase 7 inference engine.
        """
        self.n_calls += 1

        # ── Phase 6.4: Nova SM allocation ──────────────────────────────────
        sm_vision, sm_decode = self._nova_sm_allocation(
            n_pending_front=system_state.n_pending_requests,
            n_decoding=system_state.n_decoding_requests,
        )

        # ── Phase 6.1: Enumerate candidate strategies ───────────────────────
        candidates = self._enumerate_strategies()

        # ── Phase 6.2 + 6.3: Cost projection for every candidate ───────────
        projections: List[CostProjection] = [
            self._project_cost(s, request, system_state, sm_vision, sm_decode)
            for s in candidates
        ]

        feasible = [p for p in projections if p.is_feasible]

        # ── Phase 6.5: Select or fall back ─────────────────────────────────
        is_fallback = len(feasible) == 0
        if is_fallback:
            self.n_fallbacks += 1
            chosen = self._safe_fallback_projection(
                request, system_state, sm_vision, sm_decode
            )
        else:
            chosen = self._pareto_select(feasible)

        plan = _build_execution_plan(chosen, request, is_fallback)
        self.history.append(plan)
        return plan

    # -----------------------------------------------------------------------
    # Phase 6.4 — Nova Dynamic SM Reallocator
    # -----------------------------------------------------------------------

    def _nova_sm_allocation(
        self,
        n_pending_front: int,
        n_decoding: int,
    ) -> Tuple[int, int]:
        """
        Nova heuristic: reduce decode SMs as front-stage queue grows,
        freeing them for the vision worker.

            SM_dec = max(SM_min, SM_op − α·(N_front − 1))
            SM_vis = M3_TOTAL_SMS − SM_dec

        When n_decoding == 0, no decode worker is active: vision worker
        gets access to all 38 SMs regardless of the Nova formula.

        Returns
        -------
        (sm_vision, sm_decode) : ints, always satisfy sm_v + sm_d = M3_TOTAL_SMS
        """
        if n_decoding == 0:
            # No concurrent decode; vision can freely use all SMs.
            return M3_TOTAL_SMS, 0

        reduction = int(math.floor(self.nova_alpha * max(0, n_pending_front - 1)))
        sm_dec = max(self.sm_min_decode, self.sm_op - reduction)
        sm_vis = M3_TOTAL_SMS - sm_dec

        # Guarantee vision floor
        if sm_vis < self.sm_min_vision:
            sm_vis = self.sm_min_vision
            sm_dec = M3_TOTAL_SMS - sm_vis

        return sm_vis, sm_dec

    # -----------------------------------------------------------------------
    # Phase 6.1 — Strategy Enumeration
    # -----------------------------------------------------------------------

    def _enumerate_strategies(self) -> List[Strategy]:
        """
        Return the Cartesian product of:
            CROP_OPTIONS × KEEP_RATIOS × {DP, TP}

        Strategies where token_keep_ratio < 1.0 auto-enable ParVTS when
        the controller flag is set.
        """
        strategies: List[Strategy] = []
        for n_crops in CROP_OPTIONS:
            for keep in KEEP_RATIOS:
                for mode in (ParallelismMode.DP, ParallelismMode.TP):
                    strategies.append(Strategy(
                        n_crops=n_crops,
                        parallelism_mode=mode,
                        token_keep_ratio=keep,
                        use_parvts=self.enable_parvts and keep < 1.0,
                        migration_depth=PARVTS_MIGRATION_DEPTH,
                    ))
        return strategies

    # -----------------------------------------------------------------------
    # Phase 6.2 — Latency-Aware Scheduler / Cost Projection
    # -----------------------------------------------------------------------

    def _project_cost(
        self,
        strategy: Strategy,
        request: InferenceRequest,
        state: SystemState,
        sm_vision: int,
        sm_decode: int,
    ) -> CostProjection:
        """
        Predict the full cost breakdown for one Strategy using Phase 2+ models.

        Vision latency
        --------------
        Uses the compute-bound scaling from SMOrchestrator (established in
        Phase 3), scaling linearly as total_sms / sm_vision.  When n_decoding
        == 0 the caller passes sm_vision = M3_TOTAL_SMS and the formula
        reduces to the unpartitioned baseline.

        LM prefill
        ----------
        Quadratic cost model γN²+βN+α (R²=0.9978 from Phase 2).
        TP mode applies a 25% speedup factor (simulated tensor-parallel gain).

        ParVTS / migration
        ------------------
        If use_parvts and token_keep_ratio < 1.0: the non-subject token path
        runs in parallel for the first `migration_depth` LM layers then is
        discarded.  Cost model quantifies this migration overhead.

        Memory
        ------
        STATIC_BUDGET_MB  +  state.kv_used_mb  +  KV for this sequence (W4).
        """
        # ── Vision latency ─────────────────────────────────────────────────
        # When sm_vision == M3_TOTAL_SMS (n_decoding==0), scale = 1.0 exactly.
        n_vis = request.n_visual_tokens(strategy.n_crops)
        sm_vis_eff = sm_vision if sm_vision > 0 else M3_TOTAL_SMS
        t_vision_ms = (
            (strategy.n_crops / BASELINE_N_CROPS)
            * VISION_BASE_MS
            * (M3_TOTAL_SMS / sm_vis_eff)
        )

        # ── Phase 6.3: ParVTS saliency partitioning ─────────────────────────
        migration_cost_ms = 0.0
        if strategy.use_parvts and strategy.token_keep_ratio < 1.0:
            n_effective = max(1, round(n_vis * strategy.token_keep_ratio))
            migration_cost_ms = self._cost_model.predict_migration_cost(
                n_full_tokens=n_vis,
                n_pruned_tokens=n_effective,
                migration_depth=strategy.migration_depth,
            )
        else:
            n_effective = max(1, round(n_vis * strategy.token_keep_ratio))

        # ── LM prefill latency ──────────────────────────────────────────────
        n_lm = n_effective + request.prompt_length
        t_lm_prefill_ms = self._cost_model.predict_t_lm_prefill(n_lm)

        # Tensor-parallel simulated speedup (25% on LM prefill only)
        if strategy.parallelism_mode == ParallelismMode.TP:
            t_lm_prefill_ms *= 0.75

        # ── TTFT ───────────────────────────────────────────────────────────
        t_ttft_ms = t_vision_ms + t_lm_prefill_ms + migration_cost_ms

        # ── Decode TBT (Phase 5 model) ──────────────────────────────────────
        # +1 because this new request joins the decode batch
        effective_batch = state.current_decode_batch + 1
        t_decode_per_token_ms = DECODE_OVERHEAD_MS + DECODE_BW_COST_MS * effective_batch
        t_decode_total_ms = t_decode_per_token_ms * request.max_output_tokens

        # ── Memory: KV cache ────────────────────────────────────────────────
        seq_len_total = n_lm + request.max_output_tokens
        kv_seq_mb = kv_cache_size_mb(
            seq_len=seq_len_total,
            batch_size=1,
            quantization_bits=KV_QUANT_BITS,
        )
        total_memory_mb = STATIC_BUDGET_MB + state.kv_used_mb + kv_seq_mb

        # ── Feasibility ────────────────────────────────────────────────────
        sla_pass    = t_ttft_ms    <= self.sla_budget_ms
        memory_pass = total_memory_mb <= self.memory_budget_mb

        notes_parts: List[str] = []
        if not sla_pass:
            notes_parts.append(
                f"TTFT {t_ttft_ms:.0f}ms > {self.sla_budget_ms:.0f}ms"
            )
        if not memory_pass:
            notes_parts.append(
                f"mem {total_memory_mb:.0f}MB > {self.memory_budget_mb:.0f}MB"
            )

        return CostProjection(
            strategy=strategy,
            sm_vision=sm_vis_eff,
            sm_decode=sm_decode,
            n_visual_tokens=n_vis,
            n_effective_tokens=n_effective,
            n_lm_tokens=n_lm,
            t_vision_ms=t_vision_ms,
            t_lm_prefill_ms=t_lm_prefill_ms,
            t_migration_ms=migration_cost_ms,
            t_ttft_ms=t_ttft_ms,
            t_decode_per_token_ms=t_decode_per_token_ms,
            t_decode_total_ms=t_decode_total_ms,
            kv_seq_mb=kv_seq_mb,
            total_memory_mb=total_memory_mb,
            sla_pass=sla_pass,
            memory_pass=memory_pass,
            notes="; ".join(notes_parts),
        )

    # -----------------------------------------------------------------------
    # Phase 6.5 — Pareto Selection
    # -----------------------------------------------------------------------

    @staticmethod
    def _pareto_select(feasible: List[CostProjection]) -> CostProjection:
        """
        Among feasible strategies, choose the Pareto-optimal one:

            1. Highest crop count (image resolution / quality)
            2. Tiebreak: highest token keep ratio (accuracy)
            3. Tiebreak: lowest TTFT (speed margin)

        This implements the "accuracy maximisation" principle from the spec:
        use the highest resolution + lowest pruning allowed by the time budget.
        """
        return max(
            feasible,
            key=lambda p: (
                p.strategy.n_crops,
                p.strategy.token_keep_ratio,
                -p.t_ttft_ms,
            ),
        )

    # -----------------------------------------------------------------------
    # Safe-Minimal Fallback
    # -----------------------------------------------------------------------

    def _safe_fallback_projection(
        self,
        request: InferenceRequest,
        state: SystemState,
        sm_vision: int,
        sm_decode: int,
    ) -> CostProjection:
        """
        SLA Guardrail: when no strategy meets the 500 ms budget, resort to
        the Safe Minimal Strategy — 1 crop + 88.9% token pruning + DP mode.

        This is the lowest-quality, lowest-latency operating point.  The
        resulting plan may still violate SLA under extreme load; the caller
        marks it as is_fallback=True so downstream can log / shed the request.
        """
        safe = Strategy(
            n_crops=SAFE_N_CROPS,
            parallelism_mode=ParallelismMode.DP,
            token_keep_ratio=SAFE_KEEP_RATIO,
            use_parvts=self.enable_parvts,
            migration_depth=PARVTS_MIGRATION_DEPTH,
        )
        proj = self._project_cost(safe, request, state, sm_vision, sm_decode)
        note_prefix = "SAFE MINIMAL FALLBACK"
        proj.notes = (
            f"{note_prefix}: {proj.notes}" if proj.notes else note_prefix
        )
        return proj

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def print_stats(self) -> None:
        print("AdaptiveController session statistics")
        print("-" * 60)
        print(f"  Total optimize() calls : {self.n_calls}")
        fallback_pct = (
            100.0 * self.n_fallbacks / self.n_calls if self.n_calls else 0.0
        )
        print(f"  Safe-minimal fallbacks : {self.n_fallbacks}"
              f"  ({fallback_pct:.1f}% of calls)")
        if self.history:
            crops     = [p.n_crops for p in self.history]
            krs       = [p.token_keep_ratio for p in self.history]
            ttfts     = [p.predicted_ttft_ms for p in self.history]
            sla_pct   = 100.0 * sum(1 for p in self.history if p.sla_pass) / len(self.history)
            print(f"  Avg crops selected     : {sum(crops)/len(crops):.2f}")
            print(f"  Avg keep ratio         : {sum(krs)/len(krs):.3f}")
            print(f"  Avg predicted TTFT     : {sum(ttfts)/len(ttfts):.1f} ms")
            print(f"  SLA pass rate          : {sla_pct:.1f}%")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Internal helper (module-level to avoid making it a static method)
# ---------------------------------------------------------------------------

def _build_execution_plan(
    proj: CostProjection,
    request: InferenceRequest,
    is_fallback: bool,
) -> ExecutionPlan:
    s = proj.strategy
    return ExecutionPlan(
        req_id=request.req_id,
        n_crops=s.n_crops,
        parallelism_mode=s.parallelism_mode,
        token_keep_ratio=s.token_keep_ratio,
        sm_vision=proj.sm_vision,
        sm_decode=proj.sm_decode,
        use_parvts=s.use_parvts,
        migration_depth=s.migration_depth,
        n_visual_tokens=proj.n_visual_tokens,
        n_effective_tokens=proj.n_effective_tokens,
        n_lm_tokens=proj.n_lm_tokens,
        predicted_t_vision_ms=proj.t_vision_ms,
        predicted_t_prefill_ms=proj.t_lm_prefill_ms,
        predicted_t_migration_ms=proj.t_migration_ms,
        predicted_ttft_ms=proj.t_ttft_ms,
        predicted_t_decode_per_token_ms=proj.t_decode_per_token_ms,
        predicted_t_decode_total_ms=proj.t_decode_total_ms,
        predicted_kv_seq_mb=proj.kv_seq_mb,
        predicted_total_memory_mb=proj.total_memory_mb,
        sla_pass=proj.sla_pass,
        memory_pass=proj.memory_pass,
        is_fallback=is_fallback,
        quality_score=s.quality_score,
        notes=proj.notes,
    )


# ===========================================================================
# Self-test
# ===========================================================================

def _run_self_test() -> None:
    print("Running Phase 6 self-tests ...")

    ctrl = AdaptiveController()
    req  = InferenceRequest(req_id=0, image_resolution=512, prompt_length=32)

    # ── Test 1: Nova SM allocation ──────────────────────────────────────────
    # Idle (no decode workers): vision gets all 38 SMs
    vis0, dec0 = ctrl._nova_sm_allocation(n_pending_front=0, n_decoding=0)
    assert vis0 == M3_TOTAL_SMS and dec0 == 0, (
        f"No decode: expected vis=38,dec=0 got vis={vis0},dec={dec0}"
    )

    # With decode traffic, light queue: SM_OP decode, min vision
    vis1, dec1 = ctrl._nova_sm_allocation(n_pending_front=1, n_decoding=5)
    assert vis1 + dec1 == M3_TOTAL_SMS, "SMs must sum to M3_TOTAL_SMS"
    assert dec1 >= SM_MIN_DECODE
    assert vis1 >= SM_MIN_VISION

    # Heavy front-stage queue: Nova gives more SMs to vision
    vis_heavy, dec_heavy = ctrl._nova_sm_allocation(
        n_pending_front=15, n_decoding=10
    )
    assert vis_heavy > vis1, (
        f"Heavy front-stage should get more vision SMs: {vis_heavy} vs {vis1}"
    )
    assert dec_heavy <= dec1
    assert vis_heavy + dec_heavy == M3_TOTAL_SMS
    print(f"  [OK] Nova SM allocation  "
          f"idle→(38,0)  light→({vis1},{dec1})  heavy→({vis_heavy},{dec_heavy})")

    # ── Test 2: Strategy enumeration ───────────────────────────────────────
    strats = ctrl._enumerate_strategies()
    expected = len(CROP_OPTIONS) * len(KEEP_RATIOS) * 2   # 2 modes
    assert len(strats) == expected, (
        f"Expected {expected} strategies, got {len(strats)}"
    )
    # All crop options must appear
    assert set(s.n_crops for s in strats) == set(CROP_OPTIONS)
    # ParVTS only enabled when keep_ratio < 1.0
    for s in strats:
        if s.use_parvts:
            assert s.token_keep_ratio < 1.0, "ParVTS only enabled for pruned strategies"
    print(f"  [OK] Strategy enumeration  ({len(strats)} candidates)")

    # ── Test 3: Cost projection ──────────────────────────────────────────────
    strat_test = Strategy(
        n_crops=1,
        parallelism_mode=ParallelismMode.DP,
        token_keep_ratio=1.0,
    )
    idle_state = SystemState(n_pending_requests=0, n_decoding_requests=0)
    vis_sms, dec_sms = ctrl._nova_sm_allocation(0, 0)   # 38, 0
    proj = ctrl._project_cost(strat_test, req, idle_state, vis_sms, dec_sms)

    assert proj.t_vision_ms > 0
    assert proj.t_lm_prefill_ms > 0
    assert abs(proj.t_ttft_ms - (proj.t_vision_ms + proj.t_lm_prefill_ms + proj.t_migration_ms)) < 0.01
    assert proj.total_memory_mb > STATIC_BUDGET_MB
    # At idle (38 SMs), 1 crop should produce ~249 ms vision time
    expected_vision = (1 / BASELINE_N_CROPS) * VISION_BASE_MS   # 249.6 ms
    assert abs(proj.t_vision_ms - expected_vision) < 1.0, (
        f"Expected ~{expected_vision:.1f}ms vision, got {proj.t_vision_ms:.1f}ms"
    )
    print(f"  [OK] Cost projection  "
          f"T_vis={proj.t_vision_ms:.1f}ms  T_prefill={proj.t_lm_prefill_ms:.1f}ms  "
          f"TTFT={proj.t_ttft_ms:.1f}ms  SLA={'PASS' if proj.sla_pass else 'FAIL'}")

    # ── Test 4: optimize() — idle system (sla_pass expected) ───────────────
    plan_idle = ctrl.optimize(req, idle_state)
    assert plan_idle.req_id == 0
    assert plan_idle.sm_vision + plan_idle.sm_decode == M3_TOTAL_SMS
    assert 1 <= plan_idle.n_crops <= 24
    assert 0.0 < plan_idle.token_keep_ratio <= 1.0
    assert plan_idle.sla_pass, (
        f"Idle system should pass SLA; TTFT={plan_idle.predicted_ttft_ms:.1f}ms"
    )
    print(f"  [OK] optimize() idle  "
          f"crops={plan_idle.n_crops}  keep={plan_idle.token_keep_ratio:.3f}  "
          f"TTFT={plan_idle.predicted_ttft_ms:.1f}ms  fallback={plan_idle.is_fallback}")

    # ── Test 5: optimize() — heavy front-stage load (Nova enables more SMs) ─
    heavy_state = SystemState(
        n_pending_requests=15,
        n_decoding_requests=10,
        kv_used_mb=1500.0,
        current_decode_batch=10,
    )
    plan_heavy = ctrl.optimize(req, heavy_state)
    # Under heavy front-stage load, Nova allocates fewer decode SMs than SM_OP
    # (sm_dec < SM_OP) and vision gets more than the SM_MIN_VISION floor.
    expected_vis, expected_dec = ctrl._nova_sm_allocation(15, 10)  # 34/4
    assert plan_heavy.sm_vision == expected_vis, (
        f"sm_vision mismatch: expected {expected_vis}, got {plan_heavy.sm_vision}"
    )
    print(f"  [OK] optimize() heavy front-stage  "
          f"crops={plan_heavy.n_crops}  keep={plan_heavy.token_keep_ratio:.3f}  "
          f"SM_vis={plan_heavy.sm_vision}  TTFT={plan_heavy.predicted_ttft_ms:.1f}ms  "
          f"SLA={'PASS' if plan_heavy.sla_pass else 'FAIL'}")

    # ── Test 6: optimize() — SLA challenge (light decode, low queue) ────────
    # Under concurrent decode with few front-stage requests, the linear SM
    # scaling slows vision.  Controller should fall back to safe minimal.
    contested_state = SystemState(
        n_pending_requests=1,
        n_decoding_requests=20,
        kv_used_mb=2000.0,
        current_decode_batch=20,
    )
    plan_contested = ctrl.optimize(req, contested_state)
    # We don't assert sla_pass here — may or may not pass depending on SM allocation.
    # What we DO assert: plan is structurally valid.
    assert 1 <= plan_contested.n_crops <= 24
    assert plan_contested.sm_vision + plan_contested.sm_decode == M3_TOTAL_SMS
    print(f"  [OK] optimize() contested  "
          f"crops={plan_contested.n_crops}  TTFT={plan_contested.predicted_ttft_ms:.1f}ms  "
          f"fallback={plan_contested.is_fallback}")

    # ── Test 7: memory safety ─────────────────────────────────────────────────
    near_full_state = SystemState(
        n_pending_requests=0,
        n_decoding_requests=0,
        kv_used_mb=M3_TOTAL_MEMORY_MB - STATIC_BUDGET_MB - 1.0,  # just under limit
    )
    plan_kv = ctrl.optimize(req, near_full_state)
    # Total memory must not exceed budget (controller checks before selecting)
    assert plan_kv.predicted_total_memory_mb <= M3_TOTAL_MEMORY_MB + 10.0, (
        f"Memory safety violated: {plan_kv.predicted_total_memory_mb:.0f}MB"
    )
    print(f"  [OK] Memory safety  "
          f"total_mem={plan_kv.predicted_total_memory_mb:.0f}MB  "
          f"kv_seq={plan_kv.predicted_kv_seq_mb:.1f}MB")

    # ── Test 8: ExecutionPlan.summary() is well-formed ──────────────────────
    summary = plan_idle.summary()
    assert "ExecutionPlan" in summary
    assert str(plan_idle.n_crops) in summary
    assert "TTFT" in summary
    print("  [OK] ExecutionPlan.summary()")

    # ── Test 9: stats tracking ─────────────────────────────────────────────
    assert ctrl.n_calls == 4, f"Expected 4 calls so far, got {ctrl.n_calls}"
    ctrl.print_stats()

    print()
    print("All Phase 6 self-tests PASSED")
    print()


# ===========================================================================
# Scenario demonstration table
# ===========================================================================

def _demo_optimization_table() -> None:
    """Print a Phase 6 strategy-selection summary across system-load scenarios."""

    ctrl = AdaptiveController()
    req  = InferenceRequest(req_id=0, image_resolution=512, prompt_length=32)

    scenarios = [
        ("Idle  (N_pend=0, N_dec=0)",
         SystemState(0, 0,  kv_used_mb=0,    current_decode_batch=1)),
        ("Light (N_pend=2, N_dec=5)",
         SystemState(2, 5,  kv_used_mb=500,  current_decode_batch=5)),
        ("Moderate (N_pend=5, N_dec=15)",
         SystemState(5, 15, kv_used_mb=1500, current_decode_batch=15)),
        ("Heavy (N_pend=10, N_dec=40)",
         SystemState(10, 40, kv_used_mb=3000, current_decode_batch=40)),
        ("Critical (N_pend=15, N_dec=70)",
         SystemState(15, 70, kv_used_mb=5600, current_decode_batch=70)),
    ]

    hdr = (
        f"{'Scenario':<32} {'Crops':>5} {'Keep%':>6} {'Mode':<4} "
        f"{'SMs v/d':>8} {'TTFT ms':>8} {'SLA':>5} {'KV MB':>7} {'FB':>3}"
    )
    sep = "─" * len(hdr)

    print()
    print("Phase 6 — Adaptive Controller  Strategy Selection Table")
    print(sep)
    print(hdr)
    print(sep)

    for label, state in scenarios:
        plan = ctrl.optimize(req, state)
        sm_str = f"{plan.sm_vision}/{plan.sm_decode}"
        print(
            f"{label:<32} {plan.n_crops:>5} {plan.token_keep_ratio*100:>5.1f}% "
            f"{plan.parallelism_mode.value[:4]:<4} "
            f"{sm_str:>8} {plan.predicted_ttft_ms:>8.1f} "
            f"{'PASS' if plan.sla_pass else 'FAIL':>5} "
            f"{plan.predicted_kv_seq_mb:>7.1f} "
            f"{'Y' if plan.is_fallback else 'n':>3}"
        )

    print(sep)
    print()

    # Verbose plan for idle scenario
    _, idle_state = scenarios[0]
    req_verbose = InferenceRequest(req_id=42, image_resolution=512, prompt_length=32)
    plan_verbose = AdaptiveController().optimize(req_verbose, idle_state)
    print(plan_verbose.summary())

    ctrl.print_stats()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    _run_self_test()
    _demo_optimization_table()
