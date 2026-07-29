"""
simulation/controller.py  —  Phase 6: Adaptive Controller

Architecture
============
AdaptiveController is a latency-aware orchestrator that solves the per-request
optimisation problem for every incoming inference request:

    max  quality_score(strategy)   subject to:
        (1)  TTFT_predicted  ≤  SLA_TTFT_MS  (500 ms)
        (2)  total_memory    ≤  M3_TOTAL_MEMORY_MB  (8 GB)
    (TTFT is used as the tiebreak among equal-quality feasible plans.)

NOTE quality_score = n_crops × token_keep_ratio is a COMPUTE PROXY (how much
visual input the model actually processes), NOT a measured accuracy metric.
No VQA benchmark backs it; treat it as an input-fidelity budget.

Draws on all previous phases:
    Phase 2  — MEASURED quadratic LM prefill cost model (fit to 4 real
               stage-isolated points, domain N ∈ [100, 1560]; superseded
               the earlier synthetic-embedding fit)
    Phase 3  — compute-share orchestrator ("SMs" are abstract shares; the M3
               base die has 10 GPU cores — see amio_constants) with a
               MEASURED vision cost model (553.5 ms/crop + 27.4 ms)
    Phase 4  — W4 KV-cache memory model
    Phase 5  — continuous batching TBT model (MEASURED batch=1 constants,
               modeled batch-scaling term)

Key algorithms
--------------
1. Strategy Enumeration         — Cartesian product over vision levers:
                                  crops × token-keep-ratio × parallelism mode.
                                  Only 4 crop settings exist on this
                                  pipeline's processor (1/5/10/17 — there is
                                  no "24 crops" mode), giving 48 candidates;
                                  the parallelism dimension is COST-NEUTRAL on
                                  a single-chip target (see below), so the
                                  effective search space is 24 strategies.
2. Nova Share Reallocator       — dynamic compute-share split between the
                                  front stage (vision/prefill) and decode
3. ParVTS                       — Parallel Vision Token Scheduling:
                                  saliency partitioning + mid-inference pruning
4. Selection                    — feasible plan maximising quality_score,
                                  TTFT as tiebreak
5. Safe-Minimal Fallback        — 1 crop + 88.9% pruning when all else fails

Single-chip parallelism disclaimer
----------------------------------
There is exactly ONE GPU in the target (Apple M3).  Single-chip tensor
parallelism is NOT modeled: the previous ×0.75 prefill "TP speedup" was a
fiction and has been removed.  ParallelismMode is kept on plans purely as an
API-compatible annotation; any real TP gain would require multiple devices,
which is out of scope for this study.

Primary public API
------------------
    plan = AdaptiveController().optimize(request, system_state)

Returns an ExecutionPlan specifying n_crops, parallelism mode, token keep
ratio and share partition, ready to hand to the Phase 7 inference engine.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# ── sibling imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import amio_constants as C

from simulation.parallelism_engine import ParallelismMode

try:
    from simulation.parallelism_engine import RESOLUTION_TO_CROPS
except ImportError:   # keep the controller importable if the map moves
    RESOLUTION_TO_CROPS = dict(C.CROP_SETTINGS)   # {384:1, 768:5, 1152:10, 1536:17}

from simulation.sm_orchestrator import SMOrchestrator
from model_calibration.cost_model import CostModel

# ---------------------------------------------------------------------------
# Hardware constants (single source of truth: amio_constants)
# ---------------------------------------------------------------------------
# "SMs" here are abstract compute shares, not hardware partitions: the M3
# base die has 10 GPU cores and Metal exposes no per-task core partitioning.
M3_TOTAL_SMS: int        = C.TOTAL_COMPUTE_SHARES     # 38 abstract shares
M3_BW_GBps: float        = C.M3_MEMORY_BW_GBPS
M3_TOTAL_MEMORY_MB: float = C.TOTAL_MEMORY_MB

# ---------------------------------------------------------------------------
# Model weight budget (measured checkpoint size + OS reserve)
# ---------------------------------------------------------------------------
MODEL_WEIGHTS_MB: float  = C.MODEL_WEIGHTS_MB     # 1390 MB, 4-bit, vision+LM
OS_OVERHEAD_MB: float    = C.OS_RESERVE_MB        # 2048 MB (assumption)
STATIC_BUDGET_MB: float  = MODEL_WEIGHTS_MB + OS_OVERHEAD_MB   # 3438 MB
KV_POOL_BUDGET_MB: float = C.KV_POOL_BUDGET_MB    # 8192 − 3438 = 4754 MB

# ---------------------------------------------------------------------------
# SLA targets
# ---------------------------------------------------------------------------
SLA_TTFT_MS: float = C.TTFT_SLA_MS    # time-to-first-token budget (500 ms)
TBT_SLA_MS: float  = C.TBT_SLA_MS     # per-token latency budget (80 ms).
# NOTE: under the MEASURED decode model TBT(1) ≈ 18-25 ms, comfortably
# under budget; TTFT_SLA_MS (500 ms) is the binding, infeasible constraint —
# measured vision cost alone (581 ms at 1 crop) already exceeds it. Plans
# carry honest sla_pass / tbt_sla_pass flags reflecting whichever holds.

# ---------------------------------------------------------------------------
# Vision encoder model — MEASURED (baseline/results_v2.json, 2026-07-28)
# ---------------------------------------------------------------------------
# T_vision(c) = VISION_MS_PER_CROP * c + VISION_FIXED_MS  (near-perfectly
# linear across the processor's four real crop settings, full-GPU trials).
# The old "5991 ms / 24 crops" figure was an unvalidated residual and the
# "24 crops" setting never existed in this pipeline — MAX_CROPS is 17.
VISION_MS_PER_CROP: float  = C.VISION_MS_PER_CROP      # 553.5
VISION_FIXED_MS: float     = C.VISION_FIXED_MS         # 27.4
BASELINE_N_CROPS: int      = C.MAX_CROPS               # 17 (processor max)
BASELINE_TOTAL_TOKENS: int = C.TOKENS_PER_CONFIG[C.MAX_CROPS]  # 1560
TOKENS_PER_CROP: float     = C.TOKENS_PER_CROP          # 81 (checkpoint config)

# ---------------------------------------------------------------------------
# Decode TBT model — MEASURED at batch=1 (per-token timestamps, 5 trials x
# 4 configs x 32 tokens). Batch scaling beyond 1 is an explicit modeling
# assumption: each additional concurrent sequence adds one more KV read per
# step (bandwidth-bound), estimated at ctx_tokens * KV_BYTES_PER_TOKEN / BW.
# ---------------------------------------------------------------------------
DECODE_OVERHEAD_MS: float = C.DECODE_OVERHEAD_MS_MEASURED   # 18.33
DECODE_KV_MS_PER_CTX_TOKEN: float = C.DECODE_KV_MS_PER_CTX_TOKEN  # 0.00320
# TBT(ctx, B) = DECODE_OVERHEAD_MS + DECODE_KV_MS_PER_CTX_TOKEN * ctx
#               + (B - 1) * (ctx * KV_BYTES_PER_TOKEN_FP16 / BW_GBps / 1e6)
#   [last term: extra concurrent sequences' KV reads, modeled not measured]

# ---------------------------------------------------------------------------
# Compute-share allocation — Nova heuristic
# ---------------------------------------------------------------------------
# SM_OP = full decode share when vision holds its minimum share
SM_OP: int         = M3_TOTAL_SMS - 8   # = 30
SM_MIN_DECODE: int = 4
SM_MIN_VISION: int = 8
NOVA_ALPHA: float  = 2.0   # shares reclaimed per additional front-stage request

# ---------------------------------------------------------------------------
# ParVTS (Parallel Vision Token Scheduling)
# ---------------------------------------------------------------------------
LM_N_LAYERS: int            = C.LM_NUM_LAYERS
PARVTS_MIGRATION_DEPTH: int = 3   # layers before mid-inference pruning

# ---------------------------------------------------------------------------
# Strategy enumeration grids
# ---------------------------------------------------------------------------
# Crop options — the real Idefics3 processor settings (measured); there is
# no "24 crops" mode in this pipeline.
CROP_OPTIONS: List[int]   = [1, 5, 10, 17]

# Fraction of visual tokens RETAINED after ParVTS pruning
# 1.0 = no pruning; 0.111 = 88.9% discarded (maximum from research)
KEEP_RATIOS: List[float]  = [1.0, 0.75, 0.5, 0.333, 0.2, 0.111]

# Safe minimal strategy values
SAFE_N_CROPS: int      = 1
SAFE_KEEP_RATIO: float = 0.111   # 88.9% pruning

# KV quantisation used at runtime
KV_QUANT_BITS: int = 4   # W4 KV cache (saves 4× vs FP16)

# Modeled assumption: expected residual output tokens per live decoding
# sequence, used to reserve KV growth headroom at admission time.
REMAINING_DECODE_TOKENS_EST: int = 60


def _kv_seq_mb(n_tokens: int, quant_bits: int = KV_QUANT_BITS) -> float:
    """KV cache MB for n_tokens at the given KV quantisation.

    Derived from amio_constants.KV_BYTES_PER_TOKEN_FP16 = 196,608 B/token
    (2 × 24 layers × 32 KV heads × 64 head_dim × 2 B — the old 110,592 figure
    was computed from the VISION encoder's hidden size and was wrong).
    """
    bytes_per_token = C.KV_BYTES_PER_TOKEN_FP16 * quant_bits / 16.0
    return n_tokens * bytes_per_token / (1024 ** 2)


def _max_crops_for_resolution(resolution_px: int) -> int:
    """Upper bound on useful crops for an input image resolution.

    Uses the same resolution→crops map as parallelism_engine
    (RESOLUTION_TO_CROPS): an image whose long edge is 512 px tiles into at
    most 9 crops — enumerating 24-crop strategies for it would model compute
    the encoder never performs.
    """
    eligible = [res for res in RESOLUTION_TO_CROPS if res >= resolution_px]
    key = min(eligible) if eligible else max(RESOLUTION_TO_CROPS)
    return RESOLUTION_TO_CROPS[key]


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class InferenceRequest:
    """Metadata for one incoming request."""
    req_id: int
    image_resolution: int         # input image long-edge (pixels); bounds the
                                  # candidate crop counts via RESOLUTION_TO_CROPS
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
    current_decode_batch: int = 0   # informational; the controller derives the
                                    # effective batch as n_decoding_requests + 1
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
        """Compute proxy: crops × keep ratio (input fidelity, NOT accuracy)."""
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
    total_memory_mb: float      # incl. reserved KV growth of live sequences

    sla_pass: bool
    memory_pass: bool
    tbt_sla_pass: bool = False
    notes: str = ""

    @property
    def is_feasible(self) -> bool:
        return self.sla_pass and self.memory_pass


@dataclass
class ExecutionPlan:
    """
    Final controller output — passed to the Phase 7 inference engine.

    Specifies n_crops, parallelism mode, token keep ratio, and share
    partition.  parallelism_mode is an annotation only: single-chip TP is
    not modeled (cost-neutral), so it never affects predicted latencies.
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
    quality_score: float         # compute proxy, NOT accuracy
    tbt_sla_pass: bool = False
    notes: str = ""

    def summary(self) -> str:
        sla_tag = "PASS" if self.sla_pass else "FAIL"
        mem_tag = "PASS" if self.memory_pass else "FAIL"
        tbt_tag = "PASS" if self.tbt_sla_pass else "FAIL"
        fb_tag = "YES — safe minimal strategy" if self.is_fallback else "no"
        lines = [
            "=" * 72,
            "Phase 6  Adaptive Controller  —  ExecutionPlan",
            "=" * 72,
            f"  Request ID         : {self.req_id}",
            f"  N crops            : {self.n_crops}  "
            f"({self.n_visual_tokens} raw visual tokens)",
            f"  Parallelism mode   : {self.parallelism_mode.value}  "
            f"(annotation only — single-chip TP not modeled)",
            f"  Token keep ratio   : {self.token_keep_ratio:.3f}  "
            f"({100*(1-self.token_keep_ratio):.1f}% pruned)  "
            f"→ {self.n_effective_tokens} effective tokens",
            f"  Total LM tokens    : {self.n_lm_tokens}",
            f"  Shares vision      : {self.sm_vision}   "
            f"Shares decode : {self.sm_decode}",
            f"  ParVTS             : {'enabled' if self.use_parvts else 'disabled'}  "
            f"  migration depth : {self.migration_depth}",
            "  " + "-" * 68,
            f"  T_vision           : {self.predicted_t_vision_ms:9.1f} ms",
            f"  T_prefill          : {self.predicted_t_prefill_ms:9.1f} ms",
            f"  T_migration        : {self.predicted_t_migration_ms:9.1f} ms",
            f"  TTFT               : {self.predicted_ttft_ms:9.1f} ms"
            f"  [{sla_tag} vs {SLA_TTFT_MS:.0f} ms SLA]",
            f"  TBT (per token)    : {self.predicted_t_decode_per_token_ms:9.1f} ms"
            f"  [{tbt_tag} vs {TBT_SLA_MS:.0f} ms SLA]",
            f"  T_decode (total)   : {self.predicted_t_decode_total_ms:9.1f} ms",
            "  " + "-" * 68,
            f"  KV cache (seq)     : {self.predicted_kv_seq_mb:9.1f} MB",
            f"  Total memory       : {self.predicted_total_memory_mb:9.1f} MB"
            f"  [{mem_tag} vs {M3_TOTAL_MEMORY_MB:.0f} MB]",
            "  " + "-" * 68,
            f"  Quality score      : {self.quality_score:.3f}  (compute proxy)",
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

        max  quality_score(strategy)   (compute proxy: crops × keep ratio)
        s.t. TTFT_predicted ≤ SLA_TTFT_MS  (500 ms)
             total_memory   ≤ M3_TOTAL_MEMORY_MB (8 GB)
        (tiebreak: lowest predicted TTFT)

    Implements:
        Phase 6.1  —  Strategy space enumeration (4 crop settings x 6 keep
                      ratios x 2 modes = 48 strategies, 24 cost-distinct;
                      the ×2 parallelism dimension is cost-neutral)
        Phase 6.2  —  Latency-aware scheduler (cost model projection)
        Phase 6.3  —  ParVTS content-adaptive token scheduling
        Phase 6.4  —  Nova dynamic share reallocator (with a modeled decode
                      contention penalty — reallocation is not free)
        Phase 6.5  —  Quality-maximising selection + safe fallback
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
            vision_ms_per_crop=VISION_MS_PER_CROP,
            vision_fixed_ms=VISION_FIXED_MS,
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
        and share partition — ready to hand to the Phase 7 inference engine.
        """
        self.n_calls += 1

        # ── Phase 6.4: Nova share allocation ────────────────────────────────
        sm_vision, sm_decode = self._nova_sm_allocation(
            n_pending_front=system_state.n_pending_requests,
            n_decoding=system_state.n_decoding_requests,
        )

        # ── Phase 6.1: Enumerate candidate strategies ───────────────────────
        # image_resolution bounds the useful crop counts: a 512 px image
        # tiles into at most 9 crops.
        max_crops = _max_crops_for_resolution(request.image_resolution)
        candidates = self._enumerate_strategies(max_crops=max_crops)

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
            chosen = self._select_best(feasible)

        plan = _build_execution_plan(chosen, request, is_fallback)
        self.history.append(plan)
        return plan

    # -----------------------------------------------------------------------
    # Phase 6.4 — Nova Dynamic Share Reallocator
    # -----------------------------------------------------------------------

    def _nova_sm_allocation(
        self,
        n_pending_front: int,
        n_decoding: int,
    ) -> Tuple[int, int]:
        """
        Nova heuristic: reduce decode shares as front-stage queue grows,
        freeing them for the vision worker.

            SM_dec = max(SM_min, SM_op − α·(N_front − 1))
            SM_vis = M3_TOTAL_SMS − SM_dec

        When n_decoding == 0, no decode worker is active: vision worker
        gets access to all 38 shares regardless of the Nova formula.

        Reallocation is NOT free: _project_cost applies a contention penalty
        to decode TBT when sm_decode drops below SM_OP.

        Returns
        -------
        (sm_vision, sm_decode) : ints, always satisfy sm_v + sm_d = M3_TOTAL_SMS
        """
        if n_decoding == 0:
            # No concurrent decode; vision can freely use all shares.
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

    def _enumerate_strategies(
        self,
        max_crops: int = BASELINE_N_CROPS,
    ) -> List[Strategy]:
        """
        Return the Cartesian product of:
            {crops ≤ max_crops} × KEEP_RATIOS × {DP, TP}

        max_crops comes from the request's image resolution (a small image
        cannot need 24 crops).  The DP/TP dimension is kept for API
        compatibility but is cost-neutral on single-chip hardware, so the
        effective strategy space is half the enumerated size.

        Strategies where token_keep_ratio < 1.0 auto-enable ParVTS when
        the controller flag is set.
        """
        crop_choices = [n for n in CROP_OPTIONS if n <= max_crops] or [min(CROP_OPTIONS)]
        strategies: List[Strategy] = []
        for n_crops in crop_choices:
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
        MEASURED linear model T_vision(c) = VISION_MS_PER_CROP*c + VISION_FIXED_MS
        (full-GPU trials, near-perfectly linear, R ratios 540-566 ms/crop),
        scaled by total_shares / sm_vision (compute-bound share-scaling is a
        modeling assumption; when n_decoding == 0 the caller passes
        sm_vision = M3_TOTAL_SMS and the scale factor is 1.0).

        LM prefill
        ----------
        Quadratic cost model γN²+βN+α (in-sample R²=0.9978 from Phase 2).
        ParallelismMode does NOT change this cost: single-chip TP is not
        modeled (there is one GPU; the former ×0.75 factor was fictitious).

        ParVTS / migration
        ------------------
        If use_parvts and token_keep_ratio < 1.0: the non-subject token path
        runs in parallel for the first `migration_depth` LM layers then is
        discarded.  Cost model quantifies this migration overhead (a modeled
        assumption that grows with migration depth).

        Decode contention
        -----------------
        The same penalty as the Phase 5 batching engine:
        TBT × clamp(sqrt(SM_OP / sm_decode), 1.0, 1.15) — Nova reallocation
        away from decode has a modeled cost, not pure upside.

        Memory
        ------
        STATIC_BUDGET_MB + state.kv_used_mb + reserved KV growth of live
        decoding sequences + KV for this sequence (W4).
        """
        # ── Vision latency ─────────────────────────────────────────────────
        # When sm_vision == M3_TOTAL_SMS (n_decoding==0), scale = 1.0 exactly.
        n_vis = request.n_visual_tokens(strategy.n_crops)
        t_vision_ms = (
            (VISION_MS_PER_CROP * strategy.n_crops + VISION_FIXED_MS)
            * (M3_TOTAL_SMS / sm_vision)
        )

        # ── Phase 6.3: ParVTS saliency partitioning ─────────────────────────
        migration_cost_ms = 0.0
        n_effective = max(1, round(n_vis * strategy.token_keep_ratio))
        if strategy.use_parvts and strategy.token_keep_ratio < 1.0:
            migration_cost_ms = self._cost_model.predict_migration_cost(
                n_full_tokens=n_vis,
                n_pruned_tokens=n_effective,
                migration_depth=strategy.migration_depth,
            )

        # ── LM prefill latency ──────────────────────────────────────────────
        # (parallelism_mode intentionally has no effect — see docstring)
        n_lm = n_effective + request.prompt_length
        t_lm_prefill_ms = self._cost_model.predict_t_lm_prefill(n_lm)

        # ── TTFT ───────────────────────────────────────────────────────────
        t_ttft_ms = t_vision_ms + t_lm_prefill_ms + migration_cost_ms

        # ── Decode TBT (measured batch=1 model + modeled batch scaling) ─────
        # +1 because this new request joins the decode batch
        effective_batch = state.n_decoding_requests + 1
        ctx_tokens = n_lm + request.max_output_tokens // 2   # mid-decode context
        kv_read_ms_per_extra_seq = (
            ctx_tokens * C.KV_BYTES_PER_TOKEN_FP16 / (C.M3_MEMORY_BW_GBPS * 1e6)
        )
        tbt_raw = (
            DECODE_OVERHEAD_MS
            + DECODE_KV_MS_PER_CTX_TOKEN * ctx_tokens
            + (effective_batch - 1) * kv_read_ms_per_extra_seq
        )
        if sm_decode > 0:
            contention = min(1.15, max(1.0, math.sqrt(SM_OP / sm_decode)))
        else:
            contention = 1.0   # no live decode contention to model
        t_decode_per_token_ms = tbt_raw * contention
        t_decode_total_ms = t_decode_per_token_ms * request.max_output_tokens

        # ── Memory: KV cache + reserved growth of live sequences ────────────
        seq_len_total = n_lm + request.max_output_tokens
        kv_seq_mb = _kv_seq_mb(seq_len_total)
        kv_growth_reserve_mb = (
            state.n_decoding_requests * _kv_seq_mb(REMAINING_DECODE_TOKENS_EST)
        )
        total_memory_mb = (
            STATIC_BUDGET_MB + state.kv_used_mb + kv_growth_reserve_mb + kv_seq_mb
        )

        # ── Feasibility ────────────────────────────────────────────────────
        sla_pass     = t_ttft_ms       <= self.sla_budget_ms
        memory_pass  = total_memory_mb <= self.memory_budget_mb
        tbt_sla_pass = t_decode_per_token_ms <= TBT_SLA_MS   # honest; False
                                                             # under current model

        notes_parts: List[str] = []
        if not sla_pass:
            notes_parts.append(
                f"TTFT {t_ttft_ms:.0f}ms > {self.sla_budget_ms:.0f}ms"
            )
        if not memory_pass:
            notes_parts.append(
                f"mem {total_memory_mb:.0f}MB > {self.memory_budget_mb:.0f}MB"
            )
        if not tbt_sla_pass:
            notes_parts.append(
                f"TBT {t_decode_per_token_ms:.0f}ms > {TBT_SLA_MS:.0f}ms (advisory)"
            )

        return CostProjection(
            strategy=strategy,
            sm_vision=sm_vision,
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
            tbt_sla_pass=tbt_sla_pass,
            notes="; ".join(notes_parts),
        )

    # -----------------------------------------------------------------------
    # Phase 6.5 — Selection
    # -----------------------------------------------------------------------

    @staticmethod
    def _select_best(feasible: List[CostProjection]) -> CostProjection:
        """
        Among feasible strategies, choose the one maximising quality_score
        (the crops × keep-ratio compute proxy), breaking ties by lowest
        predicted TTFT.

        This replaces the previous lexicographic max over (crops, keep,
        −TTFT), which was not a Pareto rule and could pick a plan with a
        LOWER quality score than another feasible plan (e.g. 24 crops ×
        keep 0.111 = 2.66 over 21 crops × keep 1.0 = 21).
        """
        return max(
            feasible,
            key=lambda p: (p.strategy.quality_score, -p.t_ttft_ms),
        )

    # Backwards-compatible alias for the old (misnamed) selection entry point.
    _pareto_select = _select_best

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
        the Safe Minimal Strategy — 1 crop + 88.9% token pruning.

        The parallelism mode is DP purely as an annotation: both modes cost
        the same (single-chip TP is not modeled), so there is no "slower
        mode" to avoid.

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
            tbt_pct   = 100.0 * sum(1 for p in self.history if p.tbt_sla_pass) / len(self.history)
            print(f"  Avg crops selected     : {sum(crops)/len(crops):.2f}")
            print(f"  Avg keep ratio         : {sum(krs)/len(krs):.3f}")
            print(f"  Avg predicted TTFT     : {sum(ttfts)/len(ttfts):.1f} ms")
            print(f"  TTFT SLA pass rate     : {sla_pct:.1f}%  "
                  f"(500ms is infeasible on this hardware — vision alone at "
                  f"1 crop is ~581ms; reported honestly)")
            print(f"  TBT SLA pass rate      : {tbt_pct:.1f}%")
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
        tbt_sla_pass=proj.tbt_sla_pass,
        notes=proj.notes,
    )


# ===========================================================================
# Self-test
# ===========================================================================

def _run_self_test() -> None:
    print("Running Phase 6 self-tests ...")

    ctrl = AdaptiveController()
    req  = InferenceRequest(req_id=0, image_resolution=512, prompt_length=32)

    # ── Test 1: Nova share allocation ────────────────────────────────────────
    # Idle (no decode workers): vision gets all 38 shares
    vis0, dec0 = ctrl._nova_sm_allocation(n_pending_front=0, n_decoding=0)
    assert vis0 == M3_TOTAL_SMS and dec0 == 0, (
        f"No decode: expected vis=38,dec=0 got vis={vis0},dec={dec0}"
    )

    # With decode traffic, light queue: SM_OP decode, min vision
    vis1, dec1 = ctrl._nova_sm_allocation(n_pending_front=1, n_decoding=5)
    assert vis1 + dec1 == M3_TOTAL_SMS, "Shares must sum to M3_TOTAL_SMS"
    assert dec1 >= SM_MIN_DECODE
    assert vis1 >= SM_MIN_VISION

    # Heavy front-stage queue: Nova gives more shares to vision
    vis_heavy, dec_heavy = ctrl._nova_sm_allocation(
        n_pending_front=15, n_decoding=10
    )
    assert vis_heavy > vis1, (
        f"Heavy front-stage should get more vision shares: {vis_heavy} vs {vis1}"
    )
    assert dec_heavy <= dec1
    assert vis_heavy + dec_heavy == M3_TOTAL_SMS
    print(f"  [OK] Nova share allocation  "
          f"idle→(38,0)  light→({vis1},{dec1})  heavy→({vis_heavy},{dec_heavy})")

    # ── Test 2: Strategy enumeration ───────────────────────────────────────
    strats = ctrl._enumerate_strategies()
    expected = len(CROP_OPTIONS) * len(KEEP_RATIOS) * 2   # 2 (cost-neutral) modes
    assert len(strats) == expected, (
        f"Expected {expected} strategies, got {len(strats)}"
    )
    # All crop options must appear
    assert set(s.n_crops for s in strats) == set(CROP_OPTIONS)
    # ParVTS only enabled when keep_ratio < 1.0
    for s in strats:
        if s.use_parvts:
            assert s.token_keep_ratio < 1.0, "ParVTS only enabled for pruned strategies"
    # Resolution bound: a 512 px image tiles into at most 5 crops (the
    # processor's next setting at/above 512, size.longest_edge=768)
    bound_512 = _max_crops_for_resolution(512)
    strats_512 = ctrl._enumerate_strategies(max_crops=bound_512)
    assert max(s.n_crops for s in strats_512) <= bound_512, (
        f"512 px image must not enumerate >{bound_512}-crop strategies"
    )
    print(f"  [OK] Strategy enumeration  ({len(strats)} candidates, "
          f"24 cost-distinct; 512px bounded to ≤{bound_512} crops)")

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
    # At idle (38 shares), 1 crop should match the measured linear model
    expected_vision = VISION_MS_PER_CROP * 1 + VISION_FIXED_MS   # ~580.9 ms
    assert abs(proj.t_vision_ms - expected_vision) < 1.0, (
        f"Expected ~{expected_vision:.1f}ms vision, got {proj.t_vision_ms:.1f}ms"
    )
    # Parallelism mode is cost-neutral: TP projection must match DP exactly
    strat_tp = Strategy(n_crops=1, parallelism_mode=ParallelismMode.TP,
                        token_keep_ratio=1.0)
    proj_tp = ctrl._project_cost(strat_tp, req, idle_state, vis_sms, dec_sms)
    assert abs(proj_tp.t_ttft_ms - proj.t_ttft_ms) < 1e-9, (
        "Single-chip TP must not change predicted cost"
    )
    print(f"  [OK] Cost projection  "
          f"T_vis={proj.t_vision_ms:.1f}ms  T_prefill={proj.t_lm_prefill_ms:.1f}ms  "
          f"TTFT={proj.t_ttft_ms:.1f}ms  SLA={'PASS' if proj.sla_pass else 'FAIL'}  "
          f"(TP == DP cost)")

    # ── Test 4: optimize() — idle system ───────────────────────────────────
    # HONEST FINDING: the measured vision cost alone (580.9 ms at 1 crop,
    # idle/full-shares) already exceeds the 500 ms TTFT SLA, so NO strategy
    # is SLA-feasible on this hardware — the controller must report that
    # honestly (sla_pass=False) rather than have it papered over, and must
    # fall back to the safe-minimal strategy since the feasible set is empty.
    plan_idle = ctrl.optimize(req, idle_state)
    assert plan_idle.req_id == 0
    assert plan_idle.sm_vision + plan_idle.sm_decode == M3_TOTAL_SMS
    assert 1 <= plan_idle.n_crops <= BASELINE_N_CROPS
    assert 0.0 < plan_idle.token_keep_ratio <= 1.0
    projs_all = [
        ctrl._project_cost(s, req, idle_state, *ctrl._nova_sm_allocation(0, 0))
        for s in ctrl._enumerate_strategies(_max_crops_for_resolution(req.image_resolution))
    ]
    any_feasible = any(p.is_feasible for p in projs_all)
    assert not any_feasible, (
        "Expected the 500ms TTFT SLA to be infeasible under the measured "
        "vision cost model at every crop setting — if this now passes, "
        "the SLA is achievable and this test (and is_fallback below) "
        "should be revisited."
    )
    assert plan_idle.is_fallback, (
        "With an empty feasible set the controller must select the safe "
        "minimal strategy (1 crop, 88.9% pruning) and flag is_fallback"
    )
    assert not plan_idle.sla_pass, (
        f"sla_pass must be honestly False when TTFT={plan_idle.predicted_ttft_ms:.1f}"
        f"ms > {SLA_TTFT_MS:.0f}ms SLA"
    )
    assert plan_idle.n_crops == SAFE_N_CROPS
    assert abs(plan_idle.token_keep_ratio - SAFE_KEEP_RATIO) < 1e-9
    print(f"  [OK] optimize() idle  "
          f"crops={plan_idle.n_crops}  keep={plan_idle.token_keep_ratio:.3f}  "
          f"quality={plan_idle.quality_score:.2f}  "
          f"TTFT={plan_idle.predicted_ttft_ms:.1f}ms  "
          f"SLA={'PASS' if plan_idle.sla_pass else 'FAIL (infeasible on this hardware)'}  "
          f"fallback={plan_idle.is_fallback}")

    # ── Test 5: optimize() — heavy front-stage load (Nova reallocates) ──────
    heavy_state = SystemState(
        n_pending_requests=15,
        n_decoding_requests=10,
        kv_used_mb=1500.0,
        current_decode_batch=10,
    )
    plan_heavy = ctrl.optimize(req, heavy_state)
    expected_vis, _expected_dec = ctrl._nova_sm_allocation(15, 10)  # 34/4
    assert plan_heavy.sm_vision == expected_vis, (
        f"sm_vision mismatch: expected {expected_vis}, got {plan_heavy.sm_vision}"
    )
    # Decode contention penalty must be applied (sm_decode=4 < SM_OP=30)
    ctx_tokens = plan_heavy.n_lm_tokens + req.max_output_tokens // 2
    kv_read_ms_per_extra_seq = (
        ctx_tokens * C.KV_BYTES_PER_TOKEN_FP16 / (C.M3_MEMORY_BW_GBPS * 1e6)
    )
    raw_tbt = (
        DECODE_OVERHEAD_MS
        + DECODE_KV_MS_PER_CTX_TOKEN * ctx_tokens
        + heavy_state.n_decoding_requests * kv_read_ms_per_extra_seq
    )
    assert plan_heavy.predicted_t_decode_per_token_ms > raw_tbt, (
        "Nova reallocation away from decode must carry a modeled TBT penalty"
    )
    assert plan_heavy.predicted_t_decode_per_token_ms <= raw_tbt * 1.15 + 1e-9
    print(f"  [OK] optimize() heavy front-stage  "
          f"crops={plan_heavy.n_crops}  keep={plan_heavy.token_keep_ratio:.3f}  "
          f"SM_vis={plan_heavy.sm_vision}  TTFT={plan_heavy.predicted_ttft_ms:.1f}ms  "
          f"TBT={plan_heavy.predicted_t_decode_per_token_ms:.1f}ms (penalised)")

    # ── Test 6: optimize() — SLA challenge (light queue, heavy decode) ──────
    contested_state = SystemState(
        n_pending_requests=1,
        n_decoding_requests=20,
        kv_used_mb=2000.0,
        current_decode_batch=20,
    )
    plan_contested = ctrl.optimize(req, contested_state)
    # We don't assert sla_pass here — may or may not pass depending on shares.
    # What we DO assert: plan is structurally valid.
    assert 1 <= plan_contested.n_crops <= 24
    assert plan_contested.sm_vision + plan_contested.sm_decode == M3_TOTAL_SMS
    print(f"  [OK] optimize() contested  "
          f"crops={plan_contested.n_crops}  TTFT={plan_contested.predicted_ttft_ms:.1f}ms  "
          f"fallback={plan_contested.is_fallback}")

    # ── Test 7: memory safety ────────────────────────────────────────────────
    # (a) Non-fallback plans must sit strictly within the memory budget.
    moderate_state = SystemState(
        n_pending_requests=0,
        n_decoding_requests=0,
        kv_used_mb=1000.0,
    )
    plan_mod = ctrl.optimize(req, moderate_state)
    if not plan_mod.is_fallback:
        assert plan_mod.predicted_total_memory_mb <= M3_TOTAL_MEMORY_MB, (
            f"Non-fallback plan exceeds memory budget: "
            f"{plan_mod.predicted_total_memory_mb:.0f}MB"
        )
        assert plan_mod.memory_pass
    # (b) Near-full KV: every strategy is memory-infeasible → fallback plan,
    #     honestly flagged (memory_pass=False, is_fallback=True).
    near_full_state = SystemState(
        n_pending_requests=0,
        n_decoding_requests=0,
        kv_used_mb=M3_TOTAL_MEMORY_MB - STATIC_BUDGET_MB - 1.0,  # just under limit
    )
    plan_kv = ctrl.optimize(req, near_full_state)
    assert plan_kv.is_fallback and not plan_kv.memory_pass, (
        "Near-full KV must produce an honestly-flagged fallback plan"
    )
    print(f"  [OK] Memory safety  "
          f"moderate: total={plan_mod.predicted_total_memory_mb:.0f}MB "
          f"(fallback={plan_mod.is_fallback})  "
          f"near-full: fallback flagged, mem_pass={plan_kv.memory_pass}")

    # ── Test 8: ExecutionPlan.summary() is well-formed ──────────────────────
    summary = plan_idle.summary()
    assert "ExecutionPlan" in summary
    assert str(plan_idle.n_crops) in summary
    assert "TTFT" in summary
    assert "TBT" in summary
    print("  [OK] ExecutionPlan.summary()")

    # ── Test 9: stats tracking ─────────────────────────────────────────────
    assert ctrl.n_calls == 5, f"Expected 5 calls so far, got {ctrl.n_calls}"
    # TBT SLA is evaluated and honestly recorded (False under current model)
    assert all(hasattr(p, "tbt_sla_pass") for p in ctrl.history)
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
         SystemState(0, 0,  kv_used_mb=0,    current_decode_batch=0)),
        ("Light (N_pend=2, N_dec=5)",
         SystemState(2, 5,  kv_used_mb=500,  current_decode_batch=5)),
        ("Moderate (N_pend=5, N_dec=15)",
         SystemState(5, 15, kv_used_mb=1500, current_decode_batch=15)),
        ("Heavy (N_pend=10, N_dec=40)",
         SystemState(10, 40, kv_used_mb=3000, current_decode_batch=40)),
        ("Critical (N_pend=15, N_dec=70)",
         SystemState(15, 70, kv_used_mb=4600, current_decode_batch=70)),
    ]

    hdr = (
        f"{'Scenario':<32} {'Crops':>5} {'Keep%':>6} {'Mode':<4} "
        f"{'Shr v/d':>8} {'TTFT ms':>8} {'SLA':>5} {'TBT ms':>7} {'KV MB':>7} {'FB':>3}"
    )
    sep = "─" * len(hdr)

    print()
    print("Phase 6 — Adaptive Controller  Strategy Selection Table")
    print("(quality = compute proxy; TP/DP modes are cost-neutral annotations)")
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
            f"{plan.predicted_t_decode_per_token_ms:>7.1f} "
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
