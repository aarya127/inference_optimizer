"""
SM Orchestrator — Adaptive Compute-Share Partitioning

HARDWARE NOTE (honesty): the Apple M3 base die has 10 GPU cores.  "SM" is
NVIDIA nomenclature and Metal/MLX exposes no per-task core partitioning.
The 38 "SMs" used throughout this module are retained ONLY as abstract
*modeled compute shares* (see amio_constants.TOTAL_COMPUTE_SHARES): they
express fractional compute priority between two concurrent workers, not
hardware partitions.

This module models how to split the 38 compute shares between two workers:
  • Vision worker  — prefill / vision encoder (latency-critical)
  • Decode worker  — auto-regressive token generation (bandwidth-bound)

Strategy
--------
The decode worker is memory-bandwidth bound and needs few compute shares to
keep the memory bus saturated.  We use a "drain-first" heuristic:

    SM_dec = clamp(SM_min_dec + α·(N_pending − 1), SM_min_dec, total − SM_min_vis)

As concurrent decode pressure grows, the decode worker's share GROWS
(reclaiming shares from the vision worker) until the vision floor
(`sm_min_vision`) is hit.  This matches the Phase 4 controller usage.

Additionally, we model pipeline overlap — if the decode worker and vision
worker run truly concurrently on distinct share partitions, the end-to-end
latency approaches max(T_vision, T_decode) rather than T_vision + T_decode.
Overlap savings are capped by shared memory bandwidth: two concurrent
streams cannot move more bytes per second than the bus provides, so
pipelined time ≥ (bytes_vision + bytes_decode) / BW.

Phase 4 additions
-----------------
  DECODE_STRATEGIES      — single module-level definition of each KV/quant
                           strategy (used by BOTH analyses below).
  predict_tbt_ms()       — shared bandwidth-based decode TBT model.
  BatchExpansionResult   — models how PagedAttention + KV quantisation
                           change the maximum feasible batch size.
  decode_starvation_analysis()  — finds the batch size ceiling where TBT
                                  exceeds the 80 ms threshold.

TBT model (MEASURED at batch=1, baseline/results_v2.json):

    TBT(b, ctx) = DECODE_OVERHEAD_MS_MEASURED
                  + DECODE_KV_MS_PER_CTX_TOKEN × ctx × (kv_bytes / KV_FP16)
                  + (b − 1) × (ctx × kv_bytes/token / BW)

  - The measured batch-1 fit is TBT(ctx) = 18.33 + 0.00320·ctx ms
    (per-token timestamps over 160 tokens/config on the M3-8GB target).
    The 18.33 ms overhead already INCLUDES the per-step weight read
    (~14.6 ms modeled for the 1390 MiB W4 blob) — it is not added again.
  - Batch scaling is UNMEASURED: each extra sequence is modeled as its own
    KV-prefix read, ≈ 3.0 ms/seq at 1548 ctx FP16 KV (196,608 B/tok ÷
    100 GB/s) — a MODELED ASSUMPTION, flagged as such.
  - Quantized-KV strategies scale both KV terms by their kv_bytes ratio
    (modeled, unvalidated on MLX/M3).

Under the measured model TBT(1, 1560) ≈ 23.3 ms, so the 80 ms TBT SLA is
comfortably met at moderate batch (SLA ceiling ≈ 19 sequences at 1560-token
FP16 KV contexts).  The old 83.75 + weight-read model (TBT(1) ≈ 101 ms,
"SLA unmeetable") was never calibrated and is SUPERSEDED.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Allow `import amio_constants` when run as a script from anywhere
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import amio_constants

# ---------------------------------------------------------------------------
# Hardware / model constants (single source of truth: amio_constants)
# ---------------------------------------------------------------------------
# 38 = abstract modeled compute shares, NOT hardware units (M3 has 10 GPU
# cores; see module docstring and amio_constants).  Name kept for API
# stability with controller / batching_engine.
M3_TOTAL_SMs: int = amio_constants.TOTAL_COMPUTE_SHARES
M3_GPU_CORES: int = amio_constants.M3_GPU_CORES
M3_BANDWIDTH_GBps: float = amio_constants.M3_MEMORY_BW_GBPS

# SUPERSEDED: the old Phase-1 single-trial 217.7 ms/1926.5 ms decode figures
# were never calibrated; the MEASURED batch-1 model is
# TBT(ctx) = 18.33 + 0.00320·ctx ms (≈23.3 ms at 1548 ctx).
TBT_HUMAN_THRESHOLD_MS: float = amio_constants.TBT_SLA_MS   # 80 ms

# KV cache constants (corrected: 2 × 24 layers × 32 heads × 64 head_dim × 2 B)
KV_BYTES_PER_TOKEN_FP16: int = amio_constants.KV_BYTES_PER_TOKEN_FP16  # 196,608
KV_BYTES_PER_TOKEN_W8: int = amio_constants.KV_BYTES_PER_TOKEN_W8      #  98,304
KV_BYTES_PER_TOKEN_W4: int = amio_constants.KV_BYTES_PER_TOKEN_W4      #  49,152
KV_POOL_BUDGET_MB: float = amio_constants.KV_POOL_BUDGET_MB            # 4754 MB
MAX_SEQ_LEN: int = 2048

# Decode TBT model constants — MEASURED (baseline/results_v2.json)
DECODE_OVERHEAD_MS: float = amio_constants.DECODE_OVERHEAD_MS_MEASURED  # 18.33
DECODE_KV_MS_PER_CTX_TOKEN: float = amio_constants.DECODE_KV_MS_PER_CTX_TOKEN  # 0.00320
# NOTE: the measured 18.33 ms overhead already includes the per-step weight
# read (~14.6 ms modeled for the 1390 MiB W4 blob) — MODEL_WEIGHT_BYTES is
# retained ONLY for bandwidth-contention accounting, never added to TBT.
MODEL_WEIGHT_BYTES: float = amio_constants.MODEL_WEIGHTS_MB * 1024 ** 2  # 1390 MiB, W4
# Default decode context: the measured 17-crop total input token count.
DEFAULT_DECODE_CTX: int = amio_constants.TOKENS_PER_CONFIG[amio_constants.MAX_CROPS]  # 1560

# Vision encoder weight traffic (for the bandwidth-contention bound).
# SigLIP-SO400M: ≈15.93 M params/layer × 27 layers at 4 bits ≈ 205 MiB.
_SIGLIP_PARAMS_PER_LAYER = 4 * 1152 ** 2 + 2 * 1152 * 4 * 1152
VISION_WEIGHT_BYTES_W4: float = amio_constants.VISION_NUM_LAYERS * _SIGLIP_PARAMS_PER_LAYER * 0.5


# ---------------------------------------------------------------------------
# Shared decode TBT model
# ---------------------------------------------------------------------------

def kv_read_ms_per_seq(
    seq_len: int,
    kv_bytes_per_tok: int = KV_BYTES_PER_TOKEN_FP16,
    bandwidth_gbps: float = M3_BANDWIDTH_GBps,
) -> float:
    """Per-step KV-cache read time for ONE sequence of `seq_len` tokens (ms)."""
    return seq_len * kv_bytes_per_tok / (bandwidth_gbps * 1e9) * 1000.0


def predict_tbt_ms(
    batch_size: int = 1,
    seq_len: int = DEFAULT_DECODE_CTX,
    kv_bytes_per_tok: int = KV_BYTES_PER_TOKEN_FP16,
    overhead_ms: float = DECODE_OVERHEAD_MS,
    bandwidth_gbps: float = M3_BANDWIDTH_GBps,
) -> float:
    """
    Decode step time (ms), anchored to the MEASURED batch-1 fit:

        TBT(b, ctx) = overhead + kv_slope × ctx × (kv_bytes / KV_FP16)
                      + (b − 1) × (ctx × kv_bytes/tok / BW)

      - overhead = 18.33 ms and kv_slope = 0.00320 ms/ctx-token are MEASURED
        (amio_constants DECODE_OVERHEAD_MS_MEASURED / DECODE_KV_MS_PER_CTX_TOKEN).
        The overhead already includes the per-step weight read — no separate
        weight term is added (that would double-count).
      - Batch scaling is UNMEASURED: each extra sequence adds one modeled
        KV-prefix read (≈3.0 ms/seq at 1548 ctx, FP16 KV) — a MODELED
        ASSUMPTION.
      - Quantized KV scales both KV terms by kv_bytes/KV_FP16 (modeled).
    """
    if batch_size <= 0:
        return 0.0
    kv_ratio = kv_bytes_per_tok / KV_BYTES_PER_TOKEN_FP16
    batch1_kv_ms = DECODE_KV_MS_PER_CTX_TOKEN * seq_len * kv_ratio
    extra_seq_ms = (batch_size - 1) * kv_read_ms_per_seq(
        seq_len, kv_bytes_per_tok, bandwidth_gbps
    )
    return overhead_ms + batch1_kv_ms + extra_seq_ms


# ---------------------------------------------------------------------------
# Decode strategies — defined ONCE, used by decode_starvation_analysis()
# AND batch_expansion_summary().  Each strategy is fully characterised by
# its KV bytes/token and its allocator (paged vs contiguous); TBT effects
# follow from kv_bytes_per_tok via predict_tbt_ms() — there are no separate
# ad-hoc speedup factors.
# ---------------------------------------------------------------------------
DECODE_STRATEGIES: dict = {
    "baseline_fp16": {
        "label": "Contiguous FP16",
        "kv_bytes_per_tok": KV_BYTES_PER_TOKEN_FP16,
        "paged": False,
        "note": "contiguous allocator reserves MAX_SEQ_LEN per sequence",
    },
    "paged_fp16": {
        "label": "Paged FP16",
        "kv_bytes_per_tok": KV_BYTES_PER_TOKEN_FP16,
        "paged": True,
        "note": "paged allocator, exact-fit; same KV bytes as baseline",
    },
    "paged_w4a8_gar": {
        "label": "Paged + 8-bit KV",
        "kv_bytes_per_tok": KV_BYTES_PER_TOKEN_W8,
        "paged": True,
        "note": "modeled 8-bit KV cache (unvalidated on MLX/M3)",
    },
    "paged_w4": {
        "label": "Paged + 4-bit KV",
        "kv_bytes_per_tok": KV_BYTES_PER_TOKEN_W4,
        "paged": True,
        "note": "modeled 4-bit KV cache (unvalidated on MLX/M3)",
    },
}


@dataclass
class SMAllocation:
    """Result of one compute-share partitioning decision."""
    sm_vision: int           # shares assigned to vision/prefill worker
    sm_decode: int           # shares assigned to decode worker

    # Predicted execution times under this allocation
    t_vision_ms: float       # Vision encoder wall-clock (ms)
    t_decode_ms: float       # Decode wall-clock (ms)
    t_overlap_ms: float      # Pipelined wall-clock (ms) — kept for compat
    t_sequential_ms: float   # Naive sequential baseline (ms)
    t_pipelined_ms: float    # max(vision, decode, bandwidth floor) (ms)

    overlap_savings_ms: float  # t_sequential - t_pipelined
    bandwidth_floor_ms: float = 0.0  # (bytes_vision + bytes_decode) / BW

    @property
    def overlap_savings_pct(self) -> float:
        if self.t_sequential_ms <= 0:
            return 0.0
        return self.overlap_savings_ms / self.t_sequential_ms * 100.0


class SMOrchestrator:
    """
    Adaptive compute-share partitioner for concurrent vision + decode work.

    ("SM" naming retained for API stability — these are abstract modeled
    compute shares, not hardware units; M3 has 10 GPU cores.)

    Parameters
    ----------
    total_sms : int
        Total modeled compute shares (default 38, see amio_constants).
    sm_min_decode : int
        Minimum shares reserved for the decode worker (floor guarantee).
    sm_min_vision : int
        Minimum shares reserved for the vision worker.
    reclaim_alpha : float
        Rate at which the decode worker's share GROWS as pending decode
        requests increase:  SM_dec = SM_min_dec + α·(N_pending − 1),
        clamped so the vision worker keeps at least `sm_min_vision`.
    vision_ms_per_crop, vision_fixed_ms : float
        MEASURED vision model T_vision(c) = 553.5·c + 27.4 ms at full GPU
        (amio_constants VISION_MS_PER_CROP / VISION_FIXED_MS); compute-share
        scaling multiplies this measured full-GPU time.
    baseline_t_decode_ms : float or None
        Baseline decode step latency.  None (default) uses the shared
        measured-anchored TBT model (predict_tbt_ms).
    """

    def __init__(
        self,
        total_sms: int = M3_TOTAL_SMs,
        sm_min_decode: int = 4,
        sm_min_vision: int = 8,
        reclaim_alpha: float = 2.0,
        vision_ms_per_crop: float = amio_constants.VISION_MS_PER_CROP,
        vision_fixed_ms: float = amio_constants.VISION_FIXED_MS,
        baseline_t_decode_ms: Optional[float] = None,
    ):
        self.total_sms = total_sms
        self.sm_min_decode = sm_min_decode
        self.sm_min_vision = sm_min_vision
        self.reclaim_alpha = reclaim_alpha
        self.vision_ms_per_crop = vision_ms_per_crop
        self.vision_fixed_ms = vision_fixed_ms
        # None → use the shared measured-anchored TBT model
        self.baseline_t_decode_ms = baseline_t_decode_ms

        # Sanity check
        assert sm_min_decode + sm_min_vision <= total_sms, (
            f"sm_min_decode ({sm_min_decode}) + sm_min_vision ({sm_min_vision}) "
            f"exceeds total_sms ({total_sms})"
        )

    # ------------------------------------------------------------------
    # Latency scaling models
    # ------------------------------------------------------------------

    def _scale_vision_latency(self, sm_count: int, n_crops: int = amio_constants.MAX_CROPS) -> float:
        """
        Predict vision encoder latency given share allocation and crop count.

        Base cost is the MEASURED full-GPU model T_vision(c) = 553.5·c + 27.4
        ms (near-perfectly linear in crop count, per-crop ratios 540–566).
        Compute-bound share scaling T ∝ 1/share_count multiplies the measured
        full-GPU time (the share scaling itself is a modeling assumption).
        """
        base = self.vision_ms_per_crop * n_crops + self.vision_fixed_ms
        return base * (self.total_sms / sm_count)

    def _scale_decode_latency(
        self,
        sm_count: int,
        n_pending: int = 1,
        seq_len: int = DEFAULT_DECODE_CTX,
        kv_bytes_per_tok: int = KV_BYTES_PER_TOKEN_FP16,
    ) -> float:
        """
        Predict decode step latency given share allocation.

        Base step time comes from the shared bandwidth model
        predict_tbt_ms(n_pending, ...) — batching the pending sequences
        shares the per-step weight read.  Share count matters only mildly
        for a bandwidth-bound kernel; we keep a square-root saturation
        model for the compute-share sensitivity.
        """
        if self.baseline_t_decode_ms is not None:
            base = self.baseline_t_decode_ms * n_pending
        else:
            base = predict_tbt_ms(
                batch_size=max(n_pending, 1),
                seq_len=seq_len,
                kv_bytes_per_tok=kv_bytes_per_tok,
            )
        sm_ref = max(self.sm_min_decode, 1)
        # Sqrt saturation: halving shares below reference costs ~1.41×
        scale = math.sqrt(sm_ref / max(sm_count, 1))
        return base * max(scale, 0.5)

    # ------------------------------------------------------------------
    # Core allocation logic
    # ------------------------------------------------------------------

    def allocate(
        self,
        n_pending_decode: int = 1,
        n_crops: int = amio_constants.MAX_CROPS,
        seq_len: int = DEFAULT_DECODE_CTX,
        kv_bytes_per_tok: int = KV_BYTES_PER_TOKEN_FP16,
    ) -> SMAllocation:
        """
        Compute a compute-share partition for the current workload.

        Parameters
        ----------
        n_pending_decode : int
            Number of ongoing auto-regressive decode sequences.
        n_crops : int
            Number of vision encoder crops to process in this prefill step.
        seq_len : int
            Context length of decoding sequences (feeds the TBT model).
        kv_bytes_per_tok : int
            KV bytes/token of the active strategy (feeds the TBT model).

        Returns
        -------
        SMAllocation with timing predictions.
        """
        # Drain-first heuristic: decode share GROWS with pending pressure
        sm_decode_raw = self.sm_min_decode + int(
            self.reclaim_alpha * max(n_pending_decode - 1, 0)
        )
        sm_decode = min(sm_decode_raw, self.total_sms - self.sm_min_vision)
        sm_decode = max(sm_decode, self.sm_min_decode)

        sm_vision = self.total_sms - sm_decode
        sm_vision = max(sm_vision, self.sm_min_vision)

        # Re-clamp decode after vision floor is guaranteed
        sm_decode = self.total_sms - sm_vision

        t_vision = self._scale_vision_latency(sm_vision, n_crops)
        t_decode = self._scale_decode_latency(
            sm_decode, n_pending_decode, seq_len, kv_bytes_per_tok
        )

        # --- Bandwidth-contention bound on pipelining ---
        # Even with perfect overlap, the two workers share one memory bus:
        #   pipelined time ≥ (bytes_vision + bytes_decode) / BW.
        # bytes_vision: encoder weights streamed once per crop (activations
        # ignored — a lower bound).  bytes_decode: one decode step's traffic.
        bytes_vision = n_crops * VISION_WEIGHT_BYTES_W4
        bytes_decode = (
            MODEL_WEIGHT_BYTES
            + max(n_pending_decode, 1) * seq_len * kv_bytes_per_tok
        )
        bandwidth_floor_ms = (
            (bytes_vision + bytes_decode) / (M3_BANDWIDTH_GBps * 1e9) * 1000.0
        )

        t_sequential = t_vision + t_decode
        t_pipelined = max(t_vision, t_decode, bandwidth_floor_ms)
        overlap_savings = t_sequential - t_pipelined

        return SMAllocation(
            sm_vision=sm_vision,
            sm_decode=sm_decode,
            t_vision_ms=t_vision,
            t_decode_ms=t_decode,
            t_overlap_ms=t_pipelined,
            t_sequential_ms=t_sequential,
            t_pipelined_ms=t_pipelined,
            overlap_savings_ms=overlap_savings,
            bandwidth_floor_ms=bandwidth_floor_ms,
        )

    def predict_stage_overlap_savings(
        self,
        sm_vision: int,
        sm_decode: int,
        t_vision_ms: float,
        t_concurrent_ms: Optional[float] = None,
        t_lm_ms: Optional[float] = None,
        bytes_vision: Optional[float] = None,
        bytes_concurrent: Optional[float] = None,
    ) -> float:
        """
        Given explicit share counts and stage latencies, return pipeline
        savings (ms) from overlapping the vision stage with a concurrent
        stage (typically decode; `t_lm_ms` is accepted as a legacy alias
        for the second stage's latency).

        Naive savings would be
            savings = T_vision + T_conc − max(T_vision, T_conc)
                    = min(T_vision, T_conc)
        which is guaranteed positive and models zero memory contention.
        We therefore cap savings by the shared-bandwidth bound: the
        pipelined time can never be less than
            (bytes_vision + bytes_concurrent) / BW,
        because both workers share one memory bus.  If byte estimates are
        not supplied, defaults of one full-encoder weight stream and one
        decode step (weights + one 1560-token FP16 KV read) are used.
        An over-subscription penalty is also applied when the two workers'
        share requests exceed the total.
        """
        t_other = t_concurrent_ms if t_concurrent_ms is not None else t_lm_ms
        if t_other is None:
            t_other = 0.0

        overlap_fraction = 1.0
        total_sms_used = sm_vision + sm_decode
        if total_sms_used > self.total_sms:
            # Share over-subscription reduces effective overlap
            overlap_fraction = self.total_sms / total_sms_used

        if bytes_vision is None:
            bytes_vision = VISION_WEIGHT_BYTES_W4
        if bytes_concurrent is None:
            bytes_concurrent = (
                MODEL_WEIGHT_BYTES + DEFAULT_DECODE_CTX * KV_BYTES_PER_TOKEN_FP16
            )

        bandwidth_floor_ms = (
            (bytes_vision + bytes_concurrent) / (M3_BANDWIDTH_GBps * 1e9) * 1000.0
        )

        t_sequential = t_vision_ms + t_other
        t_pipelined = max(t_vision_ms, t_other, bandwidth_floor_ms)
        raw_savings = max(0.0, t_sequential - t_pipelined)
        return raw_savings * overlap_fraction

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_allocation(self, alloc: SMAllocation) -> None:
        """Pretty-print an SMAllocation."""
        print("Compute-share allocation (abstract shares, not hardware units)")
        print("-" * 60)
        print(f"  Vision worker : {alloc.sm_vision:2d} shares")
        print(f"  Decode worker : {alloc.sm_decode:2d} shares")
        print()
        print(f"  T_vision      : {alloc.t_vision_ms:8.1f} ms")
        print(f"  T_decode      : {alloc.t_decode_ms:8.1f} ms")
        print(f"  Sequential    : {alloc.t_sequential_ms:8.1f} ms")
        print(f"  BW floor      : {alloc.bandwidth_floor_ms:8.1f} ms")
        print(f"  Pipelined     : {alloc.t_pipelined_ms:8.1f} ms")
        print(f"  Savings       : {alloc.overlap_savings_ms:8.1f} ms  "
              f"({alloc.overlap_savings_pct:.1f}%)")


# ---------------------------------------------------------------------------
# Phase 4: Decode Starvation Analysis
# ---------------------------------------------------------------------------

@dataclass
class BatchExpansionResult:
    """
    Models how PagedAttention + KV quantisation change the feasible batch.

    Attributes
    ----------
    strategy        : label (see DECODE_STRATEGIES)
    kv_bytes_per_tok: KV cache bytes per token under this strategy
    max_batch_by_mem: maximum batch that fits in KV pool (memory ceiling)
    max_batch_by_tbt: maximum batch where TBT ≤ TBT_HUMAN_THRESHOLD_MS
                      (0 when even batch=1 exceeds the threshold)
    effective_max_batch: min(max_batch_by_mem, max_batch_by_tbt)
    tbt_at_max_batch_ms: predicted TBT at max_batch_by_mem
    memory_efficiency_x: memory-ceiling expansion vs contiguous baseline
    """
    strategy: str
    kv_bytes_per_tok: int
    max_batch_by_mem: int
    max_batch_by_tbt: int
    effective_max_batch: int
    tbt_at_max_batch_ms: float
    memory_efficiency_x: float
    notes: str = ""


@dataclass
class DecodeStarvationResult:
    """
    Identifies at what batch size the decode worker is "starved" (TBT > 80 ms).
    """
    strategy: str
    tbt_curve: List[float]       # ms/token for batch_sizes 1..N
    starvation_at_batch: int     # first batch_size where TBT > threshold
    tbt_at_starvation_ms: float
    safe_batch_range: List[int]  # batch sizes where TBT ≤ threshold
    throughput_tokens_per_s: List[float]  # tokens/s for each batch size


def decode_starvation_analysis(
    seq_len: int = DEFAULT_DECODE_CTX,
    max_batch: int = 32,
    tbt_threshold_ms: float = TBT_HUMAN_THRESHOLD_MS,
    strategies: dict | None = None,
) -> dict[str, DecodeStarvationResult]:
    """
    Determine the batch size ceiling for each KV strategy under the shared
    measured-anchored TBT model (predict_tbt_ms):

        TBT(b) = 18.33 + 0.00320·ctx·(kv/FP16) + (b−1) × (ctx × kv_bytes/tok / BW)

    Weights are streamed once per step (their cost lives inside the measured
    18.33 ms overhead), so throughput b / TBT(b) INCREASES with batch size.
    `seq_len` and each strategy's `kv_bytes_per_tok` set the per-sequence term.

    Under the MEASURED constants TBT(1, 1560) ≈ 23.3 ms, well below the 80 ms
    threshold — the SLA-safe batch ceiling (≈19 for FP16 KV at 1560 ctx) is a
    computed result, not an assumption.  (The extra-sequence term is a
    modeled, unmeasured assumption.)

    Parameters
    ----------
    seq_len       : context tokens per request
    max_batch     : upper bound for sweep
    tbt_threshold_ms : TBT SLA (default 80 ms)
    strategies    : optional {name: {"kv_bytes_per_tok": int, ...}} override;
                    defaults to the module-level DECODE_STRATEGIES table.

    Returns
    -------
    dict of strategy → DecodeStarvationResult
    """
    strats = strategies or DECODE_STRATEGIES
    results: dict[str, DecodeStarvationResult] = {}

    for name, spec in strats.items():
        kv_bpt = spec.get("kv_bytes_per_tok", KV_BYTES_PER_TOKEN_FP16)

        tbt_curve: List[float] = []
        throughput: List[float] = []
        starvation_at = max_batch + 1  # assume no starvation by default
        tbt_at_starv = 0.0
        safe_batches: List[int] = []

        for b in range(1, max_batch + 1):
            tbt_b = predict_tbt_ms(b, seq_len=seq_len, kv_bytes_per_tok=kv_bpt)
            tbt_curve.append(round(tbt_b, 2))
            throughput.append(round(1000.0 / tbt_b * b, 3))  # tokens/s total

            if tbt_b <= tbt_threshold_ms:
                safe_batches.append(b)
            elif starvation_at > max_batch:
                starvation_at = b
                tbt_at_starv = tbt_b

        results[name] = DecodeStarvationResult(
            strategy=name,
            tbt_curve=tbt_curve,
            starvation_at_batch=starvation_at if starvation_at <= max_batch else -1,
            tbt_at_starvation_ms=round(tbt_at_starv, 2),
            safe_batch_range=safe_batches,
            throughput_tokens_per_s=throughput,
        )

    return results


def batch_expansion_summary(
    seq_len: int = DEFAULT_DECODE_CTX,
) -> List[BatchExpansionResult]:
    """
    Return a BatchExpansionResult for each strategy in DECODE_STRATEGIES,
    showing how PagedAttention + KV quantisation change the feasible batch
    ceiling on M3 (pool budget: amio_constants.KV_POOL_BUDGET_MB = 4754 MB).

    Memory ceilings:
      contiguous — reserves MAX_SEQ_LEN per sequence (allocator policy)
      paged      — allocates exact-fit at seq_len

    TBT ceilings come from the shared measured-anchored predict_tbt_ms model:
    TBT(1, 1560) ≈ 23.3 ms, so the 80 ms threshold now admits a real batch
    ceiling per strategy (computed below).  The old "unmeetable at any batch"
    conclusion rested on the superseded 83.75 ms overhead constant.
    """
    pool_bytes = KV_POOL_BUDGET_MB * (1024 ** 2)

    contiguous_max_batch_mem = int(
        pool_bytes / (KV_BYTES_PER_TOKEN_FP16 * MAX_SEQ_LEN)
    )

    results = []
    for name, spec in DECODE_STRATEGIES.items():
        kv_bpt = spec["kv_bytes_per_tok"]
        if spec["paged"]:
            max_by_mem = int(pool_bytes / (kv_bpt * seq_len))
        else:
            max_by_mem = int(pool_bytes / (kv_bpt * MAX_SEQ_LEN))

        # Largest b with TBT(b) ≤ threshold (0 if even b=1 exceeds it):
        # headroom above the measured batch-1 step, spent in modeled
        # per-extra-sequence KV reads.
        headroom_ms = TBT_HUMAN_THRESHOLD_MS - predict_tbt_ms(
            1, seq_len=seq_len, kv_bytes_per_tok=kv_bpt
        )
        per_seq_ms = kv_read_ms_per_seq(seq_len, kv_bpt)
        if headroom_ms < 0:
            max_by_tbt = 0
        elif per_seq_ms > 0:
            max_by_tbt = 1 + int(headroom_ms / per_seq_ms)
        else:
            max_by_tbt = max_by_mem

        eff = min(max_by_mem, max_by_tbt)
        tbt_at_mem_ceiling = predict_tbt_ms(
            max(max_by_mem, 1), seq_len=seq_len, kv_bytes_per_tok=kv_bpt
        )
        mem_eff_x = round(max_by_mem / max(contiguous_max_batch_mem, 1), 2)

        results.append(BatchExpansionResult(
            strategy=spec["label"],
            kv_bytes_per_tok=kv_bpt,
            max_batch_by_mem=max_by_mem,
            max_batch_by_tbt=max_by_tbt,
            effective_max_batch=eff,
            tbt_at_max_batch_ms=round(tbt_at_mem_ceiling, 2),
            memory_efficiency_x=mem_eff_x,
            notes=spec["note"],
        ))

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AMIO Phase 3 + 4 — SM Orchestrator Self-Test")
    print("(38 'SMs' = abstract modeled compute shares; M3 has 10 GPU cores)")
    print("=" * 70)

    orchestrator = SMOrchestrator(
        total_sms=M3_TOTAL_SMs,
        sm_min_decode=4,
        sm_min_vision=8,
        reclaim_alpha=2.0,
    )

    scenarios = [
        ("Prefill only, no decode pressure",  1,  17),
        ("Light decode (2 pending)",           2,  17),
        ("Heavy decode (8 pending)",           8,  17),
        ("Reduced crops (10), heavy decode",   6,  10),
    ]

    for label, n_pending, n_crops in scenarios:
        print(f"\nScenario: {label}")
        alloc = orchestrator.allocate(n_pending_decode=n_pending, n_crops=n_crops)
        orchestrator.print_allocation(alloc)

    # --- Phase 4: Batch expansion ---
    print("\n" + "=" * 70)
    print("Phase 4 — Batch Expansion: PagedAttention + KV quantisation")
    print("=" * 70)
    _tbt1 = predict_tbt_ms(1)
    print(f"\n  TBT model (MEASURED batch-1 fit): TBT(b, ctx) = "
          f"{DECODE_OVERHEAD_MS:.2f} + {DECODE_KV_MS_PER_CTX_TOKEN:.5f}·ctx "
          f"+ (b−1) × KV-read (modeled)")
    _verdict = ("SLA met at batch 1" if _tbt1 <= TBT_HUMAN_THRESHOLD_MS
                else "SLA exceeded even at batch 1")
    print(f"  TBT(1, FP16 KV, {DEFAULT_DECODE_CTX} tok) = "
          f"{_tbt1:.1f} ms  |  threshold: {TBT_HUMAN_THRESHOLD_MS:.0f} ms  "
          f"→ {_verdict} (computed, not asserted)")
    print(f"  KV pool: {KV_POOL_BUDGET_MB:.0f} MB  |  seq_len={DEFAULT_DECODE_CTX}")
    print()
    print(f"  {'Strategy':<22} {'KVbpt':>6} {'MaxBatch(mem)':>13} "
          f"{'MaxBatch(TBT)':>13} {'EffectiveBatch':>14} {'TBT@memCeil':>11} {'MemGain':>7}")
    print("  " + "-" * 92)
    for r in batch_expansion_summary():
        print(
            f"  {r.strategy:<22} "
            f"{r.kv_bytes_per_tok//1024:>4}KB  "
            f"{r.max_batch_by_mem:>13}  "
            f"{r.max_batch_by_tbt:>13}  "
            f"{r.effective_max_batch:>14}  "
            f"{r.tbt_at_max_batch_ms:>10.1f}  "
            f"{r.memory_efficiency_x:>6.1f}×"
        )

    # --- Phase 4: Decode starvation ---
    print()
    print("  Decode Starvation Analysis (batch_size → TBT; throughput rises with batch)")
    print(f"  {'Strategy':<22} {'TBT@1':>7} {'TBT@8':>7} {'TBT@16':>7} "
          f"{'tok/s@1':>8} {'tok/s@16':>8} {'StarveAt':>8}")
    print("  " + "-" * 74)
    starvation = decode_starvation_analysis(max_batch=16)
    for name, r in starvation.items():
        starve = r.starvation_at_batch if r.starvation_at_batch > 0 else ">16"
        print(
            f"  {name:<22} "
            f"{r.tbt_curve[0]:>7.1f} "
            f"{r.tbt_curve[7]:>7.1f} "
            f"{r.tbt_curve[15]:>7.1f} "
            f"{r.throughput_tokens_per_s[0]:>8.2f} "
            f"{r.throughput_tokens_per_s[15]:>8.2f} "
            f"{str(starve):>8}"
        )

    print("\nSM orchestrator (Phase 3 + 4) self-test complete")
