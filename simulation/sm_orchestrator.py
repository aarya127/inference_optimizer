"""
SM Orchestrator — Adaptive Shader-Multiprocessor Partitioning

Apple M3 exposes 38 GPU shader-multiprocessors (SMs, or "execution units"
in Metal terminology).  On M3 the unified memory allows two concurrent
Metal command queues to execute genuinely in parallel, sharing SMs.

This module models how to split the 38 SMs between two concurrent workers:
  • Vision worker  — prefill / vision encoder (latency-critical)
  • Decode worker  — auto-regressive token generation (bandwidth-bound)

Strategy
--------
The decode worker is memory-bandwidth bound and requires very few SMs to keep
the memory bus saturated.  We therefore use a "drain-first" heuristic:

    SM_dec = clamp(SM_min, SM_op - α·(N_pending - 1), SM_op)

where N_pending is the number of decode requests waiting.  As concurrent
decode pressure grows we reclaim SMs from the vision worker only up to a
minimum floor (`SM_min`), preserving enough compute for the vision pipeline.

Additionally, we model pipeline overlap — if the decode worker and vision
worker can run truly concurrently on distinct SM partitions, the end-to-end
latency is max(T_vision, T_decode) rather than T_vision + T_decode, saving
significant wall-clock time.

Phase 4 additions
-----------------
  BatchExpansionResult   — models how PagedAttention + W4A8 increase
                           the maximum feasible batch size.
  decode_starvation_analysis()  — finds the batch size ceiling where TBT
                                  exceeds the 80 ms human-perceivable threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# M3 hardware constants
# ---------------------------------------------------------------------------
M3_TOTAL_SMs: int = 38           # Apple M3 GPU execution units
M3_BANDWIDTH_GBps: float = 100.0 # Memory bandwidth
M3_CLOCK_GHz: float = 1.398      # GPU clock (M3 base)

# Phase 1 baseline facts (results.json)
BASELINE_TBT_MS: float = 217.7           # ms/token at batch=1
BASELINE_SPIKE_TBT_MS: float = 1926.5   # TBT spike at heavy load
TBT_HUMAN_THRESHOLD_MS: float = 80.0    # human-perceivable lag boundary

# KV cache constants (from results.json + Phase 4 analysis)
KV_BYTES_PER_TOKEN_FP16: int = 110_592  # 2 × 24 layers × 1152 hidden × 2 B
KV_POOL_BUDGET_MB: float = 5742.0       # M3 budget after weights + OS
MAX_SEQ_LEN: int = 2048


@dataclass
class SMAllocation:
    """Result of one SM partitioning decision."""
    sm_vision: int           # SMs assigned to vision/prefill worker
    sm_decode: int           # SMs assigned to decode worker
    sm_idle: int             # Unused (fragmentation / rounding)

    # Predicted execution times under this allocation
    t_vision_ms: float       # Vision encoder wall-clock (ms)
    t_decode_ms: float       # Decode wall-clock (ms)
    t_overlap_ms: float      # Overlap savings if pipelined (ms)
    t_sequential_ms: float   # Naive sequential baseline (ms)
    t_pipelined_ms: float    # Pipelined wall-clock = max(vision, decode) (ms)

    utilization_pct: float   # (sm_vision + sm_decode) / total * 100
    overlap_savings_ms: float  # t_sequential - t_pipelined

    @property
    def overlap_savings_pct(self) -> float:
        if self.t_sequential_ms <= 0:
            return 0.0
        return self.overlap_savings_ms / self.t_sequential_ms * 100.0


class SMOrchestrator:
    """
    Adaptive SM partitioner for concurrent vision + decode execution.

    Parameters
    ----------
    total_sms : int
        Total GPU SM / execution-unit count (default: M3 = 38).
    sm_min_decode : int
        Minimum SMs reserved for the decode worker (floor guarantee).
    sm_min_vision : int
        Minimum SMs reserved for the vision worker.
    reclaim_alpha : float
        Rate at which decode SMs grow as pending requests increase.
        SM_dec = SM_min + alpha * N_pending  (clamped to max budget)
    baseline_t_vision_ms : float
        Calibrated baseline vision encoder latency at 24 crops with all SMs.
    baseline_t_decode_ms : float
        Baseline decode latency per step (memory-bandwidth bound).
    """

    def __init__(
        self,
        total_sms: int = M3_TOTAL_SMs,
        sm_min_decode: int = 4,
        sm_min_vision: int = 8,
        reclaim_alpha: float = 2.0,
        baseline_t_vision_ms: float = 5991.0,
        baseline_t_decode_ms: float = 1111.0,  # ~0.9 tok/s → ~1111 ms/tok
    ):
        self.total_sms = total_sms
        self.sm_min_decode = sm_min_decode
        self.sm_min_vision = sm_min_vision
        self.reclaim_alpha = reclaim_alpha
        self.baseline_t_vision_ms = baseline_t_vision_ms
        self.baseline_t_decode_ms = baseline_t_decode_ms

        # Sanity check
        assert sm_min_decode + sm_min_vision <= total_sms, (
            f"sm_min_decode ({sm_min_decode}) + sm_min_vision ({sm_min_vision}) "
            f"exceeds total_sms ({total_sms})"
        )

    # ------------------------------------------------------------------
    # Latency scaling models
    # ------------------------------------------------------------------

    def _scale_vision_latency(self, sm_count: int, n_crops: int = 24) -> float:
        """
        Predict vision encoder latency given SM allocation and crop count.

        Compute-bound scaling: T ∝ 1 / SM_count (to first order).
        We also scale linearly in crop count vs the 24-crop baseline.
        """
        baseline_sms = self.total_sms   # baseline measured at full SM count
        scaled = self.baseline_t_vision_ms * (n_crops / 24) * (baseline_sms / sm_count)
        return scaled

    def _scale_decode_latency(self, sm_count: int, n_pending: int = 1) -> float:
        """
        Predict decode latency given SM allocation.

        Decode is memory-bandwidth bound, not compute bound.  SM count
        matters only insofar as it affects memory-controller utilisation.
        We model a mild square-root benefit beyond SM_min to capture
        diminishing returns on a bandwidth-bound kernel.
        """
        sm_ref = max(self.sm_min_decode, 1)
        # Sqrt saturation model: doubling SMs from minimum gives ~1.41× speedup
        scale = math.sqrt(sm_ref / max(sm_count, 1))
        return self.baseline_t_decode_ms * n_pending * max(scale, 0.5)

    # ------------------------------------------------------------------
    # Core allocation logic
    # ------------------------------------------------------------------

    def allocate(
        self,
        n_pending_decode: int = 1,
        n_crops: int = 24,
    ) -> SMAllocation:
        """
        Compute an SM partition for the current workload.

        Parameters
        ----------
        n_pending_decode : int
            Number of ongoing auto-regressive decode sequences.
        n_crops : int
            Number of vision encoder crops to process in this prefill step.

        Returns
        -------
        SMAllocation with timing predictions.
        """
        # Drain-first heuristic: allocate decode SMs first
        sm_decode_raw = self.sm_min_decode + int(
            self.reclaim_alpha * max(n_pending_decode - 1, 0)
        )
        sm_decode = min(sm_decode_raw, self.total_sms - self.sm_min_vision)
        sm_decode = max(sm_decode, self.sm_min_decode)

        sm_vision = self.total_sms - sm_decode
        sm_vision = max(sm_vision, self.sm_min_vision)

        # Re-clamp decode after vision floor is guaranteed
        sm_decode = self.total_sms - sm_vision
        sm_idle = max(0, self.total_sms - sm_vision - sm_decode)

        t_vision = self._scale_vision_latency(sm_vision, n_crops)
        t_decode = self._scale_decode_latency(sm_decode, n_pending_decode)

        t_sequential = t_vision + t_decode
        t_pipelined = max(t_vision, t_decode)   # true pipeline overlap
        overlap_savings = t_sequential - t_pipelined

        utilization_pct = (sm_vision + sm_decode) / self.total_sms * 100.0

        return SMAllocation(
            sm_vision=sm_vision,
            sm_decode=sm_decode,
            sm_idle=sm_idle,
            t_vision_ms=t_vision,
            t_decode_ms=t_decode,
            t_overlap_ms=t_pipelined,
            t_sequential_ms=t_sequential,
            t_pipelined_ms=t_pipelined,
            utilization_pct=utilization_pct,
            overlap_savings_ms=overlap_savings,
        )

    def predict_stage_overlap_savings(
        self,
        sm_vision: int,
        sm_decode: int,
        t_vision_ms: float,
        t_lm_ms: float,
    ) -> float:
        """
        Given explicit SM counts and stage latencies, return pipeline savings (ms).

        The formula is straightforward:
            savings = T_vision + T_lm - max(T_vision, T_lm)
                    = min(T_vision, T_lm)
        but we also apply a partial-overlap penalty when both stages share SMs
        (because the memory bus becomes a bottleneck for simultaneous access).
        """
        overlap_fraction = 1.0
        total_sms_used = sm_vision + sm_decode
        if total_sms_used > self.total_sms:
            # SM over-subscription reduces effective overlap
            overlap_fraction = self.total_sms / total_sms_used

        raw_savings = min(t_vision_ms, t_lm_ms)
        return raw_savings * overlap_fraction

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_allocation(self, alloc: SMAllocation) -> None:
        """Pretty-print an SMAllocation."""
        print("SM Allocation")
        print("-" * 60)
        print(f"  Vision worker : {alloc.sm_vision:2d} SMs")
        print(f"  Decode worker : {alloc.sm_decode:2d} SMs")
        print(f"  Idle          : {alloc.sm_idle:2d} SMs")
        print(f"  Utilisation   : {alloc.utilization_pct:.1f}%")
        print()
        print(f"  T_vision      : {alloc.t_vision_ms:8.1f} ms")
        print(f"  T_decode      : {alloc.t_decode_ms:8.1f} ms")
        print(f"  Sequential    : {alloc.t_sequential_ms:8.1f} ms")
        print(f"  Pipelined     : {alloc.t_pipelined_ms:8.1f} ms")
        print(f"  Savings       : {alloc.overlap_savings_ms:8.1f} ms  "
              f"({alloc.overlap_savings_pct:.1f}%)")


# ---------------------------------------------------------------------------
# Phase 4: Decode Starvation Analysis
# ---------------------------------------------------------------------------

@dataclass
class BatchExpansionResult:
    """
    Models how PagedAttention + W4A8 expand the maximum feasible batch size.

    Attributes
    ----------
    strategy        : label ("baseline_fp16", "paged_fp16", "paged_w4a8", etc.)
    kv_bytes_per_tok: KV cache bytes per token under this strategy
    max_batch_by_mem: maximum batch that fits in KV pool (memory ceiling)
    max_batch_by_tbt: maximum batch where TBT ≤ TBT_HUMAN_THRESHOLD_MS
    effective_max_batch: min(max_batch_by_mem, max_batch_by_tbt)
    tbt_at_max_batch_ms: predicted TBT at effective_max_batch
    memory_efficiency_x: batch expansion vs baseline
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
    seq_len: int = 1548,
    max_batch: int = 32,
    tbt_threshold_ms: float = TBT_HUMAN_THRESHOLD_MS,
    strategies: dict | None = None,
) -> dict[str, DecodeStarvationResult]:
    """
    Determine the batch size ceiling for each quantisation + KV strategy.

    TBT model:
        T_decode ≈ base_tbt_ms × batch_size
        (memory-bandwidth bound: each request adds one weight read pass)

    Batch expansion from PagedAttention: no memory effect on TBT (KV reads
    scale with batch anyway), but PagedAttention allows MORE requests to fit
    in memory, so the effective concurrency is higher.

    W4A8 reduces base_tbt_ms by the quantisation speedup factor.

    Parameters
    ----------
    seq_len       : tokens per request
    max_batch     : upper bound for sweep
    tbt_threshold_ms : TBT SLA (default 80 ms)
    strategies    : dict of {name: base_tbt_ms} overrides

    Returns
    -------
    dict of strategy → DecodeStarvationResult
    """
    default_strategies = {
        "baseline_fp16":  BASELINE_TBT_MS,         # 217.7 ms / tok
        "paged_fp16":     BASELINE_TBT_MS,          # PagedAttention: same TBT, more capacity
        "paged_w4a8_gar": BASELINE_TBT_MS / 3.2,   # ~3.2× speedup from W4A8+GAR
        "paged_w4":       BASELINE_TBT_MS / 2.5,   # W4A16 only: ~2.5× speedup
    }
    strats = strategies or default_strategies
    results: dict[str, DecodeStarvationResult] = {}

    # KV bytes per token for each strategy
    kv_bytes_map = {
        "baseline_fp16":  KV_BYTES_PER_TOKEN_FP16,
        "paged_fp16":     KV_BYTES_PER_TOKEN_FP16,
        "paged_w4a8_gar": KV_BYTES_PER_TOKEN_FP16 // 2,  # A8 KV cache (FP8)
        "paged_w4":       KV_BYTES_PER_TOKEN_FP16,
    }

    for name, base_tbt in strats.items():
        kv_bpt = kv_bytes_map.get(name, KV_BYTES_PER_TOKEN_FP16)

        tbt_curve: List[float] = []
        throughput: List[float] = []
        starvation_at = max_batch + 1  # assume no starvation by default
        tbt_at_starv = 0.0
        safe_batches: List[int] = []

        for b in range(1, max_batch + 1):
            # TBT scales linearly with batch in memory-bandwidth bound regime
            tbt_b = base_tbt * b
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
    seq_len: int = 1548,
) -> List[BatchExpansionResult]:
    """
    Return a BatchExpansionResult for each strategy showing how PagedAttention
    + W4A8 expand the feasible batch size ceiling on M3.
    """
    pool_bytes = KV_POOL_BUDGET_MB * (1024 ** 2)
    baseline_bpt = KV_BYTES_PER_TOKEN_FP16

    strategies = [
        # (label, kv_bpt, base_tbt_ms)
        ("Contiguous FP16", baseline_bpt,         BASELINE_TBT_MS),
        ("Paged FP16",      baseline_bpt,         BASELINE_TBT_MS),
        ("Paged W4 KV",     baseline_bpt // 4,    BASELINE_TBT_MS),
        ("Paged W4A8+GAR",  baseline_bpt // 2,    BASELINE_TBT_MS / 3.2),
    ]

    baseline_max_batch_mem = int(
        pool_bytes / (baseline_bpt * seq_len + baseline_bpt * MAX_SEQ_LEN)
    )
    # Simple contiguous uses max_seq_len allocation always
    contiguous_max_batch_mem = int(
        pool_bytes / (baseline_bpt * MAX_SEQ_LEN)
    )

    results = []
    for label, kv_bpt, base_tbt in strategies:
        if "Contiguous" in label:
            max_by_mem = contiguous_max_batch_mem
        else:
            # Paged: only allocate for actual seq_len, not max
            max_by_mem = int(pool_bytes / (kv_bpt * seq_len))

        max_by_tbt = max(1, int(TBT_HUMAN_THRESHOLD_MS / base_tbt))
        eff = min(max_by_mem, max_by_tbt)
        tbt_at_eff = base_tbt * eff
        # Memory efficiency = batch capacity gain vs contiguous baseline (memory only)
        mem_eff_x = round(max_by_mem / max(contiguous_max_batch_mem, 1), 2)

        results.append(BatchExpansionResult(
            strategy=label,
            kv_bytes_per_tok=kv_bpt,
            max_batch_by_mem=max_by_mem,
            max_batch_by_tbt=max_by_tbt,
            effective_max_batch=eff,
            tbt_at_max_batch_ms=round(tbt_at_eff, 2),
            memory_efficiency_x=mem_eff_x,
            notes=(
                f"{kv_bpt//1024}KB/tok KV, "
                f"TBT={base_tbt:.1f}ms/tok"
            ),
        ))

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AMIO Phase 3 + 4 — SM Orchestrator Self-Test")
    print("=" * 70)

    orchestrator = SMOrchestrator(
        total_sms=M3_TOTAL_SMs,
        sm_min_decode=4,
        sm_min_vision=8,
        reclaim_alpha=2.0,
        baseline_t_vision_ms=5991.0,
        baseline_t_decode_ms=1111.0,
    )

    scenarios = [
        ("Prefill only, no decode pressure",  1,  24),
        ("Light decode (2 pending)",           2,  24),
        ("Heavy decode (8 pending)",           8,  24),
        ("Reduced crops (12), heavy decode",   6,  12),
    ]

    for label, n_pending, n_crops in scenarios:
        print(f"\nScenario: {label}")
        alloc = orchestrator.allocate(n_pending_decode=n_pending, n_crops=n_crops)
        orchestrator.print_allocation(alloc)

    # --- Phase 4: Batch expansion ---
    print("\n" + "=" * 70)
    print("Phase 4 — Batch Expansion: PagedAttention + W4A8")
    print("=" * 70)
    print(f"\n  Baseline TBT: {BASELINE_TBT_MS:.1f} ms/tok  |  "
          f"TBT threshold: {TBT_HUMAN_THRESHOLD_MS:.0f} ms  |  "
          f"KV pool: {KV_POOL_BUDGET_MB:.0f} MB  |  seq_len=1548")
    print()
    print(f"  {'Strategy':<22} {'KVbpt':>6} {'MaxBatch(mem)':>13} "
          f"{'MaxBatch(TBT)':>13} {'EffectiveBatch':>14} {'TBT@eff':>7} {'Gain':>5}")
    print("  " + "-" * 80)
    for r in batch_expansion_summary():
        print(
            f"  {r.strategy:<22} "
            f"{r.kv_bytes_per_tok//1024:>4}KB  "
            f"{r.max_batch_by_mem:>13}  "
            f"{r.max_batch_by_tbt:>13}  "
            f"{r.effective_max_batch:>14}  "
            f"{r.tbt_at_max_batch_ms:>6.1f}  "
            f"{r.memory_efficiency_x:>4.1f}×"
        )

    # --- Phase 4: Decode starvation ---
    print()
    print("  Decode Starvation Analysis (batch_size → TBT)")
    print(f"  {'Strategy':<22} {'StarveAt':>8} {'TBT@starve':>10} {'SafeBatches':>12}")
    print("  " + "-" * 56)
    starvation = decode_starvation_analysis(max_batch=16)
    for name, r in starvation.items():
        starve = r.starvation_at_batch if r.starvation_at_batch > 0 else ">16"
        safe = str(r.safe_batch_range) if r.safe_batch_range else "none"
        print(
            f"  {name:<22} "
            f"{str(starve):>8}  "
            f"{r.tbt_at_starvation_ms:>9.1f}  "
            f"{safe}"
        )

    print("\n✅ SM orchestrator (Phase 3 + 4) self-test complete")
