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
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# M3 hardware constants
# ---------------------------------------------------------------------------
M3_TOTAL_SMs: int = 38           # Apple M3 GPU execution units
M3_BANDWIDTH_GBps: float = 100.0 # Memory bandwidth
M3_CLOCK_GHz: float = 1.398      # GPU clock (M3 base)


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
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AMIO Phase 3 — SM Orchestrator Self-Test")
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

    print("\n✅ SM orchestrator self-test complete")
