"""
Parallelism Engine — Phase 3 Orchestrator

Integrates the three Phase 3 sub-systems into a single planning API:

    from simulation.parallelism_engine import ParallelismEngine

    engine = ParallelismEngine()
    plan   = engine.plan(resolution=512, n_pending_requests=2, sla_budget_ms=500)
    print(plan)

The engine consults (in order):
  1. ResolutionScaler      — find minimum crop count that fits the SLA
  2. TPSimulator           — compare TP / DP / HYBRID communication cost
  3. SMOrchestrator        — partition M3 SMs between vision and decode workers

The output `InferenceExecutionPlan` contains the recommended settings and
predicted end-to-end latency for a single prefill + decode step.

Design note: all numbers are *analytical predictions*, not wall-clock
measurements.  The engine is a planning oracle; the actual inference system
would feed these recommendations to the model runtime.
"""

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Allow sibling / parent imports when run as a script
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulation.tp_simulator import (
    ParallelismMode,
    ParallelismCostResult,
    compare_parallelism_modes,
    BASELINE_T_VISION_MS as TP_BASELINE_VISION,
    BASELINE_T_LM_MS as TP_BASELINE_LM,
    BASELINE_N_CROPS as TP_BASELINE_CROPS,
)
from simulation.sm_orchestrator import SMOrchestrator, SMAllocation, M3_TOTAL_SMs
from simulation.resolution_scaler import (
    ResolutionScaler,
    ScalingPlan,
    BASELINE_T_VISION_MS as RS_BASELINE_VISION,
    BASELINE_N_CROPS as RS_BASELINE_CROPS,
)

# ---------------------------------------------------------------------------
# Resolution → crop-count table (SmolVLM tiling)
# ---------------------------------------------------------------------------
RESOLUTION_TO_CROPS: dict[int, int] = {
    224:  1,
    336:  4,
    448:  6,
    512:  9,
    756: 13,
    1008: 21,
    1512: 24,   # = calibration baseline
}


def _nearest_crop_count(resolution_px: int) -> int:
    """Map an arbitrary pixel resolution to the nearest crop count entry."""
    best_res = min(RESOLUTION_TO_CROPS.keys(), key=lambda r: abs(r - resolution_px))
    return RESOLUTION_TO_CROPS[best_res]


# ---------------------------------------------------------------------------
# Execution plan dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceExecutionPlan:
    """
    Complete execution plan for one prefill + decode request.

    Fields
    ------
    parallelism_mode    : recommended ParallelismMode (TP / DP / HYBRID)
    sm_vision           : SMs allocated to vision/prefill worker
    sm_decode           : SMs allocated to decode worker
    n_crops             : number of image crops to encode
    resolution_fraction : fraction of max resolution (1512 px = 1.0)
    lm_pruning_ratio    : fraction of visual tokens retained for LM (1.0 = none)
    total_visual_tokens : visual token count fed into LM
    predicted_t_vision_ms : vision encoder latency under plan
    predicted_t_lm_ms     : LM prefill latency under plan
    predicted_t_decode_ms : decode latency per token step
    predicted_ttft_ms     : total time-to-first-token = T_vision + T_lm
    sla_budget_ms         : SLA target this plan was optimised for
    sla_pass              : True if predicted_ttft_ms ≤ sla_budget_ms
    throughput_gain_pct   : latency reduction vs naive sequential baseline
    overlap_savings_ms    : pipeline overlap savings from SM partitioning
    parallelism_detail    : ParallelismCostResult for the chosen mode
    sm_allocation         : SMAllocation object
    scaling_plan          : ResolutionScaler ScalingPlan
    notes                 : human-readable explanation
    """

    # Parallelism recommendation
    parallelism_mode: ParallelismMode
    sm_vision: int
    sm_decode: int

    # Resolution / token budget
    n_crops: int
    resolution_fraction: float
    lm_pruning_ratio: float
    total_visual_tokens: int

    # Timing predictions
    predicted_t_vision_ms: float
    predicted_t_lm_ms: float
    predicted_t_decode_ms: float
    predicted_ttft_ms: float

    # SLA
    sla_budget_ms: float
    sla_pass: bool

    # Gains
    throughput_gain_pct: float
    overlap_savings_ms: float

    # Provenance
    parallelism_detail: ParallelismCostResult
    sm_allocation: SMAllocation
    scaling_plan: ScalingPlan
    notes: str = ""

    def summary(self) -> str:
        flag = "PASS" if self.sla_pass else "FAIL"
        lines = [
            "=" * 70,
            "AMIO Inference Execution Plan",
            "=" * 70,
            f"  SLA target       : {self.sla_budget_ms:.0f} ms   [{flag}]",
            f"  Predicted TTFT   : {self.predicted_ttft_ms:.1f} ms",
            "",
            "  Parallelism",
            f"    mode           : {self.parallelism_mode.value}",
            f"    SM vision      : {self.sm_vision}",
            f"    SM decode      : {self.sm_decode}",
            f"    overlap savings: {self.overlap_savings_ms:.1f} ms",
            "",
            "  Resolution / Crops",
            f"    n_crops        : {self.n_crops}  (baseline 24)",
            f"    res fraction   : {self.resolution_fraction:.3f}×",
            f"    visual tokens  : {self.total_visual_tokens}  (baseline 1548)",
            f"    LM pruning     : {self.lm_pruning_ratio:.3f}  (1.0 = no pruning)",
            "",
            "  Stage Latencies",
            f"    T_vision       : {self.predicted_t_vision_ms:.1f} ms",
            f"    T_lm_prefill   : {self.predicted_t_lm_ms:.1f} ms",
            f"    T_decode/tok   : {self.predicted_t_decode_ms:.1f} ms",
            "",
            f"  Throughput gain  : {self.throughput_gain_pct:+.1f}% vs sequential",
            f"  Notes            : {self.notes}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ParallelismEngine:
    """
    Phase 3 planning oracle.

    Integrates ResolutionScaler + TPSimulator + SMOrchestrator to produce
    an InferenceExecutionPlan for any (resolution, n_pending, sla) triple.

    Parameters
    ----------
    total_sms : int
        GPU SM count (default: M3 = 38).
    decode_budget_fraction : float
        Fraction of SLA reserved for decode (default 10%).
    """

    def __init__(
        self,
        total_sms: int = M3_TOTAL_SMs,
        decode_budget_fraction: float = 0.10,
    ):
        self.total_sms = total_sms
        self.decode_budget_fraction = decode_budget_fraction
        self._sm_orch = SMOrchestrator(total_sms=total_sms)
        self._res_scaler = ResolutionScaler()

    # ------------------------------------------------------------------
    # Main planning entry point
    # ------------------------------------------------------------------

    def plan(
        self,
        resolution: int = 512,
        n_pending_requests: int = 1,
        sla_budget_ms: float = 500.0,
    ) -> InferenceExecutionPlan:
        """
        Produce a complete InferenceExecutionPlan.

        Parameters
        ----------
        resolution : int
            Input image resolution in pixels (longest side).
        n_pending_requests : int
            Number of concurrent decode sequences already in flight.
        sla_budget_ms : float
            TTFT SLA target in milliseconds.

        Returns
        -------
        InferenceExecutionPlan
        """
        # --- 0. Initialise scaler with correct SLA ---
        scaler = ResolutionScaler(
            sla_budget_ms=sla_budget_ms,
            baseline_t_vision_ms=RS_BASELINE_VISION,
            baseline_n_crops=RS_BASELINE_CROPS,
        )

        # --- 1. Resolution scaler: find optimal crop count ---
        scaling_plan = scaler.find_optimal_crops(
            n_pending_requests=n_pending_requests,
            decode_budget_fraction=self.decode_budget_fraction,
        )

        # Cap crops at what the requested resolution naturally produces
        max_crops_for_res = _nearest_crop_count(resolution)
        n_crops = min(scaling_plan.n_crops, max_crops_for_res)

        # Recompute vision latency for the selected crop count
        t_vision = scaler.predict_t_vision(n_crops)
        total_tokens = max(
            1, int(1548 * (n_crops / RS_BASELINE_CROPS))
        )
        pruned_tokens = int(total_tokens * scaling_plan.lm_pruning_ratio)
        pruned_tokens = max(pruned_tokens, 1)
        t_lm = scaler.predict_t_lm(pruned_tokens)

        # --- 2. Parallelism mode comparison ---
        para_results = compare_parallelism_modes(
            t_vision_ms=t_vision,
            t_lm_ms=t_lm,
            n_crops=n_crops,
            n_workers=2,
            tp_size=2,
            seq_len=pruned_tokens,
        )
        chosen_mode_key = para_results["recommended"].mode.name  # "TP"/"DP"/"HYBRID"
        chosen_result: ParallelismCostResult = para_results[chosen_mode_key]

        # Use the parallelism-adjusted total latency for vision+LM
        t_vision_adjusted = (
            chosen_result.t_total_ms
            * (t_vision / (t_vision + t_lm))
            if (t_vision + t_lm) > 0 else t_vision
        )
        t_lm_adjusted = chosen_result.t_total_ms - t_vision_adjusted

        # --- 3. SM Orchestrator: partition SMs ---
        sm_alloc = SMOrchestrator(total_sms=self.total_sms).allocate(
            n_pending_decode=n_pending_requests,
            n_crops=n_crops,
        )

        # Rescale vision latency for actual SM allotment
        # (Already included in SMAllocation.t_vision_ms)
        t_vision_final = sm_alloc.t_vision_ms
        t_lm_final = t_lm_adjusted  # LM runs after vision in current design

        # Decode latency estimate (memory-bandwidth bound; ~0.9 tok/s on M3)
        t_decode_ms = 1111.0  # ms per generated token at baseline

        t_ttft = t_vision_final + t_lm_final

        # Overlap savings from pipelined SM execution
        overlap_savings = self._sm_orch.predict_stage_overlap_savings(
            sm_vision=sm_alloc.sm_vision,
            sm_decode=sm_alloc.sm_decode,
            t_vision_ms=t_vision_final,
            t_lm_ms=t_decode_ms * n_pending_requests,
        )

        sla_pass = t_ttft <= sla_budget_ms

        # Throughput gain vs naive baseline (8,483 ms TTFT)
        baseline_ttft = RS_BASELINE_VISION + scaler.predict_t_lm(1548)
        throughput_gain = (baseline_ttft - t_ttft) / baseline_ttft * 100.0

        res_fraction = math.sqrt(n_crops / RS_BASELINE_CROPS)
        pruning_ratio = min(scaling_plan.lm_pruning_ratio, 1.0)

        notes = (
            f"{chosen_mode_key} parallelism; "
            f"{n_crops}/{RS_BASELINE_CROPS} crops; "
            f"{pruned_tokens}/{total_tokens} tokens retained; "
            f"SMs {sm_alloc.sm_vision}V/{sm_alloc.sm_decode}D"
        )

        return InferenceExecutionPlan(
            parallelism_mode=chosen_result.mode,
            sm_vision=sm_alloc.sm_vision,
            sm_decode=sm_alloc.sm_decode,
            n_crops=n_crops,
            resolution_fraction=res_fraction,
            lm_pruning_ratio=pruning_ratio,
            total_visual_tokens=pruned_tokens,
            predicted_t_vision_ms=t_vision_final,
            predicted_t_lm_ms=t_lm_final,
            predicted_t_decode_ms=t_decode_ms,
            predicted_ttft_ms=t_ttft,
            sla_budget_ms=sla_budget_ms,
            sla_pass=sla_pass,
            throughput_gain_pct=throughput_gain,
            overlap_savings_ms=overlap_savings,
            parallelism_detail=chosen_result,
            sm_allocation=sm_alloc,
            scaling_plan=scaling_plan,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Batch sweep for analysis
    # ------------------------------------------------------------------

    def sweep_scenarios(
        self,
        resolutions: list[int] | None = None,
        pending_counts: list[int] | None = None,
        sla_budget_ms: float = 500.0,
    ) -> list[InferenceExecutionPlan]:
        """
        Return plans for all (resolution, n_pending) combinations.
        """
        resolutions = resolutions or [224, 512, 1008, 1512]
        pending_counts = pending_counts or [1, 2, 4, 8]
        plans = []
        for res in resolutions:
            for n in pending_counts:
                plans.append(self.plan(
                    resolution=res,
                    n_pending_requests=n,
                    sla_budget_ms=sla_budget_ms,
                ))
        return plans


# ---------------------------------------------------------------------------
# Self-test / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AMIO Phase 3 — Parallelism Engine Demonstration")
    print("=" * 70)

    engine = ParallelismEngine()

    scenarios = [
        ("224 px, 1 pending",  224,  1),
        ("512 px, 1 pending",  512,  1),
        ("512 px, 2 pending",  512,  2),
        ("512 px, 4 pending",  512,  4),
        ("1008 px, 1 pending", 1008, 1),
        ("1512 px, 1 pending", 1512, 1),
    ]

    results_table = []
    for label, res, n_pend in scenarios:
        p = engine.plan(resolution=res, n_pending_requests=n_pend, sla_budget_ms=500.0)
        flag = "PASS" if p.sla_pass else "FAIL"
        results_table.append((label, p, flag))

    # Compact table
    print(f"\n{'Scenario':<26} {'Mode':<8} {'Crops':>5} {'Tokens':>6} "
          f"{'T_vis':>7} {'T_lm':>7} {'TTFT':>7} {'Gain':>6} {'SLA':>4}")
    print("-" * 80)
    for label, p, flag in results_table:
        print(
            f"{label:<26} "
            f"{p.parallelism_mode.name:<8} "
            f"{p.n_crops:>5} "
            f"{p.total_visual_tokens:>6} "
            f"{p.predicted_t_vision_ms:>7.1f} "
            f"{p.predicted_t_lm_ms:>7.1f} "
            f"{p.predicted_ttft_ms:>7.1f} "
            f"{p.throughput_gain_pct:>+6.1f}% "
            f"{flag}"
        )

    # Verbose plan for the 512px, 1-pending case
    print()
    plan_512 = engine.plan(resolution=512, n_pending_requests=1, sla_budget_ms=500.0)
    print(plan_512.summary())

    print("\nParallelism engine demonstration complete")
