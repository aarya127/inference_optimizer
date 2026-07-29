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
                             (HYPOTHETICAL multi-device projections only)
  3. SMOrchestrator        — partition modeled compute shares between the
                             vision and decode workers

The output `InferenceExecutionPlan` contains the recommended settings and
predicted end-to-end latency for a single prefill + decode step.

COHERENCE NOTE (fixes the earlier TTFT splice): `predicted_ttft_ms` is
computed from ONE coherent single-M3 model — share-scaled vision plus the
UNADJUSTED LM prefill cost model.  The TP/DP comparison is exposed only as
a separate, clearly-labeled multi-device projection
(`parallelism_detail` / `multi_device_projection_ttft_ms`) and is NEVER
folded into the M3 TTFT: single-chip "TP gains" are not realizable (M3 is
a single-GPU part; see simulation/tp_simulator.py).

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

import amio_constants as C

from simulation.tp_simulator import (
    ParallelismMode,
    ParallelismCostResult,
    compare_parallelism_modes,
)
from simulation.sm_orchestrator import (
    SMOrchestrator,
    SMAllocation,
    M3_TOTAL_SMs,
    predict_tbt_ms,
)
from simulation.resolution_scaler import (
    ResolutionScaler,
    ScalingPlan,
    max_crops_for_resolution,
)

# ---------------------------------------------------------------------------
# Processor-size policy map (MEASURED; amio_constants.CROP_SETTINGS).
# Keys are the processor's size.longest_edge settings; these are the ONLY
# crop counts that exist ({1, 5, 10, 17}; the old 24-crop entry never did).
# ---------------------------------------------------------------------------
RESOLUTION_TO_CROPS: dict[int, int] = dict(C.CROP_SETTINGS)   # {384:1, 768:5, 1152:10, 1536:17}

BASELINE_N_CROPS: int = C.MAX_CROPS                           # 17
BASELINE_TOTAL_TOKENS: int = C.TOKENS_PER_CONFIG[C.MAX_CROPS] # 1560


def _nearest_crop_count(resolution_px: int) -> int:
    """
    Serving policy (documented): cap crops at the processor setting whose
    size.longest_edge ≥ min(image_resolution, 1536) — splitting a 384 px
    image into 17 crops would model compute the encoder never performs.
    E.g. 600 px → 768 setting → 5 crops; 1200 px → 1536 → 17 crops.
    Delegates to resolution_scaler.max_crops_for_resolution.
    """
    return max_crops_for_resolution(resolution_px)


# ---------------------------------------------------------------------------
# Execution plan dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceExecutionPlan:
    """
    Complete execution plan for one prefill + decode request.

    Fields
    ------
    parallelism_mode    : recommended ParallelismMode (TP / DP / HYBRID) —
                          a MULTI-DEVICE PROJECTION recommendation only;
                          it does not affect predicted_ttft_ms
    sm_vision           : compute shares allocated to vision/prefill worker
    sm_decode           : compute shares allocated to decode worker
    n_crops             : number of image crops to encode
    resolution_fraction : fraction of max resolution (1512 px = 1.0)
    lm_pruning_ratio    : fraction of visual tokens retained for LM (1.0 = none)
    total_visual_tokens : visual token count fed into LM
    predicted_t_vision_ms : vision encoder latency under plan (share-scaled)
    predicted_t_lm_ms     : LM prefill latency under plan (UNADJUSTED cost
                            model — no TP discount)
    predicted_t_decode_ms : decode latency per token step (shared TBT model)
    predicted_ttft_ms     : single-M3 TTFT = share-scaled T_vision + T_lm,
                            from ONE coherent model; contains NO multi-device
                            TP/DP adjustment
    sla_budget_ms         : SLA target this plan was optimised for
    sla_pass              : True if predicted_ttft_ms ≤ sla_budget_ms
    throughput_gain_pct   : TOTAL latency reduction vs naive baseline
                            (= iso-quality + quality-tradeoff components)
    iso_quality_gain_pct  : gain vs baseline at UNCHANGED quality (same
                            24 crops / full tokens) — pure systems effect
    quality_tradeoff_gain_pct : remainder of the total gain, bought by crop
                            reduction / token pruning (quality tradeoff)
    overlap_savings_ms    : pipeline overlap savings from share partitioning
    multi_device_projection_ttft_ms : hypothetical vision+LM total under the
                            recommended TP/DP mode — projection ONLY, never
                            part of predicted_ttft_ms
    parallelism_detail    : ParallelismCostResult (multi-device projection)
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
    iso_quality_gain_pct: float = 0.0
    quality_tradeoff_gain_pct: float = 0.0
    multi_device_projection_ttft_ms: float = 0.0

    def summary(self) -> str:
        flag = "PASS" if self.sla_pass else "FAIL"
        lines = [
            "=" * 70,
            "AMIO Inference Execution Plan",
            "=" * 70,
            f"  SLA target       : {self.sla_budget_ms:.0f} ms   [{flag}]",
            f"  Predicted TTFT   : {self.predicted_ttft_ms:.1f} ms "
            f"(single-M3 model: share-scaled vision + unadjusted LM)",
            "",
            "  Compute-share partition (abstract shares, not hardware units)",
            f"    shares vision  : {self.sm_vision}",
            f"    shares decode  : {self.sm_decode}",
            f"    overlap savings: {self.overlap_savings_ms:.1f} ms",
            "",
            "  Multi-device projection (HYPOTHETICAL — M3 is single-GPU;",
            "  never folded into the TTFT above)",
            f"    mode           : {self.parallelism_mode.value}",
            f"    projected total: {self.multi_device_projection_ttft_ms:.1f} ms",
            "",
            "  Resolution / Crops",
            f"    n_crops        : {self.n_crops}  (baseline {BASELINE_N_CROPS})",
            f"    res fraction   : {self.resolution_fraction:.3f}×",
            f"    visual tokens  : {self.total_visual_tokens}  "
            f"(baseline {BASELINE_TOTAL_TOKENS})",
            f"    LM pruning     : {self.lm_pruning_ratio:.3f}  (1.0 = no pruning)",
            "",
            "  Stage Latencies",
            f"    T_vision       : {self.predicted_t_vision_ms:.1f} ms",
            f"    T_lm_prefill   : {self.predicted_t_lm_ms:.1f} ms",
            f"    T_decode/tok   : {self.predicted_t_decode_ms:.1f} ms",
            "",
            f"  Total gain       : {self.throughput_gain_pct:+.1f}% vs sequential baseline",
            f"    iso-quality    : {self.iso_quality_gain_pct:+.1f}% "
            f"(systems only, same {BASELINE_N_CROPS} crops)",
            f"    quality-cost   : {self.quality_tradeoff_gain_pct:+.1f}% (bought by crop/token reduction)",
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
        # (a per-call ResolutionScaler is built in plan() with the caller's
        # SLA; no engine-level scaler instance is kept)

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
        # --- 0. Initialise scaler with correct SLA (measured cost models) ---
        scaler = ResolutionScaler(sla_budget_ms=sla_budget_ms)

        # --- 1. Resolution scaler: find optimal crop count ---
        # The request's image resolution bounds the useful crop settings
        # (processor-size policy: smallest longest_edge ≥ min(res, 1536)).
        max_crops_for_res = _nearest_crop_count(resolution)
        scaling_plan = scaler.find_optimal_crops(
            n_pending_requests=n_pending_requests,
            decode_budget_fraction=self.decode_budget_fraction,
            max_crops=max_crops_for_res,
        )
        n_crops = min(scaling_plan.n_crops, max_crops_for_res)

        # Recompute vision latency for the selected crop count (measured
        # linear model) and LM tokens from the measured per-config totals.
        t_vision = scaler.predict_t_vision(n_crops)
        total_tokens = C.TOKENS_PER_CONFIG.get(
            n_crops, C.TOKENS_PER_CROP * n_crops + 19
        )
        pruned_tokens = int(total_tokens * scaling_plan.lm_pruning_ratio)
        pruned_tokens = max(pruned_tokens, 1)
        t_lm = scaler.predict_t_lm(pruned_tokens)

        # --- 2. Parallelism mode comparison ---
        # HYPOTHETICAL MULTI-DEVICE PROJECTION ONLY: compared and reported,
        # but NEVER folded into the single-M3 TTFT below (single-chip TP
        # gains are not realizable — see simulation/tp_simulator.py).
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

        # --- 3. SM Orchestrator: partition compute shares ---
        sm_alloc = SMOrchestrator(total_sms=self.total_sms).allocate(
            n_pending_decode=n_pending_requests,
            n_crops=n_crops,
        )

        # --- 4. ONE coherent single-M3 TTFT model ---
        # share-scaled vision (from the SM model) + UNADJUSTED LM prefill
        # (Phase-2 cost model).  No TP/DP adjustment enters here.
        t_vision_final = sm_alloc.t_vision_ms
        t_lm_final = t_lm

        # Decode latency per step from the shared bandwidth-based TBT model
        t_decode_ms = predict_tbt_ms(batch_size=max(n_pending_requests, 1))

        t_ttft = t_vision_final + t_lm_final

        # Overlap savings from pipelined execution: vision overlaps with the
        # CONCURRENT DECODE step (passed via t_concurrent_ms — the earlier
        # code shipped decode time in a parameter named t_lm_ms).
        overlap_savings = self._sm_orch.predict_stage_overlap_savings(
            sm_vision=sm_alloc.sm_vision,
            sm_decode=sm_alloc.sm_decode,
            t_vision_ms=t_vision_final,
            t_concurrent_ms=t_decode_ms,
        )

        sla_pass = t_ttft <= sla_budget_ms

        # --- Gains vs naive baseline (17 crops / 1560 tokens, full shares), ---
        # --- decomposed honestly ---
        # iso-quality: same max crops / full tokens, systems effects only
        # (here: the share-scaled vision worker — usually a PENALTY when
        # decode steals shares).  quality-tradeoff: the remainder, bought by
        # crop reduction / token pruning.
        baseline_vision_ms = (
            C.VISION_MS_PER_CROP * BASELINE_N_CROPS + C.VISION_FIXED_MS
        )
        baseline_ttft = baseline_vision_ms + scaler.predict_t_lm(BASELINE_TOTAL_TOKENS)
        t_iso_quality = (
            self._sm_orch._scale_vision_latency(sm_alloc.sm_vision, BASELINE_N_CROPS)
            + scaler.predict_t_lm(BASELINE_TOTAL_TOKENS)
        )
        throughput_gain = (baseline_ttft - t_ttft) / baseline_ttft * 100.0
        iso_quality_gain = (baseline_ttft - t_iso_quality) / baseline_ttft * 100.0
        quality_tradeoff_gain = throughput_gain - iso_quality_gain

        res_fraction = math.sqrt(n_crops / BASELINE_N_CROPS)
        pruning_ratio = min(scaling_plan.lm_pruning_ratio, 1.0)

        notes = (
            f"{chosen_mode_key} recommended (multi-device projection only); "
            f"{n_crops}/{BASELINE_N_CROPS} crops; "
            f"{pruned_tokens}/{total_tokens} tokens retained; "
            f"shares {sm_alloc.sm_vision}V/{sm_alloc.sm_decode}D"
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
            iso_quality_gain_pct=iso_quality_gain,
            quality_tradeoff_gain_pct=quality_tradeoff_gain,
            multi_device_projection_ttft_ms=chosen_result.t_total_ms,
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
