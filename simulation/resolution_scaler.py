"""
Resolution Scaler — Dynamic Crop Scaling for Vision Encoder Cost Reduction

Phase 2 revealed the SigLIP vision encoder dominates TTFT at 5,991 ms (71%),
and that this cost is essentially fixed regardless of pixel resolution because
SmolVLM's processor always produces 1,548 visual tokens via its tiling scheme.

The only way to reduce T_vision is to reduce the NUMBER OF CROPS processed
by the encoder.  This module derives optimal (crop_count, resolution_fraction)
pairs from a FastVLM-inspired Pareto curve model:

    T_vision ∝ N_crops          (linear: each crop is independently encoded)

So halving the crop count halves vision encoder latency, at the cost of reduced
image fidelity.  The scaler finds the smallest N_crops that, combined with LM
token pruning, satisfies the SLA budget.

SmolVLM tiling baseline
-----------------------
At 512 × 512 pixels:   1 global thumb +  8 local crops = 9 crops total
At 756 × 756 pixels:   1 global thumb + 12 local crops = 13 crops total
At 1008 × 1008 pixels: 1 global thumb + 20 local crops = 21 crops total
At 1512 × 1512 pixels: 1 global thumb + 24 local crops = 25 crops total
(The calibration run produced 24 crops; the "+1 global thumb" is already
 included in the 5,991 ms measurement.)

For planning purposes we model N_crops as a continuous variable in [1, 24]
and apply fractional scaling to the baseline latency.
"""

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Allow imports from sibling directories
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Calibrated constants (Phase 2 ground truth)
# ---------------------------------------------------------------------------
BASELINE_T_VISION_MS: float = 5991.0   # 24-crop sequential latency (ms)
BASELINE_N_CROPS: int = 24              # crops at calibration resolution
BASELINE_TOKENS_PER_CROP: int = 64     # SigLIP patch tokens per crop (approx)
BASELINE_TOTAL_TOKENS: int = 1548      # total visual tokens fed to LM

# Quadratic LM cost model coefficients (Phase 2)
GAMMA: float = 2.0957439189669464e-05  # ms / token²
BETA: float = 1.590525232336679        # ms / token
ALPHA: float = -20.081460624741705     # ms (offset)

SLA_BUDGET_MS: float = 500.0


@dataclass
class ScalingPlan:
    """Result of one resolution-scaling decision."""
    n_crops: int
    tokens_per_crop: int
    total_visual_tokens: int
    resolution_fraction: float   # fraction of 1512 px full resolution

    t_vision_ms: float           # predicted vision encoder latency
    t_lm_ms: float               # predicted LM prefill latency (post-pruning)
    t_total_ms: float            # T_vision + T_lm (decode not included)

    lm_pruning_ratio: float      # fraction of tokens KEPT in LM (1.0 = no pruning)
    sla_pass: bool               # True if t_total ≤ SLA_BUDGET_MS
    latency_reduction_pct: float # vs full-resolution baseline (8,483 ms TTFT)

    notes: str = ""


class ResolutionScaler:
    """
    Derive the minimum-crop scaling plan that satisfies an SLA budget.

    Approach
    --------
    1. Enumerate crop counts from 1 to `baseline_n_crops`.
    2. For each crop count, compute T_vision (linear model).
    3. Compute the remaining LM budget: budget_lm = SLA - T_vision.
    4. Use the quadratic cost model to find the maximum token count N_lm
       that fits within budget_lm.
    5. Express N_lm as a pruning ratio relative to total_visual_tokens.
    6. Return the plan with the highest N_crops (best image quality) that
       passes the SLA.

    Parameters
    ----------
    sla_budget_ms : float
        End-to-end TTFT target in milliseconds (default 500 ms).
    baseline_t_vision_ms : float
        Calibrated baseline vision latency at `baseline_n_crops`.
    baseline_n_crops : int
        Crop count at calibration time.
    gamma, beta, alpha : float
        Quadratic LM cost model coefficients from Phase 2.
    """

    def __init__(
        self,
        sla_budget_ms: float = SLA_BUDGET_MS,
        baseline_t_vision_ms: float = BASELINE_T_VISION_MS,
        baseline_n_crops: int = BASELINE_N_CROPS,
        gamma: float = GAMMA,
        beta: float = BETA,
        alpha: float = ALPHA,
    ):
        self.sla_budget_ms = sla_budget_ms
        self.baseline_t_vision_ms = baseline_t_vision_ms
        self.baseline_n_crops = baseline_n_crops
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Latency prediction helpers
    # ------------------------------------------------------------------

    def predict_t_vision(self, n_crops: float) -> float:
        """Vision encoder latency — scales linearly with crop count."""
        return self.baseline_t_vision_ms * (n_crops / self.baseline_n_crops)

    def predict_t_lm(self, n_tokens: float) -> float:
        """LM prefill latency from Phase 2 quadratic model."""
        return self.gamma * n_tokens ** 2 + self.beta * n_tokens + self.alpha

    def _max_lm_tokens_for_budget(self, budget_ms: float) -> float:
        """
        Invert the quadratic: find N such that T_lm(N) = budget_ms.

            gamma * N² + beta * N + (alpha - budget_ms) = 0
        """
        if budget_ms <= 0:
            return 0.0
        a = self.gamma
        b = self.beta
        c = self.alpha - budget_ms
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return 0.0
        n = (-b + math.sqrt(discriminant)) / (2 * a)
        return max(n, 0.0)

    # ------------------------------------------------------------------
    # Core planning method
    # ------------------------------------------------------------------

    def find_optimal_crops(
        self,
        n_pending_requests: int = 1,
        decode_budget_fraction: float = 0.1,
    ) -> ScalingPlan:
        """
        Find the highest-quality (most crops) plan that fits the SLA.

        Parameters
        ----------
        n_pending_requests : int
            Decode pressure — increases decode latency, tightening vision+LM budget.
        decode_budget_fraction : float
            Fraction of SLA_BUDGET reserved for decode (default 10%).

        Returns
        -------
        ScalingPlan — the recommended operating point.
        """
        # Reserve decode budget
        decode_reserve_ms = self.sla_budget_ms * decode_budget_fraction
        effective_budget_ms = self.sla_budget_ms - decode_reserve_ms

        best_plan: Optional[ScalingPlan] = None

        # Enumerate from high quality (many crops) to low quality (few crops)
        for n_crops in range(self.baseline_n_crops, 0, -1):
            t_vision = self.predict_t_vision(n_crops)
            budget_lm = effective_budget_ms - t_vision

            if budget_lm <= 0:
                continue  # vision alone exceeds budget; try fewer crops

            # Max tokens that fit in remaining LM budget
            max_tokens = self._max_lm_tokens_for_budget(budget_lm)

            # Visual tokens scale linearly with crops
            total_visual_tokens = int(
                BASELINE_TOTAL_TOKENS * (n_crops / self.baseline_n_crops)
            )
            total_visual_tokens = max(total_visual_tokens, 1)

            pruning_ratio = min(max_tokens / total_visual_tokens, 1.0)
            n_lm_tokens = min(max_tokens, total_visual_tokens)
            t_lm = self.predict_t_lm(n_lm_tokens)
            t_total = t_vision + t_lm

            sla_pass = t_total <= effective_budget_ms
            baseline_total = self.baseline_t_vision_ms + self.predict_t_lm(
                BASELINE_TOTAL_TOKENS
            )
            reduction_pct = (baseline_total - t_total) / baseline_total * 100.0

            tokens_per_crop = max(
                1, int(BASELINE_TOKENS_PER_CROP * (n_crops / self.baseline_n_crops))
            )
            # Resolution fraction: crops ∝ area ∝ (res_fraction)²
            res_fraction = math.sqrt(n_crops / self.baseline_n_crops)

            plan = ScalingPlan(
                n_crops=n_crops,
                tokens_per_crop=tokens_per_crop,
                total_visual_tokens=total_visual_tokens,
                resolution_fraction=res_fraction,
                t_vision_ms=t_vision,
                t_lm_ms=t_lm,
                t_total_ms=t_total,
                lm_pruning_ratio=pruning_ratio,
                sla_pass=sla_pass,
                latency_reduction_pct=reduction_pct,
                notes=(
                    f"{n_crops} crops @ {res_fraction:.2f}× resolution; "
                    f"pruning ratio {pruning_ratio:.2f}"
                ),
            )

            if sla_pass:
                # First passing plan = highest quality that fits
                return plan

            # Track the closest non-passing plan as a fallback
            if best_plan is None or t_total < best_plan.t_total_ms:
                best_plan = plan

        # No plan passes — return the best non-passing plan
        return best_plan or ScalingPlan(
            n_crops=1,
            tokens_per_crop=BASELINE_TOKENS_PER_CROP,
            total_visual_tokens=64,
            resolution_fraction=math.sqrt(1 / self.baseline_n_crops),
            t_vision_ms=self.predict_t_vision(1),
            t_lm_ms=0.0,
            t_total_ms=self.predict_t_vision(1),
            lm_pruning_ratio=0.0,
            sla_pass=False,
            latency_reduction_pct=0.0,
            notes="No plan satisfies SLA even with 1 crop",
        )

    def sweep(self, n_crops_range: range | None = None) -> list[ScalingPlan]:
        """
        Return a ScalingPlan for every crop count in range (for analysis).
        """
        plans = []
        crop_range = n_crops_range or range(1, self.baseline_n_crops + 1)
        for n in crop_range:
            t_vision = self.predict_t_vision(n)
            total_tokens = max(
                1, int(BASELINE_TOTAL_TOKENS * (n / self.baseline_n_crops))
            )
            t_lm = self.predict_t_lm(total_tokens)
            t_total = t_vision + t_lm
            res_f = math.sqrt(n / self.baseline_n_crops)
            baseline_total = self.baseline_t_vision_ms + self.predict_t_lm(
                BASELINE_TOTAL_TOKENS
            )
            reduction_pct = (baseline_total - t_total) / baseline_total * 100.0
            plans.append(
                ScalingPlan(
                    n_crops=n,
                    tokens_per_crop=max(1, int(BASELINE_TOKENS_PER_CROP * (n / self.baseline_n_crops))),
                    total_visual_tokens=total_tokens,
                    resolution_fraction=res_f,
                    t_vision_ms=t_vision,
                    t_lm_ms=t_lm,
                    t_total_ms=t_total,
                    lm_pruning_ratio=1.0,
                    sla_pass=t_total <= self.sla_budget_ms,
                    latency_reduction_pct=reduction_pct,
                )
            )
        return plans


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AMIO Phase 3 — Resolution Scaler Self-Test")
    print("=" * 70)

    scaler = ResolutionScaler(sla_budget_ms=500.0)

    # Show the Pareto sweep
    print("\nCrop → Latency Pareto curve")
    print(f"  {'N_crops':>7}  {'T_vision':>9}  {'T_lm':>9}  {'T_total':>9}  {'SLA?':>5}")
    print("  " + "-" * 55)
    for plan in scaler.sweep():
        flag = "✅" if plan.sla_pass else "❌"
        print(
            f"  {plan.n_crops:>7}  "
            f"{plan.t_vision_ms:>8.1f}  "
            f"{plan.t_lm_ms:>8.1f}  "
            f"{plan.t_total_ms:>8.1f}  "
            f"  {flag}"
        )

    print()
    print("Optimal plan (SLA=500 ms, 0 pending decode):")
    plan = scaler.find_optimal_crops(n_pending_requests=0)
    print(f"  n_crops         : {plan.n_crops}")
    print(f"  resolution frac : {plan.resolution_fraction:.3f}×")
    print(f"  total tokens    : {plan.total_visual_tokens}")
    print(f"  LM pruning ratio: {plan.lm_pruning_ratio:.3f}")
    print(f"  T_vision        : {plan.t_vision_ms:.1f} ms")
    print(f"  T_lm            : {plan.t_lm_ms:.1f} ms")
    print(f"  T_total         : {plan.t_total_ms:.1f} ms")
    print(f"  SLA pass        : {plan.sla_pass}")
    print(f"  Reduction       : {plan.latency_reduction_pct:.1f}% vs baseline")
    print(f"  Notes           : {plan.notes}")

    print("\n✅ Resolution scaler self-test complete")
