"""
Resolution Scaler — Dynamic Crop Scaling for Vision Encoder Cost Reduction

MEASURED PREMISE (baseline/results_v2.json, direct stage isolation on the
M3-8GB target):

    T_vision(c) = 553.5·c + 27.4 ms     (near-perfectly linear in crop count;
                                         per-crop ratios 540–566 across 1–17)

The old caveats about an unvalidated 5,991 ms residual and a never-exercised
resolution→crop control are RESOLVED: crop count was driven directly via the
Idefics3 processor (do_image_splitting / size.longest_edge) and the linearity
is now a measurement, not an assumption.

Processor crop settings (the ONLY ones that exist — "24 crops" never did):

    size.longest_edge   384   768   1152   1536
    crop count            1     5     10     17     (MAX_CROPS = 17)
    total input tokens  100   466    922   1560     (81 image tokens/crop)

Given the measured models, the scaler finds the largest crop setting that,
combined with LM token pruning, satisfies the SLA budget.  Under the
measured numbers even the 1-crop vision stage (≈581 ms) exceeds the 500 ms
TTFT SLA, so the honest default output is "no plan passes" — reported as
such, not tuned away.
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

import amio_constants

# ---------------------------------------------------------------------------
# Constants (single source of truth: amio_constants — MEASURED section)
# ---------------------------------------------------------------------------
VISION_MS_PER_CROP: float = amio_constants.VISION_MS_PER_CROP   # 553.5 (MEASURED)
VISION_FIXED_MS: float = amio_constants.VISION_FIXED_MS         # 27.4
MAX_CROPS: int = amio_constants.MAX_CROPS                       # 17
CROP_SETTINGS: dict = amio_constants.CROP_SETTINGS      # longest_edge → crops
TOKENS_PER_CONFIG: dict = amio_constants.TOKENS_PER_CONFIG  # crops → total tokens
CROP_OPTIONS: list[int] = sorted(TOKENS_PER_CONFIG)             # [1, 5, 10, 17]

# Tokens per crop is a FIXED encoder property (81, from the checkpoint's
# image_seq_len) — CONFIRMED by the v2 campaign: image tokens = 81 × crops
# exactly (17 crops → 1377 image tokens, 1560 total input tokens).
BASELINE_TOKENS_PER_CROP: int = amio_constants.TOKENS_PER_CROP  # 81
BASELINE_N_CROPS: int = MAX_CROPS
BASELINE_TOTAL_TOKENS: int = TOKENS_PER_CONFIG[MAX_CROPS]       # 1560

# Quadratic LM cost model coefficients — MEASURED fit
# (valid on PREFILL_DOMAIN = (100, 1560); see predict_t_lm)
GAMMA: float = amio_constants.PREFILL_GAMMA
BETA: float = amio_constants.PREFILL_BETA
ALPHA: float = amio_constants.PREFILL_ALPHA

# Decode step cost used to model pending-request pressure — MEASURED batch-1
# model evaluated at the maximum-context config:
#   TBT(ctx) = 18.33 + 0.00320·ctx  →  ≈23.3 ms at 1560 ctx
DECODE_TBT_MS: float = (
    amio_constants.DECODE_OVERHEAD_MS_MEASURED
    + amio_constants.DECODE_KV_MS_PER_CTX_TOKEN * BASELINE_TOTAL_TOKENS
)

SLA_BUDGET_MS: float = amio_constants.TTFT_SLA_MS  # 500.0


@dataclass
class ScalingPlan:
    """Result of one resolution-scaling decision."""
    n_crops: int
    tokens_per_crop: int
    total_visual_tokens: int
    resolution_fraction: float   # fraction of the 1536 px maximum longest_edge

    t_vision_ms: float           # predicted vision encoder latency (measured model)
    t_lm_ms: float               # predicted LM prefill latency (post-pruning)
    t_total_ms: float            # T_vision + T_lm (decode not included)

    lm_pruning_ratio: float      # fraction of tokens KEPT in LM (1.0 = no pruning)
    sla_pass: bool               # True if t_total ≤ the EFFECTIVE budget
                                 # (SLA minus the decode reserve) in
                                 # find_optimal_crops(); sweep() checks the
                                 # raw SLA budget instead
    latency_reduction_pct: float # vs the 17-crop full-quality baseline

    notes: str = ""


def _longest_edge_for_crops(n_crops: int) -> int:
    """Processor size.longest_edge that produces `n_crops` crops."""
    for edge, crops in sorted(CROP_SETTINGS.items()):
        if crops == n_crops:
            return edge
    return max(CROP_SETTINGS)


def max_crops_for_resolution(image_resolution_px: int) -> int:
    """
    Serving policy: cap crops at the setting whose longest_edge covers the
    image — there is no point splitting a 384 px image into 17 crops.

    Rule: choose the SMALLEST size.longest_edge ≥ min(image_resolution, 1536)
    and return its crop count (a 512 px image → 768 setting → 5 crops;
    anything ≥ 1536 px → 17 crops).
    """
    target = min(max(1, image_resolution_px), max(CROP_SETTINGS))
    eligible = [edge for edge in CROP_SETTINGS if edge >= target]
    edge = min(eligible) if eligible else max(CROP_SETTINGS)
    return CROP_SETTINGS[edge]


class ResolutionScaler:
    """
    Derive the minimum-crop scaling plan that satisfies an SLA budget.

    Approach
    --------
    1. Enumerate the processor's crop settings {1, 5, 10, 17} (descending).
    2. For each crop count, compute T_vision with the MEASURED linear model.
    3. Compute the remaining LM budget: budget_lm = SLA - T_vision.
    4. Use the measured quadratic cost model to find the maximum token count
       N_lm that fits within budget_lm.
    5. Express N_lm as a pruning ratio relative to that config's total tokens.
    6. Return the plan with the highest N_crops (best image quality) that
       passes the SLA — or, honestly, the best NON-passing plan when none
       does (the measured 1-crop floor is ≈581 ms vision alone).

    Parameters
    ----------
    sla_budget_ms : float
        End-to-end TTFT target in milliseconds (default 500 ms).
    vision_ms_per_crop, vision_fixed_ms : float
        Measured vision model coefficients (T = per_crop·c + fixed).
    gamma, beta, alpha : float
        Measured quadratic LM cost model coefficients.
    """

    def __init__(
        self,
        sla_budget_ms: float = SLA_BUDGET_MS,
        vision_ms_per_crop: float = VISION_MS_PER_CROP,
        vision_fixed_ms: float = VISION_FIXED_MS,
        gamma: float = GAMMA,
        beta: float = BETA,
        alpha: float = ALPHA,
    ):
        self.sla_budget_ms = sla_budget_ms
        self.vision_ms_per_crop = vision_ms_per_crop
        self.vision_fixed_ms = vision_fixed_ms
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Latency prediction helpers
    # ------------------------------------------------------------------

    def predict_t_vision(self, n_crops: float) -> float:
        """
        Vision encoder latency — MEASURED linear model:
        T_vision(c) = 553.5·c + 27.4 ms (full GPU, no contention).
        """
        return self.vision_ms_per_crop * n_crops + self.vision_fixed_ms

    def predict_t_lm(self, n_tokens: float) -> float:
        """
        LM prefill latency from the MEASURED quadratic fit, clamped ≥ 0.

        Valid on amio_constants.PREFILL_DOMAIN = (100, 1560); predictions
        outside that range are extrapolations.  The measured α = +244.6 ms
        is positive, so the clamp is a safety net only.
        """
        raw = self.gamma * n_tokens ** 2 + self.beta * n_tokens + self.alpha
        return max(raw, 0.0)

    def _max_lm_tokens_for_budget(self, budget_ms: float) -> float:
        """
        Invert the quadratic: find N such that T_lm(N) = budget_ms.

            gamma * N² + beta * N + (alpha - budget_ms) = 0
        """
        if budget_ms <= self.alpha:
            return 0.0
        a = self.gamma
        b = self.beta
        c = self.alpha - budget_ms
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return 0.0
        n = (-b + math.sqrt(discriminant)) / (2 * a)
        return max(n, 0.0)

    def _total_tokens(self, n_crops: int) -> int:
        """Total input tokens for a crop setting (measured table; falls back
        to 81·c + 19 structural tokens off-table)."""
        return TOKENS_PER_CONFIG.get(
            n_crops, BASELINE_TOKENS_PER_CROP * n_crops + 19
        )

    # ------------------------------------------------------------------
    # Core planning method
    # ------------------------------------------------------------------

    def find_optimal_crops(
        self,
        n_pending_requests: int = 1,
        decode_budget_fraction: float = 0.1,
        max_crops: int = MAX_CROPS,
    ) -> ScalingPlan:
        """
        Find the highest-quality (most crops) plan that fits the SLA.

        Parameters
        ----------
        n_pending_requests : int
            Decode pressure — each pending decode sequence is modeled as
            costing one decode step (≈23.3 ms measured at 1560 ctx) of the
            latency budget, tightening the vision+LM budget accordingly.
        decode_budget_fraction : float
            Minimum fraction of SLA_BUDGET reserved for decode (default 10%).
            The actual reserve is max(fraction × SLA, n_pending × TBT).
        max_crops : int
            Upper crop bound (e.g. from max_crops_for_resolution()).

        Returns
        -------
        ScalingPlan — the recommended operating point.  Under the measured
        numbers no plan meets a 500 ms budget (1-crop vision alone is
        ≈581 ms); the best non-passing plan is returned with sla_pass=False.
        """
        # Reserve decode budget: at least the static fraction, grown by
        # modeled pending decode cost (n_pending × TBT per docstring).
        decode_reserve_ms = max(
            self.sla_budget_ms * decode_budget_fraction,
            max(n_pending_requests, 0) * DECODE_TBT_MS,
        )
        effective_budget_ms = self.sla_budget_ms - decode_reserve_ms

        best_plan: Optional[ScalingPlan] = None
        candidates = [c for c in CROP_OPTIONS if c <= max_crops] or [min(CROP_OPTIONS)]

        # Enumerate from high quality (many crops) to low quality (few crops)
        for n_crops in sorted(candidates, reverse=True):
            t_vision = self.predict_t_vision(n_crops)
            budget_lm = effective_budget_ms - t_vision

            total_visual_tokens = self._total_tokens(n_crops)

            if budget_lm <= 0:
                # Vision alone exceeds budget — record the honest non-passing
                # plan (zero LM budget → full pruning) and try fewer crops.
                plan = self._build_plan(
                    n_crops, total_visual_tokens,
                    t_vision=t_vision, t_lm=0.0, pruning_ratio=0.0,
                    sla_pass=False,
                    note=f"vision alone ({t_vision:.0f}ms) exceeds effective "
                         f"budget ({effective_budget_ms:.0f}ms)",
                )
                if best_plan is None or plan.t_total_ms < best_plan.t_total_ms:
                    best_plan = plan
                continue

            # Max tokens that fit in remaining LM budget
            max_tokens = self._max_lm_tokens_for_budget(budget_lm)
            pruning_ratio = min(max_tokens / total_visual_tokens, 1.0)
            n_lm_tokens = min(max_tokens, total_visual_tokens)
            t_lm = self.predict_t_lm(n_lm_tokens)
            t_total = t_vision + t_lm
            sla_pass = t_total <= effective_budget_ms and max_tokens >= 1

            plan = self._build_plan(
                n_crops, total_visual_tokens,
                t_vision=t_vision, t_lm=t_lm, pruning_ratio=pruning_ratio,
                sla_pass=sla_pass,
                note=f"{n_crops} crops "
                     f"(longest_edge {_longest_edge_for_crops(n_crops)}); "
                     f"pruning ratio {pruning_ratio:.2f}",
            )

            if sla_pass:
                # First passing plan = highest quality that fits
                return plan

            # Track the closest non-passing plan as a fallback
            if best_plan is None or plan.t_total_ms < best_plan.t_total_ms:
                best_plan = plan

        assert best_plan is not None
        best_plan.notes = f"No plan satisfies SLA — best effort: {best_plan.notes}"
        return best_plan

    def _build_plan(
        self,
        n_crops: int,
        total_visual_tokens: int,
        t_vision: float,
        t_lm: float,
        pruning_ratio: float,
        sla_pass: bool,
        note: str,
    ) -> ScalingPlan:
        t_total = t_vision + t_lm
        baseline_total = self.predict_t_vision(BASELINE_N_CROPS) + self.predict_t_lm(
            BASELINE_TOTAL_TOKENS
        )
        reduction_pct = (baseline_total - t_total) / baseline_total * 100.0
        res_fraction = _longest_edge_for_crops(n_crops) / max(CROP_SETTINGS)
        return ScalingPlan(
            n_crops=n_crops,
            tokens_per_crop=BASELINE_TOKENS_PER_CROP,
            total_visual_tokens=total_visual_tokens,
            resolution_fraction=res_fraction,
            t_vision_ms=t_vision,
            t_lm_ms=t_lm,
            t_total_ms=t_total,
            lm_pruning_ratio=pruning_ratio,
            sla_pass=sla_pass,
            latency_reduction_pct=reduction_pct,
            notes=note,
        )

    def sweep(self, crop_options: list[int] | None = None) -> list[ScalingPlan]:
        """
        Return a ScalingPlan for every processor crop setting (for analysis).
        """
        plans = []
        for n in sorted(crop_options or CROP_OPTIONS):
            t_vision = self.predict_t_vision(n)
            total_tokens = self._total_tokens(n)
            t_lm = self.predict_t_lm(total_tokens)
            plan = self._build_plan(
                n, total_tokens,
                t_vision=t_vision, t_lm=t_lm, pruning_ratio=1.0,
                sla_pass=(t_vision + t_lm) <= self.sla_budget_ms,
                note="",
            )
            plans.append(plan)
        return plans


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AMIO Phase 3 — Resolution Scaler Self-Test (MEASURED models)")
    print("=" * 70)
    print(f"T_vision(c) = {VISION_MS_PER_CROP}·c + {VISION_FIXED_MS} ms  "
          f"(measured); crop settings: {dict(sorted(CROP_SETTINGS.items()))}")

    scaler = ResolutionScaler(sla_budget_ms=500.0)

    # Show the crop-setting sweep
    print("\nCrop-setting → Latency curve (full tokens, no pruning)")
    print(f"  {'N_crops':>7}  {'Tokens':>6}  {'T_vision':>9}  {'T_lm':>9}  "
          f"{'T_total':>9}  {'SLA?':>5}")
    print("  " + "-" * 60)
    for plan in scaler.sweep():
        flag = "PASS" if plan.sla_pass else "FAIL"
        print(
            f"  {plan.n_crops:>7}  "
            f"{plan.total_visual_tokens:>6}  "
            f"{plan.t_vision_ms:>8.1f}  "
            f"{plan.t_lm_ms:>8.1f}  "
            f"{plan.t_total_ms:>8.1f}  "
            f"  {flag}"
        )

    # Consistency check vs measured TTFT anchors (stage sums)
    t17 = scaler.predict_t_vision(17) + scaler.predict_t_lm(1560)
    t1 = scaler.predict_t_vision(1) + scaler.predict_t_lm(100)
    print(f"\n  Model vs measured stage sums: "
          f"1 crop {t1:.0f}ms (measured {amio_constants.TTFT_MS_MEASURED_1_CROP:.0f}ms), "
          f"17 crops {t17:.0f}ms (measured {amio_constants.TTFT_MS_MEASURED_17_CROP:.0f}ms)")
    assert abs(t1 - amio_constants.TTFT_MS_MEASURED_1_CROP) < 100, "1-crop anchor drifted"
    assert abs(t17 - amio_constants.TTFT_MS_MEASURED_17_CROP) < 700, "17-crop anchor drifted"

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
    print(f"  Reduction       : {plan.latency_reduction_pct:.1f}% vs 17-crop baseline")
    print(f"  Notes           : {plan.notes}")

    # Honest expectation under measured numbers: even 1-crop vision (~581 ms)
    # exceeds the 500 ms budget, so NO plan can pass — assert that honesty.
    assert not plan.sla_pass, (
        "With measured vision costs (1 crop ≈ 581 ms) no plan can meet "
        "500 ms — a passing plan indicates broken constants"
    )
    assert plan.n_crops == 1, "Best-effort plan should be the 1-crop minimum"
    print("  [OK] Honest infeasibility: no crop setting meets 500 ms "
          "(1-crop vision alone ≈ 581 ms)")

    # A relaxed budget must produce a passing plan (sanity of the search)
    relaxed = ResolutionScaler(sla_budget_ms=2000.0).find_optimal_crops(0)
    assert relaxed.sla_pass and relaxed.n_crops >= 1
    print(f"  [OK] Relaxed 2000 ms budget passes with {relaxed.n_crops} crop(s), "
          f"T_total={relaxed.t_total_ms:.0f} ms, keep={relaxed.lm_pruning_ratio:.2f}")

    # Resolution policy: crops capped by the image's useful longest_edge
    for res, expect in ((300, 1), (384, 1), (512, 5), (900, 10), (1536, 17), (4000, 17)):
        got = max_crops_for_resolution(res)
        assert got == expect, f"policy({res}px) = {got}, expected {expect}"
    print("  [OK] Resolution→crop policy: cap at smallest longest_edge ≥ "
          "min(resolution, 1536)")

    # Decode pressure genuinely tightens the budget
    print("\nDecode-pressure sweep (n_pending × TBT reserved from budget):")
    print(f"  {'n_pending':>9}  {'reserve ms':>10}  {'n_crops':>7}  "
          f"{'T_total':>8}  {'SLA?':>5}")
    for n_pend in (0, 1, 2, 4):
        reserve = max(500.0 * 0.10, n_pend * DECODE_TBT_MS)
        p = scaler.find_optimal_crops(n_pending_requests=n_pend)
        flag = "PASS" if p.sla_pass else "FAIL"
        print(f"  {n_pend:>9}  {reserve:>10.1f}  {p.n_crops:>7}  "
              f"{p.t_total_ms:>8.1f}  {flag:>5}")

    print("\nNOTE: T_vision(c) = 553.5·c + 27.4 ms is MEASURED "
          "(baseline/results_v2.json); the 500 ms TTFT SLA is infeasible at "
          "every crop setting — reported honestly.")
    print("\nResolution scaler self-test complete")
