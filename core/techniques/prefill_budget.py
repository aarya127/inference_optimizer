"""Prefill token budget: how many input tokens fit under a TTFT SLA.

Generalizes the SLA-safe-token-count logic that previously existed only as
a one-off calculation inside model_calibration/calibrate_cost_model.py
(SmolVLM-specific). Here it is written once against the ModelHardwareProfile
interface, so it produces correct, different answers for any profile with a
calibrated PrefillCostModel -- demonstrated against both SmolVLM/MLX and
Qwen2.5/llama.cpp in core/demo.py.

Safety margin (added after core/TIER4_CLOSED_LOOP_FINDINGS.md): a
recommendation that targets the SLA exactly is fragile. Qwen/llama.cpp's
closed-loop test predicted 500.1ms against a 500ms SLA (0.5% error, well
within its 5.7% mean LOOCV) and still measured 502.8ms on a fresh real
run -- missing the SLA despite an excellent cost model, purely because the
recommendation left zero headroom.

`recommend()` defaults `safety_margin_pct` to the profile's own
`loocv_max_ape_pct` (the worst single held-out point from leave-one-out
cross-validation) -- not the mean. The mean was tried first and found
insufficient: a margin sized to it gets breached about as often as real
error exceeds that mean, BY CONSTRUCTION, and this was confirmed directly
-- qwen25_05b_mlx_profile's mean-based margin (2.0%, from its 2.05% mean
LOOCV) missed its SLA on two separate real runs with 2.4-2.6% real error
(TIER4_CLOSED_LOOP_FINDINGS.md's second follow-up). The max is itself
still just a resampling estimate from a handful of calibration points, not
a hard worst case -- it helped close that specific gap (Qwen/llama.cpp's
max LOOCV AE was 31.7%, vs a 5.7% mean; after the sync-audit recalibration
it is 3.4%, which is NOT enough -- LOOCV can't see the n_batch chunk
steps the calibration grid never samples, see
core/SYNC_AUDIT_FINDINGS.md section 6) and brought SmolVLM/MLX's margin
far closer to its real 41-108% closed-loop error (max LOOCV AE 59.9%, vs a
20.7% mean) -- but it is still not a guarantee, particularly for SmolVLM,
whose real-world unreliability has repeatedly exceeded even generous
historical estimates. Falls back to `loocv_mape_pct` (mean) only for
profiles calibrated before per-point LOOCV data was retained, and to 0.0
when neither exists.
"""

from __future__ import annotations

from typing import Optional

from core.profile import ModelHardwareProfile
from core.technique import Recommendation, Technique


class PrefillBudgetTechnique(Technique):
    name = "prefill_token_budget"

    def applies_to(self, profile: ModelHardwareProfile) -> bool:
        return profile.prefill is not None

    def _default_margin(self, profile: ModelHardwareProfile) -> tuple[float, str]:
        if profile.prefill.loocv_max_ape_pct is not None:
            return profile.prefill.loocv_max_ape_pct, "profile max LOOCV AE"
        if profile.prefill.loocv_mape_pct is not None:
            return profile.prefill.loocv_mape_pct, "profile mean LOOCV MAPE (max LOOCV AE not available)"
        return 0.0, "no LOOCV data available"

    def recommend(
        self,
        profile: ModelHardwareProfile,
        sla_ms: float,
        safety_margin_pct: Optional[float] = None,
    ) -> Recommendation:
        if not self.applies_to(profile):
            return Recommendation(False, None, None,
                                   "profile has no calibrated prefill cost model")

        margin_source = "caller-specified"
        if safety_margin_pct is None:
            safety_margin_pct, margin_source = self._default_margin(profile)

        if not (0.0 <= safety_margin_pct < 100.0):
            return Recommendation(True, None, None,
                                   f"safety_margin_pct must be in [0, 100); got {safety_margin_pct}")

        effective_budget_ms = sla_ms * (1.0 - safety_margin_pct / 100.0)

        n = profile.prefill.max_tokens_for_budget(effective_budget_ms)
        margin_note = (
            f"{safety_margin_pct:.1f}% safety margin applied ({margin_source}) "
            f"-> effective budget {effective_budget_ms:.1f}ms"
        )
        if n is None:
            return Recommendation(
                True, None, None,
                f"no token count meets a {sla_ms:.0f}ms budget ({margin_note}); "
                f"(alpha={profile.prefill.alpha:.1f}ms fixed cost alone "
                f"{'exceeds it' if profile.prefill.alpha >= effective_budget_ms else 'does not, but the quadratic has no positive root'})",
            )

        lo, hi = profile.prefill.domain_n
        in_domain = lo <= n <= hi
        domain_note = (
            f"within the calibrated domain [{lo:.0f}, {hi:.0f}]"
            if in_domain else
            f"OUTSIDE the calibrated domain [{lo:.0f}, {hi:.0f}] -- "
            f"extrapolation, treat with reduced confidence"
        )
        return Recommendation(
            True, round(n), n,
            f"{profile.model_id} on {profile.backend}: max {n:.0f} tokens "
            f"under a {sla_ms:.0f}ms SLA ({margin_note}), {domain_note}",
        )
