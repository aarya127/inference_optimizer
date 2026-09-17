"""Prefill token budget: how many input tokens fit under a TTFT SLA.

Generalizes the SLA-safe-token-count logic that previously existed only as
a one-off calculation inside model_calibration/calibrate_cost_model.py
(SmolVLM-specific). Here it is written once against the ModelHardwareProfile
interface, so it produces correct, different answers for any profile with a
calibrated PrefillCostModel -- demonstrated against both SmolVLM/MLX and
Qwen2.5/llama.cpp in core/demo.py.
"""

from __future__ import annotations

from core.profile import ModelHardwareProfile
from core.technique import Recommendation, Technique


class PrefillBudgetTechnique(Technique):
    name = "prefill_token_budget"

    def applies_to(self, profile: ModelHardwareProfile) -> bool:
        return profile.prefill is not None

    def recommend(self, profile: ModelHardwareProfile, sla_ms: float) -> Recommendation:
        if not self.applies_to(profile):
            return Recommendation(False, None, None,
                                   "profile has no calibrated prefill cost model")

        n = profile.prefill.max_tokens_for_budget(sla_ms)
        if n is None:
            return Recommendation(
                True, None, None,
                f"no token count meets a {sla_ms:.0f}ms budget "
                f"(alpha={profile.prefill.alpha:.1f}ms fixed cost alone "
                f"{'exceeds it' if profile.prefill.alpha >= sla_ms else 'does not, but the quadratic has no positive root'})",
            )

        lo, hi = profile.prefill.domain_n
        in_domain = lo <= n <= hi
        domain_note = (
            f"within the calibrated domain [{lo:.0f}, {hi:.0f}] "
            f"(LOOCV MAPE {profile.prefill.loocv_mape_pct:.1f}%)"
            if in_domain else
            f"OUTSIDE the calibrated domain [{lo:.0f}, {hi:.0f}] -- "
            f"extrapolation, treat with reduced confidence"
        )
        return Recommendation(
            True, round(n), n,
            f"{profile.model_id} on {profile.backend}: max {n:.0f} tokens "
            f"under a {sla_ms:.0f}ms prefill budget, {domain_note}",
        )
