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
within its 5.7% LOOCV) and still measured 502.8ms on a fresh real run --
missing the SLA despite an excellent cost model, purely because the
recommendation left zero headroom. Rather than inventing a new margin
constant, `recommend()` defaults `safety_margin_pct` to the profile's own
`loocv_mape_pct` -- the honest, already-measured estimate of how far off
this profile's predictions tend to run. This helps exactly the failure
mode it targets (boundary fragility on an otherwise-accurate model, as
seen for Qwen) -- it does NOT fix a profile whose real-world error exceeds
its own LOOCV, which is exactly SmolVLM/MLX's problem
(TIER4_CLOSED_LOOP_FINDINGS.md: 41-72% real error vs a 20.7% LOOCV
estimate). A margin derived from a benchmark that itself underestimates
the model's real unreliability cannot compensate for that; it is applied
here anyway, honestly, and still shown coming up short for SmolVLM in
TIER4_CLOSED_LOOP_FINDINGS.md's follow-up.
"""

from __future__ import annotations

from typing import Optional

from core.profile import ModelHardwareProfile
from core.technique import Recommendation, Technique


class PrefillBudgetTechnique(Technique):
    name = "prefill_token_budget"

    def applies_to(self, profile: ModelHardwareProfile) -> bool:
        return profile.prefill is not None

    def recommend(
        self,
        profile: ModelHardwareProfile,
        sla_ms: float,
        safety_margin_pct: Optional[float] = None,
    ) -> Recommendation:
        if not self.applies_to(profile):
            return Recommendation(False, None, None,
                                   "profile has no calibrated prefill cost model")

        if safety_margin_pct is None:
            # Default to this profile's own honest LOOCV MAPE, when known,
            # rather than a fabricated constant. 0.0 (no margin) only when
            # the profile has no LOOCV estimate at all.
            safety_margin_pct = profile.prefill.loocv_mape_pct or 0.0

        if not (0.0 <= safety_margin_pct < 100.0):
            return Recommendation(True, None, None,
                                   f"safety_margin_pct must be in [0, 100); got {safety_margin_pct}")

        effective_budget_ms = sla_ms * (1.0 - safety_margin_pct / 100.0)

        n = profile.prefill.max_tokens_for_budget(effective_budget_ms)
        margin_note = (
            f"{safety_margin_pct:.1f}% safety margin applied "
            f"({'profile LOOCV MAPE' if safety_margin_pct == (profile.prefill.loocv_mape_pct or 0.0) else 'caller-specified'}) "
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
