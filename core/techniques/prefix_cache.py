"""Prefix/prompt-cache reuse savings, from REAL measured data only.

Answers: "if a request shares a cached prefix (e.g. a system prompt) of
length P with a prior request, how much prefill time does reusing that
cache actually save for a new suffix of length S?" Only ever answers from
real measurements (ModelHardwareProfile.prefix_cache_measurements,
populated by baseline/measure_llamacpp_prefix_cache.py) at the specific
suffix lengths actually measured -- it does not interpolate or extrapolate
to an unmeasured suffix length, and reports "not applicable" for every
profile without such data (every profile except
smollm2_360m_llamacpp_profile, as of this writing).

See core/TIER7_PREFIX_CACHE_FINDINGS.md for why this mattered to get
right: an unsynchronized version of the underlying measurement produced
numbers off by 3-4x in both directions before a real GPU-dispatch
synchronization bug was found and fixed.
"""

from __future__ import annotations

from core.profile import ModelHardwareProfile
from core.technique import Recommendation, Technique


class PrefixCacheTechnique(Technique):
    name = "prefix_cache_savings"

    def applies_to(self, profile: ModelHardwareProfile) -> bool:
        return bool(profile.prefix_cache_measurements)

    def recommend(self, profile: ModelHardwareProfile, suffix_tokens: int) -> Recommendation:
        if not self.applies_to(profile):
            return Recommendation(
                False, None, None,
                f"{profile.model_id} on {profile.backend}: no real prefix-cache "
                f"measurements exist for this profile",
            )

        exact = next((m for m in profile.prefix_cache_measurements
                      if m.suffix_tokens == suffix_tokens), None)
        if exact is None:
            measured = sorted(m.suffix_tokens for m in profile.prefix_cache_measurements)
            return Recommendation(
                True, None, None,
                f"suffix_tokens={suffix_tokens} was not directly measured "
                f"(measured suffix lengths: {measured}) -- not interpolated "
                f"or extrapolated",
            )

        return Recommendation(
            True, round(exact.savings_pct, 1), exact.savings_pct,
            f"{profile.model_id} on {profile.backend}: reusing a cached "
            f"{exact.prefix_tokens}-token prefix saves {exact.savings_pct:.1f}% "
            f"of prefill time for a {suffix_tokens}-token suffix "
            f"(MEASURED: {exact.cold_ms:.1f}ms cold vs {exact.warm_ms:.1f}ms "
            f"with the prefix cached)",
        )
