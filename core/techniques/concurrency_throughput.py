"""Max concurrent sequences under a decode (TBT) SLA -- MODELED batch
extension, not measured.

Every decode measurement in this repo (both baseline/results_v2.json for
SmolVLM and baseline/results_llamacpp.json for Qwen2.5) is batch=1 only.
This technique extends the measured batch=1 DecodeCostModel with the same
modeled per-extra-sequence KV-read term amio_constants.py already documents
for SmolVLM ("theoretical per-extra-sequence KV-read cost at 1548 ctx is
~3.0 ms = 196,608 B/tok * 1548 / 100 GB/s") -- generalized to any profile's
real kv_bytes_per_token and bandwidth_gbps, and applied honestly to BOTH
profiles with the same MODELED, not MEASURED, label. It answers a different
question from KVCacheBudgetTechnique: that one asks "how much context fits
in memory"; this one asks "how many sequences can decode together before
the SLA breaks", assuming memory is not the binding constraint.
"""

from __future__ import annotations

from core.profile import ModelHardwareProfile
from core.technique import Recommendation, Technique


class ConcurrencyThroughputTechnique(Technique):
    name = "concurrency_throughput"

    def applies_to(self, profile: ModelHardwareProfile) -> bool:
        return (profile.decode is not None
                and profile.kv_bytes_per_token is not None
                and profile.bandwidth_gbps is not None)

    def _extra_seq_cost_ms(self, profile: ModelHardwareProfile, ctx_tokens: float) -> float:
        """Modeled cost of one additional concurrent sequence's KV read at
        this context length, in ms."""
        bytes_per_step = ctx_tokens * profile.kv_bytes_per_token
        return bytes_per_step / (profile.bandwidth_gbps * 1e9) * 1000.0

    def recommend(
        self,
        profile: ModelHardwareProfile,
        tbt_sla_ms: float,
        ctx_tokens: float,
    ) -> Recommendation:
        if not self.applies_to(profile):
            return Recommendation(False, None, None,
                                   "profile lacks decode/kv_bytes_per_token/bandwidth data")

        base_tbt = profile.decode.predict_ms(ctx_tokens)
        if base_tbt > tbt_sla_ms:
            return Recommendation(
                True, 0, 0.0,
                f"{profile.model_id} on {profile.backend}: MODELED batch=1 "
                f"TBT alone ({base_tbt:.1f}ms at ctx={ctx_tokens:.0f}) already "
                f"exceeds the {tbt_sla_ms:.0f}ms SLA -- 0 concurrent sequences feasible",
            )

        extra = self._extra_seq_cost_ms(profile, ctx_tokens)
        if extra <= 0:
            return Recommendation(True, None, None,
                                   "non-positive modeled per-sequence cost")

        max_n = 1 + (tbt_sla_ms - base_tbt) / extra
        n = int(max_n)

        return Recommendation(
            True, n, max_n,
            f"{profile.model_id} on {profile.backend}: MODELED (not measured "
            f"-- batch=1 is the only measured point) up to {n} concurrent "
            f"sequences at ctx={ctx_tokens:.0f} under a {tbt_sla_ms:.0f}ms TBT "
            f"SLA (batch=1 TBT={base_tbt:.1f}ms, +{extra:.2f}ms/extra sequence)",
        )
