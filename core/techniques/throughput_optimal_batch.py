"""Throughput-optimal batch size, from REAL measured concurrency data only.

A different question from ConcurrencyThroughputTechnique: that one asks
"what's the largest batch that keeps per-sequence latency under an SLA"
(and, per core/TIER6_REAL_CONCURRENCY_FINDINGS.md, answers it from a
MODELED formula now known to be substantially too optimistic). This asks
"which of the batch sizes I've actually measured gets the most total
tokens/sec out of the GPU" -- a throughput, not latency, question -- and
only ever answers it from real data (ModelHardwareProfile.concurrency_measurements,
populated by baseline/measure_llamacpp_native_batching.py's real native
multi-sequence batching runs). It does not extrapolate to batch sizes that
were never measured, and reports "not applicable" for every profile where
no such real measurement exists -- which, as of this writing, is every
profile except smollm2_360m_llamacpp_profile.
"""

from __future__ import annotations

from typing import Optional

from core.profile import ModelHardwareProfile
from core.technique import Recommendation, Technique


class ThroughputOptimalBatchTechnique(Technique):
    name = "throughput_optimal_batch"

    def applies_to(self, profile: ModelHardwareProfile) -> bool:
        return bool(profile.concurrency_measurements) and len(profile.concurrency_measurements) > 1

    def recommend(
        self,
        profile: ModelHardwareProfile,
        max_tbt_ms: Optional[float] = None,
    ) -> Recommendation:
        if not self.applies_to(profile):
            return Recommendation(
                False, None, None,
                f"{profile.model_id} on {profile.backend}: no real native-batching "
                f"measurements exist for this profile",
            )

        candidates = profile.concurrency_measurements
        if max_tbt_ms is not None:
            candidates = tuple(c for c in candidates if c.measured_tbt_ms <= max_tbt_ms)
            if not candidates:
                return Recommendation(
                    True, None, None,
                    f"no measured batch size keeps TBT <= {max_tbt_ms:.1f}ms "
                    f"(measured range: "
                    f"{min(c.measured_tbt_ms for c in profile.concurrency_measurements):.1f}-"
                    f"{max(c.measured_tbt_ms for c in profile.concurrency_measurements):.1f}ms)",
                )

        best = max(candidates, key=lambda c: c.batch_size / (c.measured_tbt_ms / 1000.0))
        throughput = best.batch_size / (best.measured_tbt_ms / 1000.0)

        return Recommendation(
            True, best.batch_size, throughput,
            f"{profile.model_id} on {profile.backend}: batch={best.batch_size} "
            f"maximizes MEASURED throughput ({throughput:.1f} tok/s aggregate, "
            f"{best.measured_tbt_ms:.1f}ms/token/sequence at ctx={best.ctx_tokens}) "
            f"among {len(profile.concurrency_measurements)} real measured batch "
            f"sizes -- not extrapolated beyond what was actually measured",
        )
