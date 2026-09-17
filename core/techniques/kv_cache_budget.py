"""KV-cache memory budget: max context length per sequence under a memory
cap and a target number of concurrent sequences.

Uses `kv_bytes_per_token`, which is derived directly from each model's real
architecture (layers, KV heads, head dim, dtype) -- not fitted or assumed.
Applies identically to both profiles despite a 16x difference in the
underlying constant (SmolVLM: 196,608 B/token, full MHA, 32 KV heads;
Qwen2.5-0.5B: 12,288 B/token, GQA, 2 KV heads) -- the technique code does
not know or care about that difference, only the profile's real number.
"""

from __future__ import annotations

from core.profile import ModelHardwareProfile
from core.technique import Recommendation, Technique


class KVCacheBudgetTechnique(Technique):
    name = "kv_cache_budget"

    def applies_to(self, profile: ModelHardwareProfile) -> bool:
        return profile.kv_bytes_per_token is not None

    def recommend(
        self,
        profile: ModelHardwareProfile,
        memory_budget_mb: float,
        n_concurrent_sequences: int = 1,
    ) -> Recommendation:
        if not self.applies_to(profile):
            return Recommendation(False, None, None,
                                   "profile has no kv_bytes_per_token")
        if n_concurrent_sequences < 1:
            return Recommendation(True, None, None,
                                   "n_concurrent_sequences must be >= 1")

        budget_bytes = memory_budget_mb * 2**20
        bytes_per_seq = budget_bytes / n_concurrent_sequences
        max_ctx = bytes_per_seq / profile.kv_bytes_per_token

        return Recommendation(
            True, int(max_ctx), max_ctx,
            f"{profile.model_id} on {profile.backend}: "
            f"{profile.kv_bytes_per_token:.0f} B/token KV cache -> "
            f"{max_ctx:.0f} max context tokens/sequence at "
            f"{n_concurrent_sequences} concurrent sequence(s) under a "
            f"{memory_budget_mb:.0f} MiB KV budget",
        )
