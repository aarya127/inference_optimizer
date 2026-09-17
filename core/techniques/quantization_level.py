"""Quantization level choice, from REAL measured (size, speed, quality) data.

Deliberately the mirror image of the original study's fabricated
quantization claims (REVIEW_FINDINGS.md 1.9, 1.12): a manufactured 2x FP8
speedup with no such GPU path, and W4A8 applied on top of a baseline that
was already 4-bit. This technique only ever recommends among quant levels
that were actually measured for the given profile
(baseline/measure_quantization_tradeoff.py); it reports "not applicable"
rather than guessing when no such data exists -- which is exactly the case
for SmolVLM (ships pre-quantized, only one level ever measured).
"""

from __future__ import annotations

from typing import Optional

from core.profile import ModelHardwareProfile
from core.technique import Recommendation, Technique


class QuantizationLevelTechnique(Technique):
    name = "quantization_level_choice"

    def applies_to(self, profile: ModelHardwareProfile) -> bool:
        return bool(profile.quant_levels) and len(profile.quant_levels) > 1

    def recommend(
        self,
        profile: ModelHardwareProfile,
        max_file_size_mb: Optional[float] = None,
        max_perplexity: Optional[float] = None,
    ) -> Recommendation:
        if not self.applies_to(profile):
            return Recommendation(
                False, None, None,
                f"{profile.model_id} on {profile.backend}: no multi-level "
                f"quantization data measured for this profile (ships "
                f"pre-quantized, or the comparison was never run)",
            )

        candidates = dict(profile.quant_levels)
        if max_file_size_mb is not None:
            candidates = {k: v for k, v in candidates.items()
                          if v.file_size_mb <= max_file_size_mb}
        if max_perplexity is not None:
            candidates = {k: v for k, v in candidates.items()
                          if v.perplexity <= max_perplexity}

        if not candidates:
            return Recommendation(
                True, None, None,
                f"no measured quant level satisfies "
                f"max_file_size_mb={max_file_size_mb}, "
                f"max_perplexity={max_perplexity} -- constraints too tight "
                f"for the {len(profile.quant_levels)} levels measured",
            )

        # Among levels satisfying the constraints, prefer the smallest file
        # (fastest to load / least memory) -- ties broken by lower perplexity.
        best_name, best = min(
            candidates.items(),
            key=lambda kv: (kv[1].file_size_mb, kv[1].perplexity),
        )
        return Recommendation(
            True, best_name, best.file_size_mb,
            f"{profile.model_id} on {profile.backend}: '{best_name}' "
            f"({best.file_size_mb:.0f} MiB, perplexity {best.perplexity:.2f}, "
            f"prefill@{best.reference_n}={best.prefill_ms_at_reference_n:.1f}ms) "
            f"is the smallest measured level meeting the given constraints",
        )
