"""Demonstrates the Technique abstraction: the same technique code, run
against two different (model, backend) profiles, applies and recommends
differently -- because the profiles' real calibrated data differs, not
because of any per-model branching in the technique code itself.

Usage:
    scope/venv_phase0/bin/python core/demo.py
"""

from core.profile import load_all_profiles
from core.techniques.concurrency_throughput import ConcurrencyThroughputTechnique
from core.techniques.kv_cache_budget import KVCacheBudgetTechnique
from core.techniques.prefill_budget import PrefillBudgetTechnique
from core.techniques.quantization_level import QuantizationLevelTechnique


def main():
    profiles = load_all_profiles()
    techniques = [PrefillBudgetTechnique(), QuantizationLevelTechnique(),
                  KVCacheBudgetTechnique(), ConcurrencyThroughputTechnique()]

    for profile_name, profile in profiles.items():
        print(f"\n{'=' * 70}")
        print(f"Profile: {profile_name}  ({profile.model_id} on {profile.backend})")
        print(f"{'=' * 70}")

        for tech in techniques:
            applies = tech.applies_to(profile)
            print(f"\n[{tech.name}] applies_to={applies}")
            if tech.name == "prefill_token_budget":
                for sla in (500.0, 2000.0):
                    rec = tech.recommend(profile, sla_ms=sla)
                    print(f"  SLA={sla:.0f}ms -> {rec.rationale}")
            elif tech.name == "quantization_level_choice":
                rec = tech.recommend(profile, max_file_size_mb=700.0)
                print(f"  max_file_size_mb=700 -> {rec.rationale}")
                rec = tech.recommend(profile, max_perplexity=15.0)
                print(f"  max_perplexity=15.0 -> {rec.rationale}")
            elif tech.name == "kv_cache_budget":
                for n_seq in (1, 8):
                    rec = tech.recommend(profile, memory_budget_mb=2048.0,
                                          n_concurrent_sequences=n_seq)
                    print(f"  budget=2048MiB, n_seq={n_seq} -> {rec.rationale}")
            elif tech.name == "concurrency_throughput":
                rec = tech.recommend(profile, tbt_sla_ms=50.0, ctx_tokens=1024.0)
                print(f"  tbt_sla=50ms, ctx=1024 -> {rec.rationale}")


if __name__ == "__main__":
    main()
