"""The closed loop: recommend -> execute for real -> verify.

Tier 3's core/demo.py only prints what a Technique would recommend. This
runs that recommendation through a real Executor and reports whether the
prediction actually held on a fresh, live run -- not just against the
historical LOOCV score the cost model was calibrated with. This is meant
to be read as a demonstration of the closed loop, not a production
autotuner (see core/README.md for what's still missing).

Usage:
    scope/venv_phase0/bin/python -m core.optimize --profile qwen_llamacpp --sla-ms 500
    scope/venv_phase0/bin/python -m core.optimize --profile smolvlm_mlx --sla-ms 500
"""

from __future__ import annotations

import argparse

from core.executors.llamacpp_executor import LlamaCppExecutor
from core.executors.mlx_executor import MLXExecutor
from core.profile import load_all_profiles
from core.techniques.prefill_budget import PrefillBudgetTechnique
from core.techniques.quantization_level import QuantizationLevelTechnique

EXECUTORS = [LlamaCppExecutor(), MLXExecutor()]


def optimize_and_verify(profile_name: str, profile, sla_ms: float,
                         max_file_size_mb: float = None) -> None:
    print(f"\n{'=' * 70}")
    print(f"Profile: {profile_name}  ({profile.model_id} on {profile.backend})")
    print(f"{'=' * 70}")

    quant_tech = QuantizationLevelTechnique()
    quant_level = None
    if quant_tech.applies_to(profile):
        rec = quant_tech.recommend(profile, max_file_size_mb=max_file_size_mb)
        print(f"[quantization_level_choice] {rec.rationale}")
        quant_level = rec.choice
    else:
        print(f"[quantization_level_choice] not applicable to this profile")

    prefill_tech = PrefillBudgetTechnique()
    rec = prefill_tech.recommend(profile, sla_ms=sla_ms)
    print(f"[prefill_token_budget] {rec.rationale}")
    if not rec.applicable or rec.choice is None:
        print("  -> no recommended token budget; skipping execution")
        return
    n_tokens = rec.choice

    executor = next((e for e in EXECUTORS if e.supports(profile)), None)
    if executor is None:
        print(f"  -> no Executor available for backend {profile.backend!r}")
        return

    print(f"\nExecuting real prefill at N={n_tokens}"
          f"{f' (quant={quant_level})' if quant_level else ''}...")
    result = executor.run_prefill(profile, n_tokens, quant_level=quant_level)

    met_sla = result.measured_prefill_ms <= sla_ms
    print(f"  predicted: {result.predicted_prefill_ms:.1f} ms")
    print(f"  MEASURED:  {result.measured_prefill_ms:.1f} ms  "
          f"(prediction error: {result.error_pct:.1f}%)")
    print(f"  SLA ({sla_ms:.0f}ms) actually met on this real run: {met_sla}")
    if result.note:
        print(f"  note: {result.note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["smolvlm_mlx", "qwen_llamacpp", "all"],
                     default="all")
    ap.add_argument("--sla-ms", type=float, default=500.0)
    ap.add_argument("--max-file-size-mb", type=float, default=700.0)
    args = ap.parse_args()

    profiles = load_all_profiles()
    names = [args.profile] if args.profile != "all" else list(profiles.keys())
    for name in names:
        if name not in profiles:
            print(f"skipping {name}: profile data not present")
            continue
        optimize_and_verify(name, profiles[name], args.sla_ms, args.max_file_size_mb)


if __name__ == "__main__":
    main()
