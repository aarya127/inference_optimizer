"""Fit T_prefill(N) for the llama.cpp/Qwen2.5-0.5B calibration campaign, using
the identical weighted-quadratic + honest LOOCV methodology already applied
to the MLX/SmolVLM data (model_calibration/calibrate_cost_model.py,
model_calibration/refit_prefill_v3.py) -- the point of Tier 2 is to check
whether that METHODOLOGY and the quadratic FUNCTIONAL FORM generalize to a
second model on a second backend, not to compare raw speed across models.

Usage:
    scope/venv_phase0/bin/python model_calibration/fit_llamacpp_prefill.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent


def main():
    data = json.loads((ROOT / "baseline" / "results_llamacpp.json").read_text())
    points = data["points"]

    n_arr = np.array([p["n_tokens"] for p in points], dtype=float)
    t_arr = np.array([p["lm_prefill"]["mean_ms"] for p in points], dtype=float)
    std_arr = np.array([max(p["lm_prefill"]["std_ms"], 1e-3) for p in points], dtype=float)
    w = 1.0 / std_arr

    coeffs = np.polyfit(n_arr, t_arr, 2, w=w)
    gamma, beta, alpha = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    y_pred = np.polyval(coeffs, n_arr)
    ss_res = np.sum((t_arr - y_pred) ** 2)
    ss_tot = np.sum((t_arr - np.mean(t_arr)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
    insample_mape = float(np.mean(np.abs(y_pred - t_arr) / t_arr) * 100)

    loocv_apes = []
    for i in range(len(n_arr)):
        mask = np.arange(len(n_arr)) != i
        c_i = np.polyfit(n_arr[mask], t_arr[mask], 2, w=w[mask])
        pred_i = float(np.polyval(c_i, n_arr[i]))
        loocv_apes.append(abs(pred_i - t_arr[i]) / t_arr[i] * 100)
    loocv_mape = float(np.mean(loocv_apes))

    print(f"llama.cpp / Qwen2.5-0.5B-Instruct-Q4_K_M, Metal GPU offload")
    print(f"N points: {len(points)}  (domain N in [{n_arr.min():.0f}, {n_arr.max():.0f}])")
    print(f"\n{'N':>6s} {'T_prefill (ms)':>15s} {'std':>8s} {'CV%':>6s}")
    for p in points:
        cv = p["lm_prefill"]["std_ms"] / p["lm_prefill"]["mean_ms"] * 100
        print(f"{p['n_tokens']:6d} {p['lm_prefill']['mean_ms']:15.1f} "
              f"{p['lm_prefill']['std_ms']:8.1f} {cv:6.1f}")

    print(f"\nFitted: T_prefill(N) = {gamma:.6e}*N^2 + {beta:.6f}*N + {alpha:.2f}")
    print(f"  R^2 (in-sample)  = {r2:.4f}")
    print(f"  MAPE (in-sample) = {insample_mape:.2f}%")
    print(f"  MAPE (LOOCV)     = {loocv_mape:.2f}%")
    print(f"\nCompare against MLX/SmolVLM 4-point fit: R^2=0.9996 in-sample, "
          f"LOOCV=20.7% (see amio_constants.py / README.md).")

    # Decode (TBT) vs context length -- same qualitative question as
    # DECODE_OVERHEAD_MS_MEASURED / DECODE_KV_MS_PER_CTX_TOKEN for SmolVLM:
    # is decode roughly flat / mildly linear in context length?
    ctx = n_arr
    tbt = np.array([p["decode_tbt"]["mean_ms"] for p in points], dtype=float)
    tbt_coeffs = np.polyfit(ctx, tbt, 1)
    print(f"\nDecode TBT(ctx) ~= {tbt_coeffs[1]:.3f} + {tbt_coeffs[0]:.6f}*ctx  ms"
          f"  (batch=1; compare to SmolVLM's 18.33 + 0.0032*ctx)")

    out = {
        "gamma": gamma, "beta": beta, "alpha": alpha, "r2": r2,
        "insample_mape_pct": insample_mape, "loocv_mape_pct": loocv_mape,
        "n_points": len(points),
        "domain_n": [float(n_arr.min()), float(n_arr.max())],
        "decode_tbt_overhead_ms": float(tbt_coeffs[1]),
        "decode_tbt_per_ctx_token_ms": float(tbt_coeffs[0]),
    }
    out_path = ROOT / "model_calibration" / "llamacpp_prefill_fit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
