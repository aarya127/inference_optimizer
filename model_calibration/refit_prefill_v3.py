"""Refit T_prefill(N) on the combined v2 (crop-derived) + v3 (text-length-
derived) calibration points -- both measured via the identical real code
path (get_input_embeddings -> make_prompt_cache -> language_model prefill,
mx.eval on actual outputs). See baseline/measure_prefill_text_calibration.py
for why the text-length points are a valid densification and not a repeat
of the discredited synthetic-embedding approach in calibrate_cost_model.py.

Reports the same honest metrics as calibrate_cost_model.py: in-sample MAPE
(optimistic, scored on the fit's own points) AND leave-one-out
cross-validation MAPE (the generalization estimate) -- with many more
points than the previous 4, LOOCV is a much more meaningful number here.

Usage:
    scope/venv_phase0/bin/python model_calibration/refit_prefill_v3.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_points():
    v2 = json.loads((ROOT / "baseline" / "results_v2.json").read_text())
    points = []
    for label, e in v2["configs"].items():
        points.append({
            "n_tokens": e["total_input_tokens"],
            "t_prefill_ms": e["lm_prefill"]["mean_ms"],
            "t_prefill_std_ms": e["lm_prefill"]["std_ms"],
            "source": f"results_v2.json:{label}",
        })

    v3_path = ROOT / "baseline" / "prefill_text_calibration.json"
    if v3_path.exists():
        v3 = json.loads(v3_path.read_text())
        for p in v3["points"]:
            points.append({
                "n_tokens": p["n_tokens"],
                "t_prefill_ms": p["lm_prefill"]["mean_ms"],
                "t_prefill_std_ms": p["lm_prefill"]["std_ms"],
                "source": f"prefill_text_calibration.json:repeats={p['filler_repeats']}",
            })
    else:
        print(f"WARNING: {v3_path} not found -- fitting on the original 4 "
              f"points only (run measure_prefill_text_calibration.py first).")

    points.sort(key=lambda p: p["n_tokens"])
    return points


def fit_and_report(points):
    n_arr = np.array([p["n_tokens"] for p in points], dtype=float)
    t_arr = np.array([p["t_prefill_ms"] for p in points], dtype=float)
    std_arr = np.array([max(p["t_prefill_std_ms"], 1e-3) for p in points], dtype=float)
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

    print(f"N points: {len(points)}  (domain N in [{n_arr.min():.0f}, {n_arr.max():.0f}])")
    print(f"\n{'N':>6s} {'T_prefill (ms)':>15s} {'std':>8s}  source")
    for p in points:
        print(f"{p['n_tokens']:6d} {p['t_prefill_ms']:15.1f} "
              f"{p['t_prefill_std_ms']:8.1f}  {p['source']}")

    print(f"\nFitted: T_prefill(N) = {gamma:.6e}*N^2 + {beta:.6f}*N + {alpha:.2f}")
    print(f"  R^2 (in-sample)  = {r2:.4f}")
    print(f"  MAPE (in-sample) = {insample_mape:.2f}%")
    print(f"  MAPE (LOOCV)     = {loocv_mape:.2f}%   <- compare against the "
          f"previous 4-point fit's 20.7%")

    return {
        "gamma": gamma, "beta": beta, "alpha": alpha,
        "r2": r2, "insample_mape_pct": insample_mape,
        "loocv_mape_pct": loocv_mape,
        "n_points": len(points),
        "domain_n": [float(n_arr.min()), float(n_arr.max())],
        "points": points,
    }


def main():
    points = load_points()
    result = fit_and_report(points)
    out = ROOT / "model_calibration" / "prefill_fit_v3.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
