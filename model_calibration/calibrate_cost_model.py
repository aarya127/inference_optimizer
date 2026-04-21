#!/usr/bin/env python3
"""
Model Calibration — T_prefill vs Token Count
=============================================
Bypass SmolVLM's tiling processor and measure T_prefill as a function of token count.
Goal: Solve for γ in T_prefill = γ·N_tokens² + β·N + α, then build a predictive cost model.

Methodology:
1. Load SmolVLM via mlx_vlm.load()
2. Inject synthetic embeddings directly into the language model (bypasses tiling)
3. Measure full forward-pass time (prefill) at N = 128..1548 tokens
4. Fit quadratic: T_prefill(N) = γ·N² + β·N + α
5. Validate against Phase 1 data point (N=1548, T_prefill=8489ms)

Key insight from idefics3/language.py:
    LanguageModel.__call__(inputs, inputs_embeds=None, ...)
    When inputs_embeds is provided, the embed_tokens lookup is skipped.
    This lets us inject arbitrary-size embeddings without the vision processor.
"""

import sys, os, time, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import numpy as np
from mlx_vlm import load

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/SmolVLM-Instruct-4bit"
CALIB_RESULTS = ROOT / "model_calibration" / "calibration_results.json"

# Token counts to test (bypassing tiling)
TOKEN_COUNTS = [128, 256, 512, 768, 1024, 1280, 1548]
N_TRIALS = 3        # average over 3 runs to reduce variance
N_WARMUP  = 2       # warmup runs (discarded) to trigger MLX JIT compilation

@dataclass
class CalibrationDataPoint:
    n_tokens: int
    t_prefill_ms: float
    t_prefill_stdev_ms: float = 0.0
    trial_times: List[float] = field(default_factory=list)

def load_model():
    """Load SmolVLM model. Returns (model, hidden_size)."""
    print("Loading model...")
    t0 = time.perf_counter()
    model, processor = load(MODEL_ID)
    mx.eval()
    load_ms = (time.perf_counter() - t0) * 1e3
    hidden_size = model.language_model.config.hidden_size
    print(f"  Model loaded in {load_ms:.0f}ms")
    print(f"  LM hidden_size = {hidden_size}")
    print(f"  LM num_layers  = {model.language_model.num_hidden_layers}")
    return model, hidden_size

def create_synthetic_embeddings(n_tokens: int, hidden_size: int) -> mx.array:
    """
    Create unit-normalised Gaussian embeddings of shape [1, n_tokens, hidden_size].

    SmolVLM's vision encoder outputs normalised embeddings projected to the LM
    hidden size.  Random unit-normalised vectors are a good proxy — they exercise
    the same compute path without requiring a real image.
    """
    embeddings = mx.random.normal(shape=(1, n_tokens, hidden_size))
    norms = mx.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
    return embeddings / norms


def measure_prefill_latency(lm, n_tokens: int, hidden_size: int) -> float:
    """
    Measure T_prefill for a given synthetic token count.

    Injects random unit-normalised embeddings (shape [1, n_tokens, hidden_size])
    directly into the LanguageModel, bypassing vision encoder and tiling.

    The LanguageModel signature (idefics3/language.py):
        __call__(self, inputs, inputs_embeds=None, mask=None, cache=None)
    When inputs_embeds is provided, embed_tokens is skipped entirely.
    """
    syn_embeds = create_synthetic_embeddings(n_tokens, hidden_size)

    mx.eval()          # flush any pending MLX graph before timing
    t0 = time.perf_counter()
    out = lm(inputs=None, inputs_embeds=syn_embeds)
    mx.eval(out.logits)
    return (time.perf_counter() - t0) * 1e3


def calibrate() -> dict:
    model, hidden_size = load_model()
    lm = model.language_model

    # ── Warmup (triggers MLX JIT / Metal shader compilation) ─────────────────
    print(f"\nWarming up ({N_WARMUP}×{TOKEN_COUNTS[0]} tokens)...")
    for _ in range(N_WARMUP):
        measure_prefill_latency(lm, TOKEN_COUNTS[0], hidden_size)
    print("  Warmup done.")

    print("\n" + "=" * 70)
    print("CALIBRATION: T_prefill vs Token Count")
    print("=" * 70)

    calibration_data: List[CalibrationDataPoint] = []

    for n_tok in TOKEN_COUNTS:
        print(f"\nN_tokens = {n_tok:4d}  ...", end="", flush=True)
        trial_times = []

        for _ in range(N_TRIALS):
            try:
                t = measure_prefill_latency(lm, n_tok, hidden_size)
                trial_times.append(t)
                print(".", end="", flush=True)
            except Exception as e:
                print(f"ERR({e})", end="", flush=True)

        if trial_times:
            mean_t = float(np.mean(trial_times))
            std_t  = float(np.std(trial_times))
            print(f"  T_prefill = {mean_t:.1f} ± {std_t:.1f} ms")
            calibration_data.append(CalibrationDataPoint(
                n_tokens=n_tok,
                t_prefill_ms=mean_t,
                t_prefill_stdev_ms=std_t,
                trial_times=trial_times,
            ))
        else:
            print("  ALL TRIALS FAILED")

    if len(calibration_data) < 3:
        raise RuntimeError("Too few successful measurements to fit a polynomial.")

    # ── Fit T_prefill = γ·N² + β·N + α ───────────────────────────────────────
    n_arr = np.array([d.n_tokens      for d in calibration_data])
    t_arr = np.array([d.t_prefill_ms  for d in calibration_data])

    coeffs = np.polyfit(n_arr, t_arr, 2)
    gamma, beta, alpha = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    y_pred  = np.polyval(coeffs, n_arr)
    ss_res  = np.sum((t_arr - y_pred) ** 2)
    ss_tot  = np.sum((t_arr - np.mean(t_arr)) ** 2)
    r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    print("\n" + "=" * 70)
    print("FITTED EQUATION: T_prefill = γ·N² + β·N + α")
    print("=" * 70)
    print(f"  γ (quadratic) = {gamma:.6e}  ms/token²")
    print(f"  β (linear)    = {beta:.6f}  ms/token")
    print(f"  α (constant)  = {alpha:.2f}  ms")
    print(f"  R²            = {r2:.4f}  {'PASS' if r2 >= 0.95 else 'WARN (below 0.95 threshold)'}")

    # ── Validate against Phase 1 ground truth ────────────────────────────────
    N_BASELINE = 1548
    T_BASELINE = 8489.27   # ms, from baseline/results.json
    t_pred_baseline = float(np.polyval(coeffs, N_BASELINE))
    error_pct = abs(t_pred_baseline - T_BASELINE) / T_BASELINE * 100

    print(f"\nValidation vs Phase 1 baseline (N={N_BASELINE}, T={T_BASELINE:.1f}ms):")
    print(f"  Predicted : {t_pred_baseline:.1f} ms")
    print(f"  Error     : {error_pct:.1f}%  {'PASS' if error_pct < 15 else 'WARN'}")

    # ── SLA-safe token count ──────────────────────────────────────────────────
    SLA_MS = 500.0
    disc = beta**2 - 4 * gamma * (alpha - SLA_MS)
    n_sla_str = "N/A (SLA already impossible at α cost)"
    n_sla_val = None
    if disc >= 0 and gamma > 0:
        # Take the smaller (positive) root: N where T_prefill = SLA
        n_sla_val = float((-beta - np.sqrt(disc)) / (2 * gamma))
        if n_sla_val <= 0:
            n_sla_val = float((-beta + np.sqrt(disc)) / (2 * gamma))
        if n_sla_val > 0:
            ratio = n_sla_val / N_BASELINE
            n_sla_str = f"{n_sla_val:.0f} tokens  ({ratio:.1%} of baseline, {1/ratio:.1f}× compression)"

    print(f"\nSLA ({SLA_MS:.0f}ms)-safe max token count: {n_sla_str}")

    return {
        "gamma": gamma,
        "beta":  beta,
        "alpha": alpha,
        "r2":    r2,
        "hidden_size": hidden_size,
        "n_trials": N_TRIALS,
        "calibration_points": [
            {
                "n_tokens":           d.n_tokens,
                "t_prefill_ms":       d.t_prefill_ms,
                "t_prefill_stdev_ms": d.t_prefill_stdev_ms,
                "trial_times":        d.trial_times,
            }
            for d in calibration_data
        ],
        "validation": {
            "n_baseline":       N_BASELINE,
            "t_baseline_actual": T_BASELINE,
            "t_baseline_pred":  t_pred_baseline,
            "error_pct":        error_pct,
        },
        "sla_safe_n": n_sla_val,
    }


def main():
    CALIB_RESULTS.parent.mkdir(parents=True, exist_ok=True)

    result = calibrate()

    with open(CALIB_RESULTS, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved → {CALIB_RESULTS}")
    print("\nNext: run  python model_calibration/cost_model.py  to load these coefficients.")


if __name__ == "__main__":
    main()
