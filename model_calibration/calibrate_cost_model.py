#!/usr/bin/env python3
"""
Model Calibration — T_prefill vs Token Count
=============================================
Bypass SmolVLM's tiling processor and measure T_prefill as a function of token count.
Goal: Solve for γ in T_prefill = γ·N_tokens² + β·N + α, then build a predictive cost model.

Methodology:
1. Load SmolVLM via mlx_vlm.load() (parameters materialised with mx.eval).
2. Inject synthetic embeddings directly into the language model (bypasses tiling).
3. Per token count N: one discarded warm-up trial (Metal kernel compilation is
   SHAPE-dependent, so warming up only at N=128 is not enough), then n_trials
   timed trials, each synchronised on the actual output tensor.
4. Fit quadratic T_prefill(N) = γ·N² + β·N + α, WEIGHTED by 1/std per point.
5. Report leave-one-out cross-validation MAPE (fit on 6 points, predict the
   7th, cycle) alongside the in-sample MAPE — clearly labeled.
6. If the unconstrained fit yields α < 0, also report a constrained fit with
   α clamped to 0 (γ, β refit by weighted least squares) and a domain warning.

The fitted model is only valid on N ∈ [128, 1548]
(amio_constants.PREFILL_DOMAIN); consumers must clamp predictions ≥ 0.

Key insight from idefics3/language.py:
    LanguageModel.__call__(inputs, inputs_embeds=None, ...)
    When inputs_embeds is provided, the embed_tokens lookup is skipped.
    This lets us inject arbitrary-size embeddings without the vision processor.
"""

import sys, time, json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import numpy as np
from mlx_vlm import load

import amio_constants as C

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "mlx-community/SmolVLM-Instruct-4bit"
CALIB_RESULTS = ROOT / "model_calibration" / "calibration_results.json"

# Token counts to test (bypassing tiling)
TOKEN_COUNTS = [128, 256, 512, 768, 1024, 1280, 1548]
N_TRIALS_DEFAULT = 5   # timed trials per shape (raised from 3)
N_WARMUP = 2           # global warm-up runs at the smallest shape

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
    # mx.eval() with no arguments is a NO-OP: it synchronises nothing.
    # Materialise the actual parameter arrays so lazy weight loading does not
    # leak into the first timed forward pass.
    mx.eval(model.parameters())
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

    Synchronisation: mx.eval is called ON THE ACTUAL ARRAYS — the input
    embeddings before starting the timer (so their generation is not timed)
    and the output logits before stopping it (so the full forward pass is
    forced; a bare mx.eval() call is a no-op and times nothing).

    The LanguageModel signature (idefics3/language.py):
        __call__(self, inputs, inputs_embeds=None, mask=None, cache=None)
    When inputs_embeds is provided, embed_tokens is skipped entirely.
    """
    syn_embeds = create_synthetic_embeddings(n_tokens, hidden_size)
    mx.eval(syn_embeds)   # materialise inputs before timing starts

    t0 = time.perf_counter()
    out = lm(inputs=None, inputs_embeds=syn_embeds)
    out_arr = out.logits if hasattr(out, "logits") else out
    mx.eval(out_arr)      # force the computation to complete before stopping
    return (time.perf_counter() - t0) * 1e3


def _weighted_quadratic_fit(n_arr: np.ndarray, t_arr: np.ndarray,
                            std_arr: np.ndarray):
    """Weighted polyfit with w = 1/std (zero/near-zero stds floored)."""
    w = 1.0 / np.maximum(std_arr, 1e-3)
    coeffs = np.polyfit(n_arr, t_arr, 2, w=w)
    return coeffs, w


def _constrained_fit_alpha_zero(n_arr: np.ndarray, t_arr: np.ndarray,
                                w: np.ndarray):
    """Weighted least-squares fit of T = γ·N² + β·N with α clamped to 0."""
    A = np.column_stack([n_arr.astype(float) ** 2, n_arr.astype(float)])
    Aw = A * w[:, None]
    tw = t_arr * w
    sol, *_ = np.linalg.lstsq(Aw, tw, rcond=None)
    return float(sol[0]), float(sol[1])   # gamma, beta


def calibrate(n_trials: int = N_TRIALS_DEFAULT) -> dict:
    model, hidden_size = load_model()
    lm = model.language_model

    # ── Global warm-up (triggers MLX JIT / Metal shader compilation) ─────────
    print(f"\nGlobal warm-up ({N_WARMUP}×{TOKEN_COUNTS[0]} tokens)...")
    for _ in range(N_WARMUP):
        measure_prefill_latency(lm, TOKEN_COUNTS[0], hidden_size)
    print("  Warm-up done.")

    print("\n" + "=" * 70)
    print(f"CALIBRATION: T_prefill vs Token Count  ({n_trials} trials/shape)")
    print("=" * 70)

    calibration_data: List[CalibrationDataPoint] = []

    for n_tok in TOKEN_COUNTS:
        print(f"\nN_tokens = {n_tok:4d}  ...", end="", flush=True)
        trial_times = []

        # Per-shape warm-up: Metal kernel compilation is shape-dependent, so
        # the first trial at each N absorbs compile time — discard it.
        try:
            measure_prefill_latency(lm, n_tok, hidden_size)
            print("w", end="", flush=True)
        except Exception as e:
            print(f"WARMUP-ERR({e})", end="", flush=True)

        for _ in range(n_trials):
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

    if len(calibration_data) < 4:
        raise RuntimeError("Too few successful measurements to fit + cross-validate.")

    # ── Weighted fit: T_prefill = γ·N² + β·N + α ─────────────────────────────
    n_arr   = np.array([d.n_tokens           for d in calibration_data], dtype=float)
    t_arr   = np.array([d.t_prefill_ms       for d in calibration_data])
    std_arr = np.array([d.t_prefill_stdev_ms for d in calibration_data])

    coeffs, w = _weighted_quadratic_fit(n_arr, t_arr, std_arr)
    gamma, beta, alpha = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    y_pred  = np.polyval(coeffs, n_arr)
    ss_res  = np.sum((t_arr - y_pred) ** 2)
    ss_tot  = np.sum((t_arr - np.mean(t_arr)) ** 2)
    r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    # In-sample MAPE (same points the fit was computed on — NOT held out)
    insample_mape = float(np.mean(np.abs(y_pred - t_arr) / t_arr) * 100)

    # Leave-one-out cross-validation MAPE (honest generalisation estimate:
    # fit on all-but-one point, predict the held-out point, cycle).
    loocv_apes = []
    for i in range(len(n_arr)):
        mask = np.arange(len(n_arr)) != i
        c_i = np.polyfit(n_arr[mask], t_arr[mask], 2, w=w[mask])
        pred_i = float(np.polyval(c_i, n_arr[i]))
        loocv_apes.append(abs(pred_i - t_arr[i]) / t_arr[i] * 100)
    loocv_mape = float(np.mean(loocv_apes))

    print("\n" + "=" * 70)
    print("FITTED EQUATION: T_prefill = γ·N² + β·N + α   (weighted by 1/std)")
    print("=" * 70)
    print(f"  γ (quadratic) = {gamma:.6e}  ms/token²")
    print(f"  β (linear)    = {beta:.6f}  ms/token")
    print(f"  α (constant)  = {alpha:.2f}  ms")
    print(f"  R² (in-sample)      = {r2:.4f}  {'PASS' if r2 >= 0.95 else 'WARN (below 0.95 threshold)'}")
    print(f"  MAPE (in-sample)    = {insample_mape:.2f}%  (fit and scored on the same points)")
    print(f"  MAPE (LOOCV)        = {loocv_mape:.2f}%  (leave-one-out cross-validation)")

    # ── Constrained fit when α < 0 ────────────────────────────────────────────
    constrained = None
    if alpha < 0:
        gamma_c, beta_c = _constrained_fit_alpha_zero(n_arr, t_arr, w)
        pred_c = gamma_c * n_arr ** 2 + beta_c * n_arr
        mape_c = float(np.mean(np.abs(pred_c - t_arr) / t_arr) * 100)
        constrained = {"gamma": gamma_c, "beta": beta_c, "alpha": 0.0,
                       "insample_mape_pct": mape_c}
        print(f"\n  WARNING: unconstrained α = {alpha:.2f} < 0 — the raw polynomial")
        print(f"  goes negative for small N; the model is only valid on "
              f"N ∈ [{C.PREFILL_DOMAIN[0]}, {C.PREFILL_DOMAIN[1]}].")
        print(f"  Constrained fit (α clamped to 0, weighted LSQ):")
        print(f"    γ = {gamma_c:.6e}  β = {beta_c:.6f}  "
              f"(in-sample MAPE {mape_c:.2f}%)")

    # ── Context vs Phase 1 end-to-end measurement ─────────────────────────────
    # NOTE: this is NOT a validation of the LM fit.  The Phase 1 number is the
    # END-TO-END prefill (vision + LM); this calibration measures the LM only.
    # The gap is the unvalidated "vision residual" (amio_constants.
    # VISION_BASE_MS_UNVALIDATED) — an attribution, not a measurement.
    N_BASELINE = 1548
    T_BASELINE = 8489.27   # ms end-to-end, from baseline/results.json
    t_pred_baseline = float(np.polyval(coeffs, N_BASELINE))
    error_pct = abs(t_pred_baseline - T_BASELINE) / T_BASELINE * 100

    print(f"\nContext vs Phase 1 END-TO-END prefill (N={N_BASELINE}, T={T_BASELINE:.1f}ms):")
    print(f"  LM-only prediction : {t_pred_baseline:.1f} ms")
    print(f"  Gap                : {error_pct:.1f}%  — expected: the end-to-end number")
    print(f"  includes the vision stage; the gap IS the unvalidated vision residual.")

    # ── SLA-safe token count ──────────────────────────────────────────────────
    # Two budgets, clearly labeled:
    #  (a) LM-prefill-only: how many tokens the LM alone can prefill in the SLA.
    #  (b) Including the (unvalidated) vision residual: SLA − 5991 ms < 0, so
    #      the end-to-end SLA is unmeetable without vision optimisation.
    SLA_MS = C.TTFT_SLA_MS

    def _solve_n_for_budget(budget_ms: float):
        if budget_ms <= 0 or gamma <= 0:
            return None
        disc = beta ** 2 - 4 * gamma * (alpha - budget_ms)
        if disc < 0:
            return None
        n = float((-beta + np.sqrt(disc)) / (2 * gamma))   # the positive root
        return n if n > 0 else None

    n_sla_lm_only = _solve_n_for_budget(SLA_MS)
    vision_budget = SLA_MS - C.VISION_BASE_MS_UNVALIDATED
    n_sla_with_vision = _solve_n_for_budget(vision_budget)

    if n_sla_lm_only:
        ratio = n_sla_lm_only / N_BASELINE
        print(f"\nSLA ({SLA_MS:.0f}ms)-safe max token count "
              f"[LM-prefill-only budget — EXCLUDES vision]: "
              f"{n_sla_lm_only:.0f} tokens  ({ratio:.1%} of baseline, "
              f"{1/ratio:.1f}× compression)")
    else:
        print(f"\nSLA ({SLA_MS:.0f}ms)-safe max token count "
              f"[LM-prefill-only budget]: N/A")
    if n_sla_with_vision is None:
        print(f"Including the unvalidated vision residual "
              f"({C.VISION_BASE_MS_UNVALIDATED:.0f}ms), the remaining LM budget is "
              f"{vision_budget:.0f}ms — the end-to-end SLA is unmeetable by "
              f"token pruning alone.")

    return {
        "gamma": gamma,
        "beta":  beta,
        "alpha": alpha,
        "r2":    r2,
        "insample_mape_pct": insample_mape,
        "loocv_mape_pct":    loocv_mape,
        "fit_weighting": "1/std per point (weighted polyfit)",
        "constrained_fit_alpha0": constrained,   # populated when α < 0
        "valid_domain_n": list(C.PREFILL_DOMAIN),
        "hidden_size": hidden_size,
        "n_trials": n_trials,
        "per_shape_warmup": True,
        "calibration_points": [
            {
                "n_tokens":           d.n_tokens,
                "t_prefill_ms":       d.t_prefill_ms,
                "t_prefill_stdev_ms": d.t_prefill_stdev_ms,
                "trial_times":        d.trial_times,
            }
            for d in calibration_data
        ],
        # Kept for backwards compatibility; NOT a validation of the LM fit —
        # the baseline is end-to-end (vision + LM), the fit is LM-only, and
        # the gap is the unvalidated vision residual.
        "validation": {
            "n_baseline":        N_BASELINE,
            "t_baseline_actual": T_BASELINE,
            "t_baseline_pred":   t_pred_baseline,
            "error_pct":         error_pct,
            "note": ("End-to-end baseline vs LM-only prediction; the gap is "
                     "the unvalidated vision residual, not model error."),
        },
        # LM-prefill-only budget: EXCLUDES the vision stage.  Including the
        # unvalidated 5991 ms vision residual, no token count meets the SLA.
        "sla_safe_n": n_sla_lm_only,
        "sla_safe_n_label": "LM-prefill-only budget (excludes vision stage)",
        "sla_safe_n_including_vision_residual": n_sla_with_vision,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate the LM prefill cost model")
    parser.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT,
                        help=f"Timed trials per token count (default {N_TRIALS_DEFAULT})")
    args = parser.parse_args()

    CALIB_RESULTS.parent.mkdir(parents=True, exist_ok=True)

    result = calibrate(n_trials=args.trials)

    with open(CALIB_RESULTS, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved → {CALIB_RESULTS}")
    print("\nNext: run  python model_calibration/cost_model.py  to load these coefficients.")


if __name__ == "__main__":
    main()
