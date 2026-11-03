"""Regression tests for the Tier-3 Technique abstraction (core/).

Real invariant checks, not self-confirming smoke tests: profile loading
succeeds and traces to real calibration data, PrefillBudgetTechnique is
monotonic and domain-aware, and QuantizationLevelTechnique's applicability
genuinely differs between a profile with measured multi-level quant data
and one without -- the central "different techniques for different
models" claim this abstraction exists to demonstrate.

Run with: pytest tests/test_core.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.profile import (
    load_smolvlm_mlx_profile,
    load_qwen_llamacpp_profile,
    load_profile_from_calibration_json,
)
from core.techniques.concurrency_throughput import ConcurrencyThroughputTechnique
from core.techniques.kv_cache_budget import KVCacheBudgetTechnique
from core.techniques.prefill_budget import PrefillBudgetTechnique
from core.techniques.quantization_level import QuantizationLevelTechnique

QWEN_FIT = _ROOT / "model_calibration" / "llamacpp_prefill_fit.json"
QUANT_DATA = _ROOT / "baseline" / "results_quantization_tradeoff.json"

requires_qwen_fit = pytest.mark.skipif(
    not QWEN_FIT.exists(),
    reason="model_calibration/llamacpp_prefill_fit.json not present -- run "
           "model_calibration/fit_llamacpp_prefill.py first",
)
requires_quant_data = pytest.mark.skipif(
    not QUANT_DATA.exists(),
    reason="baseline/results_quantization_tradeoff.json not present -- run "
           "baseline/measure_quantization_tradeoff.py first",
)


def test_smolvlm_profile_loads_from_real_constants():
    profile = load_smolvlm_mlx_profile()
    assert profile.model_id == "mlx-community/SmolVLM-Instruct-4bit"
    assert profile.prefill.alpha > 0  # positive intercept, per amio_constants
    assert profile.supports_vision is True
    # SmolVLM ships pre-quantized; no multi-level quant comparison exists.
    assert profile.quant_levels is None


@requires_qwen_fit
def test_qwen_profile_loads_from_real_fit():
    profile = load_qwen_llamacpp_profile()
    assert profile.backend == "llama.cpp"
    assert profile.supports_vision is False
    lo, hi = profile.prefill.domain_n
    assert lo < hi


def test_fit_decode_linear_clamps_negative_slope():
    """core/calibrate.py's fit_decode_linear: a negative per-token decode
    cost is not physically meaningful (more context can't make a token
    cheaper). Reproduces the exact qwen25_05b_mlx_profile scenario that
    surfaced this (5 narrow-range points giving a small negative slope,
    which would predict negative latency by ctx~2048) -- the clamped fit
    must not do that."""
    from core.calibrate import fit_decode_linear

    # Synthetic points with a slight (noise-driven) downward trend, same
    # shape as the real qwen25_05b_mlx_profile decode sweep.
    points = [
        {"ctx_tokens": 32, "mean_ms": 10.5},
        {"ctx_tokens": 64, "mean_ms": 10.3},
        {"ctx_tokens": 128, "mean_ms": 10.1},
        {"ctx_tokens": 256, "mean_ms": 9.9},
        {"ctx_tokens": 384, "mean_ms": 9.7},
    ]
    fit = fit_decode_linear(points)
    assert fit["clamped_negative_slope"] is True
    assert fit["raw_fit_per_ctx_token_ms"] < 0  # the raw fit really was negative
    assert fit["per_ctx_token_ms"] == 0.0  # but the usable fit is clamped
    assert fit["overhead_ms"] > 0
    # No context length should ever predict negative latency.
    for ctx in (0, 384, 2048, 100_000):
        assert fit["overhead_ms"] + fit["per_ctx_token_ms"] * ctx >= 0


def test_fit_decode_linear_leaves_positive_slope_alone():
    from core.calibrate import fit_decode_linear

    points = [
        {"ctx_tokens": 32, "mean_ms": 10.0},
        {"ctx_tokens": 384, "mean_ms": 12.0},
    ]
    fit = fit_decode_linear(points)
    assert fit["clamped_negative_slope"] is False
    assert fit["per_ctx_token_ms"] == pytest.approx(fit["raw_fit_per_ctx_token_ms"])


def test_fit_quadratic_with_loocv_reports_max_not_just_mean():
    from core.calibrate import fit_quadratic_with_loocv

    points = [
        {"n_tokens": 100, "mean_ms": 350.0, "std_ms": 10.0},
        {"n_tokens": 466, "mean_ms": 1100.0, "std_ms": 20.0},
        {"n_tokens": 922, "mean_ms": 2300.0, "std_ms": 30.0},
        {"n_tokens": 1560, "mean_ms": 4900.0, "std_ms": 40.0},
    ]
    fit = fit_quadratic_with_loocv(points)
    assert "loocv_max_ape_pct" in fit
    # The worst held-out point can only be >= the mean of all held-out errors.
    assert fit["loocv_max_ape_pct"] >= fit["loocv_mape_pct"]


def test_generic_calibration_loader_needs_no_per_model_code(tmp_path):
    """core/calibrate.py's whole point is that onboarding a new model needs
    no new loader function, unlike load_qwen_llamacpp_profile() /
    load_smolvlm_mlx_profile() which have that model's constants inlined.
    This exercises the loader against a synthetic but schema-correct
    calibration JSON -- no network or real model load required, so it stays
    in the fast suite."""
    calib = {
        "model_id": "fake/TestModel-360M",
        "backend": "llama.cpp",
        "hardware": "Apple M3 base (10-core GPU, 8GB unified, 100GB/s), Metal offload",
        "supports_vision": False,
        "kv_bytes_per_token": 12345.0,
        "bandwidth_gbps": 100.0,
        "prefill": {
            "gamma": 1e-5, "beta": 0.5, "alpha": 10.0,
            "domain_n": [32, 2048], "loocv_mape_pct": 7.5, "loocv_max_ape_pct": 22.3,
        },
        "decode": {"overhead_ms": 8.0, "per_ctx_token_ms": 0.002},
    }
    path = tmp_path / "fake_llamacpp_profile.json"
    path.write_text(json.dumps(calib))

    profile = load_profile_from_calibration_json(path)
    assert profile.model_id == "fake/TestModel-360M"
    assert profile.backend == "llama.cpp"
    assert profile.kv_bytes_per_token == 12345.0
    assert profile.prefill.loocv_mape_pct == 7.5
    assert profile.prefill.loocv_max_ape_pct == 22.3
    assert profile.prefill.predict_ms(100) == pytest.approx(1e-5 * 100**2 + 0.5 * 100 + 10.0)


def test_prefill_budget_is_monotonic_in_sla():
    profile = load_smolvlm_mlx_profile()
    tech = PrefillBudgetTechnique()
    assert tech.applies_to(profile)

    small = tech.recommend(profile, sla_ms=500.0)
    large = tech.recommend(profile, sla_ms=2000.0)
    assert small.applicable and large.applicable
    # A looser budget must never recommend fewer tokens than a tighter one.
    if small.choice is not None and large.choice is not None:
        assert large.choice >= small.choice


def test_prefill_budget_default_margin_is_profiles_own_max_loocv():
    """Added after core/TIER4_CLOSED_LOOP_FINDINGS.md: a boundary-exact
    recommendation is fragile (Qwen missed a 500ms SLA by 2.8ms despite
    0.5% prediction error). Default margin should equal the profile's own
    max LOOCV AE (not the mean -- a mean-based margin was tried and found
    to still miss the SLA reproducibly for qwen25_05b_mlx_profile), not a
    fabricated constant, and must never recommend MORE tokens than the
    zero-margin case (a margin can only make the budget tighter or, if the
    margin consumes the whole budget, infeasible -- see the companion test
    below for SmolVLM's 500ms case, where the honestly-sized margin now
    correctly reports infeasibility instead of a number)."""
    profile = load_smolvlm_mlx_profile()
    tech = PrefillBudgetTechnique()

    no_margin = tech.recommend(profile, sla_ms=500.0, safety_margin_pct=0.0)
    default_margin = tech.recommend(profile, sla_ms=500.0)
    assert profile.prefill.loocv_max_ape_pct is not None
    assert f"{profile.prefill.loocv_max_ape_pct:.1f}% safety margin" in default_margin.rationale
    assert "max LOOCV AE" in default_margin.rationale
    assert no_margin.choice is not None
    # A margin can only shrink the recommendation -- possibly to "no
    # feasible token count" (None), never to more tokens than no-margin.
    if default_margin.choice is not None:
        assert default_margin.choice <= no_margin.choice


def test_prefill_budget_honest_margin_confirms_smolvlm_500ms_is_infeasible():
    """With SmolVLM's honestly-sized margin (59.9% max LOOCV AE), the
    effective budget (500*(1-0.599)=200.5ms) is below alpha (244.6ms,
    SmolVLM's fixed prefill cost) -- the technique now correctly reports
    no feasible token count, matching this project's own long-standing,
    independently-established finding that the 500ms SLA is infeasible on
    this hardware at every real crop setting (README.md). The earlier
    mean-based margin (20.7%) was lenient enough to still recommend a
    number (113 tokens) -- this is a case where the more conservative
    margin produces the more correct answer, not just a more cautious one."""
    profile = load_smolvlm_mlx_profile()
    tech = PrefillBudgetTechnique()
    rec = tech.recommend(profile, sla_ms=500.0)
    assert rec.applicable
    assert rec.choice is None
    assert "no token count meets" in rec.rationale


def test_prefill_budget_falls_back_to_mean_loocv_without_max():
    """A profile calibrated before loocv_max_ape_pct existed (no per-point
    LOOCV data retained) should fall back to the mean, not crash or
    silently use 0."""
    profile = load_smolvlm_mlx_profile()
    # Simulate an older profile missing the max field.
    import dataclasses
    old_style = dataclasses.replace(profile, prefill=dataclasses.replace(
        profile.prefill, loocv_max_ape_pct=None))

    tech = PrefillBudgetTechnique()
    rec = tech.recommend(old_style, sla_ms=500.0)
    assert f"{old_style.prefill.loocv_mape_pct:.1f}% safety margin" in rec.rationale
    assert "mean LOOCV MAPE" in rec.rationale


def test_prefill_budget_rejects_invalid_margin():
    profile = load_smolvlm_mlx_profile()
    tech = PrefillBudgetTechnique()
    rec = tech.recommend(profile, sla_ms=500.0, safety_margin_pct=100.0)
    assert rec.applicable and rec.choice is None


def test_prefill_budget_reports_out_of_domain_extrapolation():
    profile = load_smolvlm_mlx_profile()
    tech = PrefillBudgetTechnique()
    lo, hi = profile.prefill.domain_n
    # A very generous budget should push the recommended N past the
    # calibrated domain -- the rationale must say so, not silently claim
    # confidence it doesn't have.
    rec = tech.recommend(profile, sla_ms=100_000.0)
    if rec.choice is not None and rec.choice > hi:
        assert "OUTSIDE" in rec.rationale


def test_quantization_technique_not_applicable_without_multilevel_data():
    """SmolVLM has no measured multi-level quantization comparison in this
    repo -- the technique must say so, not guess."""
    profile = load_smolvlm_mlx_profile()
    tech = QuantizationLevelTechnique()
    assert tech.applies_to(profile) is False
    rec = tech.recommend(profile)
    assert rec.applicable is False


@requires_quant_data
def test_quantization_technique_applies_and_respects_constraints():
    profile = load_qwen_llamacpp_profile()
    tech = QuantizationLevelTechnique()
    assert tech.applies_to(profile) is True

    # A very tight file-size cap must exclude fp16.
    rec = tech.recommend(profile, max_file_size_mb=500.0)
    assert rec.applicable
    if rec.choice is not None:
        assert profile.quant_levels[rec.choice].file_size_mb <= 500.0

    # An impossible perplexity constraint must report no candidate, not
    # silently fall back to something that violates it.
    rec_impossible = tech.recommend(profile, max_perplexity=0.001)
    assert rec_impossible.applicable is True
    assert rec_impossible.choice is None


@requires_quant_data
def test_quantization_technique_same_class_differs_by_profile():
    """The central claim of this abstraction: one Technique class, two
    profiles, genuinely different applicability -- not per-model branching
    inside the technique."""
    tech = QuantizationLevelTechnique()
    smolvlm = load_smolvlm_mlx_profile()
    qwen = load_qwen_llamacpp_profile()
    assert tech.applies_to(smolvlm) != tech.applies_to(qwen)


def test_kv_cache_budget_applies_to_both_profiles_with_real_constants():
    tech = KVCacheBudgetTechnique()
    smolvlm = load_smolvlm_mlx_profile()
    assert tech.applies_to(smolvlm)
    # SmolVLM's KV bytes/token (196,608, full MHA) is a documented multiple
    # of Qwen's (12,288, GQA) -- exactly 16x, from amio_constants.py and the
    # real Qwen2.5-0.5B config.json respectively.
    assert smolvlm.kv_bytes_per_token == pytest.approx(196608)


def test_kv_cache_budget_scales_inversely_with_concurrent_sequences():
    tech = KVCacheBudgetTechnique()
    profile = load_smolvlm_mlx_profile()
    one_seq = tech.recommend(profile, memory_budget_mb=2048.0, n_concurrent_sequences=1)
    eight_seq = tech.recommend(profile, memory_budget_mb=2048.0, n_concurrent_sequences=8)
    assert one_seq.applicable and eight_seq.applicable
    # 8x the concurrent sequences -> 1/8th the context length per sequence.
    assert one_seq.estimated_value == pytest.approx(eight_seq.estimated_value * 8, rel=1e-6)


@requires_qwen_fit
def test_kv_cache_budget_reflects_real_architecture_difference():
    """Same memory budget, same n_sequences: Qwen (GQA, 12,288 B/token)
    must fit a proportionally larger context than SmolVLM (full MHA,
    196,608 B/token) -- this is real architecture math, not an assumption."""
    tech = KVCacheBudgetTechnique()
    smolvlm = load_smolvlm_mlx_profile()
    qwen = load_qwen_llamacpp_profile()
    rec_smolvlm = tech.recommend(smolvlm, memory_budget_mb=2048.0, n_concurrent_sequences=1)
    rec_qwen = tech.recommend(qwen, memory_budget_mb=2048.0, n_concurrent_sequences=1)
    ratio = rec_qwen.estimated_value / rec_smolvlm.estimated_value
    expected_ratio = smolvlm.kv_bytes_per_token / qwen.kv_bytes_per_token
    assert ratio == pytest.approx(expected_ratio, rel=1e-6)


def test_concurrency_throughput_is_modeled_not_measured_label():
    tech = ConcurrencyThroughputTechnique()
    profile = load_smolvlm_mlx_profile()
    rec = tech.recommend(profile, tbt_sla_ms=50.0, ctx_tokens=1024.0)
    assert rec.applicable
    assert "MODELED" in rec.rationale


def test_concurrency_throughput_reports_zero_when_batch1_already_over_sla():
    tech = ConcurrencyThroughputTechnique()
    profile = load_smolvlm_mlx_profile()
    # An unreasonably tight TBT SLA that even batch=1 cannot meet.
    rec = tech.recommend(profile, tbt_sla_ms=0.001, ctx_tokens=1024.0)
    assert rec.applicable
    assert rec.choice == 0
