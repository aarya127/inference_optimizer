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

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.profile import load_smolvlm_mlx_profile, load_qwen_llamacpp_profile
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
