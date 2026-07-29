"""Phase 4 regression tests — KV backends, W4A8, vision offload, batching.

Rewritten from a self-confirming smoke script (asserted whatever the
formulas produced, including several since-fixed bugs) into real pytest
tests that check invariants which would actually fail on a regression:
directional relationships and cross-module consistency, not the exact
numeric outputs of the modules' own formulas.

Run with: pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import amio_constants as C
from simulation.kv_manager import (
    ContiguousBackend,
    PagedBackend,
    kv_cache_size_mb,
    KV_BYTES_PER_TOKEN,
)
from simulation.w4a8_quantizer import W4A8Analyzer, SCHEMES
from simulation.vision_offloader import VisionOffloader
from simulation.sm_orchestrator import (
    decode_starvation_analysis,
    batch_expansion_summary,
    KV_BYTES_PER_TOKEN_FP16 as SM_KV_BYTES_PER_TOKEN,
)
from model_calibration.cost_model import CostModel


# ---------------------------------------------------------------------------
# Cross-module constant consistency (would have caught the 1152-vs-2048
# hidden size and 110,592-vs-196,608 KV/token drift from the original review)
# ---------------------------------------------------------------------------

def test_kv_bytes_per_token_consistent_across_modules():
    assert KV_BYTES_PER_TOKEN == C.KV_BYTES_PER_TOKEN_FP16
    assert SM_KV_BYTES_PER_TOKEN == C.KV_BYTES_PER_TOKEN_FP16
    # 2 (K+V) x 24 layers x 32 kv_heads x 64 head_dim x 2 bytes (FP16)
    assert C.KV_BYTES_PER_TOKEN_FP16 == 2 * 24 * 32 * 64 * 2 == 196_608


def test_prefill_domain_matches_measured_data():
    assert C.PREFILL_DOMAIN == (100, 1560)
    assert C.PREFILL_ALPHA > 0, (
        "measured intercept should be physically sensible (positive); "
        "the earlier synthetic fit had a negative alpha"
    )


def test_cost_model_predictions_are_nonnegative_across_domain():
    cm = CostModel()
    for n in [1, 32, 100, 500, 1560, 4000]:
        assert cm.predict_t_lm_prefill(n) >= 0.0


# ---------------------------------------------------------------------------
# KV backends: paged must waste meaningfully less than contiguous, but this
# is a comparison of two ASSUMED allocation policies, not measured allocator
# behavior (see kv_manager module docstring).
# ---------------------------------------------------------------------------

def test_paged_backend_wastes_less_than_contiguous():
    cb, pb = ContiguousBackend(), PagedBackend()
    r_c = cb.allocate(1548)
    r_p = pb.allocate(1548)
    assert r_p.fragmentation_pct < r_c.fragmentation_pct
    assert r_p.fragmentation_pct < 5.0, "paged (exact-fit-to-block) waste should stay small"


def test_kv_cache_size_scales_linearly_with_tokens():
    small = kv_cache_size_mb(100, batch_size=1, quantization_bits=16)
    large = kv_cache_size_mb(1000, batch_size=1, quantization_bits=16)
    assert large == pytest.approx(small * 10, rel=1e-6)


def test_kv_cache_size_quantization_reduces_footprint():
    fp16 = kv_cache_size_mb(1548, batch_size=1, quantization_bits=16)
    w4 = kv_cache_size_mb(1548, batch_size=1, quantization_bits=4)
    assert w4 == pytest.approx(fp16 / 4, rel=1e-6)


# ---------------------------------------------------------------------------
# W4A8 quantizer: the double-counted "2.5x+ gain vs FP16" bug (analyzing an
# already-4-bit baseline as if it were FP16) meant the correct comparison
# vs the actual shipped W4A16 baseline is a much smaller, honest gain.
# ---------------------------------------------------------------------------

def test_w4a8_gain_vs_w4a16_baseline_is_modest():
    an = W4A8Analyzer()
    result = an.analyze(SCHEMES["w4a8_gar"])
    # The shipped checkpoint IS w4a16; comparing w4a8 to it should show a
    # modest gain (activation quantization only), not the >2x figure that
    # came from double-counting the weight compression the baseline already had.
    assert 0.9 < result.tbt_gain_vs_w4a16 < 1.5, (
        f"got {result.tbt_gain_vs_w4a16}x — if this changed materially, check "
        f"for the double-count bug (comparing against a synthetic FP16 "
        f"baseline instead of the real w4a16 checkpoint)"
    )
    assert result.memory_reduction_x > 0


def test_w4a8_bandwidth_utilization_is_physically_plausible():
    an = W4A8Analyzer()
    result = an.analyze(SCHEMES["w4a16"])
    # Under the MEASURED TBT model (~18-25 ms), the shipped W4A16 checkpoint
    # runs at ~60% of the 100 GB/s unified-memory bandwidth budget — neither
    # negligible (as the old, badly-mistimed 217.7 ms baseline implied ~5%
    # utilization) nor impossible (>100% would indicate a modeling error).
    assert 0.0 < result.bw_utilization_pct <= 100.0
    assert result.bw_bound == (result.bw_utilization_pct > 50.0)


# ---------------------------------------------------------------------------
# Vision offloader: freed memory is only real via SSD-backed swap on unified
# memory (see module docstring) — check the model uses SSD bandwidth, not
# the 100 GB/s DRAM figure, and that the swap margin is still positive.
# ---------------------------------------------------------------------------

def test_vision_offload_uses_ssd_bandwidth_not_dram():
    off = VisionOffloader(k_physical_layers=2)
    result = off.simulate()
    assert result.bandwidth_available_GBps < 20.0, (
        "offload swap-in must be modeled at SSD read speed, not the 100 "
        "GB/s unified-memory DRAM bandwidth (freeing layers to \"CPU\" on "
        "unified memory does not free physical memory)"
    )
    assert result.zero_overhead
    assert result.memory_saved_mb > 0


# ---------------------------------------------------------------------------
# Batching / starvation analysis: must run without crashing and produce a
# non-degenerate strategy table (regression guard for the TBT-linear-in-
# batch bug that made batching pointless by construction).
# ---------------------------------------------------------------------------

def test_decode_starvation_analysis_runs_and_is_batch_sensitive():
    star = decode_starvation_analysis(max_batch=8)
    assert "paged_w4a8_gar" in star
    curve = star["paged_w4a8_gar"].tbt_curve
    assert len(curve) >= 2
    assert curve[-1] > curve[0], (
        "TBT should increase with batch size (more concurrent KV reads); "
        "a flat curve indicates batch size has no effect on decode cost"
    )


def test_batch_expansion_summary_strategies_are_well_formed():
    exp = batch_expansion_summary()
    labels = [r.strategy for r in exp]
    assert len(labels) == len(set(labels)), "strategy names must be unique"
    assert len(exp) >= 4


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", str(Path(__file__)), "-v"]))
