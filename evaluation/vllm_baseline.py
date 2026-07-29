#!/usr/bin/env python3
"""
Extension 1 — vLLM / CUDA Baseline Comparison
===============================================
Explores how the same crop-count endpoints would project to CUDA hardware.

We cannot run vLLM on Apple M3 (no CUDA). Instead, we build an analytical
model of A100 SXM4 40 GB performance using:
  1. Roofline scaling from measured M3 numbers (hardware ratio).
  2. The corrected measured M3 stage sums at 17 crops and 1 crop.

This is a modeled projection, not an A100/vLLM measurement. Because both M3
endpoints are divided by the same assumed hardware ratio, preserving their
speedup on A100 is algebraic and cannot establish hardware independence.

Output
------
  figures/fig5_vllm_comparison.png
  Prints comparison table to stdout.
  Appends Extension 1 section to FINAL_REPORT.md (if --write-report passed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import amio_constants as C
from simulation.kv_manager import ContiguousBackend, PagedBackend

# ── Output paths ──────────────────────────────────────────────────────────────
FIG_PATH    = _ROOT / "figures" / "fig5_vllm_comparison.png"
REPORT_PATH = _ROOT / "FINAL_REPORT.md"

# =============================================================================
# Hardware constants
# =============================================================================

# ── Apple M3 (measured baseline) ─────────────────────────────────────────────
M3_COMPUTE_TFLOPS      = 3.6      # FP16 GPU
M3_BANDWIDTH_GBps      = 100.0    # unified memory
M3_MEMORY_GB           = 8.0

# ── NVIDIA A100 SXM4 40 GB (analytical reference) ─────────────────────────────
A100_COMPUTE_TFLOPS    = 312.0    # FP16 tensor core peak
A100_BANDWIDTH_GBps    = 1_555.0  # HBM2e
A100_MEMORY_GB         = 40.0

COMPUTE_RATIO   = A100_COMPUTE_TFLOPS / M3_COMPUTE_TFLOPS    # 86.7×
BANDWIDTH_RATIO = A100_BANDWIDTH_GBps / M3_BANDWIDTH_GBps     # 15.55×

# =============================================================================
# Corrected SmolVLM cost model inputs
# =============================================================================

CROPS_BASELINE          = C.MAX_CROPS
CROPS_AMIO              = 1
TOKENS_BASELINE         = C.TOKENS_PER_CONFIG[CROPS_BASELINE]
TOKENS_AMIO             = C.TOKENS_PER_CONFIG[CROPS_AMIO]
KV_FP16_BYTES_PER_TOKEN = C.KV_BYTES_PER_TOKEN_FP16
KV_W4_BYTES_PER_TOKEN   = C.KV_BYTES_PER_TOKEN_W4
MODEL_WEIGHTS_GB_W4     = C.MODEL_WEIGHTS_MB / 1000.0


# =============================================================================
# Analytical performance projections
# =============================================================================

def project_compute(m3_ms: float) -> float:
    """Scale a compute-bound latency from M3 to A100."""
    return m3_ms / COMPUTE_RATIO


def project_bandwidth(m3_ms: float) -> float:
    """Scale a bandwidth-bound latency from M3 to A100."""
    return m3_ms / BANDWIDTH_RATIO


def max_concurrent_seqs(memory_gb: float, weights_gb: float,
                         kv_bytes_per_token: int,
                         avg_seq_len: int = TOKENS_BASELINE) -> int:
    """
    Maximum concurrent sequences fitting in memory.

    available_kv_bytes = (memory_gb - weights_gb) × 1e9
    max_seqs = available_kv_bytes / (avg_seq_len × kv_bytes_per_token)
    """
    available = (memory_gb - weights_gb) * 1e9
    return int(available / (avg_seq_len * kv_bytes_per_token))


def tbt_ms(b: int, tbt_b1: float, bw_cost_per_seq: float) -> float:
    """TBT(B) = tbt_b1 + (bw_cost_per_seq × (B - 1))  [linear bandwidth model]."""
    return tbt_b1 + bw_cost_per_seq * (b - 1)


def throughput_tok_per_s(b: int, tbt: float) -> float:
    return (b / tbt) * 1_000.0   # tbt in ms → tok/s


# =============================================================================
# Build the comparison table
# =============================================================================

def build_comparison() -> dict:
    """
    Returns a dict of {system_label: {metric: value}} for 4 systems:
      1. M3 MLX max-crop — measured 17-crop stage sum, modeled contiguous KV
      2. M3 MLX min-crop — measured 1-crop stage sum, modeled paged W4 KV
      3. A100 max-crop projection — M3 endpoint scaled by peak FLOPS ratio
      4. A100 min-crop projection — same scaling assumption

    The two M3 TTFT values are measurements. Fragmentation/capacity and every
    A100 value are analytical policy projections.
    """
    # ── 1. M3 maximum-crop measured endpoint ─────────────────────────────────
    m3_naive_ttft_ms    = C.TTFT_MS_MEASURED_17_CROP
    m3_naive_tbt_ms     = (
        C.DECODE_OVERHEAD_MS_MEASURED
        + C.DECODE_KV_MS_PER_CTX_TOKEN * TOKENS_BASELINE
    )
    m3_naive_kv_frag    = ContiguousBackend().allocate(
        TOKENS_BASELINE
    ).fragmentation_pct
    m3_naive_max_seqs   = max_concurrent_seqs(
        M3_MEMORY_GB, MODEL_WEIGHTS_GB_W4, KV_FP16_BYTES_PER_TOKEN)
    m3_naive_thru       = throughput_tok_per_s(1, m3_naive_tbt_ms)

    # ── 2. M3 minimum-crop measured endpoint ─────────────────────────────────
    # This is the measured 1-crop stage sum, not a measurement of the service.
    m3_amio_ttft_ms     = C.TTFT_MS_MEASURED_1_CROP
    m3_amio_tbt_ms      = (
        C.DECODE_OVERHEAD_MS_MEASURED
        + C.DECODE_KV_MS_PER_CTX_TOKEN * TOKENS_AMIO
    )
    m3_amio_kv_frag     = PagedBackend().allocate(
        TOKENS_AMIO
    ).fragmentation_pct
    m3_amio_max_seqs    = max_concurrent_seqs(
        M3_MEMORY_GB, MODEL_WEIGHTS_GB_W4, KV_W4_BYTES_PER_TOKEN)
    m3_amio_thru        = throughput_tok_per_s(1, m3_amio_tbt_ms)

    # ── 3. A100/vLLM maximum-crop modeled projection ─────────────────────────
    a100_naive_ttft_ms    = project_compute(m3_naive_ttft_ms)
    a100_naive_tbt_ms     = project_bandwidth(m3_naive_tbt_ms)
    a100_naive_kv_frag    = PagedBackend().allocate(
        TOKENS_BASELINE
    ).fragmentation_pct
    a100_naive_max_seqs   = max_concurrent_seqs(
        A100_MEMORY_GB, MODEL_WEIGHTS_GB_W4, KV_FP16_BYTES_PER_TOKEN)
    a100_naive_thru       = throughput_tok_per_s(1, a100_naive_tbt_ms)

    # ── 4. A100/vLLM minimum-crop modeled projection ─────────────────────────
    a100_amio_ttft_ms    = project_compute(m3_amio_ttft_ms)
    a100_amio_tbt_ms     = project_bandwidth(m3_amio_tbt_ms)
    a100_amio_kv_frag    = PagedBackend().allocate(TOKENS_AMIO).fragmentation_pct
    a100_amio_max_seqs   = max_concurrent_seqs(
        A100_MEMORY_GB, MODEL_WEIGHTS_GB_W4, KV_W4_BYTES_PER_TOKEN)
    a100_amio_thru       = throughput_tok_per_s(1, a100_amio_tbt_ms)

    # ── Crop-endpoint ratio (A100 equality is by construction) ────────────────
    m3_algo_speedup   = m3_naive_ttft_ms / m3_amio_ttft_ms
    # Equal by construction: both endpoints use the same compute scaling.
    a100_algo_speedup = a100_naive_ttft_ms / a100_amio_ttft_ms

    results = {
        "M3 MLX\nNaive": {
            "ttft_ms":       m3_naive_ttft_ms,
            "tbt_ms":        m3_naive_tbt_ms,
            "kv_frag_pct":   m3_naive_kv_frag,
            "max_seqs":      m3_naive_max_seqs,
            "throughput":    m3_naive_thru,
            "algo_speedup":  1.0,
            "hw":            "M3 measured",
            "provenance":    "MEASURED TTFT; MODELED memory policy",
            "color":         "#d62728",
        },
        "M3 MLX\nAMIO": {
            "ttft_ms":       m3_amio_ttft_ms,
            "tbt_ms":        m3_amio_tbt_ms,
            "kv_frag_pct":   m3_amio_kv_frag,
            "max_seqs":      m3_amio_max_seqs,
            "throughput":    m3_amio_thru,
            "algo_speedup":  m3_algo_speedup,
            "hw":            "M3 measured",
            "provenance":    "MEASURED TTFT; MODELED memory policy",
            "color":         "#2ca02c",
        },
        "A100 vLLM\nNaive": {
            "ttft_ms":       a100_naive_ttft_ms,
            "tbt_ms":        a100_naive_tbt_ms,
            "kv_frag_pct":   a100_naive_kv_frag,
            "max_seqs":      a100_naive_max_seqs,
            "throughput":    a100_naive_thru,
            "algo_speedup":  1.0,
            "hw":            "A100 SXM4 projection",
            "provenance":    "MODELED projection",
            "color":         "#ff7f0e",
        },
        "A100 vLLM\n+AMIO": {
            "ttft_ms":       a100_amio_ttft_ms,
            "tbt_ms":        a100_amio_tbt_ms,
            "kv_frag_pct":   a100_amio_kv_frag,
            "max_seqs":      a100_amio_max_seqs,
            "throughput":    a100_amio_thru,
            "algo_speedup":  a100_algo_speedup,
            "hw":            "A100 SXM4 projection",
            "provenance":    "MODELED projection",
            "color":         "#1f77b4",
        },
    }
    return results


# =============================================================================
# Figure
# =============================================================================

def make_figure(results: dict) -> None:
    labels = list(results.keys())
    colors = [r["color"] for r in results.values()]

    ttft       = [r["ttft_ms"]      for r in results.values()]
    algo_gain  = [r["algo_speedup"] for r in results.values()]
    max_seqs   = [r["max_seqs"]     for r in results.values()]
    kv_frag    = [r["kv_frag_pct"]  for r in results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Extension 1 — M3 Measurements and A100/vLLM Modeled Projection",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Panel 1: TTFT (log scale) ────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(labels, ttft, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("TTFT  (ms, log scale)", fontsize=10)
    ax.set_title("Time-to-First-Token", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.1f}"))
    for bar, val in zip(bars, ttft):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                f"{val:,.1f} ms",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylim(1, max(ttft) * 6)
    # Annotate hardware speedup
    ax.annotate("", xy=(0.5, ttft[2]), xytext=(0.5, ttft[0]),
                arrowprops=dict(arrowstyle="<->", color="grey", lw=1.2))
    ax.text(0.72, (ttft[0] * ttft[2])**0.5,
            f"Hardware\n{COMPUTE_RATIO:.0f}× faster",
            fontsize=7.5, color="grey", va="center")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: Algorithm speedup (hardware-normalized) ─────────────────────
    ax = axes[1]
    pair_labels = ["M3 endpoints\n(measured)", "A100 endpoints\n(modeled)"]
    pair_speedups = [
        results["M3 MLX\nAMIO"]["algo_speedup"],
        results["A100 vLLM\n+AMIO"]["algo_speedup"],
    ]
    pair_colors = ["#2ca02c", "#1f77b4"]
    bars2 = ax.bar(pair_labels, pair_speedups, color=pair_colors,
                   edgecolor="black", linewidth=0.6, width=0.45)
    for bar, val in zip(bars2, pair_speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.1f}×",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_ylabel("Algorithm speedup  (vs same-hardware naive)", fontsize=10)
    ax.set_title("Crop-endpoint ratio\n(preserved by construction)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(pair_speedups) * 1.25)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    # The equality is algebraic because both endpoints use the same scaling.
    diff_pct = abs(pair_speedups[0] - pair_speedups[1]) / pair_speedups[0] * 100
    ax.text(0.5, 0.06, f"Difference: {diff_pct:.1f}% — equal by model construction",
            transform=ax.transAxes, ha="center", fontsize=8,
            color="#444", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbe6", edgecolor="#ccc"))

    # ── Panel 3: KV memory capacity (max concurrent seqs) ────────────────────
    ax = axes[2]
    bars3 = ax.bar(labels, max_seqs, color=colors, edgecolor="black",
                   linewidth=0.6, width=0.55)
    for bar, val in zip(bars3, max_seqs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:,}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Max concurrent sequences (KV budget)", fontsize=10)
    ax.set_title("KV Memory Capacity\n(W4 = 4× more room)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Annotate W4 gain
    ax.annotate("W4 quantisation\ngives 4× capacity",
                xy=(1, max_seqs[1]), xytext=(0.5, max_seqs[1] * 0.85),
                fontsize=7.5, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.1))

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED]  {FIG_PATH}")


# =============================================================================
# Console table
# =============================================================================

def print_table(results: dict) -> None:
    header = (
        f"{'System':<22} {'TTFT (ms)':>11} {'Algo ×':>8} "
        f"{'TBT (ms)':>10} {'KV Frag%':>10} {'Max Seqs':>10}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for label, r in results.items():
        flat = label.replace("\n", " ")
        print(
            f"{flat:<22} {r['ttft_ms']:>11.1f} {r['algo_speedup']:>8.1f}× "
            f"{r['tbt_ms']:>10.1f} {r['kv_frag_pct']:>9.1f}% {r['max_seqs']:>10,}"
        )
    print(sep)
    m3_algo  = results["M3 MLX\nAMIO"]["algo_speedup"]
    a100_algo = results["A100 vLLM\n+AMIO"]["algo_speedup"]
    diff_pct = abs(m3_algo - a100_algo) / m3_algo * 100
    print(f"\nCrop-endpoint ratio: M3={m3_algo:.1f}×   A100={a100_algo:.1f}×")
    print("A100 equality is by construction (same scaling factor), not validation "
          f"of hardware portability; difference={diff_pct:.1f}%.\n")


# =============================================================================
# Report section
# =============================================================================

def build_report_section(results: dict) -> str:
    """Build report markdown from build_comparison(); no duplicated literals."""
    labels = list(results)
    rows = [results[label] for label in labels]
    m3_ratio = rows[0]["ttft_ms"] / rows[1]["ttft_ms"]
    a100_ratio = rows[2]["ttft_ms"] / rows[3]["ttft_ms"]

    lines = [
        "",
        "---",
        "## Extension 1 — PyTorch + vLLM Modeled Projection",
        "",
        "> **Provenance:** the two M3 TTFT endpoints are MEASURED stage sums. "
        "All A100/vLLM numbers, memory capacities, and fragmentation values are "
        "MODELED projections; no A100 or vLLM benchmark was run.",
        "",
        "### Methodology",
        "",
        "The projection divides compute-bound M3 TTFT by the peak-FLOPS ratio "
        f"({COMPUTE_RATIO:.1f}x) and batch-1 TBT by the bandwidth ratio "
        f"({BANDWIDTH_RATIO:.2f}x). This is a deliberately simple roofline "
        "projection, not a prediction calibrated on CUDA kernels. The two crop "
        "endpoints use the same scaling factor, so their A100 ratio is preserved "
        "algebraically and cannot validate hardware independence.",
        "",
        "### Results",
        "",
        "| Metric | M3 max crop | M3 min crop | A100 max crop | A100 min crop |",
        "|--------|-------------|-------------|---------------|---------------|",
        "| Provenance | MEASURED TTFT | MEASURED TTFT | MODELED projection | MODELED projection |",
        "| Crops | {} | {} | {} | {} |".format(
            CROPS_BASELINE, CROPS_AMIO, CROPS_BASELINE, CROPS_AMIO
        ),
        "| TTFT stage sum (ms) | {:,.1f} | {:,.1f} | {:,.1f} | {:,.1f} |".format(
            *(r["ttft_ms"] for r in rows)
        ),
        "| Crop-endpoint ratio | 1x | {:.1f}x | 1x | {:.1f}x |".format(
            m3_ratio, a100_ratio
        ),
        "| TBT B=1 (ms) | {:.1f} | {:.1f} | {:.1f} | {:.1f} |".format(
            *(r["tbt_ms"] for r in rows)
        ),
        "| Allocator-policy waste | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% |".format(
            *(r["kv_frag_pct"] for r in rows)
        ),
        "| Modeled max sequences | {:,} | {:,} | {:,} | {:,} |".format(
            *(r["max_seqs"] for r in rows)
        ),
        "",
        "The M3 range is a crop/fidelity tradeoff, not a same-work acceleration: "
        "the 1-crop configuration processes far less visual input. The 500 ms SLA "
        "remains infeasible on M3 because the measured 1-crop stage sum is "
        f"{rows[1]['ttft_ms']:.0f} ms.",
        "",
        f"KV sizes are {KV_FP16_BYTES_PER_TOKEN:,} B/token (FP16) and "
        f"{KV_W4_BYTES_PER_TOKEN:,} B/token (modeled W4). Fragmentation is generated "
        "by the repository's assumed contiguous/paged policies, not measured from "
        "MLX or vLLM.",
        "",
        "### Figure",
        "",
        "![](figures/fig5_vllm_comparison.png)",
        "",
        "*Analysis generated by `evaluation/vllm_baseline.py`.*",
        "",
    ]
    return "\n".join(lines)


def append_to_report(results: dict) -> None:
    report_section = build_report_section(results)
    text = REPORT_PATH.read_text()
    if "Extension 1" in text:
        # Replace existing section
        start = text.find("\n---\n## Extension 1")
        if start != -1:
            text = text[:start] + report_section
            REPORT_PATH.write_text(text)
            print(f"[UPDATED] {REPORT_PATH}  (replaced existing Extension 1 section)")
            return
    REPORT_PATH.write_text(text.rstrip() + "\n" + report_section)
    print(f"[APPENDED] {REPORT_PATH}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    results = build_comparison()
    print_table(results)
    make_figure(results)
    append_to_report(results)


if __name__ == "__main__":
    main()
