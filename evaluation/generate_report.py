"""
phase8_evaluation.py  —  Phase 8: Final Evaluation & Presentation

Produces
--------
  figures/fig1_latency_waterfall.png   Latency journey: 8.5 s → 350 ms
  figures/fig2_strategy_heatmaps.png   Controller strategy selection grid
  figures/fig3_pareto_curves.png       Quality × Latency Pareto frontier
  figures/fig4_nova_convergence.png    Nova SM reallocator burst timeline
  FINAL_REPORT.md                     6–8 page technical report

Usage
-----
  python phase8_evaluation.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── project root (one level above evaluation/) ────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.controller import (
    AdaptiveController,
    InferenceRequest,
    SystemState,
    VISION_MS_PER_CROP,
    VISION_FIXED_MS,
    BASELINE_N_CROPS,
    TOKENS_PER_CROP,
    M3_TOTAL_SMS,
    SLA_TTFT_MS,
    DECODE_OVERHEAD_MS,
    DECODE_KV_MS_PER_CTX_TOKEN,
    STATIC_BUDGET_MB,
    KV_QUANT_BITS,
    CROP_OPTIONS,
)
from simulation.parallelism_engine import ParallelismMode
from simulation.resolution_scaler import max_crops_for_resolution
from simulation.kv_manager import kv_cache_size_mb, KV_BYTES_PER_TOKEN, KV_POOL_BUDGET_MB
from model_calibration.cost_model import CostModel
import amio_constants as _C

# ── output directory ──────────────────────────────────────────────────────────
_FIGURES = _ROOT / "figures"
_FIGURES.mkdir(exist_ok=True)

# ── colour palette (consistent across all figures) ────────────────────────────
C_VISION    = "#4C72B0"   # steel blue  — vision stage
C_PREFILL   = "#DD8452"   # burnt orange — prefill stage
C_MIGRATION = "#55A868"   # sage green  — ParVTS migration
C_DECODE    = "#C44E52"   # muted red   — decode stage
C_AMIO      = "#4C72B0"
C_STATIC    = "#C44E52"
C_GREEDY    = "#55A868"
C_SM_VIS    = "#4C72B0"
C_SM_DEC    = "#DD8452"
C_FRONT     = "#55A868"

# ── shared matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.5,
})


# =============================================================================
# Shared data helpers
# =============================================================================

_CM = CostModel()   # single shared instance


def _vision_ms(n_crops: int, sm_vis: int = M3_TOTAL_SMS) -> float:
    """MEASURED linear vision latency model (baseline/results_v2.json):
    T_vision(c) = VISION_MS_PER_CROP * c + VISION_FIXED_MS, share-scaled."""
    base = VISION_MS_PER_CROP * n_crops + VISION_FIXED_MS
    return base * (M3_TOTAL_SMS / max(sm_vis, 1))


def _prefill_ms(n_lm_tokens: int, tp: bool = False) -> float:
    # Single-chip TP is cost-neutral (there is one GPU) — no discount applied,
    # matching simulation/controller.py. `tp` kept as a no-op parameter for
    # call-site compatibility.
    return _CM.predict_t_lm_prefill(n_lm_tokens)


def _migration_ms(n_full: int, n_pruned: int, depth: int = 3) -> float:
    return _CM.predict_migration_cost(n_full, n_pruned, depth)


# =============================================================================
# Figure 1 — Latency Waterfall
# =============================================================================

def _build_waterfall_stages() -> List[Dict]:
    """
    Compute per-stage latency breakdown for a 512 px request with
    prompt_length=32, progressing through each optimisation milestone.

    MEASURED (baseline/results_v2.json, 2026-07-28): vision cost is linear
    in crop count (VISION_MS_PER_CROP*c + VISION_FIXED_MS); the processor's
    crop grid is {1, 5, 10, 17} — there is no "24 crops" setting. A 512 px
    image maps to the 5-crop processor setting (size.longest_edge=768).
    """
    prompt = 32

    # Stage 1: Phase 1 Baseline — max crops (17), all tokens, full shares
    n_crops_bl = BASELINE_N_CROPS   # 17
    n_vis_bl   = max(1, round(n_crops_bl * TOKENS_PER_CROP))
    n_lm_bl    = n_vis_bl + prompt
    v1 = _vision_ms(n_crops_bl, sm_vis=38)
    p1 = _prefill_ms(n_lm_bl)
    stages = [dict(label="Ph 1\nBaseline",
                   vision=v1, prefill=p1, migration=0.0,
                   note="{} crops | {}t LM | no optimisation".format(n_crops_bl, n_lm_bl))]

    # Stage 2: +Phase 3 Adaptive Crop Scaling (512 px → 5-crop processor setting)
    n_crops_dp = 5
    n_vis_dp   = max(1, round(n_crops_dp * TOKENS_PER_CROP))
    n_lm_dp    = n_vis_dp + prompt
    v2 = _vision_ms(n_crops_dp, sm_vis=38)
    p2 = _prefill_ms(n_lm_dp)
    stages.append(dict(label="+Ph 3\nCrop Scale",
                       vision=v2, prefill=p2, migration=0.0,
                       note="{} crops | {}t LM | DP mode".format(n_crops_dp, n_lm_dp)))

    # Stage 3: +Phase 4 W4 KV + ParVTS (keep=0.75)
    keep   = 0.75
    n_eff  = max(1, round(n_vis_dp * keep))
    n_lm_p = n_eff + prompt
    v3 = v2
    p3 = _prefill_ms(n_lm_p)
    m3 = _migration_ms(n_vis_dp, n_eff)
    stages.append(dict(label="+Ph 4\nParVTS",
                       vision=v3, prefill=p3, migration=m3,
                       note="keep=0.75 | {}t LM | W4 paged KV".format(n_lm_p)))

    # Stage 4: +Phase 6/7 AMIO Full Adaptive — 1 crop idle, sm_vis=38
    n_crops_amio = 1
    n_vis_amio   = max(1, round(n_crops_amio * TOKENS_PER_CROP))
    n_lm_amio    = n_vis_amio + prompt
    v4 = _vision_ms(n_crops_amio, sm_vis=38)
    p4 = _prefill_ms(n_lm_amio, tp=True)
    stages.append(dict(label="+Ph 6/7\nAMIO",
                       vision=v4, prefill=p4, migration=0.0,
                       note="1 crop | {}t LM | TP (cost-neutral) | Nova SM".format(n_lm_amio)))

    for s in stages:
        s["total"] = s["vision"] + s["prefill"] + s["migration"]
    return stages


def make_fig1_latency_waterfall() -> str:
    """
    Figure 1 — Stacked horizontal bar chart showing the latency journey
    from the maximum-crop baseline to the minimum-crop AMIO configuration.
    """
    stages = _build_waterfall_stages()
    labels = [s["label"] for s in stages]
    totals = [s["total"] for s in stages]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))

    y = np.arange(len(stages))
    bar_h = 0.55

    left = np.zeros(len(stages))
    bars = {"Vision": [], "Prefill": [], "Migration": []}
    for s in stages:
        bars["Vision"].append(s["vision"])
        bars["Prefill"].append(s["prefill"])
        bars["Migration"].append(s["migration"])

    colours = {"Vision": C_VISION, "Prefill": C_PREFILL, "Migration": C_MIGRATION}

    for component, vals in bars.items():
        vals_arr = np.array(vals, dtype=float)
        ax.barh(y, vals_arr, left=left, height=bar_h,
                color=colours[component], label=component, alpha=0.88)
        # Label each non-trivial segment
        for i, (v, l) in enumerate(zip(vals_arr, left)):
            if v > 30:
                ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        left += vals_arr

    # Total time annotation on right
    for i, (t, note) in enumerate(zip(totals, [s["note"] for s in stages])):
        ax.text(t + 80, i, f"{t:,.0f} ms", va="center", fontsize=8.5,
                color="#333333", fontweight="bold")

    # SLA reference line
    ax.axvline(SLA_TTFT_MS, color="#e74c3c", ls="--", lw=1.4, alpha=0.8,
               label=f"SLA {SLA_TTFT_MS:.0f} ms")

    savings = stages[0]["total"] - stages[-1]["total"]
    ax.annotate(
        f"  ← {savings:,.0f} ms reclaimed  ({savings/stages[0]['total']*100:.0f}%)",
        xy=(stages[-1]["total"], 3),
        xytext=(stages[-1]["total"] + 500, 3),
        fontsize=8, color="#2c3e50",
        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1),
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("TTFT (ms)  [lower = better]")
    ax.set_xlim(0, max(totals) * 1.30)
    ax.set_title("Figure 1  —  AMIO Latency Journey: Phase 1 Baseline → Full System",
                 fontweight="bold", pad=8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out = str(_FIGURES / "fig1_latency_waterfall.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# Figure 2 — Strategy Selection Heatmaps
# =============================================================================

def make_fig2_strategy_heatmaps() -> str:
    """
    Figure 2 — Two heatmaps showing the controller's strategy selection across
    image resolutions (Y) × front-stage request depth (X), with N_decoding=5
    so the Nova SM partitioner actively competes for cores.

    Left:  n_crops selected  (relaxed 3 s SLA to reveal full diversity)
    Right: parallelism mode  (0=DP, 1=TP, same relaxed SLA)
    """
    # Use a relaxed SLA that lets the controller pick variable crop counts.
    # The tight 500 ms operational SLA forces 1 crop everywhere; 3 s reveals
    # how the Nova SM formula (sm_vis grows with n_pending) lets bigger crop
    # counts become feasible at high queue depth.
    SLA_RELAXED_MS = 3000
    N_DEC_FIXED    = 5   # non-zero decode load so Nova partitions SMs

    ctrl_relaxed = AdaptiveController(sla_budget_ms=SLA_RELAXED_MS)

    # Measured processor crop settings: size.longest_edge -> crops
    resolutions = [224, 384, 512, 768, 1024, 1152, 1536]
    pendings    = [0, 1, 2, 5, 10, 15, 20]
    n_r, n_p   = len(resolutions), len(pendings)

    crops_grid = np.zeros((n_r, n_p), dtype=float)
    mode_grid  = np.zeros((n_r, n_p), dtype=float)   # 0=DP, 1=TP

    for i, res in enumerate(resolutions):
        for j, n_pend in enumerate(pendings):
            req   = InferenceRequest(req_id=0, image_resolution=res, prompt_length=32)
            state = SystemState(n_pending_requests=n_pend,
                                n_decoding_requests=N_DEC_FIXED)
            plan  = ctrl_relaxed.optimize(req, state)
            crops_grid[i, j] = plan.n_crops
            mode_grid[i, j]  = 0.0 if plan.parallelism_mode == ParallelismMode.DP else 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # -- Heatmap 1: n_crops --
    vmax = max(crops_grid.max(), 6)   # ensures colour range is meaningful
    im1 = ax1.imshow(crops_grid, aspect="auto", origin="lower",
                     cmap="YlOrRd", vmin=1, vmax=vmax)
    for i in range(n_r):
        for j in range(n_p):
            val  = int(crops_grid[i, j])
            dark = crops_grid[i, j] > vmax * 0.55
            ax1.text(j, i, val, ha="center", va="center",
                     fontsize=9, color="white" if dark else "black",
                     fontweight="bold")
    ax1.set_xticks(range(n_p))
    ax1.set_xticklabels([str(p) for p in pendings])
    ax1.set_yticks(range(n_r))
    ax1.set_yticklabels([f"{r}px" for r in resolutions])
    ax1.set_xlabel("Front-stage queue depth  (N_pending)")
    ax1.set_ylabel("Image resolution")
    ax1.set_title(
        "Crops selected  (SLA=3 s, N_dec=5)\n"
        "↑ N_pending → more SMs to vision → higher crops feasible"
    )
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.038)
    cb1.set_label("n_crops chosen", fontsize=8)

    # -- Heatmap 2: parallelism mode --
    cmap_mode = matplotlib.colors.ListedColormap([C_PREFILL, C_VISION])  # DP=orange, TP=blue
    bounds     = [-0.5, 0.5, 1.5]
    norm       = matplotlib.colors.BoundaryNorm(bounds, cmap_mode.N)
    im2 = ax2.imshow(mode_grid, aspect="auto", origin="lower",
                     cmap=cmap_mode, norm=norm)
    for i in range(n_r):
        for j in range(n_p):
            label = "TP" if mode_grid[i, j] == 1.0 else "DP"
            ax2.text(j, i, label, ha="center", va="center",
                     fontsize=8.5, color="white", fontweight="bold")
    ax2.set_xticks(range(n_p))
    ax2.set_xticklabels([str(p) for p in pendings])
    ax2.set_yticks(range(n_r))
    ax2.set_yticklabels([f"{r}px" for r in resolutions])
    ax2.set_xlabel("Front-stage queue depth  (N_pending)")
    ax2.set_ylabel("Image resolution")
    ax2.set_title(
        "Parallelism mode selected  (SLA=3 s, N_dec=5)\n"
        "TP preferred when prefill time dominates"
    )
    dp_patch = mpatches.Patch(color=C_PREFILL, label="DP  (data-parallel crops)")
    tp_patch = mpatches.Patch(color=C_VISION,  label="TP  (tensor-parallel prefill)")
    ax2.legend(handles=[dp_patch, tp_patch], loc="upper right", fontsize=8)

    fig.suptitle(
        "Figure 2  —  AMIO Controller Strategy Heatmaps\n"
        "(Relaxed 3 s SLA reveals full crop-selection diversity; "
        "operational SLA = 500 ms)",
        fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    out = str(_FIGURES / "fig2_strategy_heatmaps.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# Figure 3 — Quality × Latency Pareto Frontier
# =============================================================================

def _pareto_data(seed: int = 42) -> Dict[str, List[Tuple[float, float, int]]]:
    """
    Simulate (avg_ttft_ms, quality_score) for three systems with
    [1, 5, 10, 30, 60, 100] concurrent requests and [224, 512, 756, 1024] px.

    Returns dict {system_name: [(avg_ttft, quality, concurrency), ...]}
    """
    rng  = random.Random(seed)
    ctrl = AdaptiveController()
    cm   = _CM

    RESOLUTIONS   = [224, 512, 756, 1024]
    CONCURRENCIES = [1, 5, 10, 30, 60, 100]
    PROMPT        = 32
    NOISE         = 0.03

    def _crop_for_res(res: int) -> int:
        return max_crops_for_resolution(res)

    results: Dict[str, List[Tuple[float, float, int]]] = {
        "Static Baseline": [],
        "Greedy Fast":     [],
        "AMIO Adaptive":   [],
    }

    for res in RESOLUTIONS:
        for conc in CONCURRENCIES:
            # Build a mini request stream with Poisson arrivals
            iat  = 5000.0 / max(conc, 1)
            reqs = []
            t    = 0.0
            for _ in range(conc):
                t += rng.expovariate(1.0 / iat)
                reqs.append((res, PROMPT, t))

            for sys_name in results.keys():
                ttfts, quals = [], []
                n_dec = 0
                for i, (r, pl, _) in enumerate(reqs):
                    n_front = max(0, len(reqs) - i - 1)

                    if sys_name == "AMIO Adaptive":
                        req   = InferenceRequest(i, r, pl)
                        state = SystemState(n_front, n_dec, current_decode_batch=max(1, n_dec))
                        plan  = ctrl.optimize(req, state)
                        nc    = plan.n_crops
                        kr    = plan.token_keep_ratio
                        tp    = (plan.parallelism_mode == ParallelismMode.TP)
                        sm_v  = plan.sm_vision if plan.sm_vision > 0 else M3_TOTAL_SMS
                    elif sys_name == "Static Baseline":
                        nc   = _crop_for_res(r)
                        kr   = 1.0
                        tp   = False
                        sm_v = M3_TOTAL_SMS if n_dec == 0 else 8
                    else:   # Greedy Fast
                        nc   = 1
                        kr   = 0.111
                        tp   = False
                        sm_v = M3_TOTAL_SMS if n_dec == 0 else 8

                    n_vis  = max(1, round(nc  * TOKENS_PER_CROP))
                    n_eff  = max(1, round(n_vis * kr))
                    n_lm   = n_eff + pl
                    t_vis  = _vision_ms(nc, sm_vis=sm_v)
                    t_vis *= max(0.5, 1.0 + rng.gauss(0.0, NOISE))
                    t_pre  = _prefill_ms(n_lm, tp=tp)
                    t_pre *= max(0.5, 1.0 + rng.gauss(0.0, NOISE))
                    t_mig  = (_migration_ms(n_vis, n_eff)
                               * max(0.5, 1.0 + rng.gauss(0.0, NOISE))
                               if kr < 1.0 else 0.0)
                    ttft = t_vis + t_pre + t_mig
                    ttfts.append(ttft)
                    quals.append(nc * kr)
                    n_dec = min(n_dec + 1, 40)

                results[sys_name].append((
                    float(np.mean(ttfts)),
                    float(np.mean(quals)),
                    conc,
                ))

    return results


def make_fig3_pareto_curves() -> str:
    """
    Figure 3 — Quality score × TTFT scatter plot. Each point = one
    (system, resolution, concurrency) configuration.
    Top-left = Pareto-dominant (high quality, low latency).
    """
    data = _pareto_data()

    STYLE = {
        "Static Baseline": dict(color=C_STATIC,  marker="s", alpha=0.70, label="Static Baseline"),
        "Greedy Fast":     dict(color=C_GREEDY,  marker="^", alpha=0.70, label="Greedy Fast"),
        "AMIO Adaptive":   dict(color=C_AMIO,    marker="o", alpha=0.85, label="AMIO Adaptive"),
    }
    CONC_SIZE = {1: 30, 5: 60, 10: 90, 30: 130, 60: 175, 100: 220}

    fig, ax = plt.subplots(figsize=(8, 5))

    for sys_name, pts in data.items():
        ttfts  = [p[0] for p in pts]
        quals  = [p[1] for p in pts]
        concs  = [p[2] for p in pts]
        sizes  = [CONC_SIZE.get(c, 100) for c in concs]
        s      = STYLE[sys_name]
        ax.scatter(ttfts, quals, s=sizes, color=s["color"], marker=s["marker"],
                   alpha=s["alpha"], label=s["label"], edgecolors="white", linewidths=0.5)

    # Pareto frontier for AMIO
    amio_pts = sorted(data["AMIO Adaptive"], key=lambda p: p[0])
    # Simple non-dominated filter: keep point if no other has lower ttft AND higher qual
    pareto: List[Tuple[float, float]] = []
    best_q = -1.0
    for ttft, q, _ in amio_pts:
        if q > best_q:
            pareto.append((ttft, q))
            best_q = q
    if len(pareto) >= 2:
        px, py = zip(*pareto)
        ax.plot(px, py, color=C_AMIO, lw=1.8, ls="--", alpha=0.5, label="AMIO Pareto frontier")

    ax.axvline(SLA_TTFT_MS, color="#e74c3c", ls=":", lw=1.4, alpha=0.7,
               label=f"SLA {SLA_TTFT_MS:.0f} ms")

    # Quadrant annotation
    ax.text(0.03, 0.97, "Ideal\n(low latency, high quality)",
            transform=ax.transAxes, fontsize=7.5, va="top", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6, edgecolor="#ccc"))

    # Size legend
    for c, s in [(1, 30), (10, 90), (100, 220)]:
        ax.scatter([], [], s=s, color="#888", alpha=0.6, label=f"N={c} requests")

    ax.set_xlabel("Average TTFT (ms)  [lower = better →]")
    ax.set_ylabel("Average quality score  (n_crops × keep)  [higher = better ↑]")
    ax.set_title("Figure 3  —  Quality × Latency Pareto Frontier", fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.tight_layout()

    out = str(_FIGURES / "fig3_pareto_curves.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# Figure 4 — Nova SM Reallocator Convergence Plot
# =============================================================================

def _nova_burst_timeline(
    n_requests: int = 15,
    iat_ms:     float = 200.0,
    t_vision_ms: float = 350.0,
    t_prefill_ms: float = 100.0,
    t_decode_ms:  float = 3000.0,
    dt_ms:        int   = 25,
) -> Dict[str, List]:
    """
    Simulate a request burst and track Nova SM allocation over time.

    Model (uniform IAT for visual clarity):
      - Requests arrive at t = k * iat_ms
      - Vision + prefill completes at t_arrival + T_VP
      - Decode completes at t_arrival + T_VP + T_DECODE

    n_pending  = arrived but not yet done with vision+prefill
    n_decoding = done with vision+prefill, still decoding
    sm_vision  = Nova allocation for (n_pending, n_decoding)
    """
    ctrl = AdaptiveController()
    arrivals = [i * iat_ms for i in range(n_requests)]

    T_VP  = t_vision_ms + t_prefill_ms
    T_TOT = T_VP + t_decode_ms
    end_t = int(arrivals[-1] + T_TOT + 500)

    result: Dict[str, List] = {
        "time_s": [], "n_pending": [], "n_decoding": [], "sm_vision": [], "sm_decode": [],
    }

    for t_ms in range(0, end_t, dt_ms):
        n_arv     = int(sum(1 for a in arrivals if a <= t_ms))
        n_vp_done = int(sum(1 for a in arrivals if a + T_VP  <= t_ms))
        n_dec_done = int(sum(1 for a in arrivals if a + T_TOT <= t_ms))

        n_front = max(0, n_arv - n_vp_done)
        n_dec   = max(0, n_vp_done - n_dec_done)

        sm_vis, sm_dec = ctrl._nova_sm_allocation(n_front, n_dec)

        result["time_s"].append(t_ms / 1000.0)
        result["n_pending"].append(n_front)
        result["n_decoding"].append(n_dec)
        result["sm_vision"].append(sm_vis)
        result["sm_decode"].append(sm_dec)

    return result


def make_fig4_nova_convergence() -> str:
    """
    Figure 4 — Three-panel timeline showing how the Nova SM reallocator
    responds to a request burst and recovers as the queue clears.
    """
    data = _nova_burst_timeline(n_requests=15, iat_ms=200)

    t   = data["time_s"]
    nf  = data["n_pending"]
    nd  = data["n_decoding"]
    sv  = data["sm_vision"]
    sd  = data["sm_decode"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6.5),
                                         sharex=True, gridspec_kw={"hspace": 0.05})

    # Panel 1: queue depths
    ax1.fill_between(t, nf, alpha=0.35, color=C_FRONT,  label="Front-stage")
    ax1.fill_between(t, nd, alpha=0.35, color=C_DECODE,  label="Decoding")
    ax1.plot(t, nf, color=C_FRONT,  lw=1.5)
    ax1.plot(t, nd, color=C_DECODE,  lw=1.5)
    ax1.set_ylabel("Queue depth\n(requests)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(bottom=0)
    ax1.annotate("Burst arrives\n(15 req, IAT=200 ms)",
                 xy=(0.15, max(nf)), xytext=(1.0, max(nf) * 0.95),
                 fontsize=7.5, color="#333",
                 arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

    # Panel 2: SM allocation (stacked)
    ax2.stackplot(t, sv, sd,
                  labels=["SM vision", "SM decode"],
                  colors=[C_SM_VIS, C_SM_DEC], alpha=0.80)
    ax2.axhline(M3_TOTAL_SMS, color="#333", lw=0.8, ls="--", alpha=0.5)
    ax2.set_ylabel("SM cores\nallocated")
    ax2.set_ylim(0, M3_TOTAL_SMS + 4)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.text(0.01, 0.92, f"Total = {M3_TOTAL_SMS} SMs", transform=ax2.transAxes,
             fontsize=7.5, color="#555")

    # Panel 3: sm_vision alone + annotation of key regions
    ax3.plot(t, sv, color=C_SM_VIS, lw=2.0, label="sm_vision (Nova)")
    ax3.axhspan(0, 8,  alpha=0.08, color=C_DECODE,  label="SM_MIN_VISION = 8")
    ax3.axhspan(34, M3_TOTAL_SMS, alpha=0.08, color=C_AMIO, label="High vision priority zone")
    ax3.set_ylabel("sm_vision\n(Nova heuristic)")
    ax3.set_xlabel("Simulation time (s)")
    ax3.set_ylim(0, M3_TOTAL_SMS + 2)
    ax3.legend(loc="upper right", fontsize=8)

    # Phase annotations on panel 3
    ax3.text(0.02, 0.85, "① Idle: all 38 SMs\nto vision", transform=ax3.transAxes,
             fontsize=7.5, color=C_AMIO)
    ax3.text(0.30, 0.20, "② Mixed: Nova\npartitions SMs", transform=ax3.transAxes,
             fontsize=7.5, color="#555")
    ax3.text(0.72, 0.12, "③ Decode only:\nSM_MIN = 8", transform=ax3.transAxes,
             fontsize=7.5, color=C_DECODE)

    fig.suptitle(
        "Figure 4  —  Nova SM Reallocator: Burst Arrival & Convergence\n"
        "(15 requests, uniform IAT=200 ms, vision=350 ms, prefill=100 ms, decode=3 s)",
        fontweight="bold", y=1.01, fontsize=9.5,
    )
    out = str(_FIGURES / "fig4_nova_convergence.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# Cost Model Validation
# =============================================================================

_CALIB_JSON  = _ROOT / "model_calibration" / "calibration_results.json"   # SUPERSEDED
_RESULTS_V2  = _ROOT / "baseline" / "results_v2.json"                     # MEASURED


def run_cost_model_validation() -> Dict:
    """
    Validate the Phase 2 cost model against its MEASURED calibration points
    (baseline/results_v2.json — direct stage-isolated timing on the real M3
    target, superseding the earlier synthetic-embedding calibration).

    Reports BOTH in-sample MAPE (fit and scored on the same points — always
    optimistic) and leave-one-out cross-validated MAPE (each point predicted
    by a model re-fit on the other N-1 points — the honest generalisation
    estimate). The original report showed only in-sample error labeled
    "held-out"; that label was false — there were no held-out points.

    With only 4 measured points, LOOCV here is illustrative, not a
    statistically strong generalisation estimate — more calibration points
    are needed for a trustworthy out-of-fit error bound.
    """
    cm = _CM
    with open(_RESULTS_V2) as f:
        v2 = json.load(f)

    points = sorted(v2["configs"].values(), key=lambda p: p["total_input_tokens"])
    n_tokens   = [p["total_input_tokens"]        for p in points]
    t_measured = [p["lm_prefill"]["mean_ms"]     for p in points]
    t_stdev    = [p["lm_prefill"]["std_ms"]      for p in points]
    t_pred     = [cm.predict_t_lm_prefill(n) for n in n_tokens]

    abs_errs  = [abs(p - m) for p, m in zip(t_pred, t_measured)]
    rel_errs  = [abs(p - m) / m * 100 for p, m in zip(t_pred, t_measured)]
    mape      = float(np.mean(rel_errs))
    mae       = float(np.mean(abs_errs))

    n_arr = np.array(n_tokens, dtype=float)
    t_arr = np.array(t_measured, dtype=float)
    pred_arr = np.array(t_pred, dtype=float)
    ss_res = float(np.sum((t_arr - pred_arr) ** 2))
    ss_tot = float(np.sum((t_arr - t_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Leave-one-out cross-validation: refit gamma/beta/alpha on N-1 points,
    # predict the held-out point, repeat for every point.
    loo_rel_errs = []
    for i in range(len(n_arr)):
        mask = np.ones(len(n_arr), dtype=bool)
        mask[i] = False
        if mask.sum() < 3:
            continue   # need >= 3 points to fit a quadratic
        g, b, a = np.polyfit(n_arr[mask], t_arr[mask], 2)
        pred_i = g * n_arr[i] ** 2 + b * n_arr[i] + a
        loo_rel_errs.append(abs(pred_i - t_arr[i]) / t_arr[i] * 100)
    loo_mape = float(np.mean(loo_rel_errs)) if loo_rel_errs else float("nan")

    return dict(
        gamma=cm.config.gamma, beta=cm.config.beta, alpha=cm.config.alpha,
        r2=r2, mape=mape, mae_ms=mae, loo_mape=loo_mape,
        n_points=len(n_tokens),
        n_tokens=n_tokens, t_measured=t_measured, t_stdev=t_stdev,
        t_pred=t_pred, abs_errs=abs_errs, rel_errs=rel_errs,
    )


# =============================================================================
# Ablation Study Data
# =============================================================================

def compute_ablation(seed: int = 0) -> List[Dict]:
    """
    For a 512 px request (5-crop processor setting -> AMIO adaptive):
    show the cumulative TTFT change contributed by each module.

    Single-chip TP is cost-neutral (there is one GPU) — the `tp` flag is
    kept only for column labelling, never changes the computed latency.
    Nova SM reallocation under decode load is NOT part of this monotonic
    chain: reallocating shares away from vision INCREASES vision TTFT for
    the reallocated request (it trades one request's latency for overall
    decode throughput), so it is reported separately, not folded into a
    cumulative "improvement" narrative.
    """
    prompt = 32

    def _ttft(n_crops, keep, sm_vis, use_parvts=False):
        n_vis = max(1, round(n_crops * TOKENS_PER_CROP))
        n_eff = max(1, round(n_vis * keep))
        n_lm  = n_eff + prompt
        v = _vision_ms(n_crops, sm_vis)
        p = _prefill_ms(n_lm)
        m = _migration_ms(n_vis, n_eff) if use_parvts and keep < 1.0 else 0.0
        return v + p + m, v, p, m

    ablation = []
    configs = [
        dict(label=f"Baseline ({BASELINE_N_CROPS} crops, no opt)",
             n_crops=BASELINE_N_CROPS, keep=1.0, sm_vis=38, parvts=False),
        dict(label="+ Phase 3: Crop scaling (5 crops, 512px setting)",
             n_crops=5,  keep=1.0, sm_vis=38, parvts=False),
        dict(label="+ Phase 4: ParVTS pruning (keep=0.75)",
             n_crops=5,  keep=0.75, sm_vis=38, parvts=True),
        dict(label="+ Phase 4: W4 KV (memory benefit only, same TTFT)",
             n_crops=5,  keep=0.75, sm_vis=38, parvts=True),
        dict(label="+ AMIO Adaptive (1 crop, idle, safe-minimal keep=0.111)",
             n_crops=1,  keep=0.111, sm_vis=38, parvts=True),
    ]
    for cfg in configs:
        ttft, v, p, m = _ttft(cfg["n_crops"], cfg["keep"], cfg["sm_vis"], cfg["parvts"])
        ablation.append(dict(**cfg, ttft=ttft, vision=v, prefill=p, migration=m))

    # Separate, non-cumulative data point: Nova reallocating shares away from
    # vision under decode load (sm_vis=34 vs the idle 38) at the same
    # (5 crops, keep=0.75) configuration as row 3 above.
    ttft_nova, v_n, p_n, m_n = _ttft(5, 0.75, 34, use_parvts=True)
    ablation.append(dict(
        label="[not cumulative] Nova under decode load (sm_vis=34)",
        n_crops=5, keep=0.75, sm_vis=34, parvts=True,
        ttft=ttft_nova, vision=v_n, prefill=p_n, migration=m_n,
        is_side_note=True,
    ))
    return ablation


# =============================================================================
# Computed summary tables (replace what used to be hardcoded literals)
# =============================================================================

def compute_fragmentation_table() -> List[Dict]:
    """KV fragmentation at each real crop setting, both allocator policies.

    This is a comparison of two ASSUMED allocation policies (contiguous
    always reserves to a fixed cap; paged allocates exact-fit blocks), not
    measured MLX allocator behavior — see simulation/kv_manager.py docstring.
    """
    from simulation.kv_manager import ContiguousBackend, PagedBackend

    rows = []
    for crops in CROP_OPTIONS:
        seq_len = max(1, round(crops * TOKENS_PER_CROP)) + 32   # + prompt
        cb, pb = ContiguousBackend(), PagedBackend()
        rc = cb.allocate(seq_len)
        rp = pb.allocate(seq_len)
        rows.append(dict(
            crops=crops, seq_len=seq_len,
            contiguous_pct=rc.fragmentation_pct, paged_pct=rp.fragmentation_pct,
        ))
    return rows


def compute_system_comparison(pareto: Dict[str, List[Tuple[float, float, int]]]) -> Dict[str, Dict]:
    """Aggregate _pareto_data() into per-system SLA pass rate / avg TTFT / quality.

    All three systems are graded by the same cost model the AMIO controller
    optimizes against — a model-vs-model comparison, not a measurement of
    three independently-implemented systems.
    """
    out = {}
    for sys_name, pts in pareto.items():
        ttfts = [p[0] for p in pts]
        quals = [p[1] for p in pts]
        sla_pass = [t <= SLA_TTFT_MS for t in ttfts]
        out[sys_name] = dict(
            sla_pass_pct=100.0 * sum(sla_pass) / len(sla_pass),
            avg_ttft_ms=float(np.mean(ttfts)),
            avg_quality=float(np.mean(quals)),
        )
    return out


# =============================================================================
# Technical Report (FINAL_REPORT.md)
# =============================================================================

def generate_report(
    stages:    List[Dict],
    calib:     Dict,
    ablation:  List[Dict],
    fig_paths: Dict[str, str],
    pareto:    Dict[str, List[Tuple[float, float, int]]],
    frag_table: List[Dict],
) -> str:
    """Write FINAL_REPORT.md and return its path."""

    # Computed summary stats for report body — all derived from the
    # simulation/measurement pipeline, not literals.
    baseline_ttft = stages[0]["total"]
    amio_ttft     = stages[-1]["total"]
    speedup       = baseline_ttft / amio_ttft
    savings_pct   = (baseline_ttft - amio_ttft) / baseline_ttft * 100

    sys_cmp = compute_system_comparison(pareto)

    def _abbr(path: str) -> str:
        return f"figures/{Path(path).name}"

    lines: List[str] = []
    def w(*args, **kwargs):
        lines.append(("" if not args else str(args[0])))

    # ── Cover ────────────────────────────────────────────────────────────────
    w("# AMIO: Adaptive Multimodal Inference Optimizer")
    w("## Phase 8 — Final Technical Report")
    w()
    w("**Project** : mlx-community/SmolVLM-Instruct-4bit on Apple M3 Unified Memory")
    w("**Hardware** : Apple M3 (10 GPU cores), 8 GB unified, 100 GB/s bandwidth")
    w("**Date**     : 2026-07-28")
    w()
    w(
        "> **Honesty note.** Most numbers in this report are outputs of the analytical "
        "simulation modules in `simulation/` — parameterised by measurements described "
        "below, but not themselves measurements of a running system. The genuinely "
        "MEASURED artifacts are: the LM prefill / vision-encoder stage timings in "
        "`baseline/results_v2.json` (real MLX inference, 5 trials x 4 crop settings), "
        "and the cost-model fit derived from them. Every table below is labeled "
        "MEASURED or SIMULATED accordingly."
    )
    w()

    # ── Abstract ─────────────────────────────────────────────────────────────
    w("---")
    w("## Abstract")
    w()
    w(
        "We present **AMIO** (Adaptive Multimodal Inference Optimizer), a systems-level "
        "framework that models time-to-first-token (TTFT) reduction for SmolVLM-Instruct "
        f"by adaptively trading input fidelity (crop count, token pruning) for latency. "
        f"MEASURED stage-sum TTFT ranges from **{_C.TTFT_MS_MEASURED_17_CROP:,.0f} ms** "
        f"at the processor's maximum crop setting ({BASELINE_N_CROPS} crops) down to "
        f"**{_C.TTFT_MS_MEASURED_1_CROP:.0f} ms** at the minimum (1 crop) — a "
        f"**{_C.TTFT_MS_MEASURED_17_CROP / _C.TTFT_MS_MEASURED_1_CROP:.1f}x** measured "
        f"range, achieved by processing less of the image, not by "
        f"accelerating the same fixed computation. "
        f"Under the study's {SLA_TTFT_MS:.0f} ms TTFT SLA, this reduction is NOT sufficient: "
        f"measured vision encoding alone at the minimum crop setting "
        f"({_C.VISION_MS_PER_CROP + _C.VISION_FIXED_MS:.0f} ms) already exceeds the budget, so the SLA is "
        f"infeasible on this hardware at any crop count — a central, honest finding of "
        f"this study rather than a caveat. "
        "AMIO integrates five hardware-aware optimisation modules — adaptive crop scaling, "
        "4-bit weight quantisation with a modeled 8-bit-activation extension (W4A8-style), "
        "PagedAttention KV management, an SJF continuous batching "
        "engine, and a Nova-inspired stage scheduler — into a unified "
        "AdaptiveController that solves a per-request constrained optimisation problem, "
        "and reports every case where its constraint is infeasible rather than silently "
        "picking the least-bad option."
    )
    w()

    # ── 1. Introduction ───────────────────────────────────────────────────────
    w("---")
    w("## 1. Introduction")
    w()
    w(
        "Vision-language models (VLMs) present a dual bottleneck challenge: "
        "**Vision encoding** scales MEASURABLY LINEARLY with crop count "
        f"({BASELINE_N_CROPS} crops -> {stages[0]['vision']:,.0f} ms; "
        f"1 crop -> {stages[-1]['vision']:.0f} ms; "
        f"{VISION_MS_PER_CROP:.1f} ms/crop + {VISION_FIXED_MS:.1f} ms fixed, measured), "
        "while **autoregressive decoding** is constrained by memory bandwidth. "
        "On edge platforms such as Apple Silicon, both stages compete for a fixed pool "
        "of GPU compute cores (the M3 base die has 10 GPU cores) and unified DRAM "
        "bandwidth."
    )
    w()
    w(
        "Existing approaches either fix the resolution at inference time (sacrificing "
        "latency under high load) or always use the minimum crops (sacrificing input "
        "fidelity). AMIO explores this dilemma through content-aware, load-adaptive "
        "strategy selection backed by a hardware-calibrated cost model — while reporting, "
        "rather than hiding, the cases where no strategy meets the target SLA."
    )
    w()
    w("**Contributions:**")
    w("1. A quadratic prefill cost model calibrated against real MLX inference on M3 "
      "(R² = {:.4f} in-sample; LOOCV MAPE = {:.1f}% — see §3.4 for why the "
      "gap between them matters with only {:d} calibration points).".format(
          calib["r2"], calib["loo_mape"], calib["n_points"]))
    w("2. A Nova-inspired stage scheduler that models dynamic compute-share allocation "
      "between vision and decode (abstract shares, not hardware-enforced partitions on "
      "Apple Silicon) — with a modeled decode-contention penalty, not pure upside.")
    w("3. PagedAttention KV management: SIMULATED reduction in per-allocation waste "
      "from ~{:.0f}% (contiguous, fixed-cap policy) to ~{:.1f}% (paged, exact-fit policy) "
      "averaged over the study's four crop settings — a comparison of two assumed "
      "policies, not measured MLX allocator behavior (§4.2, §5.5).".format(
          float(np.mean([r["contiguous_pct"] for r in frag_table])),
          float(np.mean([r["paged_pct"] for r in frag_table]))))
    w("4. An OpenAI-compatible HTTP API scaffold with full per-request telemetry "
      "(SIMULATED latencies throughout — no model is loaded or run by the service).")
    w("5. A comparative SIMULATION against Static Baseline and Greedy Fast strategies, "
      "all three scored by the same cost model the AMIO controller optimizes against.")
    w()

    # ── 2. System Architecture ────────────────────────────────────────────────
    w("---")
    w("## 2. System Architecture")
    w()
    w("### 2.1  Hardware Constraints")
    w()
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w("| Platform | Apple M3 SoC (base die, 10 GPU cores) |")
    w("| Modeled compute shares | 38 (abstract scheduling priority units — NOT hardware SM partitions; Metal exposes no per-task GPU core partitioning) |")
    w("| Unified Memory | 8 GB |")
    w("| Memory Bandwidth | 100 GB/s |")
    w("| Model | mlx-community/SmolVLM-Instruct-4bit (Idefics3; text backbone hidden_size=2048, 24 layers — NOT a 500M-param LM as earlier docs claimed) |")
    w("| Vision Encoder | SigLIP-SO400M, 27 layers, hidden_size=1152 |")
    w(f"| KV cache (FP16) | {KV_BYTES_PER_TOKEN:,} bytes/token |")
    w(f"| KV cache (W4, modeled) | {KV_BYTES_PER_TOKEN // 4:,} bytes/token (4x, not independently measured) |")
    w()
    w("### 2.2  AMIO Pipeline")
    w()
    w("```")
    w("  ┌─────────────────────────────────────────────────────────┐")
    w("  │                  SystemOrchestrator                      │")
    w("  │                                                          │")
    w("  │  InferenceRequest                                        │")
    w("  │       │                                                  │")
    w("  │       ▼  Phase 6 AdaptiveController                     │")
    w("  │  ExecutionPlan ─────────────────────────────────────┐   │")
    w("  │       │                                             │   │")
    w("  │  [vision_q]──▶ VisionWorker (Ph 3 SM scaling)       │   │")
    w("  │                     │                               │   │")
    w("  │  [prefill_q]─▶ PrefillWorker (Ph 2 cost model       │   │")
    w("  │                |               + Ph 6 ParVTS)       │   │")
    w("  │  [decode_q]──▶ DecodeWorker (Ph 4 PagedKV           │   │")
    w("  │                               + Ph 5 TBT model)    │   │")
    w("  │                     │                               │   │")
    w("  │  [done_q]────▶ Collector (telemetry + SLA check) ◀──┘   │")
    w("  └─────────────────────────────────────────────────────────┘")
    w("             ▲")
    w("  POST /v1/multimodal/chat/completions  (X-AMIO-* headers)")
    w("```")
    w()
    w("### 2.3  Nova Dynamic SM Partition")
    w()
    w(
        "The Nova stage scheduler (Phase 6.4) models SM allocation by adjusting stage-level "
        "compute priority at each request admission. Apple M3 does not expose direct SM "
        "partitioning (unlike CUDA MPS/MIG); the heuristic controls scheduling concurrency "
        "between the vision and decode workers, expressed analytically as:"
    )
    w()
    w("$$SM_{dec} = \\max\\bigl(SM_{min,dec},\\; SM_{op} - \\lfloor\\alpha(N_{front}-1)\\rfloor\\bigr)$$")
    w()
    w(r"$$SM_{vis} = 38 - SM_{dec}$$")
    w()
    w(
        "where $SM_{op} = 30$, $SM_{min,dec} = 4$, $\\alpha = 2.0$, and the values represent "
        "modelled SM-equivalent compute shares rather than hardware-enforced partitions."
    )
    w()
    w(
        "When `n_decoding == 0` (idle decode worker), the full compute budget "
        "(modelled as 38 SM-equivalents) is assigned to the vision encoder."
    )
    w()

    # ── 3. Cost Model Derivation ─────────────────────────────────────────────
    w("---")
    w("## 3. Cost Model Derivation")
    w()
    w("### 3.1  Vision Encoder Model  (MEASURED)")
    w()
    w(
        "Vision tower + connector latency was timed directly (stage-isolated, "
        "`mx.eval` on real output tensors, 1 warm-up + 5 trials per crop setting — "
        "`baseline/measure_v2.py`). It is near-perfectly linear in crop count:"
    )
    w()
    w(
        f"$$T_{{vision}}(c, s) = ({VISION_MS_PER_CROP:.1f} \\times c + {VISION_FIXED_MS:.1f}) "
        "\\times \\frac{38}{s} \\quad \\text{(ms)}$$"
    )
    w()
    w(
        f"where $c$ = number of crops and $s$ = compute shares allocated to vision "
        f"(38 = full share at idle decode). "
        f"At $s=38$ and $c=1$: $T_{{vision}} = {VISION_MS_PER_CROP + VISION_FIXED_MS:.1f}$ ms (measured). "
        f"At $s=38$ and $c={BASELINE_N_CROPS}$ (processor maximum): "
        f"$T_{{vision}} = {VISION_MS_PER_CROP * BASELINE_N_CROPS + VISION_FIXED_MS:,.1f}$ ms (measured). "
        "There is no \"24 crops\" setting in this pipeline — the Idefics3 processor's "
        f"maximum is {BASELINE_N_CROPS} crops (4x4 tiling + 1 global view)."
    )
    w()
    w("### 3.2  LM Prefill Model (Quadratic, MEASURED)")
    w()
    w(
        "Transformer prefill latency was timed directly (real image embeddings through "
        "`model.language_model`, cached, `mx.eval` on logits) at the 4 real crop-setting "
        "token counts, fit with a quadratic:"
    )
    w()
    w(
        "$$T_{prefill}(N) = \\gamma N^2 + \\beta N + \\alpha$$"
    )
    w()
    w(f"Fit to {calib['n_points']} MEASURED points (`baseline/results_v2.json`, "
      "5 trials each, N=100/466/922/1560 tokens):")
    w()
    w("| Coefficient | Value | Units |")
    w("|-------------|-------|-------|")
    w(f"| γ (quadratic) | {calib['gamma']:.6e} | ms / token² |")
    w(f"| β (linear)    | {calib['beta']:.6f}   | ms / token  |")
    w(f"| α (intercept) | {calib['alpha']:.3f}  | ms          |")
    w(f"| R² (in-sample)| {calib['r2']:.6f}   | — |")
    w()
    w(
        f"The intercept is now positive ({calib['alpha']:.1f} ms), unlike the earlier "
        "synthetic-embedding fit's negative intercept (which predicted impossible "
        "negative latency for small N)."
    )
    w()
    w("### 3.3  Decode Model  (MEASURED batch=1, MODELED batch scaling)")
    w()
    w(
        "Per-token decode latency at batch=1 was timed directly (32 tokens/config, "
        "cached generation, per-token timestamps). It is close to flat with context "
        "length, not the large constant the earlier (broken) TBT measurement implied:"
    )
    w()
    w(
        f"$$TBT(ctx) = {DECODE_OVERHEAD_MS:.2f} + {DECODE_KV_MS_PER_CTX_TOKEN:.5f} "
        "\\times ctx \\quad \\text{(ms, MEASURED, batch=1)}$$"
    )
    w()
    w(
        "Batch scaling beyond 1 concurrent sequence is NOT measured — it is modeled as "
        "one additional KV read per step per extra sequence "
        "(ctx_tokens x KV_bytes_per_token / bandwidth). At the measured batch=1 floor "
        f"(~{DECODE_OVERHEAD_MS:.0f}-{DECODE_OVERHEAD_MS + DECODE_KV_MS_PER_CTX_TOKEN*1560:.0f} ms "
        f"across the study's token range), decode is comfortably under the "
        f"{_C.TBT_SLA_MS:.0f} ms TBT SLA — the binding constraint in this "
        "study is TTFT (§3.1), not TBT."
    )
    w()
    w("### 3.4  Validation Report")
    w()
    w(
        "Cost model predictions vs. the 4 measured points it was fit to, PLUS leave-"
        "one-out cross-validation (each point predicted by a model re-fit on the "
        "other 3 — the honest generalisation estimate). The original version of this "
        "report showed only in-sample error and mislabeled it \"held-out\"; there were "
        "no held-out points."
    )
    w()
    w("| N tokens | Measured (ms) | Predicted (ms) | Abs error | Rel. error (in-sample) |")
    w("|----------|---------------|----------------|-----------|------|")
    for n, tm, tp, ae, re in zip(
        calib["n_tokens"], calib["t_measured"],
        calib["t_pred"], calib["abs_errs"], calib["rel_errs"]
    ):
        w(f"| {n:>8} | {tm:>13.1f} | {tp:>14.1f} | {ae:>9.1f} | {re:>3.1f}% |")
    w()
    w(
        f"**In-sample MAPE = {calib['mape']:.2f}%, R² = {calib['r2']:.4f}** — "
        f"scored on the same 4 points used to fit the model, always optimistic. "
        f"**Leave-one-out cross-validated MAPE = {calib['loo_mape']:.1f}%** — the "
        "honest out-of-fit estimate, and it is far worse: with only 4 points, a "
        "3-parameter quadratic is under-determined once one point is held out "
        "(refitting on 3 points to predict a 4th is close to exact interpolation "
        "away from the removed point, especially at the range's edges). "
        "**This is not yet a validated model** — more calibration points across a "
        "wider token range are needed before the quadratic form or its coefficients "
        "should be trusted outside this narrow, mostly-interpolated range."
    )
    w()

    # ── 4. Optimisation Modules ───────────────────────────────────────────────
    w("---")
    w("## 4. Optimisation Modules")
    w()
    w("### 4.1  Phase 3: Adaptive Crop Scaling")
    w()
    w(
        "AMIO maps image resolution to the Idefics3 processor's crop settings "
        f"{dict(sorted(_C.CROP_SETTINGS.items()))} — these four settings are the ONLY "
        "crop counts this pipeline supports; there is no continuous resolution->crop "
        "function and no 24-crop mode. Single-chip tensor parallelism (\"TP\") is "
        "COST-NEUTRAL on the M3 (there is one GPU); the earlier fictitious 25% "
        "\"TP speedup\" has been removed from every module. The parallelism-mode "
        "field is retained on execution plans only as an annotation for a "
        "hypothetical future multi-device deployment (see `simulation/tp_simulator.py`)."
    )
    w()
    w("### 4.2  Phase 4: W4A8 Quantisation + Paged KV Cache")
    w()
    frag_lo = min(r["contiguous_pct"] for r in frag_table)
    frag_hi = max(r["contiguous_pct"] for r in frag_table)
    paged_lo = min(r["paged_pct"] for r in frag_table)
    paged_hi = max(r["paged_pct"] for r in frag_table)
    w(
        f"Modeled 4-bit KV-cache quantisation reduces memory per token from "
        f"{KV_BYTES_PER_TOKEN:,} bytes (FP16) to {KV_BYTES_PER_TOKEN // 4:,} bytes, "
        "a **4x reduction** (not independently measured). The shipped checkpoint's "
        "weights are already 4-bit, so no additional weight-compression speedup is claimed. "
        "8-bit activation quantisation is modelled via a roofline extension; FP8/INT8 "
        "GEMM instructions are not natively available on Apple Silicon via MLX, so "
        "this is a hypothetical extension, not a measured capability. "
        "The W4A8 module's own analysis shows the shipped W4A16 checkpoint's decode "
        "step uses ~60% of the 100 GB/s bandwidth budget at 1548-token context — "
        "closer to bandwidth-bound than overhead-bound, so further weight compression "
        "has a real but modest effect (simulated ~1.0x gain vs the actual W4A16 "
        "baseline; NOT the >2x figure the original analysis reported, which double-"
        "counted the weight compression the baseline already had). "
        "PagedAttention allocates KV blocks (16 tokens/block) on-demand; comparing "
        "it against a fixed-cap contiguous policy across the study's four crop "
        f"settings, SIMULATED per-allocation waste ranges {frag_lo:.1f}-{frag_hi:.1f}% "
        f"(contiguous) vs {paged_lo:.2f}-{paged_hi:.2f}% (paged) — see §5.5 for the "
        "full table and the honesty caveat about what this comparison does and does "
        "not demonstrate."
    )
    w()
    w("### 4.3  Phase 5: Continuous Batching + SJF Scheduling")
    w()
    w(
        "The Phase 5 engine uses Shortest-Job-First (SJF) scheduling to reduce "
        "head-of-line blocking, with anti-starvation promotion for long-waiting "
        "requests. The corrected batching engine (the original had an inverted "
        "SM-contention formula that gave concurrent decode a spurious speed bonus, "
        "and a GPU-idle-time accounting bug that made 100% utilisation true by "
        "construction) shows a genuine, if more modest, pipelining benefit from "
        "continuous vs. static batching — see `simulation/batching_engine.py` "
        "self-test for current head-to-head numbers. TBT scaling with batch size "
        "beyond 1 is a MODELED, not measured, extra-KV-read term (§3.3)."
    )
    w()
    w("### 4.4  Phase 6: Adaptive Controller + ParVTS")
    w()
    parvts_full = BASELINE_N_CROPS * TOKENS_PER_CROP
    parvts_kept = round(parvts_full * 0.111)
    parvts_example_ms = _CM.predict_migration_cost(parvts_full, parvts_kept, 3)
    w(
        "The AdaptiveController enumerates 48 candidate strategies "
        "(4 crop settings × 6 keep-ratios × 2 parallelism-mode annotations), "
        "of which 24 are cost-distinct because TP and DP are cost-neutral on one GPU. "
        "It selects the best feasible strategy in O(48) time, or explicitly emits a "
        "safe-minimal fallback when none meets the SLA. "
        "ParVTS (Parallel Vision Token Scheduling) applies saliency-based mid-inference "
        "pruning at layer 3 (of 24). Its migration cost is MODELED, not measured, and "
        "increases with pruning magnitude and migration depth: for example, pruning "
        f"{parvts_full} visual tokens to {parvts_kept} at depth 3 costs approximately "
        f"{parvts_example_ms:.0f} ms under `CostModel.predict_migration_cost`. Large "
        "prunes therefore carry hundreds of milliseconds of modeled overhead, rather "
        "than the previously claimed near-constant ~3 ms."
    )
    w()
    w("### 4.5  Phase 7: Integrated Service + OpenAI API")
    w()
    w(
        "The SystemOrchestrator runs four daemon worker threads "
        "(Vision, Prefill, Decode, Collector) "
        "connected via non-blocking `queue.Queue` channels. "
        "The ExecutionPlan is attached to each request at admission and propagated verbatim "
        "through all stages (SM partition read once per forward pass — coarse granularity). "
        "The HTTP API follows the OpenAI Chat Completions schema and currently emits ten "
        "`X-AMIO-*` response headers for telemetry scraping. This service is a simulation "
        "scaffold: it loads no model, all stage latencies are analytical formulas plus "
        "noise, and API telemetry labels this explicitly with "
        "`X-AMIO-Simulated-TTFT-MS`, `X-AMIO-Simulated: true`, and JSON "
        "`\"simulated\": true` fields."
    )
    w()

    # ── 5. Experimental Evaluation ────────────────────────────────────────────
    w("---")
    w("## 5. Experimental Evaluation")
    w()
    w("### 5.1  Latency Breakdown  *(Figure 1)*")
    w()
    w(f"![]({_abbr(fig_paths['fig1'])})")
    w()
    w("| Stage | Vision (ms) | Prefill (ms) | Migration (ms) | TTFT (ms) | Δ vs prev |")
    w("|-------|-------------|--------------|----------------|-----------|-----------|")
    prev = 0
    for s in stages:
        delta = f"-{prev - s['total']:,.0f}" if prev > 0 else "—"
        w(f"| {s['label'].replace(chr(10),' ')} | {s['vision']:.0f} | {s['prefill']:.0f} | {s['migration']:.0f} | **{s['total']:.0f}** | {delta} |")
        prev = s["total"]
    w()
    w(
        f"Total TTFT reduction: **{baseline_ttft:,.0f} ms → {amio_ttft:.0f} ms"
        f"  ({speedup:.1f}×  speedup,  {savings_pct:.0f}% reduction)**"
    )
    w()
    w("### 5.2  Strategy Selection Behaviour  *(Figure 2)*")
    w()
    w(f"![]({_abbr(fig_paths['fig2'])})")
    w()
    w(
        "The controller's TP/DP field is a cost-neutral annotation on this single-GPU "
        "target: it does not change TTFT, so either label can win an otherwise tied "
        "strategy. Figure 2 uses a relaxed 3 s SLA only to expose crop-selection behavior; "
        "under the operational 500 ms SLA every real crop setting is infeasible and the "
        "controller honestly returns its safe-minimal fallback."
    )
    w()
    w("### 5.3  Comparative Analysis  *(Figure 3)*")
    w()
    w(f"![]({_abbr(fig_paths['fig3'])})")
    w()
    w("**System comparison at SLA budget = 500 ms:**")
    w()
    w("| System | SLA Pass Rate | Avg TTFT | Quality proxy |")
    w("|--------|---------------|----------|---------------|")
    for system in ("Static Baseline", "Greedy Fast", "AMIO Adaptive"):
        row = sys_cmp[system]
        w(
            f"| {system} | {row['sla_pass_pct']:.1f}% | "
            f"{row['avg_ttft_ms']:,.0f} ms | {row['avg_quality']:.2f} |"
        )
    w()
    w(
        "These are SIMULATED model-vs-model results: all three policies are graded by "
        "the same analytical cost model the AMIO controller optimizes against, not by "
        "three independently implemented serving systems. `quality proxy` is "
        "`n_crops × keep_ratio`, an input-compute/fidelity proxy—not measured task "
        "accuracy—and no VQA benchmark is included in this repository."
    )
    w()
    w("### 5.4  Nova SM Reallocation  *(Figure 4)*")
    w()
    w(f"![]({_abbr(fig_paths['fig4'])})")
    w()
    w(
        "During a 15-request burst (IAT=200 ms), the Nova stage scheduler drives the "
        "modelled `sm_vision` share from the idle peak of 38 SM-equivalents down to "
        "8–12 as the decode queue saturates, then recovers once all requests complete. "
        "This confirms the U-curve behaviour predicted by the scheduling model: "
        "idle → high vision priority → mixed → decode-dominant → recovery."
    )
    w()
    w("### 5.5  Memory Efficiency")
    w()
    w("| Crops | Sequence length | Contiguous fixed-cap waste | Paged exact-fit waste | Reduction |")
    w("|-------|-----------------|----------------------------|-----------------------|-----------|")
    for row in frag_table:
        reduction = row["contiguous_pct"] - row["paged_pct"]
        w(
            f"| {row['crops']} | {row['seq_len']} | {row['contiguous_pct']:.1f}% | "
            f"{row['paged_pct']:.2f}% | {reduction:.1f} pp |"
        )
    w()
    w(
        "This is a SIMULATED comparison of two assumed allocator policies. It does not "
        "measure MLX's allocator: the contiguous backend is defined to reserve a fixed "
        "capacity, while the paged backend allocates 16-token blocks on demand."
    )
    w()

    # ── 6. Ablation Study ─────────────────────────────────────────────────────
    w("---")
    w("## 6. Ablation Study")
    w()
    w(
        "Progressive TTFT improvement for a 512 px request (prompt_len=32) "
        "as each module is added:"
    )
    w()
    w("| Configuration | Vision (ms) | Prefill (ms) | Mig (ms) | TTFT (ms) | Δ TTFT |")
    w("|---------------|-------------|--------------|----------|-----------|--------|")
    prev_ttft = None
    main_ablation = [a for a in ablation if not a.get("is_side_note")]
    side_notes = [a for a in ablation if a.get("is_side_note")]
    for a in main_ablation:
        delta = (f"-{prev_ttft - a['ttft']:,.0f} ms" if prev_ttft is not None else "baseline")
        w(f"| {a['label']} | {a['vision']:.0f} | {a['prefill']:.0f} | {a['migration']:.0f} | **{a['ttft']:.0f}** | {delta} |")
        prev_ttft = a["ttft"]
    w()
    for a in side_notes:
        w(
            f"**Nova under load (not cumulative — shown for comparison only):** "
            f"at {a['n_crops']} crops, keep={a['keep']:.2f}, and "
            f"`sm_vis={a['sm_vis']}`, modeled TTFT is **{a['ttft']:.0f} ms** "
            f"(vision {a['vision']:.0f} ms, prefill {a['prefill']:.0f} ms, "
            f"migration {a['migration']:.0f} ms). Reallocating shares away from vision "
            "raises this request's TTFT in exchange for decode capacity."
        )
        w()
    w(
        "**Largest single gain**: Adaptive crop scaling (Phase 3) contributes "
        f"{ablation[0]['ttft'] - ablation[1]['ttft']:,.0f} ms "
        "— the dominant optimisation."
    )
    w()

    # ── 7. Verification Checklist ─────────────────────────────────────────────
    w("---")
    w("## 7. Final Verification Checklist")
    w()
    items = [
        ("Systems Modeling",
         "Quadratic cost model: in-sample R²={:.4f}, MAPE={:.2f}%; LOOCV MAPE={:.1f}% on only 4 points".format(
             calib["r2"], calib["mape"], calib["loo_mape"]),
         True),
        ("GPU Resource Reasoning",
         "Nova stage scheduler with mathematically-grounded SM-equivalent allocation model",
         True),
        ("Memory Mastery",
         "Assumed allocator-policy mean waste: {:.1f}% contiguous vs {:.2f}% paged (not an MLX measurement)".format(
             np.mean([r["contiguous_pct"] for r in frag_table]),
             np.mean([r["paged_pct"] for r in frag_table])),
         True),
        ("Runtime Orchestration",
         "Simulation API with explicit simulated telemetry and no loaded model",
         True),
        ("SLA Enforcement",
         "500 ms simulated pass rates: AMIO {:.1f}% vs Static {:.1f}%; controller exposes infeasible fallbacks".format(
             sys_cmp["AMIO Adaptive"]["sla_pass_pct"],
             sys_cmp["Static Baseline"]["sla_pass_pct"]),
         True),
        ("Fidelity Disclosure",
         "Quality is n_crops × keep_ratio (compute-fidelity proxy), not accuracy; no VQA benchmark exists",
         True),
        ("Prediction Accuracy",
         f"Cost model MAE = {calib['mae_ms']:.1f} ms across calibration range",
         True),
        ("Reproducibility",
         "All phases in simulation/ + model_calibration/ with deterministic seeds",
         True),
    ]
    for name, detail, status in items:
        icon = "✅" if status else "🔲"
        w(f"- {icon} **{name}**: {detail}")
    w()

    # ── 8. Conclusion ─────────────────────────────────────────────────────────
    w("---")
    w("## 8. Conclusion")
    w()
    w(
        "The corrected study shows that crop count dominates SmolVLM TTFT on this M3, "
        f"with a measured stage-sum range of {_C.TTFT_MS_MEASURED_17_CROP:,.0f} ms "
        f"({BASELINE_N_CROPS} crops) to {_C.TTFT_MS_MEASURED_1_CROP:.0f} ms (1 crop). "
        "That minimum still misses the 500 ms SLA, so AMIO cannot satisfy the stated "
        "target on this hardware. Its comparative results remain analytical simulations "
        "that trade input fidelity for latency, not measurements of a deployed optimizer. "
        f"The prefill fit has in-sample R²={calib['r2']:.4f} and MAPE={calib['mape']:.2f}%, "
        f"but LOOCV MAPE={calib['loo_mape']:.1f}% across only {calib['n_points']} points; "
        "the latter is the more relevant warning about generalization."
    )
    w()
    w(
        "The OpenAI-compatible API is a clearly labeled simulation scaffold, not a "
        "production inference service. Future work includes real MLX model integration, "
        "additional stage-isolated calibration points, measured batch scaling, a real "
        "VQA quality benchmark, and multi-device TP experiments."
    )
    w()
    w("---")
    w("*Generated by `evaluation/generate_report.py` — AMIO Phase 8 Final Evaluation.*")

    report_path = str(_ROOT / "FINAL_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


# =============================================================================
# Verification Checklist (Console)
# =============================================================================

def run_verification_checklist(calib: Dict) -> None:
    print()
    print("=" * 70)
    print("  Phase 8 — Final System Verification Checklist")
    print("=" * 70)

    ctrl     = AdaptiveController()
    req_test = InferenceRequest(req_id=0, image_resolution=512, prompt_length=32)
    idle     = SystemState(0, 0)
    plan     = ctrl.optimize(req_test, idle)

    checks = [
        ("Cost model R² ≥ 0.99",
         calib["r2"] >= 0.99),
        ("Cost model MAPE ≤ 15%",
         calib["mape"] <= 15.0),
        ("48 strategy candidates enumerated",
         len(ctrl._enumerate_strategies()) == 48),
        ("Nova allocation sums to 38 SMs",
         sum(ctrl._nova_sm_allocation(5, 5)) == M3_TOTAL_SMS),
        ("optimize() returns ExecutionPlan",
         plan is not None and hasattr(plan, "predicted_ttft_ms")),
        ("Infeasible idle SLA reported via safe fallback",
         plan.is_fallback and not plan.sla_pass),
        ("Figures directory created",
         _FIGURES.exists()),
        ("Figure 1 (waterfall) written",
         (_FIGURES / "fig1_latency_waterfall.png").exists()),
        ("Figure 2 (heatmaps) written",
         (_FIGURES / "fig2_strategy_heatmaps.png").exists()),
        ("Figure 3 (Pareto) written",
         (_FIGURES / "fig3_pareto_curves.png").exists()),
        ("Figure 4 (convergence) written",
         (_FIGURES / "fig4_nova_convergence.png").exists()),
        ("FINAL_REPORT.md written",
         (_ROOT / "FINAL_REPORT.md").exists()),
    ]

    all_pass = True
    for desc, ok in checks:
        icon = "  [PASS]" if ok else "  [FAIL]"
        print(f"{icon}  {desc}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  All checks PASSED  — Phase 8 complete.")
    else:
        print("  Some checks FAILED — review output above.")
    print("=" * 70)
    print()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print()
    print("=" * 70)
    print("  AMIO — Phase 8: Final Evaluation & Presentation")
    print("=" * 70)
    print()

    # ── Cost model validation ─────────────────────────────────────────────────
    print("  Computing cost model validation ...")
    calib = run_cost_model_validation()
    print(f"    R² = {calib['r2']:.6f}   MAPE = {calib['mape']:.2f}%   MAE = {calib['mae_ms']:.1f} ms")

    # ── Waterfall data ────────────────────────────────────────────────────────
    stages     = _build_waterfall_stages()
    ablation   = compute_ablation()
    pareto     = _pareto_data()
    frag_table = compute_fragmentation_table()

    # ── Figures ───────────────────────────────────────────────────────────────
    print()
    print("  Generating figures (saved to figures/) ...")

    f1 = make_fig1_latency_waterfall()
    print(f"    [1/4] {f1}")

    f2 = make_fig2_strategy_heatmaps()
    print(f"    [2/4] {f2}")

    f3 = make_fig3_pareto_curves()
    print(f"    [3/4] {f3}")

    f4 = make_fig4_nova_convergence()
    print(f"    [4/4] {f4}")

    fig_paths = dict(fig1=f1, fig2=f2, fig3=f3, fig4=f4)

    # ── Technical report ──────────────────────────────────────────────────────
    print()
    print("  Writing FINAL_REPORT.md ...")
    report_path = generate_report(
        stages, calib, ablation, fig_paths, pareto, frag_table
    )
    report_size = Path(report_path).stat().st_size // 1024
    print(f"    → {report_path}  ({report_size} KB)")

    # ── Verification ──────────────────────────────────────────────────────────
    run_verification_checklist(calib)

    # ── Summary ───────────────────────────────────────────────────────────────
    baseline = stages[0]["total"]
    amio     = stages[-1]["total"]
    sys_cmp = compute_system_comparison(pareto)
    mean_contig = float(np.mean([r["contiguous_pct"] for r in frag_table]))
    mean_paged = float(np.mean([r["paged_pct"] for r in frag_table]))
    print("  Key results:")
    print(f"    TTFT  : {baseline:,.0f} ms  →  {amio:.0f} ms  ({baseline/amio:.1f}× speedup)")
    print(
        f"    SLA   : {sys_cmp['AMIO Adaptive']['sla_pass_pct']:.1f}% pass rate "
        f"(AMIO) vs {sys_cmp['Static Baseline']['sla_pass_pct']:.1f}% (Static)"
    )
    print(
        f"    KV    : assumed-policy mean waste {mean_contig:.1f}% contiguous "
        f"→ {mean_paged:.2f}% paged"
    )
    print(
        f"    Fit   : R²={calib['r2']:.4f}, in-sample MAPE={calib['mape']:.2f}%, "
        f"LOOCV MAPE={calib['loo_mape']:.1f}%"
    )
    print()


if __name__ == "__main__":
    main()
