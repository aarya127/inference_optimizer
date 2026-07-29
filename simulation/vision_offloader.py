"""
Vision Offloader — Nova-Style Asynchronous Layer Swapping (Phase 4)

MEASURED (baseline/results_v2.json): the SigLIP vision encoder + connector
costs T_vision(c) = 553.5·c + 27.4 ms (9,412 ms mean at the 17-crop maximum
— ~65% of the 14,395 ms TTFT stage sum).  It also consumes ~200 MB of GPU
memory in 4-bit format.  On M3 with 8 GB unified memory this crowds out KV
cache capacity.  (The earlier 5,991 ms / "24 crops" figures were an
unvalidated residual — superseded.)

The Nova framework solves this by keeping only K "physical layers" resident
in GPU memory at any time and asynchronously streaming in the next logical
layer while the current layer executes.

Key invariant (zero-overhead condition):
    B ≥ Layer_Size / T_forward_layer

    where B is the SWAP-IN bandwidth and T_forward_layer is the wall-clock
    time for one forward pass through a single layer.

SmolVLM SigLIP vision encoder (matches the code's constants):
    - 27 layers (SigLIP-So400M depth)
    - hidden_size = 1152, MLP ratio = 4, patch_size = 14
    - Parameter count per layer = 4×1152² + 2×1152×4608 ≈ 15.93 M params
    - 4-bit weight size per layer ≈ 15.93 M × 0.5 B ≈ 7.59 MiB
    - Full encoder weight size ≈ 27 × 7.59 ≈ 205 MiB

UNIFIED-MEMORY HONESTY (this is the load-bearing caveat):
    M3 unified memory has no CPU↔GPU DMA — CPU and GPU share the SAME
    physical DRAM.  Moving layers "GPU→CPU" therefore frees NOTHING: the
    bytes still occupy DRAM.  Memory is only genuinely freed when the
    evicted weights are made purgeable / mmap-backed and RE-READ FROM SSD
    on swap-in.  This module therefore models swap-in at SSD read
    bandwidth (`ssd_read_gbps`, default 5.0 GB/s), NOT the 100 GB/s DRAM
    figure the earlier version used.

Zero-overhead condition (M3, SSD-backed swap; MEASURED 17-crop vision time):
    B = ~5 GB/s (SSD sequential read)
    T_forward_layer = (553.5·17 + 27.4) ms / 27 ≈ 350 ms  (uniform split of
        the measured 17-crop stage time; even at 1 crop it is ≈ 21.5 ms)
    BW required = 7.96 MB / 0.350 s ≈ 0.023 GB/s
    Margin ≈ 5 / 0.023 ≈ ~220× — comfortably zero-overhead (≈27× even at
    the 1-crop minimum), two orders of magnitude below the earlier
    DRAM-based "2787×" claim.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

# Allow `import amio_constants` when run as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from amio_constants import VISION_MS_PER_CROP, VISION_FIXED_MS, MAX_CROPS


# ---------------------------------------------------------------------------
# Vision encoder constants (SigLIP-So400M in SmolVLM)
# ---------------------------------------------------------------------------
SIGLIP_N_LAYERS: int = 27
SIGLIP_HIDDEN: int = 1152
SIGLIP_MLP_RATIO: int = 4
SIGLIP_PATCH_SIZE: int = 14

# Params per layer (attention + MLP, simplified)
#   Attention: 4 × hidden² (Q, K, V, O projections)
#   MLP:       2 × hidden × hidden × MLP_ratio  (up + down)
#   Norms:     ~2 × hidden  (negligible)
PARAMS_PER_LAYER: int = (
    4 * SIGLIP_HIDDEN ** 2 +                  # attention
    2 * SIGLIP_HIDDEN * SIGLIP_HIDDEN * SIGLIP_MLP_RATIO  # MLP
)
BITS_W4: int = 4
BYTES_PER_LAYER_W4: float = PARAMS_PER_LAYER * BITS_W4 / 8  # 4-bit
BYTES_PER_LAYER_FP16: float = PARAMS_PER_LAYER * 2             # FP16

TOTAL_VISION_MB_W4: float = SIGLIP_N_LAYERS * BYTES_PER_LAYER_W4 / (1024 ** 2)
TOTAL_VISION_MB_FP16: float = SIGLIP_N_LAYERS * BYTES_PER_LAYER_FP16 / (1024 ** 2)

# Hardware
M3_BW_GBps: float = 100.0        # DRAM bandwidth (NOT used for swap-in)
SSD_READ_GBPS_DEFAULT: float = 5.0  # SSD sequential read — the honest swap path
M3_TOTAL_MB: float = 8192.0

# Vision encoder latency — MEASURED (baseline/results_v2.json):
# T_vision(c) = 553.5·c + 27.4 ms at full GPU.  Default per-layer forward
# time uses the 17-crop (maximum) stage time under a uniform per-layer
# split (the split itself is a modeling assumption; the total is measured).
BASELINE_N_CROPS: int = MAX_CROPS                                     # 17
BASELINE_T_VISION_MS: float = VISION_MS_PER_CROP * MAX_CROPS + VISION_FIXED_MS  # ≈9436.9
T_PER_LAYER_MS: float = BASELINE_T_VISION_MS / SIGLIP_N_LAYERS        # ≈349.5
# Worst case for swap overlap is the FASTEST layer time — at the 1-crop
# minimum: (553.5 + 27.4)/27 ≈ 21.5 ms/layer, still ≈27× the required swap BW.
T_PER_LAYER_MS_1_CROP: float = (VISION_MS_PER_CROP + VISION_FIXED_MS) / SIGLIP_N_LAYERS


# ---------------------------------------------------------------------------
# Nova-style physical layer buffer
# ---------------------------------------------------------------------------

@dataclass
class LayerSwapEvent:
    """Record of one async layer swap."""
    layer_idx: int           # logical layer being swapped IN
    swap_in_bytes: float     # bytes transferred
    swap_in_ms: float        # transfer latency
    forward_ms: float        # forward pass latency for this layer
    overlap_ms: float        # time the swap overlaps with prior compute
    exposed_overhead_ms: float  # max(0, swap_in_ms - forward_ms_prev)

    @property
    def is_zero_overhead(self) -> bool:
        return self.exposed_overhead_ms <= 0.01  # <0.01 ms threshold


@dataclass
class OffloadResult:
    """
    Result of a full vision encoder forward pass with layer swapping.

    Memory-honesty note: `memory_saved_mb` is only REAL savings if the
    non-resident layers are purgeable / mmap-backed and re-read from SSD
    (which is exactly what the swap timings model).  Merely tagging pages
    "CPU" on unified memory frees nothing.
    """
    k_physical_layers: int     # resident layer count
    n_layers: int              # total logical layers
    weight_format: str         # "w4" or "fp16"

    # Memory
    gpu_memory_mb: float       # DRAM used by the resident physical buffer
    cpu_memory_mb: float       # bytes of SSD-backed (purgeable) layers
    memory_saved_mb: float     # freed DRAM, real only via SSD-backed pages

    # Timing
    events: List[LayerSwapEvent]
    total_forward_ms: float    # sum of all layer forward times
    total_swap_ms: float       # sum of all swap-in times (overlapped)
    exposed_swap_overhead_ms: float  # actual stall time

    # Zero-overhead test
    bandwidth_required_GBps: float
    bandwidth_available_GBps: float
    zero_overhead: bool

    @property
    def overhead_pct(self) -> float:
        return (self.exposed_swap_overhead_ms / self.total_forward_ms) * 100.0


class VisionOffloader:
    """
    Models Nova-style asynchronous layer swapping for the SigLIP vision encoder.

    UNIFIED-MEMORY NOTE: on M3, "GPU memory" and "CPU memory" are the same
    DRAM.  Eviction only frees real memory if the evicted weights are
    purgeable / mmap-backed and re-read from SSD on swap-in, so swap
    latency is computed from `ssd_read_gbps`, not DRAM bandwidth.

    Parameters
    ----------
    k_physical_layers : int
        Number of layers pinned resident at once (default 2).
        K=1 gives minimum memory; K=n_layers disables swapping.
    weight_format : str
        "w4" (4-bit, default) or "fp16".
    ssd_read_gbps : float
        SSD sequential-read bandwidth used for swap-in (default 5.0 GB/s).
        This replaces the earlier, dishonest use of the 100 GB/s DRAM
        figure for a path that must come from storage.
    bandwidth_GBps : float
        DRAM bandwidth (kept for reference/reporting only).
    """

    def __init__(
        self,
        k_physical_layers: int = 2,
        weight_format: str = "w4",
        bandwidth_GBps: float = M3_BW_GBps,
        n_layers: int = SIGLIP_N_LAYERS,
        t_per_layer_ms: float = T_PER_LAYER_MS,
        ssd_read_gbps: float = SSD_READ_GBPS_DEFAULT,
    ):
        self.k = k_physical_layers
        self.weight_format = weight_format
        self.bandwidth_GBps = bandwidth_GBps
        self.ssd_read_gbps = ssd_read_gbps
        self.n_layers = n_layers
        self.t_per_layer_ms = t_per_layer_ms
        self._bytes_per_layer = (
            BYTES_PER_LAYER_W4 if weight_format == "w4" else BYTES_PER_LAYER_FP16
        )

    def _swap_latency_ms(self) -> float:
        """Time to swap in one layer from SSD-backed (purgeable/mmap) pages."""
        return (self._bytes_per_layer / (self.ssd_read_gbps * 1e9)) * 1000.0

    def simulate(self) -> OffloadResult:
        """
        Simulate a full forward pass through all `n_layers` with K physical slots.

        Pipeline model:
            - While layer L is executing (T_forward), asynchronously load L+1.
            - If swap-in completes before layer L finishes, overhead = 0.
            - Otherwise, stall = swap_in_ms - T_forward (exposed overhead).
        """
        swap_ms = self._swap_latency_ms()
        events: List[LayerSwapEvent] = []
        total_exposed = 0.0

        for layer_idx in range(self.n_layers):
            # First K layers are pre-loaded; subsequent layers need swap-in
            if layer_idx < self.k:
                actual_swap_ms = 0.0
                exposed = 0.0
                overlap = 0.0
            else:
                actual_swap_ms = swap_ms
                # Overlap = time the swap runs concurrently with prior layer
                overlap = min(actual_swap_ms, self.t_per_layer_ms)
                exposed = max(0.0, actual_swap_ms - self.t_per_layer_ms)
                total_exposed += exposed

            events.append(LayerSwapEvent(
                layer_idx=layer_idx,
                swap_in_bytes=self._bytes_per_layer,
                swap_in_ms=actual_swap_ms,
                forward_ms=self.t_per_layer_ms,
                overlap_ms=overlap,
                exposed_overhead_ms=exposed,
            ))

        total_forward = self.n_layers * self.t_per_layer_ms
        total_swap = sum(e.swap_in_ms for e in events)

        # GPU memory = K physical layers in buffer
        gpu_mb = self.k * self._bytes_per_layer / (1024 ** 2)
        cpu_mb = (self.n_layers - self.k) * self._bytes_per_layer / (1024 ** 2)
        total_mb = self.n_layers * self._bytes_per_layer / (1024 ** 2)
        saved_mb = total_mb - gpu_mb

        # Bandwidth actually required for zero overhead, compared against
        # the SSD read path (the honest swap-in mechanism)
        bw_required = (self._bytes_per_layer / (self.t_per_layer_ms / 1000.0)) / 1e9

        return OffloadResult(
            k_physical_layers=self.k,
            n_layers=self.n_layers,
            weight_format=self.weight_format,
            gpu_memory_mb=gpu_mb,
            cpu_memory_mb=cpu_mb,
            memory_saved_mb=saved_mb,
            events=events,
            total_forward_ms=total_forward,
            total_swap_ms=total_swap,
            exposed_swap_overhead_ms=total_exposed,
            bandwidth_required_GBps=bw_required,
            bandwidth_available_GBps=self.ssd_read_gbps,
            zero_overhead=total_exposed < 0.01,
        )

    def kv_headroom_freed_mb(self) -> float:
        """
        DRAM freed for KV cache by offloading all but K layers.

        HONESTY: this is only real if the evicted layers are purgeable /
        mmap-backed and re-read from SSD on swap-in (as this class models).
        On unified memory, merely reassigning pages "GPU→CPU" frees nothing.
        """
        total_mb = self.n_layers * self._bytes_per_layer / (1024 ** 2)
        resident_mb = self.k * self._bytes_per_layer / (1024 ** 2)
        return total_mb - resident_mb

    def zero_overhead_condition(self) -> dict:
        """
        Verify the Nova zero-overhead condition against the SSD swap path:
            B_ssd ≥ Layer_Size / T_forward_layer
        """
        # Explicit guards (the old truthiness `and` chain divided by zero
        # whenever swap latency evaluated falsy)
        if self.t_per_layer_ms <= 0:
            raise ValueError("t_per_layer_ms must be positive")
        bw_required = (
            self._bytes_per_layer / (self.t_per_layer_ms / 1000.0)
        ) / 1e9
        margin = (
            self.ssd_read_gbps / bw_required if bw_required > 0 else float("inf")
        )
        return {
            "layer_size_mb": self._bytes_per_layer / (1024 ** 2),
            "t_forward_layer_ms": self.t_per_layer_ms,
            "bw_required_GBps": round(bw_required, 4),
            "bw_available_GBps": self.ssd_read_gbps,
            "margin_x": round(margin, 1),
            "satisfied": margin >= 1.0,
            "notes": (
                f"SSD read path ({self.ssd_read_gbps:.0f} GB/s) has "
                f"{margin:.0f}× the required swap bandwidth; the earlier "
                "2787× figure used DRAM bandwidth, which frees no memory "
                "on a unified-memory chip."
            ),
        }


def sweep_k_values(
    k_values: List[int] | None = None,
    weight_format: str = "w4",
) -> List[OffloadResult]:
    """Compare different K physical layer counts."""
    ks = k_values or [1, 2, 4, 8, 14, 27]
    results = []
    for k in ks:
        offloader = VisionOffloader(k_physical_layers=k, weight_format=weight_format)
        results.append(offloader.simulate())
    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("AMIO Phase 4 — Vision Offloader (Nova Strategy) Self-Test")
    print("=" * 72)

    # Architecture summary
    print(f"\nSigLIP vision encoder:")
    print(f"  Layers         : {SIGLIP_N_LAYERS}")
    print(f"  Hidden size    : {SIGLIP_HIDDEN}")
    print(f"  Params/layer   : {PARAMS_PER_LAYER/1e6:.2f} M")
    print(f"  Size/layer W4  : {BYTES_PER_LAYER_W4/1e6:.2f} MB")
    print(f"  Total W4       : {TOTAL_VISION_MB_W4:.1f} MB")
    print(f"  T_forward/layer: {T_PER_LAYER_MS:.1f} ms  "
          f"(measured {BASELINE_T_VISION_MS:.0f} ms at {BASELINE_N_CROPS} crops, "
          f"uniform-split; 1-crop minimum {T_PER_LAYER_MS_1_CROP:.1f} ms/layer)")

    # Zero-overhead condition — against the honest SSD swap path
    off = VisionOffloader(k_physical_layers=2)
    cond = off.zero_overhead_condition()
    print(f"\nZero-overhead condition (Nova, SSD-backed swap):")
    print(f"  Layer size     : {cond['layer_size_mb']:.3f} MB")
    print(f"  BW required    : {cond['bw_required_GBps']:.4f} GB/s")
    print(f"  BW available   : {cond['bw_available_GBps']} GB/s (SSD read)")
    print(f"  Margin         : {cond['margin_x']}×")
    print(f"  {cond['notes']}")
    cond1 = VisionOffloader(
        k_physical_layers=2, t_per_layer_ms=T_PER_LAYER_MS_1_CROP
    ).zero_overhead_condition()
    print(f"  Worst case (1-crop, {T_PER_LAYER_MS_1_CROP:.1f} ms/layer): "
          f"margin {cond1['margin_x']}× — "
          f"{'still zero-overhead' if cond1['satisfied'] else 'NOT zero-overhead'}")

    # K sweep
    print(f"\nK physical layers sweep (W4 format, SSD swap-in):")
    print(f"  {'K':>3}  {'GPU MB':>7}  {'CPU MB':>7}  {'Saved MB':>8}  "
          f"{'Stall ms':>8}  {'Overhead%':>9}  {'Zero?':>5}")
    print("  " + "-" * 55)
    for r in sweep_k_values():
        flag = "PASS" if r.zero_overhead else "FAIL"
        print(
            f"  {r.k_physical_layers:>3}  "
            f"{r.gpu_memory_mb:>7.2f}  "
            f"{r.cpu_memory_mb:>7.2f}  "
            f"{r.memory_saved_mb:>8.2f}  "
            f"{r.exposed_swap_overhead_ms:>8.3f}  "
            f"{r.overhead_pct:>8.2f}%  "
            f"{flag}"
        )

    print(f"\nRecommended: K=2  — "
          f"{VisionOffloader(k_physical_layers=2).kv_headroom_freed_mb():.1f} MB "
          f"of DRAM freed for KV cache (real only via purgeable/mmap-backed "
          f"weights re-read from SSD)")
    print("\nVision offloader self-test complete")
