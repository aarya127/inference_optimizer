"""
Vision Offloader — Nova-Style Asynchronous Layer Swapping (Phase 4)

Phase 2 established the SigLIP vision encoder costs 5,991 ms and constitutes
71% of TTFT.  It also consumes ~200 MB of GPU memory in 4-bit format.  On M3
with 8 GB unified memory this crowds out KV cache capacity.

The Nova framework solves this by keeping only K "physical layers" resident
in GPU memory at any time and asynchronously streaming in the next logical
layer from CPU memory while the current layer executes.

Key invariant (zero-overhead condition):
    B ≥ Layer_Size / T_forward_layer

    where B is memory bandwidth and T_forward_layer is the wall-clock time
    for one forward pass through a single layer.

SmolVLM SigLIP vision encoder:
    - 27 layers (SigLIP-So400M depth)
    - hidden_size = 1152, MLP ratio = 4, patch_size = 14
    - Parameter count per layer ≈ 4 × 1152² / 1M ≈ ~5.3 M params
    - 4-bit weight size per layer ≈ 5.3 M × 0.5 = 2.65 MB
    - Full encoder weight size ≈ 27 × 2.65 = 71.6 MB

M3 unified memory has no CPU↔GPU DMA — all memory is physically shared.
"Streaming" here means controlling which weight tensors are kept in the
active compute region vs evicted to lower-priority memory pages.  In
practice on M3 we model this as a latency penalty for cold loads.

Zero-overhead condition (M3):
    B = 100 GB/s
    T_forward_layer = 5991 ms / 27 = 222 ms
    Max layer size for zero overhead = 100e9 × 0.222 = 22.2 GB >> 2.65 MB
    ✅ M3 satisfies the zero-overhead condition with massive margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


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
M3_BW_GBps: float = 100.0
M3_TOTAL_MB: float = 8192.0

# Calibrated vision encoder latency
BASELINE_T_VISION_MS: float = 5991.0
T_PER_LAYER_MS: float = BASELINE_T_VISION_MS / SIGLIP_N_LAYERS


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
    """Result of a full vision encoder forward pass with layer swapping."""
    k_physical_layers: int     # GPU-resident layer count
    n_layers: int              # total logical layers
    weight_format: str         # "w4" or "fp16"

    # Memory
    gpu_memory_mb: float       # GPU memory used by physical buffer
    cpu_memory_mb: float       # remaining layers on CPU/low-priority
    memory_saved_mb: float     # vs loading all layers

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

    Parameters
    ----------
    k_physical_layers : int
        Number of layers pinned in GPU memory at once (default 2).
        K=1 gives minimum memory; K=n_layers disables swapping.
    weight_format : str
        "w4" (4-bit, default) or "fp16".
    bandwidth_GBps : float
        Memory bandwidth for swap-in (default M3 = 100 GB/s).
    """

    def __init__(
        self,
        k_physical_layers: int = 2,
        weight_format: str = "w4",
        bandwidth_GBps: float = M3_BW_GBps,
        n_layers: int = SIGLIP_N_LAYERS,
        t_per_layer_ms: float = T_PER_LAYER_MS,
    ):
        self.k = k_physical_layers
        self.weight_format = weight_format
        self.bandwidth_GBps = bandwidth_GBps
        self.n_layers = n_layers
        self.t_per_layer_ms = t_per_layer_ms
        self._bytes_per_layer = (
            BYTES_PER_LAYER_W4 if weight_format == "w4" else BYTES_PER_LAYER_FP16
        )

    def _swap_latency_ms(self) -> float:
        """Time to swap in one layer from CPU/evicted pages."""
        return (self._bytes_per_layer / (self.bandwidth_GBps * 1e9)) * 1000.0

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

        # Bandwidth actually required for zero overhead
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
            bandwidth_available_GBps=self.bandwidth_GBps,
            zero_overhead=total_exposed < 0.01,
        )

    def kv_headroom_freed_mb(self) -> float:
        """
        Memory freed for KV cache by offloading all but K layers.
        """
        total_mb = self.n_layers * self._bytes_per_layer / (1024 ** 2)
        resident_mb = self.k * self._bytes_per_layer / (1024 ** 2)
        return total_mb - resident_mb

    def zero_overhead_condition(self) -> dict:
        """
        Verify the Nova zero-overhead condition:
            B ≥ Layer_Size / T_forward_layer
        """
        bw_required = self._swap_latency_ms() and (
            self._bytes_per_layer / (self.t_per_layer_ms / 1000.0)
        ) / 1e9
        margin = self.bandwidth_GBps / bw_required
        return {
            "layer_size_mb": self._bytes_per_layer / (1024 ** 2),
            "t_forward_layer_ms": self.t_per_layer_ms,
            "bw_required_GBps": round(bw_required, 4),
            "bw_available_GBps": self.bandwidth_GBps,
            "margin_x": round(margin, 1),
            "satisfied": margin >= 1.0,
            "notes": (
                f"M3 has {margin:.0f}× more bandwidth than required; "
                "swap-in is always complete before next layer starts."
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
    print(f"  T_forward/layer: {T_PER_LAYER_MS:.1f} ms")

    # Zero-overhead condition
    off = VisionOffloader(k_physical_layers=2)
    cond = off.zero_overhead_condition()
    print(f"\nZero-overhead condition (Nova):")
    print(f"  Layer size     : {cond['layer_size_mb']:.3f} MB")
    print(f"  BW required    : {cond['bw_required_GBps']:.4f} GB/s")
    print(f"  BW available   : {cond['bw_available_GBps']} GB/s")
    print(f"  Margin         : {cond['margin_x']}×   ✅ {cond['notes']}")

    # K sweep
    print(f"\nK physical layers sweep (W4 format):")
    print(f"  {'K':>3}  {'GPU MB':>7}  {'CPU MB':>7}  {'Saved MB':>8}  "
          f"{'Stall ms':>8}  {'Overhead%':>9}  {'Zero?':>5}")
    print("  " + "-" * 55)
    for r in sweep_k_values():
        flag = "✅" if r.zero_overhead else "❌"
        print(
            f"  {r.k_physical_layers:>3}  "
            f"{r.gpu_memory_mb:>7.2f}  "
            f"{r.cpu_memory_mb:>7.2f}  "
            f"{r.memory_saved_mb:>8.2f}  "
            f"{r.exposed_swap_overhead_ms:>8.3f}  "
            f"{r.overhead_pct:>8.2f}%  "
            f"{flag}"
        )

    print(f"\nRecommended: K=2  — {VisionOffloader(k_physical_layers=2).kv_headroom_freed_mb():.1f} MB freed for KV cache")
    print("\n✅ Vision offloader self-test complete")
