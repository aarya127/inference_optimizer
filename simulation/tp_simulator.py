"""
Tensor Parallelism Simulation for Single-GPU Systems

Simulates multi-GPU TP communication overhead on M3 Mac by injecting
artificial delays at synchronization points (All-Reduce operations).

This allows training an adaptive controller on single-GPU hardware that
understands multi-GPU communication costs.

IMPORTANT — HYPOTHETICAL MULTI-DEVICE PROJECTIONS ONLY
------------------------------------------------------
The Apple M3 is a SINGLE-GPU chip.  Every "TP/DP gain" computed here models
a hypothetical system with `tp_size` (or `n_workers`) separate devices; the
analytical formula t_total = t_compute / tp_size + t_comm *assumes* the
compute speedup it reports.  None of these gains are realizable on a single
M3 and they must never be folded into single-M3 latency predictions.  They
are retained strictly as multi-device projections (e.g. MLX distributed
over multiple Macs).

Phase 3 extensions:
  - ParallelismMode enum (TP / DP / HYBRID)
  - Analytical cost functions — compute_tp_cost(), compute_dp_cost()
  - compare_parallelism_modes() returning recommended mode + gain %
"""

import time
import math
import os
import sys
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# Allow `import amio_constants` when run as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import amio_constants

# Printed/embedded in every mode-comparison output (see module docstring).
MULTI_DEVICE_DISCLAIMER = (
    "HYPOTHETICAL MULTI-DEVICE PROJECTION — M3 is a single-GPU chip; "
    "these TP/DP gains assume tp_size separate devices and are NOT "
    "realizable single-M3 gains."
)

# mlx is only required for the injection-based simulation (Phase 0 code).
# The Phase 3 analytical functions do not need it.
try:
    import mlx.core as mx
    _MX_ARRAY = mx.array
    _MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore[assignment]
    _MX_ARRAY = Any  # type: ignore[assignment]
    _MLX_AVAILABLE = False


class OperationType(Enum):
    """Types of operations requiring all-reduce"""
    QKV_PROJECTION = "qkv_projection"
    ATTENTION_OUTPUT = "attention_output"
    FFN_INTERMEDIATE = "ffn_intermediate"
    FFN_OUTPUT = "ffn_output"
    LAYER_NORM = "layer_norm"
    PROJECTION = "projection"


class ParallelismMode(Enum):
    """Parallelism strategy for inference"""
    TP = "tensor_parallel"        # Split weight matrices across devices/SMs
    DP = "data_parallel"          # Each device/SM processes different crop
    HYBRID = "hybrid"             # TP for LM, DP for vision encoder crops


@dataclass
class ParallelismCostResult:
    """
    Result of a TP vs DP cost comparison.

    ALL fields describe a HYPOTHETICAL multi-device projection (see module
    docstring): `t_compute_ms` is the assumed per-device compute after an
    idealized 1/tp_size split, and `throughput_gain_pct` is the projected
    gain of that hypothetical system vs the sequential baseline — it is NOT
    a realizable single-M3 gain.
    """
    mode: ParallelismMode
    t_compute_ms: float
    t_communication_ms: float
    t_total_ms: float
    n_sync_points: int
    throughput_gain_pct: float        # projected, hypothetical multi-device
    notes: str = ""


# ---------------------------------------------------------------------------
# Analytical cost functions (no actual MLX execution required)
# ---------------------------------------------------------------------------

# SmolVLM / SigLIP vision encoder configuration (from amio_constants)
SIGLIP_N_LAYERS = amio_constants.VISION_NUM_LAYERS    # 27 (SigLIP-So400M)
SIGLIP_HIDDEN_DIM = amio_constants.VISION_HIDDEN_SIZE # 1152
SIGLIP_SYNC_PER_LAYER = 5     # QKV, attn-out, FFN-up, FFN-down, layernorm

# Visual tokens per crop (81, from checkpoint image_seq_len) — CONFIRMED by
# the v2 measurement campaign: image tokens = 81 × crops exactly
# (17 crops → 1377 image tokens; 1560 TOTAL input tokens incl. text).  The
# old "1548 tokens at 24 crops" figure came from the broken sweep — a
# 24-crop setting never existed (MAX_CROPS = 17).
TOKENS_PER_CROP = amio_constants.TOKENS_PER_CROP      # 81
TOTAL_VISUAL_TOKENS_MEASURED = TOKENS_PER_CROP * amio_constants.MAX_CROPS  # 1377
TOTAL_INPUT_TOKENS_MEASURED = amio_constants.TOKENS_PER_CONFIG[amio_constants.MAX_CROPS]  # 1560

# SmolVLM language model configuration (from amio_constants)
LM_N_LAYERS = amio_constants.LM_NUM_LAYERS            # 24 (SmolLM2-1.7B-class)
LM_HIDDEN_DIM = amio_constants.LM_HIDDEN_SIZE         # 2048
LM_SYNC_PER_LAYER = 4         # QKV, attn-out, FFN-gate/up, FFN-down

# M3 hardware constants
M3_BANDWIDTH_GBps = amio_constants.M3_MEMORY_BW_GBPS  # 100 GB/s unified memory
M3_SYNC_LATENCY_us = 50.0     # Base all-reduce / barrier overhead (µs)
M3_SYNC_OVERHEAD_us = 20.0    # Extra per-sync coordination overhead (µs)

# MEASURED baselines (baseline/results_v2.json).  The old 5991 ms residual /
# "24 crops" attribution is superseded (see amio_constants provenance notes).
BASELINE_N_CROPS = amio_constants.MAX_CROPS            # 17 (processor maximum)
BASELINE_T_VISION_MS = (
    amio_constants.VISION_MS_PER_CROP * BASELINE_N_CROPS
    + amio_constants.VISION_FIXED_MS
)                                                      # ≈9436.9 ms at 17 crops
# T_lm_prefill at N=1560 tokens from the MEASURED quadratic fit (≈4975 ms;
# measured mean 4983 ms).
BASELINE_T_LM_MS = (
    amio_constants.PREFILL_GAMMA * TOTAL_INPUT_TOKENS_MEASURED ** 2
    + amio_constants.PREFILL_BETA * TOTAL_INPUT_TOKENS_MEASURED
    + amio_constants.PREFILL_ALPHA
)


def _sync_cost_us(tensor_bytes: int,
                  base_us: float = M3_SYNC_LATENCY_us,
                  overhead_us: float = M3_SYNC_OVERHEAD_us,
                  bandwidth_GBps: float = M3_BANDWIDTH_GBps) -> float:
    """Latency (µs) for a single All-Reduce of `tensor_bytes` bytes."""
    bandwidth_us = (tensor_bytes / (bandwidth_GBps * 1e9)) * 1e6
    return base_us + bandwidth_us + overhead_us


def compute_tp_cost(
    t_compute_ms: float,
    n_layers: int,
    hidden_dim: int,
    sync_per_layer: int,
    seq_len: int = TOTAL_INPUT_TOKENS_MEASURED,
    tp_size: int = 2,
) -> ParallelismCostResult:
    """
    Analytical cost of Tensor-Parallel execution on a HYPOTHETICAL system
    with `tp_size` separate devices.

    TP shards every weight matrix across `tp_size` workers.  After every
    collective op, an All-Reduce is needed.  The payload of each All-Reduce
    is proportional to hidden_dim * seq_len (FP16 = 2 bytes).

    CAVEAT (honesty): the compute term below is `t_compute_ms / tp_size` —
    i.e. the 1/tp_size speedup is an ASSUMPTION of the projection, not a
    derived result, and the M3 is a single-GPU chip on which no such split
    exists.  Treat the returned gains strictly as multi-device projections.

    Parameters
    ----------
    t_compute_ms : baseline sequential compute time (ms)
    n_layers     : number of transformer layers
    hidden_dim   : model hidden dimension
    sync_per_layer: number of All-Reduce calls per layer
    seq_len      : sequence length at this stage
    tp_size      : hypothetical TP degree (2 = projected two-device system)

    Returns
    -------
    ParallelismCostResult (hypothetical multi-device projection)
    """
    tensor_bytes = hidden_dim * seq_len * 2  # FP16
    cost_per_sync_us = _sync_cost_us(tensor_bytes)
    n_sync_total = n_layers * sync_per_layer
    t_comm_ms = (n_sync_total * cost_per_sync_us) / 1000.0

    # ASSUMED idealized split: compute ≈ t_compute_ms / tp_size on tp_size
    # hypothetical devices, with communication overhead added back.
    t_total_ms = t_compute_ms / tp_size + t_comm_ms
    gain_pct = (
        (t_compute_ms - t_total_ms) / t_compute_ms * 100.0
        if t_compute_ms > 0 else 0.0
    )

    return ParallelismCostResult(
        mode=ParallelismMode.TP,
        t_compute_ms=t_compute_ms / tp_size,
        t_communication_ms=t_comm_ms,
        t_total_ms=t_total_ms,
        n_sync_points=n_sync_total,
        throughput_gain_pct=gain_pct,
        notes=(f"TP-{tp_size} (hypothetical multi-device): {n_sync_total} sync pts, "
               f"{t_comm_ms:.1f} ms comm overhead"),
    )


def compute_dp_cost(
    t_compute_ms: float,
    n_crops: int,
    n_workers: int = 2,
    hidden_dim: int = SIGLIP_HIDDEN_DIM,
    tokens_per_crop: int = TOKENS_PER_CROP,
) -> ParallelismCostResult:
    """
    Analytical cost of Data-Parallel execution over vision crops on a
    HYPOTHETICAL system with `n_workers` separate devices (see module
    docstring — not realizable on a single M3).

    In DP mode each worker processes a disjoint subset of crops
    (n_crops / n_workers each).  Results are fused with a single
    All-Gather whose payload is all crop embeddings combined.

    Parameters
    ----------
    t_compute_ms  : baseline sequential compute time (ms) for ALL crops
    n_crops       : number of image crops to process
    n_workers     : hypothetical DP degree (number of parallel devices)
    hidden_dim    : encoder output hidden dimension
    tokens_per_crop: visual tokens emitted per crop (81 per checkpoint
                     config, CONFIRMED by the v2 campaign: image tokens =
                     81 × crops exactly; 17 crops → 1377)

    Returns
    -------
    ParallelismCostResult (hypothetical multi-device projection)
    """
    all_gather_bytes = n_crops * tokens_per_crop * hidden_dim * 2  # FP16
    t_comm_ms = _sync_cost_us(all_gather_bytes) / 1000.0  # 1 AllGather

    # Each worker handles ⌈crops / workers⌉ crops in parallel
    crops_per_worker = math.ceil(n_crops / n_workers)
    # Compute scales linearly with crop count
    t_parallel_compute_ms = t_compute_ms * (crops_per_worker / n_crops)
    t_total_ms = t_parallel_compute_ms + t_comm_ms

    gain_pct = (
        (t_compute_ms - t_total_ms) / t_compute_ms * 100.0
        if t_compute_ms > 0 else 0.0
    )

    return ParallelismCostResult(
        mode=ParallelismMode.DP,
        t_compute_ms=t_parallel_compute_ms,
        t_communication_ms=t_comm_ms,
        t_total_ms=t_total_ms,
        n_sync_points=1,   # single AllGather
        throughput_gain_pct=gain_pct,
        notes=(f"DP-{n_workers}: {crops_per_worker} crops/worker, "
               f"1 AllGather ({all_gather_bytes/1024:.1f} KB)"),
    )


def compare_parallelism_modes(
    t_vision_ms: float = BASELINE_T_VISION_MS,
    t_lm_ms: float = BASELINE_T_LM_MS,
    n_crops: int = BASELINE_N_CROPS,
    n_workers: int = 2,
    tp_size: int = 2,
    seq_len: int = TOTAL_INPUT_TOKENS_MEASURED,
) -> Dict[str, ParallelismCostResult]:
    """
    Compare TP, DP, and HYBRID parallelism modes analytically.

    ALL results are HYPOTHETICAL MULTI-DEVICE PROJECTIONS (see
    MULTI_DEVICE_DISCLAIMER): they model `tp_size`/`n_workers` separate
    devices and must not be interpreted as realizable single-M3 gains.
    Any printed summary of these results must state this explicitly.

    Returns a dict with keys 'TP', 'DP', 'HYBRID' and as a bonus,
    'recommended' pointing to the lowest-latency option.
    """
    # --- TP for vision encoder ---
    # Total visual tokens = n_crops × 81 (per-crop token count from the
    # checkpoint config, confirmed by the v2 measurement campaign).
    tp_vision = compute_tp_cost(
        t_compute_ms=t_vision_ms,
        n_layers=SIGLIP_N_LAYERS,
        hidden_dim=SIGLIP_HIDDEN_DIM,
        sync_per_layer=SIGLIP_SYNC_PER_LAYER,
        seq_len=n_crops * TOKENS_PER_CROP,
        tp_size=tp_size,
    )
    # TP for LM
    tp_lm = compute_tp_cost(
        t_compute_ms=t_lm_ms,
        n_layers=LM_N_LAYERS,
        hidden_dim=LM_HIDDEN_DIM,
        sync_per_layer=LM_SYNC_PER_LAYER,
        seq_len=seq_len,
        tp_size=tp_size,
    )
    tp_total = ParallelismCostResult(
        mode=ParallelismMode.TP,
        t_compute_ms=tp_vision.t_compute_ms + tp_lm.t_compute_ms,
        t_communication_ms=tp_vision.t_communication_ms + tp_lm.t_communication_ms,
        t_total_ms=tp_vision.t_total_ms + tp_lm.t_total_ms,
        n_sync_points=tp_vision.n_sync_points + tp_lm.n_sync_points,
        throughput_gain_pct=(
            (t_vision_ms + t_lm_ms - tp_vision.t_total_ms - tp_lm.t_total_ms)
            / (t_vision_ms + t_lm_ms) * 100.0
        ),
        notes="TP applied to both vision encoder and LM",
    )

    # --- DP for vision crops only ---
    dp_vision = compute_dp_cost(
        t_compute_ms=t_vision_ms,
        n_crops=n_crops,
        n_workers=n_workers,
        hidden_dim=SIGLIP_HIDDEN_DIM,
    )
    dp_total = ParallelismCostResult(
        mode=ParallelismMode.DP,
        t_compute_ms=dp_vision.t_compute_ms + t_lm_ms,
        t_communication_ms=dp_vision.t_communication_ms,
        t_total_ms=dp_vision.t_total_ms + t_lm_ms,
        n_sync_points=1,
        throughput_gain_pct=(
            (t_vision_ms + t_lm_ms - dp_vision.t_total_ms - t_lm_ms)
            / (t_vision_ms + t_lm_ms) * 100.0
        ),
        notes="DP over vision crops; LM runs sequentially after",
    )

    # --- HYBRID: DP for vision, TP for LM ---
    hybrid_total = ParallelismCostResult(
        mode=ParallelismMode.HYBRID,
        t_compute_ms=dp_vision.t_compute_ms + tp_lm.t_compute_ms,
        t_communication_ms=dp_vision.t_communication_ms + tp_lm.t_communication_ms,
        t_total_ms=dp_vision.t_total_ms + tp_lm.t_total_ms,
        n_sync_points=1 + tp_lm.n_sync_points,
        throughput_gain_pct=(
            (t_vision_ms + t_lm_ms - dp_vision.t_total_ms - tp_lm.t_total_ms)
            / (t_vision_ms + t_lm_ms) * 100.0
        ),
        notes="DP vision crops + TP language model",
    )

    results = {
        "TP": tp_total,
        "DP": dp_total,
        "HYBRID": hybrid_total,
    }
    best_key = min(results, key=lambda k: results[k].t_total_ms)
    results["recommended"] = results[best_key]

    return results


@dataclass
class CommunicationConfig:
    """
    Configuration for TP communication simulation.

    The per-byte cost is DERIVED from `bandwidth_gbps` so the two can never
    contradict: at 100 GB/s one byte takes 0.01 ns (the earlier hardcoded
    10 ns/B was a 1000× error — it corresponds to 0.1 GB/s).
    Pass `per_byte_latency_ns` explicitly only to override the derivation.
    """
    base_latency_us: float = 50.0        # Base synchronization overhead
    bandwidth_gbps: float = M3_BANDWIDTH_GBps  # link bandwidth (GB/s)
    per_byte_latency_ns: Optional[float] = None  # derived if None
    sync_overhead_us: float = 20.0       # Barrier/coordination overhead
    tp_size: int = 2                     # Simulated TP degree

    def __post_init__(self):
        if self.per_byte_latency_ns is None:
            # ns per byte = 1 / (GB/s)  →  0.01 ns/B at 100 GB/s
            self.per_byte_latency_ns = 1.0 / self.bandwidth_gbps


@dataclass
class SyncPoint:
    """Record of a synchronization point"""
    operation: OperationType
    layer_idx: int
    tensor_size_bytes: int
    latency_us: float
    timestamp: float
    component: str = "unknown"   # "vision" or "language" — explicit tag,
                                 # never guessed from layer_idx


class TPSimulator:
    """
    Simulate Tensor Parallelism communication overhead
    
    Injects artificial delays at All-Reduce synchronization points to model
    the communication cost of multi-GPU tensor parallel inference.
    """
    
    def __init__(self, config: CommunicationConfig = None):
        self.config = config or CommunicationConfig()
        self.enabled = False
        self.sync_history: List[SyncPoint] = []

    def enable(self):
        """Enable TP simulation"""
        if not _MLX_AVAILABLE:
            raise RuntimeError("mlx is not installed; TP injection simulation requires mlx.")
        self.enabled = True
        print("TP simulation enabled")
        
    def disable(self):
        """Disable TP simulation"""
        self.enabled = False
        print("TP simulation disabled")
        
    def _calculate_latency(self, tensor_size_bytes: int) -> float:
        """
        Calculate all-reduce latency for given tensor size
        
        Formula: latency = base + (size * per_byte) + sync_overhead
        
        Args:
            tensor_size_bytes: Size of tensor in bytes
            
        Returns:
            Latency in microseconds
        """
        bandwidth_latency = tensor_size_bytes * self.config.per_byte_latency_ns / 1000.0
        total_latency = (
            self.config.base_latency_us +
            bandwidth_latency +
            self.config.sync_overhead_us
        )
        return total_latency
    
    def simulate_all_reduce(
        self,
        tensor: Any,
        operation: OperationType,
        layer_idx: int = 0,
        component: str = "unknown",
    ) -> Any:
        """
        Simulate all-reduce operation with artificial delay.

        The payload is always computed from the ACTUAL tensor (size ×
        itemsize), so it scales with seq_len exactly like the analytical
        path — the earlier hardcoded per-operation table silently assumed
        seq_len=1 payloads and has been removed.

        Args:
            tensor: Input tensor
            operation: Type of operation
            layer_idx: Layer index for tracking
            component: "vision" or "language" — explicit component tag

        Returns:
            Same tensor after simulated delay
        """
        if not self.enabled:
            return tensor

        # Payload from real tensor shape (includes seq_len).  Layer norm
        # is a pure barrier: no data transfer.
        if operation == OperationType.LAYER_NORM:
            tensor_size_bytes = 0
        else:
            tensor_size_bytes = tensor.size * tensor.itemsize
        latency_us = self._calculate_latency(tensor_size_bytes)

        # Record sync point
        sync_point = SyncPoint(
            operation=operation,
            layer_idx=layer_idx,
            tensor_size_bytes=tensor_size_bytes,
            latency_us=latency_us,
            timestamp=time.time(),
            component=component,
        )
        self.sync_history.append(sync_point)
        
        # Force evaluation and inject delay
        if _MLX_AVAILABLE:
            mx.eval(tensor)
        time.sleep(latency_us / 1e6)  # Convert microseconds to seconds
        
        return tensor
    
    def simulate_vision_encoder_layer(
        self,
        layer_output: Any,
        layer_idx: int
    ) -> Any:
        """
        Simulate all synchronization points in a vision encoder layer
        
        Vision encoder layer has 5 sync points:
        1. QKV projection
        2. Attention output
        3. FFN intermediate
        4. FFN output
        5. Layer norm sync
        
        Args:
            layer_output: Output tensor from layer
            layer_idx: Layer index
            
        Returns:
            Output tensor after all simulated delays
        """
        if not self.enabled:
            return layer_output
        
        # Simulate all 5 sync points
        for op_type in [
            OperationType.QKV_PROJECTION,
            OperationType.ATTENTION_OUTPUT,
            OperationType.FFN_INTERMEDIATE,
            OperationType.FFN_OUTPUT,
            OperationType.LAYER_NORM
        ]:
            layer_output = self.simulate_all_reduce(
                layer_output,
                op_type,
                layer_idx,
                component="vision",
            )

        return layer_output
    
    def simulate_language_model_layer(
        self,
        layer_output: Any,
        layer_idx: int
    ) -> Any:
        """
        Simulate synchronization points in a language model layer
        
        Language model layer has 4 sync points:
        1. QKV projection
        2. Attention output
        3. FFN gate/up projection
        4. FFN down projection
        
        Args:
            layer_output: Output tensor from layer
            layer_idx: Layer index
            
        Returns:
            Output tensor after simulated delays
        """
        if not self.enabled:
            return layer_output
        
        # Simulate all 4 sync points
        for op_type in [
            OperationType.QKV_PROJECTION,
            OperationType.ATTENTION_OUTPUT,
            OperationType.FFN_INTERMEDIATE,  # Gate/up
            OperationType.FFN_OUTPUT         # Down
        ]:
            layer_output = self.simulate_all_reduce(
                layer_output,
                op_type,
                layer_idx,
                component="language",
            )

        return layer_output
    
    def get_total_overhead_ms(self) -> float:
        """Calculate total communication overhead from history"""
        return sum(sp.latency_us for sp in self.sync_history) / 1000.0
    
    def get_overhead_by_operation(self) -> Dict[OperationType, float]:
        """Get overhead breakdown by operation type"""
        overhead = {}
        for op_type in OperationType:
            overhead[op_type] = sum(
                sp.latency_us for sp in self.sync_history 
                if sp.operation == op_type
            ) / 1000.0
        return overhead
    
    def get_overhead_by_layer(self) -> Dict[int, float]:
        """Get overhead breakdown by layer"""
        overhead = {}
        for sp in self.sync_history:
            if sp.layer_idx not in overhead:
                overhead[sp.layer_idx] = 0.0
            overhead[sp.layer_idx] += sp.latency_us / 1000.0
        return overhead
    
    def reset_history(self):
        """Clear synchronization history"""
        self.sync_history = []
    
    def print_summary(self):
        """Print summary of simulated communication overhead"""
        print("\n" + "=" * 80)
        print("TP Simulation Summary")
        print("=" * 80)
        
        total_overhead = self.get_total_overhead_ms()
        print(f"Total Communication Overhead: {total_overhead:.2f} ms")
        print(f"Number of Sync Points: {len(self.sync_history)}")
        print(f"TP Size: {self.config.tp_size}")
        
        print("\nOverhead by Operation Type:")
        print("-" * 80)
        overhead_by_op = self.get_overhead_by_operation()
        for op_type, overhead_ms in sorted(
            overhead_by_op.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if overhead_ms > 0:
                percentage = (overhead_ms / total_overhead) * 100
                print(f"  {op_type.value:25s}: {overhead_ms:7.2f} ms ({percentage:5.1f}%)")
        
        print("\nOverhead by Component (from explicit SyncPoint tags):")
        print("-" * 80)
        components: Dict[str, float] = {}
        for sp in self.sync_history:
            components[sp.component] = (
                components.get(sp.component, 0.0) + sp.latency_us / 1000.0
            )
        for comp, overhead_ms in sorted(components.items()):
            print(f"  {comp:<16}: {overhead_ms:.2f} ms")

        print("=" * 80)


class VisionEncoderWithTP:
    """Vision encoder wrapper with TP simulation"""
    
    def __init__(self, num_layers: int, tp_simulator: TPSimulator):
        self.num_layers = num_layers
        self.tp_simulator = tp_simulator
        
    def forward(self, x: Any) -> Any:
        """Forward pass with TP simulation"""
        for layer_idx in range(self.num_layers):
            # Simulate layer computation
            # In real implementation, this would be actual layer forward pass
            
            # Simulate TP overhead for this layer
            x = self.tp_simulator.simulate_vision_encoder_layer(x, layer_idx)
        
        return x


class LanguageModelWithTP:
    """Language model wrapper with TP simulation"""
    
    def __init__(self, num_layers: int, tp_simulator: TPSimulator):
        self.num_layers = num_layers
        self.tp_simulator = tp_simulator
        
    def forward(self, x: Any) -> Any:
        """Forward pass with TP simulation"""
        for layer_idx in range(self.num_layers):
            # Simulate TP overhead for this layer.  No layer-index offset:
            # SyncPoints carry an explicit component tag instead.
            x = self.tp_simulator.simulate_language_model_layer(x, layer_idx)

        return x


if __name__ == "__main__":
    """Test TP simulation and Phase 3 parallelism cost comparison"""
    print("=" * 80)
    print("AMIO Phase 3 - TP Simulation + Parallelism Cost Comparison")
    print("=" * 80)
    print()

    # --- Analytical cost comparison (no MLX execution needed) ---
    print("Analytical Parallelism Cost Comparison")
    print(f"NOTE: {MULTI_DEVICE_DISCLAIMER}")
    print("-" * 80)
    results = compare_parallelism_modes(
        t_vision_ms=BASELINE_T_VISION_MS,
        t_lm_ms=BASELINE_T_LM_MS,
        n_crops=BASELINE_N_CROPS,
        n_workers=2,
        tp_size=2,
        seq_len=TOTAL_INPUT_TOKENS_MEASURED,
    )
    baseline_total = BASELINE_T_VISION_MS + BASELINE_T_LM_MS
    print(f"  Baseline (sequential): {baseline_total:.1f} ms")
    print()
    for key in ("TP", "DP", "HYBRID"):
        r = results[key]
        print(f"  {key:6s}: compute={r.t_compute_ms:7.1f} ms  "
              f"comm={r.t_communication_ms:7.1f} ms  "
              f"total={r.t_total_ms:7.1f} ms  "
              f"gain={r.throughput_gain_pct:+5.1f}%")
        print(f"          {r.notes}")
    print()
    print(f"  Recommended: {results['recommended'].mode.value.upper()}")
    print()

    # --- Original TP injection test (Phase 0) ---
    print("=" * 80)
    print("AMIO Phase 0 - TP Injection Simulation Test")
    print("=" * 80)
    print()

    if not _MLX_AVAILABLE:
        print("  [WARN] mlx not available — skipping injection simulation.")
        print("     Install mlx (scope/venv_phase0) to run Phase 0 tests.")
        sys.exit(0)

    # Create simulator — per-byte cost derived from bandwidth (0.01 ns/B
    # at 100 GB/s; the old hardcoded 10 ns/B was a 1000× error)
    config = CommunicationConfig(
        base_latency_us=50.0,
        bandwidth_gbps=M3_BANDWIDTH_GBps,
        sync_overhead_us=20.0,
        tp_size=2
    )
    simulator = TPSimulator(config)
    print(f"Derived per-byte latency: {config.per_byte_latency_ns:.4f} ns/B "
          f"({config.bandwidth_gbps:.0f} GB/s)")

    # Example latencies for real SmolVLM payloads (FP16, prefill-scale:
    # 1377 image tokens / 1560 total input tokens at 17 crops)
    print(f"\nExample All-Reduce latencies (SmolVLM shapes, "
          f"seq_len={TOTAL_INPUT_TOKENS_MEASURED}):")
    print("-" * 80)
    for label, nbytes in [
        (f"vision hidden ({SIGLIP_HIDDEN_DIM}) x {TOTAL_VISUAL_TOKENS_MEASURED} tok",
         SIGLIP_HIDDEN_DIM * TOTAL_VISUAL_TOKENS_MEASURED * 2),
        (f"LM hidden ({LM_HIDDEN_DIM}) x {TOTAL_INPUT_TOKENS_MEASURED} tok",
         LM_HIDDEN_DIM * TOTAL_INPUT_TOKENS_MEASURED * 2),
        (f"LM hidden ({LM_HIDDEN_DIM}) x 1 tok (decode)", LM_HIDDEN_DIM * 2),
    ]:
        print(f"  {label:40s}: {simulator._calculate_latency(nbytes):8.1f} us")

    print("\n")

    # Test vision encoder simulation — real SmolVLM/SigLIP shapes:
    # 27 layers, hidden 1152, 1377 image tokens at the 17-crop maximum
    print(f"Simulating Vision Encoder ({SIGLIP_N_LAYERS} layers, "
          f"hidden {SIGLIP_HIDDEN_DIM}):")
    print("-" * 80)
    simulator.enable()
    simulator.reset_history()

    x = mx.zeros((1, TOTAL_VISUAL_TOKENS_MEASURED, SIGLIP_HIDDEN_DIM))

    start_time = time.time()
    vision_encoder = VisionEncoderWithTP(
        num_layers=SIGLIP_N_LAYERS, tp_simulator=simulator
    )
    x = vision_encoder.forward(x)
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"Elapsed time: {elapsed_ms:.1f} ms")
    print(f"Simulated overhead: {simulator.get_total_overhead_ms():.1f} ms")

    print("\n")

    # Test language model simulation — real SmolVLM LM shapes:
    # 24 layers, hidden 2048 (SmolLM2-1.7B-class)
    print(f"Simulating Language Model ({LM_N_LAYERS} layers, "
          f"hidden {LM_HIDDEN_DIM}):")
    print("-" * 80)

    x = mx.zeros((1, 64, LM_HIDDEN_DIM))  # [batch, seq_len, hidden_dim]

    start_time = time.time()
    language_model = LanguageModelWithTP(
        num_layers=LM_N_LAYERS, tp_simulator=simulator
    )
    x = language_model.forward(x)
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"Elapsed time: {elapsed_ms:.1f} ms")
    print(f"Simulated overhead (vision + LM): "
          f"{simulator.get_total_overhead_ms():.1f} ms")

    # Print summary (components split by explicit tags)
    simulator.print_summary()

    print("\nPhase 3 tp_simulator extension complete")
