"""
Tensor Parallelism Simulation for Single-GPU Systems

Simulates multi-GPU TP communication overhead on M3 Mac by injecting
artificial delays at synchronization points (All-Reduce operations).

This allows training an adaptive controller on single-GPU hardware that
understands multi-GPU communication costs.

Phase 3 extensions:
  - ParallelismMode enum (TP / DP / HYBRID)
  - Analytical cost functions — compute_tp_cost(), compute_dp_cost()
  - compare_parallelism_modes() returning recommended mode + gain %
"""

import time
import math
import sys
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

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
    """Result of a TP vs DP cost comparison"""
    mode: ParallelismMode
    t_compute_ms: float
    t_communication_ms: float
    t_total_ms: float
    n_sync_points: int
    throughput_gain_pct: float        # vs naive sequential baseline
    notes: str = ""


# ---------------------------------------------------------------------------
# Analytical cost functions (no actual MLX execution required)
# ---------------------------------------------------------------------------

# SmolVLM / SigLIP vision encoder configuration
SIGLIP_N_LAYERS = 27          # SigLIP-So400M depth (SmolVLM uses)
SIGLIP_HIDDEN_DIM = 1152      # SigLIP hidden dimension
SIGLIP_SYNC_PER_LAYER = 5     # QKV, attn-out, FFN-up, FFN-down, layernorm

# SmolVLM language model configuration
LM_N_LAYERS = 24              # Qwen2-style LM depth
LM_HIDDEN_DIM = 2048          # Calibrated in Phase 2
LM_SYNC_PER_LAYER = 4         # QKV, attn-out, FFN-gate/up, FFN-down

# M3 hardware constants
M3_BANDWIDTH_GBps = 100.0     # Unified memory bandwidth
M3_SYNC_LATENCY_us = 50.0     # Base all-reduce / barrier overhead (µs)
M3_SYNC_OVERHEAD_us = 20.0    # Extra per-sync coordination overhead (µs)

# Phase 2 calibrated baseline
BASELINE_T_VISION_MS = 5991.0   # T_vision at 24 crops, sequential
BASELINE_T_LM_MS = 2492.0       # T_lm_prefill at N=1548 tokens
BASELINE_N_CROPS = 24


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
    seq_len: int = 1548,
    tp_size: int = 2,
) -> ParallelismCostResult:
    """
    Analytical cost of Tensor-Parallel execution.

    TP shards every weight matrix across `tp_size` workers.  After every
    collective op, an All-Reduce is needed.  The payload of each All-Reduce
    is proportional to hidden_dim * seq_len (FP16 = 2 bytes).

    Parameters
    ----------
    t_compute_ms : baseline sequential compute time (ms)
    n_layers     : number of transformer layers
    hidden_dim   : model hidden dimension
    sync_per_layer: number of All-Reduce calls per layer
    seq_len      : sequence length at this stage
    tp_size      : TP degree (set to 2 for M3 simulation)

    Returns
    -------
    ParallelismCostResult
    """
    tensor_bytes = hidden_dim * seq_len * 2  # FP16
    cost_per_sync_us = _sync_cost_us(tensor_bytes)
    n_sync_total = n_layers * sync_per_layer
    t_comm_ms = (n_sync_total * cost_per_sync_us) / 1000.0

    # TP splits compute evenly when weight matrices are column/row sharded,
    # so theoretical compute ≈ t_compute_ms / tp_size, but communication
    # overhead is added back.
    t_total_ms = t_compute_ms / tp_size + t_comm_ms
    gain_pct = (t_compute_ms - t_total_ms) / t_compute_ms * 100.0

    return ParallelismCostResult(
        mode=ParallelismMode.TP,
        t_compute_ms=t_compute_ms / tp_size,
        t_communication_ms=t_comm_ms,
        t_total_ms=t_total_ms,
        n_sync_points=n_sync_total,
        throughput_gain_pct=gain_pct,
        notes=(f"TP-{tp_size}: {n_sync_total} sync pts, "
               f"{t_comm_ms:.1f} ms comm overhead"),
    )


def compute_dp_cost(
    t_compute_ms: float,
    n_crops: int,
    n_workers: int = 2,
    hidden_dim: int = SIGLIP_HIDDEN_DIM,
    tokens_per_crop: int = 256,
) -> ParallelismCostResult:
    """
    Analytical cost of Data-Parallel execution over vision crops.

    In DP mode each worker processes a disjoint subset of crops
    (n_crops / n_workers each).  Results are fused with a single
    All-Gather whose payload is all crop embeddings combined.

    Parameters
    ----------
    t_compute_ms  : baseline sequential compute time (ms) for ALL crops
    n_crops       : number of image crops to process
    n_workers     : DP degree (number of parallel workers / SMs partitions)
    hidden_dim    : encoder output hidden dimension
    tokens_per_crop: patch tokens emitted per crop

    Returns
    -------
    ParallelismCostResult
    """
    all_gather_bytes = n_crops * tokens_per_crop * hidden_dim * 2  # FP16
    t_comm_ms = _sync_cost_us(all_gather_bytes) / 1000.0  # 1 AllGather

    # Each worker handles ⌈crops / workers⌉ crops in parallel
    crops_per_worker = math.ceil(n_crops / n_workers)
    # Compute scales linearly with crop count
    t_parallel_compute_ms = t_compute_ms * (crops_per_worker / n_crops)
    t_total_ms = t_parallel_compute_ms + t_comm_ms

    gain_pct = (t_compute_ms - t_total_ms) / t_compute_ms * 100.0

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
    seq_len: int = 1548,
) -> Dict[str, ParallelismCostResult]:
    """
    Compare TP, DP, and HYBRID parallelism modes analytically.

    Returns a dict with keys 'TP', 'DP', 'HYBRID' and as a bonus,
    'recommended' pointing to the lowest-latency option.
    """
    # --- TP for vision encoder ---
    tp_vision = compute_tp_cost(
        t_compute_ms=t_vision_ms,
        n_layers=SIGLIP_N_LAYERS,
        hidden_dim=SIGLIP_HIDDEN_DIM,
        sync_per_layer=SIGLIP_SYNC_PER_LAYER,
        seq_len=n_crops * 256,   # approximate total patch tokens
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
    """Configuration for TP communication simulation"""
    base_latency_us: float = 50.0        # Base synchronization overhead
    per_byte_latency_ns: float = 10.0    # Network bandwidth (100 GB/s)
    sync_overhead_us: float = 20.0       # Barrier/coordination overhead
    tp_size: int = 2                     # Simulated TP degree


@dataclass
class SyncPoint:
    """Record of a synchronization point"""
    operation: OperationType
    layer_idx: int
    tensor_size_bytes: int
    latency_us: float
    timestamp: float


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
        
        # Pre-computed latencies for common operations (microseconds)
        self.operation_latencies = self._compute_operation_latencies()
        
    def enable(self):
        """Enable TP simulation"""
        if not _MLX_AVAILABLE:
            raise RuntimeError("mlx is not installed; TP injection simulation requires mlx.")
        self.enabled = True
        print("✅ TP simulation enabled")
        
    def disable(self):
        """Disable TP simulation"""
        self.enabled = False
        print("⏸️  TP simulation disabled")
        
    def _compute_operation_latencies(self) -> Dict[OperationType, float]:
        """Pre-compute latencies for common operation types"""
        # Tensor sizes for different operations (bytes, assuming fp16)
        tensor_sizes = {
            # Vision encoder (hidden_dim=1024)
            OperationType.QKV_PROJECTION: 6144,      # 3 * 1024 * 2
            OperationType.ATTENTION_OUTPUT: 2048,    # 1024 * 2
            OperationType.FFN_INTERMEDIATE: 8192,    # 4096 * 2
            OperationType.FFN_OUTPUT: 2048,          # 1024 * 2
            OperationType.LAYER_NORM: 0,             # No data transfer, just sync
            OperationType.PROJECTION: 8192,          # Projection layer
        }
        
        latencies = {}
        for op_type, size_bytes in tensor_sizes.items():
            latencies[op_type] = self._calculate_latency(size_bytes)
            
        return latencies
    
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
        layer_idx: int = 0
    ) -> Any:
        """
        Simulate all-reduce operation with artificial delay
        
        Args:
            tensor: Input tensor
            operation: Type of operation
            layer_idx: Layer index for tracking
            
        Returns:
            Same tensor after simulated delay
        """
        if not self.enabled:
            return tensor
        
        # Get pre-computed latency or calculate it
        if operation in self.operation_latencies:
            latency_us = self.operation_latencies[operation]
        else:
            tensor_size_bytes = tensor.size * tensor.itemsize
            latency_us = self._calculate_latency(tensor_size_bytes)
        
        # Record sync point
        sync_point = SyncPoint(
            operation=operation,
            layer_idx=layer_idx,
            tensor_size_bytes=tensor.size * tensor.itemsize,
            latency_us=latency_us,
            timestamp=time.time()
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
                layer_idx
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
                layer_idx
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
        
        print("\nOverhead by Component:")
        print("-" * 80)
        # Estimate component breakdown
        vision_overhead = sum(sp.latency_us for sp in self.sync_history 
                             if sp.layer_idx < 24) / 1000.0
        language_overhead = sum(sp.latency_us for sp in self.sync_history 
                               if sp.layer_idx >= 24) / 1000.0
        
        print(f"  Vision Encoder: {vision_overhead:.2f} ms")
        print(f"  Language Model: {language_overhead:.2f} ms")
        
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
            # Simulate TP overhead for this layer
            x = self.tp_simulator.simulate_language_model_layer(
                x, 
                layer_idx + 24  # Offset for vision encoder layers
            )
        
        return x


if __name__ == "__main__":
    """Test TP simulation and Phase 3 parallelism cost comparison"""
    print("=" * 80)
    print("AMIO Phase 3 - TP Simulation + Parallelism Cost Comparison")
    print("=" * 80)
    print()

    # --- Analytical cost comparison (no MLX execution needed) ---
    print("Analytical Parallelism Cost Comparison")
    print("-" * 80)
    results = compare_parallelism_modes(
        t_vision_ms=BASELINE_T_VISION_MS,
        t_lm_ms=BASELINE_T_LM_MS,
        n_crops=BASELINE_N_CROPS,
        n_workers=2,
        tp_size=2,
        seq_len=1548,
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
    print(f"  ✅ Recommended: {results['recommended'].mode.value.upper()}")
    print()

    # --- Original TP injection test (Phase 0) ---
    print("=" * 80)
    print("AMIO Phase 0 - TP Injection Simulation Test")
    print("=" * 80)
    print()

    if not _MLX_AVAILABLE:
        print("  ⚠️  mlx not available — skipping injection simulation.")
        print("     Install mlx (scope/venv_phase0) to run Phase 0 tests.")
        sys.exit(0)

    # Create simulator
    config = CommunicationConfig(
        base_latency_us=50.0,
        per_byte_latency_ns=10.0,
        sync_overhead_us=20.0,
        tp_size=2
    )
    simulator = TPSimulator(config)

    # Test latency calculations
    print("Operation Latencies (microseconds):")
    print("-" * 80)
    for op_type, latency in simulator.operation_latencies.items():
        print(f"  {op_type.value:25s}: {latency:6.1f} us")

    print("\n")

    # Test vision encoder simulation
    print("Simulating Vision Encoder (24 layers):")
    print("-" * 80)
    simulator.enable()
    simulator.reset_history()

    # Dummy tensor
    x = mx.zeros((1, 576, 1024))  # [batch, patches, hidden_dim]

    start_time = time.time()
    vision_encoder = VisionEncoderWithTP(num_layers=24, tp_simulator=simulator)
    x = vision_encoder.forward(x)
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"Elapsed time: {elapsed_ms:.1f} ms")
    print(f"Simulated overhead: {simulator.get_total_overhead_ms():.1f} ms")

    print("\n")

    # Test language model simulation
    print("Simulating Language Model (32 layers):")
    print("-" * 80)
    simulator.reset_history()

    x = mx.zeros((1, 64, 4096))  # [batch, seq_len, hidden_dim]

    start_time = time.time()
    language_model = LanguageModelWithTP(num_layers=32, tp_simulator=simulator)
    x = language_model.forward(x)
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"Elapsed time: {elapsed_ms:.1f} ms")
    print(f"Simulated overhead: {simulator.get_total_overhead_ms():.1f} ms")

    # Print summary
    simulator.print_summary()

    print("\n✅ Phase 3 tp_simulator extension complete")
