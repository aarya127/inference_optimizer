"""
Tensor Parallelism Simulation for Single-GPU Systems

Simulates multi-GPU TP communication overhead on M3 Mac by injecting
artificial delays at synchronization points (All-Reduce operations).

This allows training an adaptive controller on single-GPU hardware that
understands multi-GPU communication costs.
"""

import time
import mlx.core as mx
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class OperationType(Enum):
    """Types of operations requiring all-reduce"""
    QKV_PROJECTION = "qkv_projection"
    ATTENTION_OUTPUT = "attention_output"
    FFN_INTERMEDIATE = "ffn_intermediate"
    FFN_OUTPUT = "ffn_output"
    LAYER_NORM = "layer_norm"
    PROJECTION = "projection"


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
        tensor: mx.array,
        operation: OperationType,
        layer_idx: int = 0
    ) -> mx.array:
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
        mx.eval(tensor)
        time.sleep(latency_us / 1e6)  # Convert microseconds to seconds
        
        return tensor
    
    def simulate_vision_encoder_layer(
        self,
        layer_output: mx.array,
        layer_idx: int
    ) -> mx.array:
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
        layer_output: mx.array,
        layer_idx: int
    ) -> mx.array:
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
        
    def forward(self, x: mx.array) -> mx.array:
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
        
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass with TP simulation"""
        for layer_idx in range(self.num_layers):
            # Simulate TP overhead for this layer
            x = self.tp_simulator.simulate_language_model_layer(
                x, 
                layer_idx + 24  # Offset for vision encoder layers
            )
        
        return x


if __name__ == "__main__":
    """Test TP simulation"""
    print("=" * 80)
    print("AMIO Phase 0 - TP Simulation Test")
    print("=" * 80)
    print()
    
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
    
    print("\n✅ TP simulation test complete")
