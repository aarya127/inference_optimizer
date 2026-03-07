"""
Core Performance Metrics for AMIO Phase 0

Defines and implements the three primary metrics for multimodal inference optimization:
1. Time-To-First-Token (TTFT) - Primary bottleneck for multimodal
2. Time-Between-Tokens (TBT) - Generation smoothness
3. Memory Fragmentation - Memory waste percentage
"""

import time
import mlx.core as mx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psutil
import json


class MetricType(Enum):
    """Types of metrics tracked"""
    TTFT = "time_to_first_token"
    TBT = "time_between_tokens"
    FRAGMENTATION = "memory_fragmentation"
    THROUGHPUT = "throughput"
    LATENCY = "end_to_end_latency"


@dataclass
class TTFTMetrics:
    """Time-To-First-Token metrics"""
    # Component breakdown
    image_preprocessing_ms: float = 0.0
    vision_encoding_ms: float = 0.0
    projection_ms: float = 0.0
    prompt_processing_ms: float = 0.0
    first_token_generation_ms: float = 0.0
    
    # Communication overhead (if TP enabled)
    tp_overhead_ms: float = 0.0
    
    # Total TTFT
    total_ttft_ms: float = 0.0
    
    # Metadata
    image_size: Tuple[int, int] = (0, 0)
    prompt_length: int = 0
    batch_size: int = 1
    
    def compute_total(self):
        """Compute total TTFT from components"""
        self.total_ttft_ms = (
            self.image_preprocessing_ms +
            self.vision_encoding_ms +
            self.projection_ms +
            self.prompt_processing_ms +
            self.first_token_generation_ms +
            self.tp_overhead_ms
        )
        return self.total_ttft_ms
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'image_preprocessing_ms': self.image_preprocessing_ms,
            'vision_encoding_ms': self.vision_encoding_ms,
            'projection_ms': self.projection_ms,
            'prompt_processing_ms': self.prompt_processing_ms,
            'first_token_generation_ms': self.first_token_generation_ms,
            'tp_overhead_ms': self.tp_overhead_ms,
            'total_ttft_ms': self.total_ttft_ms,
            'image_size': self.image_size,
            'prompt_length': self.prompt_length,
            'batch_size': self.batch_size
        }


@dataclass
class TBTMetrics:
    """Time-Between-Tokens metrics"""
    # Per-token latencies (ms)
    token_latencies: List[float] = field(default_factory=list)
    
    # Statistics
    mean_tbt_ms: float = 0.0
    median_tbt_ms: float = 0.0
    p50_tbt_ms: float = 0.0
    p90_tbt_ms: float = 0.0
    p99_tbt_ms: float = 0.0
    min_tbt_ms: float = 0.0
    max_tbt_ms: float = 0.0
    std_tbt_ms: float = 0.0
    
    # Throughput
    tokens_per_second: float = 0.0
    
    # Metadata
    num_tokens: int = 0
    batch_size: int = 1
    
    def add_token_latency(self, latency_ms: float):
        """Add a token latency measurement"""
        self.token_latencies.append(latency_ms)
        self.num_tokens += 1
    
    def compute_statistics(self):
        """Compute statistics from collected latencies"""
        if not self.token_latencies:
            return
        
        import numpy as np
        latencies = np.array(self.token_latencies)
        
        self.mean_tbt_ms = float(np.mean(latencies))
        self.median_tbt_ms = float(np.median(latencies))
        self.p50_tbt_ms = float(np.percentile(latencies, 50))
        self.p90_tbt_ms = float(np.percentile(latencies, 90))
        self.p99_tbt_ms = float(np.percentile(latencies, 99))
        self.min_tbt_ms = float(np.min(latencies))
        self.max_tbt_ms = float(np.max(latencies))
        self.std_tbt_ms = float(np.std(latencies))
        
        # Throughput (tokens/second)
        if self.mean_tbt_ms > 0:
            self.tokens_per_second = 1000.0 / self.mean_tbt_ms
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'mean_tbt_ms': self.mean_tbt_ms,
            'median_tbt_ms': self.median_tbt_ms,
            'p50_tbt_ms': self.p50_tbt_ms,
            'p90_tbt_ms': self.p90_tbt_ms,
            'p99_tbt_ms': self.p99_tbt_ms,
            'min_tbt_ms': self.min_tbt_ms,
            'max_tbt_ms': self.max_tbt_ms,
            'std_tbt_ms': self.std_tbt_ms,
            'tokens_per_second': self.tokens_per_second,
            'num_tokens': self.num_tokens,
            'batch_size': self.batch_size
        }


@dataclass
class FragmentationMetrics:
    """Memory fragmentation metrics"""
    # Memory usage (MB)
    allocated_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    wasted_memory_mb: float = 0.0
    
    # Fragmentation percentage
    fragmentation_percent: float = 0.0
    
    # Breakdown
    model_weights_mb: float = 0.0
    kv_cache_mb: float = 0.0
    activations_mb: float = 0.0
    gradients_mb: float = 0.0
    optimizer_state_mb: float = 0.0
    fragmentation_mb: float = 0.0
    
    # System memory
    system_memory_total_gb: float = 0.0
    system_memory_available_gb: float = 0.0
    system_memory_percent: float = 0.0
    
    def compute_fragmentation(self):
        """Compute fragmentation percentage"""
        if self.allocated_memory_mb > 0:
            self.wasted_memory_mb = self.allocated_memory_mb - self.used_memory_mb
            self.fragmentation_percent = (
                self.wasted_memory_mb / self.allocated_memory_mb
            ) * 100.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'allocated_memory_mb': self.allocated_memory_mb,
            'used_memory_mb': self.used_memory_mb,
            'wasted_memory_mb': self.wasted_memory_mb,
            'fragmentation_percent': self.fragmentation_percent,
            'model_weights_mb': self.model_weights_mb,
            'kv_cache_mb': self.kv_cache_mb,
            'activations_mb': self.activations_mb,
            'system_memory_total_gb': self.system_memory_total_gb,
            'system_memory_available_gb': self.system_memory_available_gb,
            'system_memory_percent': self.system_memory_percent
        }


@dataclass
class InferenceMetrics:
    """Complete inference metrics for a single request"""
    ttft: TTFTMetrics = field(default_factory=TTFTMetrics)
    tbt: TBTMetrics = field(default_factory=TBTMetrics)
    fragmentation: FragmentationMetrics = field(default_factory=FragmentationMetrics)
    
    # Request metadata
    request_id: str = ""
    timestamp: float = 0.0
    model_id: str = ""
    quantization: str = "int4"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'request_id': self.request_id,
            'timestamp': self.timestamp,
            'model_id': self.model_id,
            'quantization': self.quantization,
            'ttft': self.ttft.to_dict(),
            'tbt': self.tbt.to_dict(),
            'fragmentation': self.fragmentation.to_dict()
        }


class MetricsCollector:
    """
    Collects and tracks performance metrics during inference
    
    Usage:
        collector = MetricsCollector()
        
        # Start tracking
        collector.start_ttft_measurement()
        
        # Mark components
        collector.mark_component("vision_encoding")
        # ... do work ...
        collector.mark_component("projection")
        
        # Track TBT
        for token in generate():
            collector.start_token_measurement()
            # ... generate token ...
            collector.end_token_measurement()
        
        # Get metrics
        metrics = collector.get_metrics()
    """
    
    def __init__(self):
        self.current_metrics = InferenceMetrics()
        self.ttft_start_time = 0.0
        self.last_component_time = 0.0
        self.token_start_time = 0.0
        self.measurement_active = False
        
    def start_ttft_measurement(self, request_id: str = ""):
        """Start TTFT measurement"""
        self.ttft_start_time = time.time()
        self.last_component_time = self.ttft_start_time
        self.measurement_active = True
        self.current_metrics.request_id = request_id
        self.current_metrics.timestamp = self.ttft_start_time
        
    def mark_component(self, component_name: str) -> float:
        """
        Mark completion of a TTFT component
        
        Args:
            component_name: Name of component (e.g., "vision_encoding")
            
        Returns:
            Component duration in milliseconds
        """
        if not self.measurement_active:
            return 0.0
        
        current_time = time.time()
        duration_ms = (current_time - self.last_component_time) * 1000.0
        
        # Update corresponding field
        ttft = self.current_metrics.ttft
        if component_name == "image_preprocessing":
            ttft.image_preprocessing_ms = duration_ms
        elif component_name == "vision_encoding":
            ttft.vision_encoding_ms = duration_ms
        elif component_name == "projection":
            ttft.projection_ms = duration_ms
        elif component_name == "prompt_processing":
            ttft.prompt_processing_ms = duration_ms
        elif component_name == "first_token_generation":
            ttft.first_token_generation_ms = duration_ms
        
        self.last_component_time = current_time
        return duration_ms
    
    def end_ttft_measurement(self) -> float:
        """
        End TTFT measurement and compute total
        
        Returns:
            Total TTFT in milliseconds
        """
        if not self.measurement_active:
            return 0.0
        
        return self.current_metrics.ttft.compute_total()
    
    def start_token_measurement(self):
        """Start measuring time for next token"""
        self.token_start_time = time.time()
    
    def end_token_measurement(self) -> float:
        """
        End token measurement and record latency
        
        Returns:
            Token latency in milliseconds
        """
        latency_ms = (time.time() - self.token_start_time) * 1000.0
        self.current_metrics.tbt.add_token_latency(latency_ms)
        return latency_ms
    
    def measure_memory(self):
        """Measure current memory usage and fragmentation"""
        # System memory
        mem = psutil.virtual_memory()
        frag = self.current_metrics.fragmentation
        
        frag.system_memory_total_gb = mem.total / (1024 ** 3)
        frag.system_memory_available_gb = mem.available / (1024 ** 3)
        frag.system_memory_percent = mem.percent
        
        # TODO: Implement MLX-specific memory tracking
        # For Phase 0, use estimates based on model config
        
        # Placeholder: estimate from known model size
        frag.model_weights_mb = 4500  # INT4 LLaVA-7B
        frag.kv_cache_mb = 256        # 1K tokens
        frag.activations_mb = 400     # Forward pass
        
        frag.used_memory_mb = (
            frag.model_weights_mb +
            frag.kv_cache_mb +
            frag.activations_mb
        )
        
        # Estimate allocated (with fragmentation)
        frag.allocated_memory_mb = frag.used_memory_mb * 1.15  # 15% overhead
        
        frag.compute_fragmentation()
    
    def finalize_metrics(self) -> InferenceMetrics:
        """
        Finalize all metrics and return complete snapshot
        
        Returns:
            Complete InferenceMetrics object
        """
        # Compute TBT statistics
        self.current_metrics.tbt.compute_statistics()
        
        # Measure memory
        self.measure_memory()
        
        self.measurement_active = False
        return self.current_metrics
    
    def get_metrics(self) -> InferenceMetrics:
        """Get current metrics"""
        return self.current_metrics
    
    def reset(self):
        """Reset for next measurement"""
        self.current_metrics = InferenceMetrics()
        self.ttft_start_time = 0.0
        self.last_component_time = 0.0
        self.token_start_time = 0.0
        self.measurement_active = False


class MetricsAggregator:
    """Aggregate metrics across multiple requests"""
    
    def __init__(self):
        self.metrics_history: List[InferenceMetrics] = []
        
    def add_metrics(self, metrics: InferenceMetrics):
        """Add metrics from a single request"""
        self.metrics_history.append(metrics)
    
    def compute_summary(self) -> Dict:
        """Compute summary statistics across all requests"""
        if not self.metrics_history:
            return {}
        
        import numpy as np
        
        # Extract TTFT values
        ttft_values = [m.ttft.total_ttft_ms for m in self.metrics_history]
        
        # Extract TBT values
        tbt_values = [m.tbt.mean_tbt_ms for m in self.metrics_history 
                     if m.tbt.mean_tbt_ms > 0]
        
        # Extract fragmentation values
        frag_values = [m.fragmentation.fragmentation_percent 
                      for m in self.metrics_history
                      if m.fragmentation.fragmentation_percent > 0]
        
        summary = {
            'num_requests': len(self.metrics_history),
            'ttft': {
                'mean_ms': float(np.mean(ttft_values)) if ttft_values else 0,
                'median_ms': float(np.median(ttft_values)) if ttft_values else 0,
                'p95_ms': float(np.percentile(ttft_values, 95)) if ttft_values else 0,
                'p99_ms': float(np.percentile(ttft_values, 99)) if ttft_values else 0,
                'min_ms': float(np.min(ttft_values)) if ttft_values else 0,
                'max_ms': float(np.max(ttft_values)) if ttft_values else 0,
            },
            'tbt': {
                'mean_ms': float(np.mean(tbt_values)) if tbt_values else 0,
                'median_ms': float(np.median(tbt_values)) if tbt_values else 0,
                'p95_ms': float(np.percentile(tbt_values, 95)) if tbt_values else 0,
                'p99_ms': float(np.percentile(tbt_values, 99)) if tbt_values else 0,
            },
            'fragmentation': {
                'mean_percent': float(np.mean(frag_values)) if frag_values else 0,
                'max_percent': float(np.max(frag_values)) if frag_values else 0,
            }
        }
        
        return summary
    
    def export_to_json(self, filepath: str):
        """Export all metrics to JSON file"""
        data = {
            'summary': self.compute_summary(),
            'requests': [m.to_dict() for m in self.metrics_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    """Test metrics collection"""
    print("=" * 80)
    print("AMIO Phase 0 - Metrics Collection Test")
    print("=" * 80)
    print()
    
    # Create collector
    collector = MetricsCollector()
    
    # Simulate TTFT measurement
    print("Simulating TTFT measurement:")
    print("-" * 80)
    
    collector.start_ttft_measurement(request_id="test-001")
    
    time.sleep(0.05)
    duration = collector.mark_component("image_preprocessing")
    print(f"  Image preprocessing: {duration:.1f} ms")
    
    time.sleep(0.15)
    duration = collector.mark_component("vision_encoding")
    print(f"  Vision encoding: {duration:.1f} ms")
    
    time.sleep(0.01)
    duration = collector.mark_component("projection")
    print(f"  Projection: {duration:.1f} ms")
    
    time.sleep(0.08)
    duration = collector.mark_component("prompt_processing")
    print(f"  Prompt processing: {duration:.1f} ms")
    
    time.sleep(0.05)
    duration = collector.mark_component("first_token_generation")
    print(f"  First token generation: {duration:.1f} ms")
    
    total_ttft = collector.end_ttft_measurement()
    print(f"\n  Total TTFT: {total_ttft:.1f} ms")
    
    # Simulate TBT measurement
    print("\nSimulating TBT measurement (10 tokens):")
    print("-" * 80)
    
    for i in range(10):
        collector.start_token_measurement()
        time.sleep(0.04 + (i % 3) * 0.01)  # Vary latency
        latency = collector.end_token_measurement()
        print(f"  Token {i+1}: {latency:.1f} ms")
    
    # Finalize
    metrics = collector.finalize_metrics()
    
    print("\nFinal Metrics:")
    print("-" * 80)
    print(f"  TTFT: {metrics.ttft.total_ttft_ms:.1f} ms")
    print(f"  Mean TBT: {metrics.tbt.mean_tbt_ms:.1f} ms")
    print(f"  Throughput: {metrics.tbt.tokens_per_second:.1f} tok/s")
    print(f"  Memory Fragmentation: {metrics.fragmentation.fragmentation_percent:.1f}%")
    
    print("\n✅ Metrics collection test complete")
