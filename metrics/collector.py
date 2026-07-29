"""
Core Performance Metrics for AMIO Phase 0

NOTE: this module is currently not imported by the live service
(integrated_service.py) — it is standalone instrumentation.

Defines and implements the three primary metrics for multimodal inference optimization:
1. Time-To-First-Token (TTFT) - Primary bottleneck for multimodal
2. Time-Between-Tokens (TBT) - Generation smoothness
3. Memory Fragmentation - Memory waste percentage

Measurement policy:
- All durations use time.perf_counter() (monotonic); time.time() is used
  only for epoch timestamps.
- Total TTFT is WALL TIME from start_ttft_measurement() to
  end_ttft_measurement(); the per-component times are a breakdown and a
  warning is printed if the component sum diverges from wall time by >5%.
- Memory metrics come from the MLX allocator when mlx is importable;
  otherwise they are reported as unavailable.  No constants are ever
  emitted as measurements.
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psutil
import json

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without MLX
    mx = None
    _MLX_AVAILABLE = False


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

    # Total TTFT (WALL TIME, measured start→end; authoritative)
    total_ttft_ms: float = 0.0
    # Sum of the component durations above (breakdown; may differ slightly
    # from wall time due to un-instrumented gaps)
    component_sum_ms: float = 0.0

    # Metadata
    image_size: Tuple[int, int] = (0, 0)
    prompt_length: int = 0
    batch_size: int = 1

    def compute_component_sum(self) -> float:
        """Sum the per-component durations (breakdown, NOT the total TTFT —
        total_ttft_ms is wall time set by end_ttft_measurement)."""
        self.component_sum_ms = (
            self.image_preprocessing_ms +
            self.vision_encoding_ms +
            self.projection_ms +
            self.prompt_processing_ms +
            self.first_token_generation_ms +
            self.tp_overhead_ms
        )
        return self.component_sum_ms

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
            'component_sum_ms': self.component_sum_ms,
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
    """Memory fragmentation metrics.

    `measurement_available` is False when no real allocator statistics could
    be read (e.g. MLX not importable) — in that case the MB fields are NOT
    measurements and must not be reported as such.
    """
    measurement_available: bool = False
    source: str = "unavailable"    # e.g. "mlx_allocator"
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
            'measurement_available': self.measurement_available,
            'source': self.source,
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
    
    # Recognized TTFT components → TTFTMetrics attribute
    _COMPONENT_FIELDS = {
        "image_preprocessing": "image_preprocessing_ms",
        "vision_encoding": "vision_encoding_ms",
        "projection": "projection_ms",
        "prompt_processing": "prompt_processing_ms",
        "first_token_generation": "first_token_generation_ms",
        "tp_overhead": "tp_overhead_ms",
    }

    def __init__(self):
        self.current_metrics = InferenceMetrics()
        self.ttft_start_time = 0.0        # perf_counter seconds
        self.last_component_time = 0.0    # perf_counter seconds
        self.token_start_time = 0.0       # perf_counter seconds
        self.measurement_active = False

    def start_ttft_measurement(self, request_id: str = ""):
        """Start TTFT measurement"""
        self.ttft_start_time = time.perf_counter()
        self.last_component_time = self.ttft_start_time
        self.measurement_active = True
        self.current_metrics.request_id = request_id
        self.current_metrics.timestamp = time.time()   # epoch timestamp only

    def mark_component(self, component_name: str) -> float:
        """
        Mark completion of a TTFT component.

        Args:
            component_name: One of the recognized component names
                (see MetricsCollector._COMPONENT_FIELDS).

        Returns:
            Component duration in milliseconds.

        Raises:
            ValueError: if `component_name` is not a recognized component.
                (Previously an unknown name silently advanced the clock and
                recorded nothing — a typo meant silently lost time.)
        """
        if component_name not in self._COMPONENT_FIELDS:
            raise ValueError(
                f"Unknown TTFT component {component_name!r}. "
                f"Valid components: {sorted(self._COMPONENT_FIELDS)}"
            )
        if not self.measurement_active:
            return 0.0

        current_time = time.perf_counter()
        duration_ms = (current_time - self.last_component_time) * 1000.0

        setattr(self.current_metrics.ttft,
                self._COMPONENT_FIELDS[component_name], duration_ms)

        self.last_component_time = current_time
        return duration_ms

    def end_ttft_measurement(self) -> float:
        """
        End TTFT measurement.

        total_ttft_ms is the WALL TIME since start_ttft_measurement() —
        authoritative.  The component sum is kept as a breakdown
        (component_sum_ms); if it diverges from wall time by more than 5%
        a warning is emitted (it means un-instrumented time exists or a
        component was double-marked).

        Returns:
            Total (wall-time) TTFT in milliseconds.
        """
        if not self.measurement_active:
            return 0.0

        ttft = self.current_metrics.ttft
        wall_ms = (time.perf_counter() - self.ttft_start_time) * 1000.0
        comp_ms = ttft.compute_component_sum()
        ttft.total_ttft_ms = wall_ms

        if wall_ms > 0 and abs(wall_ms - comp_ms) / wall_ms > 0.05:
            warnings.warn(
                f"TTFT component sum ({comp_ms:.1f} ms) diverges from wall "
                f"time ({wall_ms:.1f} ms) by "
                f"{abs(wall_ms - comp_ms) / wall_ms * 100:.1f}% — "
                f"some time is un-instrumented or double-counted.",
                stacklevel=2,
            )
        return wall_ms

    def start_token_measurement(self):
        """Start measuring time for next token"""
        self.token_start_time = time.perf_counter()

    def end_token_measurement(self) -> float:
        """
        End token measurement and record latency

        Returns:
            Token latency in milliseconds
        """
        latency_ms = (time.perf_counter() - self.token_start_time) * 1000.0
        self.current_metrics.tbt.add_token_latency(latency_ms)
        return latency_ms

    def measure_memory(self):
        """Measure current memory usage from real sources only.

        System memory comes from psutil.  Allocator-level memory comes from
        the MLX allocator (mx.get_active_memory / mx.get_peak_memory /
        mx.get_cache_memory) when mlx is importable; otherwise the
        fragmentation block is marked `measurement_available=False` and no
        numbers are fabricated.  (A previous version returned hardcoded
        LLaVA-7B constants here, so "fragmentation" was always exactly
        13.04% regardless of what ran.)
        """
        # System memory (real, via psutil)
        mem = psutil.virtual_memory()
        frag = self.current_metrics.fragmentation

        frag.system_memory_total_gb = mem.total / (1024 ** 3)
        frag.system_memory_available_gb = mem.available / (1024 ** 3)
        frag.system_memory_percent = mem.percent

        if _MLX_AVAILABLE:
            try:
                active_bytes = float(mx.get_active_memory())
                cache_bytes = float(mx.get_cache_memory())
                frag.used_memory_mb = active_bytes / (1024 ** 2)
                # Allocator holds active buffers + cached (freed but retained)
                frag.allocated_memory_mb = (active_bytes + cache_bytes) / (1024 ** 2)
                frag.measurement_available = True
                frag.source = "mlx_allocator"
                frag.compute_fragmentation()
            except Exception:
                frag.measurement_available = False
                frag.source = "unavailable (mlx allocator query failed)"
        else:
            # No allocator statistics available — report unavailable rather
            # than emitting constants as measurements.
            frag.measurement_available = False
            frag.source = "unavailable (mlx not importable)"
    
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
        """Compute summary statistics across all requests.

        Tail TBT metrics (p95/p99) are computed over the POOLED per-token
        latencies of all requests — a percentile of per-request means (the
        previous behavior) systematically understates tail latency because
        averaging within a request hides its slow tokens.
        """
        if not self.metrics_history:
            return {}

        import numpy as np

        # Extract TTFT values (wall-time totals)
        ttft_values = [m.ttft.total_ttft_ms for m in self.metrics_history]

        # Pool per-token latencies across ALL requests for tail metrics
        pooled_token_latencies: List[float] = []
        for m in self.metrics_history:
            pooled_token_latencies.extend(m.tbt.token_latencies)

        # Per-request means (still useful as a central-tendency view)
        tbt_request_means = [m.tbt.mean_tbt_ms for m in self.metrics_history
                             if m.tbt.mean_tbt_ms > 0]

        # Extract fragmentation values (real measurements only)
        frag_values = [m.fragmentation.fragmentation_percent
                       for m in self.metrics_history
                       if m.fragmentation.measurement_available
                       and m.fragmentation.fragmentation_percent > 0]

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
                # mean/median of per-request means (central tendency)
                'mean_ms': float(np.mean(tbt_request_means)) if tbt_request_means else 0,
                'median_ms': float(np.median(tbt_request_means)) if tbt_request_means else 0,
                # tails from pooled per-token latencies across all requests
                'p95_ms': float(np.percentile(pooled_token_latencies, 95))
                          if pooled_token_latencies else 0,
                'p99_ms': float(np.percentile(pooled_token_latencies, 99))
                          if pooled_token_latencies else 0,
                'num_tokens_pooled': len(pooled_token_latencies),
            },
            'fragmentation': {
                'measurements_available': len(frag_values),
                'mean_percent': float(np.mean(frag_values)) if frag_values else None,
                'max_percent': float(np.max(frag_values)) if frag_values else None,
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
    print(f"  TTFT (wall): {metrics.ttft.total_ttft_ms:.1f} ms  "
          f"(component sum: {metrics.ttft.component_sum_ms:.1f} ms)")
    print(f"  Mean TBT: {metrics.tbt.mean_tbt_ms:.1f} ms")
    print(f"  Throughput: {metrics.tbt.tokens_per_second:.1f} tok/s")
    if metrics.fragmentation.measurement_available:
        print(f"  Memory Fragmentation: "
              f"{metrics.fragmentation.fragmentation_percent:.1f}%  "
              f"(source: {metrics.fragmentation.source})")
    else:
        print(f"  Memory Fragmentation: unavailable "
              f"({metrics.fragmentation.source})")

    # Unknown component names must raise (not silently advance the clock)
    try:
        collector.start_ttft_measurement("test-002")
        collector.mark_component("visoin_encoding")   # deliberate typo
        raise AssertionError("expected ValueError for unknown component")
    except ValueError:
        print("\n  PASS: unknown component name raises ValueError")

    print("\nMetrics collection test complete")
