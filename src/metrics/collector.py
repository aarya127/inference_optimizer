"""Metrics collection and tracking."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import time
import psutil
import GPUtil
from datetime import datetime


@dataclass
class SystemMetrics:
    """System-level metrics."""
    
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_utilization: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Metrics for a single benchmark run."""
    
    # Identification
    backend: str
    model_name: str
    quantization: str
    batch_size: int
    
    # Performance metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    avg_time_to_first_token_ms: float
    p50_time_to_first_token_ms: float
    p95_time_to_first_token_ms: float
    
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float
    
    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Quality metrics
    avg_tokens_generated: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Timing
    total_duration_sec: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates benchmark metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.system_metrics: List[SystemMetrics] = []
        self.latencies: List[float] = []
        self.ttfts: List[float] = []
        self.memories: List[float] = []
        self.tokens_generated: List[int] = []
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self):
        """Start collecting metrics."""
        self.start_time = time.time()
        self.system_metrics = []
        self.latencies = []
        self.ttfts = []
        self.memories = []
        self.tokens_generated = []
        self.successful_requests = 0
        self.failed_requests = 0
    
    def stop(self):
        """Stop collecting metrics."""
        self.end_time = time.time()
    
    def record_inference(
        self,
        latency_ms: float,
        ttft_ms: float,
        memory_mb: float,
        tokens_generated: int,
        success: bool = True,
    ):
        """Record metrics from a single inference.
        
        Args:
            latency_ms: Total latency in milliseconds
            ttft_ms: Time to first token in milliseconds
            memory_mb: Memory used in megabytes
            tokens_generated: Number of tokens generated
            success: Whether the inference was successful
        """
        if success:
            self.latencies.append(latency_ms)
            self.ttfts.append(ttft_ms)
            self.memories.append(memory_mb)
            self.tokens_generated.append(tokens_generated)
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    def record_system_metrics(self):
        """Record current system metrics."""
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_util = 0.0
        gpu_mem_used = 0.0
        gpu_mem_total = 0.0
        gpu_temp = 0.0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_util = gpu.load * 100
                gpu_mem_used = gpu.memoryUsed
                gpu_mem_total = gpu.memoryTotal
                gpu_temp = gpu.temperature
        except:
            pass
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            gpu_utilization=gpu_util,
            gpu_memory_used_mb=gpu_mem_used,
            gpu_memory_total_mb=gpu_mem_total,
            gpu_temperature=gpu_temp,
        )
        
        self.system_metrics.append(metrics)
    
    def compute_benchmark_metrics(
        self,
        backend: str,
        model_name: str,
        quantization: str,
        batch_size: int,
    ) -> BenchmarkMetrics:
        """Compute aggregated benchmark metrics.
        
        Args:
            backend: Backend name
            model_name: Model name
            quantization: Quantization level
            batch_size: Batch size
            
        Returns:
            Aggregated benchmark metrics
        """
        import numpy as np
        
        if not self.latencies:
            raise ValueError("No metrics collected")
        
        # Latency percentiles
        latencies_np = np.array(self.latencies)
        avg_latency = float(np.mean(latencies_np))
        p50_latency = float(np.percentile(latencies_np, 50))
        p95_latency = float(np.percentile(latencies_np, 95))
        p99_latency = float(np.percentile(latencies_np, 99))
        
        # TTFT percentiles
        ttfts_np = np.array(self.ttfts)
        avg_ttft = float(np.mean(ttfts_np))
        p50_ttft = float(np.percentile(ttfts_np, 50))
        p95_ttft = float(np.percentile(ttfts_np, 95))
        
        # Memory
        memories_np = np.array(self.memories)
        peak_memory = float(np.max(memories_np))
        avg_memory = float(np.mean(memories_np))
        
        # Throughput
        total_duration = (self.end_time - self.start_time) if self.end_time else 0
        total_tokens = sum(self.tokens_generated)
        throughput_tokens = total_tokens / total_duration if total_duration > 0 else 0
        throughput_requests = self.successful_requests / total_duration if total_duration > 0 else 0
        
        # Quality
        avg_tokens = float(np.mean(self.tokens_generated)) if self.tokens_generated else 0
        
        return BenchmarkMetrics(
            backend=backend,
            model_name=model_name,
            quantization=quantization,
            batch_size=batch_size,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            avg_time_to_first_token_ms=avg_ttft,
            p50_time_to_first_token_ms=p50_ttft,
            p95_time_to_first_token_ms=p95_ttft,
            throughput_tokens_per_sec=throughput_tokens,
            throughput_requests_per_sec=throughput_requests,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            avg_tokens_generated=avg_tokens,
            total_requests=self.successful_requests + self.failed_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            total_duration_sec=total_duration,
        )
    
    def get_summary(self) -> Dict:
        """Get summary of collected metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        import numpy as np
        
        if not self.latencies:
            return {}
        
        return {
            "total_requests": self.successful_requests + self.failed_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": float(np.mean(self.latencies)),
            "p95_latency_ms": float(np.percentile(self.latencies, 95)),
            "avg_ttft_ms": float(np.mean(self.ttfts)),
            "avg_memory_mb": float(np.mean(self.memories)),
            "peak_memory_mb": float(np.max(self.memories)),
            "total_tokens": sum(self.tokens_generated),
            "avg_tokens_per_request": float(np.mean(self.tokens_generated)),
        }
