"""Test suite for metrics collector."""

import pytest
from src.metrics.collector import MetricsCollector, BenchmarkMetrics


def test_metrics_collector_initialization():
    """Test MetricsCollector initialization."""
    collector = MetricsCollector()
    
    assert collector.latencies == []
    assert collector.ttfts == []
    assert collector.memories == []
    assert collector.successful_requests == 0
    assert collector.failed_requests == 0


def test_record_inference():
    """Test recording inference metrics."""
    collector = MetricsCollector()
    
    collector.record_inference(
        latency_ms=100.0,
        ttft_ms=10.0,
        memory_mb=500.0,
        tokens_generated=50,
        success=True,
    )
    
    assert len(collector.latencies) == 1
    assert collector.latencies[0] == 100.0
    assert collector.successful_requests == 1
    assert collector.failed_requests == 0


def test_record_failed_inference():
    """Test recording failed inference."""
    collector = MetricsCollector()
    
    collector.record_inference(
        latency_ms=0.0,
        ttft_ms=0.0,
        memory_mb=0.0,
        tokens_generated=0,
        success=False,
    )
    
    assert len(collector.latencies) == 0
    assert collector.successful_requests == 0
    assert collector.failed_requests == 1


def test_compute_benchmark_metrics():
    """Test computing benchmark metrics."""
    collector = MetricsCollector()
    collector.start()
    
    # Record some inferences
    for i in range(10):
        collector.record_inference(
            latency_ms=100.0 + i,
            ttft_ms=10.0 + i,
            memory_mb=500.0 + i,
            tokens_generated=50,
            success=True,
        )
    
    collector.stop()
    
    metrics = collector.compute_benchmark_metrics(
        backend="test",
        model_name="test-model",
        quantization="fp16",
        batch_size=1,
    )
    
    assert metrics.backend == "test"
    assert metrics.model_name == "test-model"
    assert metrics.avg_latency_ms > 0
    assert metrics.throughput_tokens_per_sec > 0
    assert metrics.successful_requests == 10


def test_get_summary():
    """Test getting summary statistics."""
    collector = MetricsCollector()
    collector.start()
    
    collector.record_inference(100.0, 10.0, 500.0, 50, True)
    collector.record_inference(120.0, 12.0, 510.0, 52, True)
    
    collector.stop()
    
    summary = collector.get_summary()
    
    assert "avg_latency_ms" in summary
    assert "total_requests" in summary
    assert summary["successful_requests"] == 2
