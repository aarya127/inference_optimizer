"""Inference Optimizer - A comprehensive benchmarking tool for LLM inference engines."""

__version__ = "0.1.0"
__author__ = "Inference Optimizer Team"

from src.backends.base import BaseBackend
from src.benchmarks.runner import BenchmarkRunner
from src.metrics.collector import MetricsCollector

__all__ = ["BaseBackend", "BenchmarkRunner", "MetricsCollector"]
