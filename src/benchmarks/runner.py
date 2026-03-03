"""Benchmark orchestration and execution."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
import time
from pathlib import Path
from tqdm import tqdm

from src.backends import get_backend
from src.backends.base import ModelConfig, InferenceRequest
from src.metrics.collector import MetricsCollector, BenchmarkMetrics


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    
    # Model configuration
    model_name: str
    model_path: Optional[str] = None
    
    # Backends to test
    backends: List[str] = None
    
    # Quantization levels to test
    quantizations: List[str] = None
    
    # Batch sizes to test
    batch_sizes: List[int] = None
    
    # Inference parameters
    num_requests: int = 100
    max_new_tokens: int = 128
    temperature: float = 0.8
    warmup_requests: int = 10
    
    # Output
    output_dir: str = "results"
    save_results: bool = True
    
    def __post_init__(self):
        """Set defaults for list fields."""
        if self.backends is None:
            self.backends = ["pytorch", "vllm"]
        if self.quantizations is None:
            self.quantizations = ["fp16"]
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8]


class BenchmarkRunner:
    """Runs benchmarks across multiple backends and configurations."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results: List[BenchmarkMetrics] = []
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run(self) -> List[BenchmarkMetrics]:
        """Run all benchmarks.
        
        Returns:
            List of benchmark results
        """
        print("=" * 80)
        print("Starting Inference Optimizer Benchmark")
        print("=" * 80)
        print(f"Model: {self.config.model_name}")
        print(f"Backends: {', '.join(self.config.backends)}")
        print(f"Quantizations: {', '.join(self.config.quantizations)}")
        print(f"Batch sizes: {', '.join(map(str, self.config.batch_sizes))}")
        print(f"Requests per config: {self.config.num_requests}")
        print("=" * 80)
        
        # Generate test prompts
        prompts = self._generate_prompts(self.config.num_requests)
        warmup_prompts = self._generate_prompts(self.config.warmup_requests)
        
        # Run benchmarks for each configuration
        total_configs = (
            len(self.config.backends) *
            len(self.config.quantizations) *
            len(self.config.batch_sizes)
        )
        
        config_idx = 0
        
        for backend_name in self.config.backends:
            for quantization in self.config.quantizations:
                for batch_size in self.config.batch_sizes:
                    config_idx += 1
                    print(f"\n[{config_idx}/{total_configs}] Testing configuration:")
                    print(f"  Backend: {backend_name}")
                    print(f"  Quantization: {quantization}")
                    print(f"  Batch size: {batch_size}")
                    
                    try:
                        metrics = self._run_single_benchmark(
                            backend_name=backend_name,
                            quantization=quantization,
                            batch_size=batch_size,
                            prompts=prompts,
                            warmup_prompts=warmup_prompts,
                        )
                        
                        self.results.append(metrics)
                        
                        # Print summary
                        print(f"  ✓ Avg Latency: {metrics.avg_latency_ms:.2f}ms")
                        print(f"  ✓ Throughput: {metrics.throughput_tokens_per_sec:.2f} tokens/s")
                        print(f"  ✓ Memory: {metrics.avg_memory_mb:.2f}MB")
                        
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
                        continue
        
        # Save results
        if self.config.save_results:
            self._save_results()
        
        print("\n" + "=" * 80)
        print("Benchmark completed!")
        print(f"Results saved to: {self.config.output_dir}")
        print("=" * 80)
        
        return self.results
    
    def _run_single_benchmark(
        self,
        backend_name: str,
        quantization: str,
        batch_size: int,
        prompts: List[str],
        warmup_prompts: List[str],
    ) -> BenchmarkMetrics:
        """Run benchmark for a single configuration.
        
        Args:
            backend_name: Backend to use
            quantization: Quantization level
            batch_size: Batch size
            prompts: List of prompts for testing
            warmup_prompts: List of prompts for warmup
            
        Returns:
            Benchmark metrics
        """
        # Create model config
        model_config = ModelConfig(
            model_name=self.config.model_name,
            model_path=self.config.model_path,
            quantization=quantization,
            max_batch_size=max(self.config.batch_sizes),
            max_seq_length=2048,
        )
        
        # Initialize backend
        backend = get_backend(backend_name, model_config)
        
        # Load model
        print("  Loading model...")
        backend.load_model()
        
        # Warmup
        print("  Warming up...")
        self._run_warmup(backend, warmup_prompts, batch_size)
        
        # Run benchmark
        print("  Running benchmark...")
        collector = MetricsCollector()
        collector.start()
        
        # Create requests
        requests = [
            InferenceRequest(
                prompt=prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            for prompt in prompts
        ]
        
        # Process in batches
        num_batches = (len(requests) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="  Progress"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(requests))
            batch_requests = requests[start_idx:end_idx]
            
            # Run batch inference
            results = backend.batch_infer(batch_requests)
            
            # Record metrics
            for result in results:
                collector.record_inference(
                    latency_ms=result.latency_ms,
                    ttft_ms=result.time_to_first_token_ms,
                    memory_mb=result.memory_used_mb,
                    tokens_generated=result.tokens_generated,
                    success=result.success,
                )
            
            # Record system metrics periodically
            if i % 10 == 0:
                collector.record_system_metrics()
        
        collector.stop()
        
        # Compute metrics
        metrics = collector.compute_benchmark_metrics(
            backend=backend_name,
            model_name=self.config.model_name,
            quantization=quantization,
            batch_size=batch_size,
        )
        
        # Unload model
        backend.unload_model()
        
        return metrics
    
    def _run_warmup(
        self,
        backend,
        warmup_prompts: List[str],
        batch_size: int,
    ):
        """Run warmup inferences.
        
        Args:
            backend: Backend instance
            warmup_prompts: Prompts for warmup
            batch_size: Batch size
        """
        requests = [
            InferenceRequest(prompt=prompt, max_new_tokens=self.config.max_new_tokens)
            for prompt in warmup_prompts
        ]
        
        num_batches = (len(requests) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(requests))
            batch_requests = requests[start_idx:end_idx]
            backend.batch_infer(batch_requests)
    
    def _generate_prompts(self, num_prompts: int) -> List[str]:
        """Generate test prompts.
        
        Args:
            num_prompts: Number of prompts to generate
            
        Returns:
            List of prompts
        """
        # Sample prompts of varying lengths
        prompt_templates = [
            "Write a short story about",
            "Explain the concept of",
            "What are the benefits of",
            "How does",
            "Describe the process of",
            "Compare and contrast",
            "What is the history of",
            "Give me a recipe for",
            "Tell me about",
            "What would happen if",
        ]
        
        topics = [
            "artificial intelligence",
            "quantum computing",
            "climate change",
            "space exploration",
            "renewable energy",
            "biotechnology",
            "machine learning",
            "blockchain technology",
            "virtual reality",
            "nanotechnology",
        ]
        
        prompts = []
        for i in range(num_prompts):
            template = prompt_templates[i % len(prompt_templates)]
            topic = topics[i % len(topics)]
            prompts.append(f"{template} {topic}.")
        
        return prompts
    
    def _save_results(self):
        """Save benchmark results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = Path(self.config.output_dir) / f"results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                [result.to_dict() for result in self.results],
                f,
                indent=2,
            )
        
        # Save as CSV
        csv_path = Path(self.config.output_dir) / f"results_{timestamp}.csv"
        self._save_csv(csv_path)
        
        print(f"\nResults saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
    
    def _save_csv(self, path: Path):
        """Save results as CSV.
        
        Args:
            path: Path to save CSV
        """
        import pandas as pd
        
        df = pd.DataFrame([result.to_dict() for result in self.results])
        df.to_csv(path, index=False)
