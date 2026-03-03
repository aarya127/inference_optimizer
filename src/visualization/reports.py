"""Report generation for benchmark results."""

from typing import List, Optional
import json
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

from src.metrics.collector import BenchmarkMetrics


class ReportGenerator:
    """Generates detailed reports from benchmark results."""
    
    def __init__(self, results: List[BenchmarkMetrics], output_dir: str = "results"):
        """Initialize report generator.
        
        Args:
            results: List of benchmark results
            output_dir: Directory to save reports
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(self, include_plots: bool = True) -> str:
        """Generate a comprehensive markdown report.
        
        Args:
            include_plots: Whether to include plot references
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start report
        lines = [
            "# Inference Optimizer Benchmark Report",
            "",
            f"**Generated:** {timestamp}",
            "",
            "## Summary",
            "",
            f"- **Total Configurations Tested:** {len(self.results)}",
            f"- **Backends:** {', '.join(set(r.backend for r in self.results))}",
            f"- **Quantizations:** {', '.join(set(r.quantization for r in self.results))}",
            f"- **Batch Sizes:** {', '.join(map(str, sorted(set(r.batch_size for r in self.results))))}",
            "",
        ]
        
        # Best performers
        lines.extend(self._generate_best_performers())
        
        # Detailed results by backend
        lines.extend(self._generate_backend_sections())
        
        # Recommendations
        lines.extend(self._generate_recommendations())
        
        # Include plots
        if include_plots:
            lines.extend([
                "## Visualizations",
                "",
                "### Latency Comparison",
                "![Latency Comparison](latency_comparison.png)",
                "",
                "### Throughput Comparison",
                "![Throughput Comparison](throughput_comparison.png)",
                "",
                "### Memory Comparison",
                "![Memory Comparison](memory_comparison.png)",
                "",
                "### Efficiency Comparison",
                "![Efficiency Comparison](efficiency_comparison.png)",
                "",
            ])
        
        # Raw data reference
        lines.extend([
            "## Raw Data",
            "",
            "Detailed results are available in the accompanying CSV and JSON files.",
            "",
        ])
        
        # Write report
        report_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text("\n".join(lines))
        
        return str(report_path)
    
    def _generate_best_performers(self) -> List[str]:
        """Generate best performers section."""
        lines = [
            "## Best Performers",
            "",
        ]
        
        # Lowest latency
        best_latency = min(self.results, key=lambda r: r.avg_latency_ms)
        lines.extend([
            "### Lowest Latency",
            f"- **Backend:** {best_latency.backend}",
            f"- **Quantization:** {best_latency.quantization}",
            f"- **Batch Size:** {best_latency.batch_size}",
            f"- **Average Latency:** {best_latency.avg_latency_ms:.2f}ms",
            f"- **P95 Latency:** {best_latency.p95_latency_ms:.2f}ms",
            "",
        ])
        
        # Highest throughput
        best_throughput = max(self.results, key=lambda r: r.throughput_tokens_per_sec)
        lines.extend([
            "### Highest Throughput",
            f"- **Backend:** {best_throughput.backend}",
            f"- **Quantization:** {best_throughput.quantization}",
            f"- **Batch Size:** {best_throughput.batch_size}",
            f"- **Throughput:** {best_throughput.throughput_tokens_per_sec:.2f} tokens/s",
            "",
        ])
        
        # Lowest memory
        best_memory = min(self.results, key=lambda r: r.avg_memory_mb)
        lines.extend([
            "### Lowest Memory Usage",
            f"- **Backend:** {best_memory.backend}",
            f"- **Quantization:** {best_memory.quantization}",
            f"- **Batch Size:** {best_memory.batch_size}",
            f"- **Memory Usage:** {best_memory.avg_memory_mb:.2f}MB",
            "",
        ])
        
        return lines
    
    def _generate_backend_sections(self) -> List[str]:
        """Generate detailed sections for each backend."""
        lines = [
            "## Detailed Results by Backend",
            "",
        ]
        
        # Group by backend
        backends = {}
        for result in self.results:
            if result.backend not in backends:
                backends[result.backend] = []
            backends[result.backend].append(result)
        
        for backend_name, backend_results in sorted(backends.items()):
            lines.extend([
                f"### {backend_name}",
                "",
            ])
            
            # Create table
            table_data = []
            for r in backend_results:
                table_data.append([
                    r.quantization,
                    r.batch_size,
                    f"{r.avg_latency_ms:.2f}",
                    f"{r.p95_latency_ms:.2f}",
                    f"{r.throughput_tokens_per_sec:.2f}",
                    f"{r.avg_memory_mb:.2f}",
                ])
            
            headers = [
                "Quantization",
                "Batch Size",
                "Avg Latency (ms)",
                "P95 Latency (ms)",
                "Throughput (tok/s)",
                "Memory (MB)",
            ]
            
            table = tabulate(table_data, headers=headers, tablefmt="pipe")
            lines.extend([table, ""])
        
        return lines
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        lines = [
            "## Recommendations",
            "",
        ]
        
        # Analyze results
        best_latency = min(self.results, key=lambda r: r.avg_latency_ms)
        best_throughput = max(self.results, key=lambda r: r.throughput_tokens_per_sec)
        best_memory = min(self.results, key=lambda r: r.avg_memory_mb)
        
        lines.extend([
            "### Use Case Recommendations",
            "",
            "#### Low Latency (Real-time Applications)",
            f"**Recommended:** {best_latency.backend} with {best_latency.quantization} quantization",
            f"- Best for: Chatbots, interactive applications",
            f"- Expected latency: ~{best_latency.avg_latency_ms:.2f}ms",
            "",
            "#### High Throughput (Batch Processing)",
            f"**Recommended:** {best_throughput.backend} with {best_throughput.quantization} quantization",
            f"- Best for: Batch inference, data processing pipelines",
            f"- Expected throughput: ~{best_throughput.throughput_tokens_per_sec:.2f} tokens/s",
            "",
            "#### Memory Constrained (Edge Devices)",
            f"**Recommended:** {best_memory.backend} with {best_memory.quantization} quantization",
            f"- Best for: Edge deployment, limited GPU memory",
            f"- Expected memory: ~{best_memory.avg_memory_mb:.2f}MB",
            "",
        ])
        
        return lines
    
    def generate_console_summary(self):
        """Print a summary to console."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        
        # Create comparison table
        table_data = []
        for r in self.results:
            table_data.append([
                r.backend,
                r.quantization,
                r.batch_size,
                f"{r.avg_latency_ms:.2f}",
                f"{r.throughput_tokens_per_sec:.2f}",
                f"{r.avg_memory_mb:.2f}",
            ])
        
        headers = [
            "Backend",
            "Quant",
            "Batch",
            "Latency (ms)",
            "Throughput (tok/s)",
            "Memory (MB)",
        ]
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("=" * 80)
