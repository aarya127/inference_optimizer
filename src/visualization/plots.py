"""Visualization tools for benchmark results."""

from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from src.metrics.collector import BenchmarkMetrics


class BenchmarkVisualizer:
    """Creates visualizations for benchmark results."""
    
    def __init__(self, results: List[BenchmarkMetrics], output_dir: str = "results"):
        """Initialize visualizer.
        
        Args:
            results: List of benchmark results
            output_dir: Directory to save plots
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        self.df = pd.DataFrame([r.to_dict() for r in results])
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_latency_comparison(self, save: bool = True) -> Optional[str]:
        """Plot latency comparison across backends.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Path to saved plot if save=True
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Latency Comparison Across Backends", fontsize=16, fontweight='bold')
        
        # Average latency by backend
        ax = axes[0, 0]
        data = self.df.groupby('backend')['avg_latency_ms'].mean().sort_values()
        data.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(data)))
        ax.set_title("Average Latency by Backend")
        ax.set_ylabel("Latency (ms)")
        ax.set_xlabel("Backend")
        ax.tick_params(axis='x', rotation=45)
        
        # Latency by quantization
        ax = axes[0, 1]
        pivot = self.df.pivot_table(
            values='avg_latency_ms',
            index='backend',
            columns='quantization',
            aggfunc='mean'
        )
        pivot.plot(kind='bar', ax=ax)
        ax.set_title("Latency by Quantization Level")
        ax.set_ylabel("Latency (ms)")
        ax.set_xlabel("Backend")
        ax.legend(title="Quantization")
        ax.tick_params(axis='x', rotation=45)
        
        # P95 latency comparison
        ax = axes[1, 0]
        data = self.df.groupby('backend')['p95_latency_ms'].mean().sort_values()
        data.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(data)))
        ax.set_title("P95 Latency by Backend")
        ax.set_ylabel("P95 Latency (ms)")
        ax.set_xlabel("Backend")
        ax.tick_params(axis='x', rotation=45)
        
        # Latency by batch size
        ax = axes[1, 1]
        for backend in self.df['backend'].unique():
            backend_data = self.df[self.df['backend'] == backend]
            grouped = backend_data.groupby('batch_size')['avg_latency_ms'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=backend)
        ax.set_title("Latency Scaling with Batch Size")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Latency (ms)")
        ax.legend()
        ax.set_xscale('log', base=2)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "latency_comparison.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        else:
            plt.show()
            return None
    
    def plot_throughput_comparison(self, save: bool = True) -> Optional[str]:
        """Plot throughput comparison across backends.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Path to saved plot if save=True
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Throughput Comparison Across Backends", fontsize=16, fontweight='bold')
        
        # Tokens per second by backend
        ax = axes[0, 0]
        data = self.df.groupby('backend')['throughput_tokens_per_sec'].mean().sort_values(ascending=False)
        data.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(data)))
        ax.set_title("Throughput by Backend")
        ax.set_ylabel("Tokens/Second")
        ax.set_xlabel("Backend")
        ax.tick_params(axis='x', rotation=45)
        
        # Throughput by quantization
        ax = axes[0, 1]
        pivot = self.df.pivot_table(
            values='throughput_tokens_per_sec',
            index='backend',
            columns='quantization',
            aggfunc='mean'
        )
        pivot.plot(kind='bar', ax=ax)
        ax.set_title("Throughput by Quantization Level")
        ax.set_ylabel("Tokens/Second")
        ax.set_xlabel("Backend")
        ax.legend(title="Quantization")
        ax.tick_params(axis='x', rotation=45)
        
        # Requests per second
        ax = axes[1, 0]
        data = self.df.groupby('backend')['throughput_requests_per_sec'].mean().sort_values(ascending=False)
        data.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(data)))
        ax.set_title("Request Throughput by Backend")
        ax.set_ylabel("Requests/Second")
        ax.set_xlabel("Backend")
        ax.tick_params(axis='x', rotation=45)
        
        # Throughput scaling with batch size
        ax = axes[1, 1]
        for backend in self.df['backend'].unique():
            backend_data = self.df[self.df['backend'] == backend]
            grouped = backend_data.groupby('batch_size')['throughput_tokens_per_sec'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=backend)
        ax.set_title("Throughput Scaling with Batch Size")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Tokens/Second")
        ax.legend()
        ax.set_xscale('log', base=2)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "throughput_comparison.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        else:
            plt.show()
            return None
    
    def plot_memory_comparison(self, save: bool = True) -> Optional[str]:
        """Plot memory usage comparison across backends.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Path to saved plot if save=True
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Memory Usage Comparison Across Backends", fontsize=16, fontweight='bold')
        
        # Average memory by backend
        ax = axes[0, 0]
        data = self.df.groupby('backend')['avg_memory_mb'].mean().sort_values()
        data.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(data)))
        ax.set_title("Average Memory Usage by Backend")
        ax.set_ylabel("Memory (MB)")
        ax.set_xlabel("Backend")
        ax.tick_params(axis='x', rotation=45)
        
        # Peak memory by backend
        ax = axes[0, 1]
        data = self.df.groupby('backend')['peak_memory_mb'].mean().sort_values()
        data.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(data)))
        ax.set_title("Peak Memory Usage by Backend")
        ax.set_ylabel("Memory (MB)")
        ax.set_xlabel("Backend")
        ax.tick_params(axis='x', rotation=45)
        
        # Memory by quantization
        ax = axes[1, 0]
        pivot = self.df.pivot_table(
            values='avg_memory_mb',
            index='backend',
            columns='quantization',
            aggfunc='mean'
        )
        pivot.plot(kind='bar', ax=ax)
        ax.set_title("Memory Usage by Quantization Level")
        ax.set_ylabel("Memory (MB)")
        ax.set_xlabel("Backend")
        ax.legend(title="Quantization")
        ax.tick_params(axis='x', rotation=45)
        
        # Memory scaling with batch size
        ax = axes[1, 1]
        for backend in self.df['backend'].unique():
            backend_data = self.df[self.df['backend'] == backend]
            grouped = backend_data.groupby('batch_size')['avg_memory_mb'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=backend)
        ax.set_title("Memory Scaling with Batch Size")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Memory (MB)")
        ax.legend()
        ax.set_xscale('log', base=2)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "memory_comparison.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        else:
            plt.show()
            return None
    
    def plot_efficiency_comparison(self, save: bool = True) -> Optional[str]:
        """Plot efficiency metrics (throughput/memory).
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Path to saved plot if save=True
        """
        # Calculate efficiency
        self.df['efficiency'] = (
            self.df['throughput_tokens_per_sec'] / self.df['avg_memory_mb']
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Efficiency Comparison Across Backends", fontsize=16, fontweight='bold')
        
        # Efficiency by backend
        ax = axes[0]
        data = self.df.groupby('backend')['efficiency'].mean().sort_values(ascending=False)
        data.plot(kind='bar', ax=ax, color=sns.color_palette("husl", len(data)))
        ax.set_title("Efficiency (Tokens/s per MB)")
        ax.set_ylabel("Efficiency")
        ax.set_xlabel("Backend")
        ax.tick_params(axis='x', rotation=45)
        
        # Scatter: throughput vs memory
        ax = axes[1]
        for backend in self.df['backend'].unique():
            backend_data = self.df[self.df['backend'] == backend]
            ax.scatter(
                backend_data['avg_memory_mb'],
                backend_data['throughput_tokens_per_sec'],
                label=backend,
                s=100,
                alpha=0.6
            )
        ax.set_title("Throughput vs Memory Usage")
        ax.set_xlabel("Memory (MB)")
        ax.set_ylabel("Throughput (Tokens/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "efficiency_comparison.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(path)
        else:
            plt.show()
            return None
    
    def plot_all(self) -> List[str]:
        """Generate all plots.
        
        Returns:
            List of paths to saved plots
        """
        paths = []
        
        print("Generating visualizations...")
        
        path = self.plot_latency_comparison(save=True)
        if path:
            paths.append(path)
            print(f"  ✓ Latency comparison: {path}")
        
        path = self.plot_throughput_comparison(save=True)
        if path:
            paths.append(path)
            print(f"  ✓ Throughput comparison: {path}")
        
        path = self.plot_memory_comparison(save=True)
        if path:
            paths.append(path)
            print(f"  ✓ Memory comparison: {path}")
        
        path = self.plot_efficiency_comparison(save=True)
        if path:
            paths.append(path)
            print(f"  ✓ Efficiency comparison: {path}")
        
        return paths
