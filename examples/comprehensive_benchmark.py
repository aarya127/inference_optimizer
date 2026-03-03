"""
Advanced example comparing multiple backends with different configurations.
"""

from src.benchmarks import BenchmarkRunner, BenchmarkConfig
from src.visualization import BenchmarkVisualizer, ReportGenerator


def main():
    """Run comprehensive benchmark across multiple backends."""
    
    # Configure comprehensive benchmark
    config = BenchmarkConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        backends=["pytorch", "vllm", "deepspeed"],
        quantizations=["fp16", "int8"],
        batch_sizes=[1, 4, 8, 16, 32],
        num_requests=200,
        max_new_tokens=128,
        warmup_requests=20,
        output_dir="results/comprehensive",
    )
    
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Backends: {', '.join(config.backends)}")
    print(f"Quantizations: {', '.join(config.quantizations)}")
    print(f"Batch sizes: {', '.join(map(str, config.batch_sizes))}")
    print(f"Total configurations: {len(config.backends) * len(config.quantizations) * len(config.batch_sizes)}")
    print("=" * 80)
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run()
    
    # Generate comprehensive analysis
    if results:
        print("\nGenerating comprehensive analysis...")
        
        # Visualizations
        visualizer = BenchmarkVisualizer(results, config.output_dir)
        plot_paths = visualizer.plot_all()
        
        # Report
        report_gen = ReportGenerator(results, config.output_dir)
        report_path = report_gen.generate_markdown_report(include_plots=True)
        report_gen.generate_console_summary()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Results directory: {config.output_dir}")
        print(f"Report: {report_path}")
        print(f"Plots generated: {len(plot_paths)}")
        print("=" * 80)


if __name__ == "__main__":
    main()
