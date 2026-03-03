"""
Simple example of running a benchmark with inference optimizer.
"""

from src.benchmarks import BenchmarkRunner, BenchmarkConfig
from src.visualization import BenchmarkVisualizer, ReportGenerator


def main():
    """Run a simple benchmark example."""
    
    # Configure benchmark
    config = BenchmarkConfig(
        model_name="gpt2",  # Small model for quick testing
        backends=["pytorch"],  # Start with PyTorch baseline
        quantizations=["fp16"],
        batch_sizes=[1, 4, 8],
        num_requests=50,  # Fewer requests for demo
        max_new_tokens=64,
        output_dir="results/example",
    )
    
    print("Starting simple benchmark example...")
    print(f"Model: {config.model_name}")
    print(f"Backends: {config.backends}")
    print(f"Batch sizes: {config.batch_sizes}")
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run()
    
    # Generate visualizations
    if results:
        print("\nGenerating visualizations...")
        visualizer = BenchmarkVisualizer(results, config.output_dir)
        visualizer.plot_all()
        
        # Generate report
        print("\nGenerating report...")
        report_gen = ReportGenerator(results, config.output_dir)
        report_path = report_gen.generate_markdown_report()
        report_gen.generate_console_summary()
        
        print(f"\n✓ Example complete! Results saved to: {config.output_dir}")
        print(f"✓ Report: {report_path}")


if __name__ == "__main__":
    main()
