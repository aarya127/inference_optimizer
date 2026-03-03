"""Command-line interface for inference optimizer."""

import click
import json
from pathlib import Path

from src.benchmarks import BenchmarkRunner, BenchmarkConfig
from src.visualization import BenchmarkVisualizer, ReportGenerator
from src.metrics import BenchmarkMetrics


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Inference Optimizer - Compare LLM inference engines."""
    pass


@cli.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="Model name or path (e.g., 'meta-llama/Llama-2-7b-hf')",
)
@click.option(
    "--backends",
    "-b",
    multiple=True,
    default=["pytorch", "vllm"],
    help="Backends to test (can be specified multiple times)",
)
@click.option(
    "--quantizations",
    "-q",
    multiple=True,
    default=["fp16"],
    help="Quantization levels to test (fp16, fp8, int8, int4)",
)
@click.option(
    "--batch-sizes",
    "-bs",
    multiple=True,
    type=int,
    default=[1, 4, 8],
    help="Batch sizes to test",
)
@click.option(
    "--num-requests",
    "-n",
    type=int,
    default=100,
    help="Number of requests per configuration",
)
@click.option(
    "--max-tokens",
    type=int,
    default=128,
    help="Maximum tokens to generate",
)
@click.option(
    "--output-dir",
    "-o",
    default="results",
    help="Output directory for results",
)
@click.option(
    "--no-visualize",
    is_flag=True,
    help="Skip visualization generation",
)
def benchmark(
    model,
    backends,
    quantizations,
    batch_sizes,
    num_requests,
    max_tokens,
    output_dir,
    no_visualize,
):
    """Run benchmarks across multiple configurations."""
    
    # Create config
    config = BenchmarkConfig(
        model_name=model,
        backends=list(backends),
        quantizations=list(quantizations),
        batch_sizes=list(batch_sizes),
        num_requests=num_requests,
        max_new_tokens=max_tokens,
        output_dir=output_dir,
    )
    
    # Run benchmarks
    runner = BenchmarkRunner(config)
    results = runner.run()
    
    # Generate visualizations
    if not no_visualize and results:
        visualizer = BenchmarkVisualizer(results, output_dir)
        visualizer.plot_all()
        
        # Generate report
        report_gen = ReportGenerator(results, output_dir)
        report_path = report_gen.generate_markdown_report()
        report_gen.generate_console_summary()
        
        click.echo(f"\n✓ Report generated: {report_path}")


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    help="Output directory (defaults to same as results file)",
)
def visualize(results_file, output_dir):
    """Generate visualizations from existing results file."""
    
    # Load results
    results_path = Path(results_file)
    with open(results_path) as f:
        data = json.load(f)
    
    results = [BenchmarkMetrics(**r) for r in data]
    
    # Set output dir
    if output_dir is None:
        output_dir = results_path.parent
    
    # Generate visualizations
    visualizer = BenchmarkVisualizer(results, str(output_dir))
    paths = visualizer.plot_all()
    
    # Generate report
    report_gen = ReportGenerator(results, str(output_dir))
    report_path = report_gen.generate_markdown_report()
    report_gen.generate_console_summary()
    
    click.echo(f"\n✓ Visualizations saved to: {output_dir}")
    click.echo(f"✓ Report generated: {report_path}")


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
def report(results_file):
    """Generate a report from existing results."""
    
    # Load results
    with open(results_file) as f:
        data = json.load(f)
    
    results = [BenchmarkMetrics(**r) for r in data]
    
    # Generate report
    output_dir = Path(results_file).parent
    report_gen = ReportGenerator(results, str(output_dir))
    report_gen.generate_console_summary()
    report_path = report_gen.generate_markdown_report()
    
    click.echo(f"\n✓ Report generated: {report_path}")


@cli.command()
@click.argument("results_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    default="results",
    help="Output directory",
)
def compare(results_files, output_dir):
    """Compare multiple benchmark results."""
    
    if len(results_files) < 2:
        click.echo("Error: Please provide at least 2 results files to compare")
        return
    
    # Load all results
    all_results = []
    for results_file in results_files:
        with open(results_file) as f:
            data = json.load(f)
            all_results.extend([BenchmarkMetrics(**r) for r in data])
    
    # Generate comparison visualizations
    visualizer = BenchmarkVisualizer(all_results, output_dir)
    visualizer.plot_all()
    
    # Generate report
    report_gen = ReportGenerator(all_results, output_dir)
    report_path = report_gen.generate_markdown_report()
    report_gen.generate_console_summary()
    
    click.echo(f"\n✓ Comparison complete!")
    click.echo(f"✓ Report generated: {report_path}")


@cli.command()
def list_backends():
    """List available backends."""
    from src.backends import BACKEND_REGISTRY
    
    click.echo("\nAvailable backends:")
    for backend_name in sorted(BACKEND_REGISTRY.keys()):
        click.echo(f"  - {backend_name}")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
