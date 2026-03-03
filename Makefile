.PHONY: help install install-dev install-all test lint format clean run-example

help:
	@echo "Inference Optimizer - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install core dependencies"
	@echo "  make install-dev      Install with development tools"
	@echo "  make install-all      Install all backends"
	@echo ""
	@echo "Development:"
	@echo "  make test            Run tests"
	@echo "  make lint            Run linters"
	@echo "  make format          Format code"
	@echo "  make clean           Clean build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make run-example     Run simple example"
	@echo ""

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	pytest tests/ -v

test-coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-example:
	python examples/simple_benchmark.py

benchmark-quick:
	inference-optimizer benchmark \
		--model gpt2 \
		--backends pytorch \
		--quantizations fp16 \
		--batch-sizes 1 4 \
		--num-requests 20 \
		--output-dir results/quick

list-backends:
	inference-optimizer list-backends
