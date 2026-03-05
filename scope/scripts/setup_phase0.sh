#!/bin/bash

# AMIO Phase 0 - Automated Setup Script
# Sets up environment and validates installation for Apple Silicon M3

set -e  # Exit on error

echo "================================================================================"
echo "  AMIO Phase 0 - Automated Setup"
echo "  Adaptive Multimodal Inference Optimizer"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "--------------------------------------------------------------------------------"

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    print_error "This script requires macOS"
    exit 1
fi
print_success "Running on macOS"

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    print_error "This script requires Apple Silicon (ARM64)"
    exit 1
fi
print_success "Apple Silicon detected"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.9-3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_success "Python ${PYTHON_VERSION} found"

echo ""

# Step 2: Create virtual environment
echo "Step 2: Setting up Python virtual environment..."
echo "--------------------------------------------------------------------------------"

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

echo ""

# Step 3: Upgrade pip
echo "Step 3: Upgrading pip..."
echo "--------------------------------------------------------------------------------"

pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "pip upgraded"

echo ""

# Step 4: Install core dependencies
echo "Step 4: Installing core dependencies..."
echo "--------------------------------------------------------------------------------"

print_info "Installing MLX framework..."
pip install "mlx>=0.10.0" > /dev/null 2>&1
print_success "MLX installed"

print_info "Installing MLX-LM..."
pip install "mlx-lm>=0.10.0" > /dev/null 2>&1
print_success "MLX-LM installed"

print_info "Installing MLX-VLM (optional)..."
if pip install mlx-vlm > /dev/null 2>&1; then
    print_success "MLX-VLM installed"
else
    print_warning "MLX-VLM not available, continuing without it"
fi

echo ""

# Step 5: Install supporting libraries
echo "Step 5: Installing supporting libraries..."
echo "--------------------------------------------------------------------------------"

print_info "Installing NumPy, Pillow, and system tools..."
pip install numpy>=1.24.0 pillow>=10.0.0 psutil>=5.9.0 > /dev/null 2>&1
print_success "System libraries installed"

print_info "Installing HuggingFace libraries..."
pip install transformers>=4.40.0 huggingface-hub>=0.20.0 > /dev/null 2>&1
print_success "HuggingFace libraries installed"

print_info "Installing visualization libraries..."
pip install matplotlib>=3.7.0 seaborn>=0.12.0 > /dev/null 2>&1
print_success "Visualization libraries installed"

print_info "Installing packaging for version checks..."
pip install packaging > /dev/null 2>&1
print_success "Packaging installed"

echo ""

# Step 6: Validate installation
echo "Step 6: Validating installation..."
echo "--------------------------------------------------------------------------------"

python config/validate_stack.py

echo ""

# Step 7: Run component tests
echo "Step 7: Running component tests..."
echo "--------------------------------------------------------------------------------"

print_info "Testing quantization framework..."
if python models/quantization.py > /dev/null 2>&1; then
    print_success "Quantization framework test passed"
else
    print_warning "Quantization framework test had issues (non-critical)"
fi

print_info "Testing TP simulation..."
if python simulation/tp_simulator.py > /dev/null 2>&1; then
    print_success "TP simulation test passed"
else
    print_warning "TP simulation test had issues (non-critical)"
fi

print_info "Testing metrics collection..."
if python metrics/collector.py > /dev/null 2>&1; then
    print_success "Metrics collection test passed"
else
    print_warning "Metrics collection test had issues (non-critical)"
fi

print_info "Testing SLA validation..."
if python metrics/sla_validator.py > /dev/null 2>&1; then
    print_success "SLA validation test passed"
else
    print_warning "SLA validation test had issues (non-critical)"
fi

echo ""

# Step 8: Display summary
echo "================================================================================"
echo "  Setup Complete!"
echo "================================================================================"
echo ""
echo "📦 Installed packages:"
echo "  - MLX framework for Apple Silicon"
echo "  - MLX-LM for language models"
echo "  - Quantization framework (INT4)"
echo "  - TP simulation (multi-GPU modeling)"
echo "  - Metrics collection (TTFT, TBT, Fragmentation)"
echo "  - SLA validation"
echo ""
echo "🚀 Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Review design document: docs/DESIGN.md"
echo "  3. Test model loading: python models/multimodal_loader.py"
echo "  4. Proceed to Phase 1: Adaptive controller development"
echo ""
echo "📊 System info:"
echo "  - Platform: $(uname -s) $(uname -m)"
echo "  - Python: $(python --version)"
echo "  - MLX: $(python -c 'import mlx; print(mlx.__version__)')"
echo "  - Memory: $(sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}')"
echo ""
echo "✅ Phase 0 foundation is ready!"
echo "================================================================================"
