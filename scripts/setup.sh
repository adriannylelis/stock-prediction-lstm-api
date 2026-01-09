#!/bin/bash
# Setup ML Environment - Python 3.13 + ML dependencies
# Run this ONCE to setup base environment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

function log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
function log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
function log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
function log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}   Stock Prediction LSTM - Setup${NC}"
echo -e "${CYAN}   Python 3.13 + ML + API${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Hardware selection (FIRST - antes de tudo)
echo -e "${YELLOW}Hardware Selection:${NC}"
echo "1) CPU (Recommended for most users)"
echo "2) NVIDIA GPU - CUDA 12.4"
echo ""
read -p "Select mode (1 or 2): " gpu_choice

case $gpu_choice in
    1)
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        MODE_MSG="CPU"
        ;;
    2)
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        MODE_MSG="CUDA 12.4"
        ;;
    *)
        log_error "Invalid choice. Use 1 or 2."
        exit 1
        ;;
esac

log_success "Hardware mode: $MODE_MSG"
echo ""

# Step 1: Find Python 3.13
log_info "Step 1/5: Looking for Python 3.13..."

PYTHON_CMD=""
for cmd in "python3.13" "python3" "python"; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        if [[ "$version" == "3.13" ]]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    log_error "Python 3.13 not found!"
    echo "Install Python 3.13 from: https://python.org"
    exit 1
fi

log_success "Found: $($PYTHON_CMD --version)"

# Step 2: Check existing venv
echo ""
log_info "Step 2/5: Checking virtual environment..."
if [ -d ".venv" ]; then
    log_warn "Virtual environment already exists at .venv/"
    log_warn "To recreate, delete manually: rm -rf .venv"
    log_warn "Then run this script again."
    echo ""
    read -p "Continue with existing .venv? (y/n): " continue_choice
    if [[ ! $continue_choice =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    SKIP_VENV_CREATE=true
else
    SKIP_VENV_CREATE=false
fi

# Step 3: Create venv (if needed)
if [ "$SKIP_VENV_CREATE" = false ]; then
    echo ""
    log_info "Step 3/5: Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    log_success "Virtual environment created"
else
    echo ""
    log_info "Step 3/5: Skipping venv creation (already exists)"
fi

# Step 4: Activate and install
echo ""
log_info "Step 4/5: Installing dependencies..."

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Upgrade pip
echo -ne "${CYAN}Upgrading pip...${NC} "
python -m pip install --upgrade pip --quiet
echo -e "${GREEN}Done${NC}"

# Install PyTorch with selected index
echo ""
echo -e "${CYAN}Installing PyTorch ($MODE_MSG)...${NC}"
pip install torch torchvision --index-url $TORCH_INDEX

# Install all dependencies
echo ""
echo -e "${CYAN}Installing dependencies (requirements.txt)...${NC}"
pip install -r requirements.txt

# Install development tools (ruff, ipython, pytest)
echo ""
echo -e "${CYAN}Installing development tools (requirements-dev.txt)...${NC}"
pip install -r requirements-dev.txt

# Install project in editable mode
echo ""
echo -e "${CYAN}Installing project in editable mode (pip install -e .)...${NC}"
pip install -e .
log_success "Project installed in editable mode"

# Step 5: Create necessary directories
echo ""
log_info "Step 5/5: Creating project directories..."
mkdir -p data/raw data/processed data/mlflow/tracking data/mlflow/artifacts
mkdir -p models artifacts/models logs
log_success "Directories created"

# Create production config if missing
if [ ! -f "configs/production_model.yaml" ]; then
    mkdir -p configs
    cat > configs/production_model.yaml << EOF
# Auto-generated production config
model_uri: null
deployed_at: null
deployed_by: setup_script
tracking_uri: "file:./data/mlflow/tracking"
version: null
metrics: {}
EOF
    log_success "Created configs/production_model.yaml"
fi

# Verify installation
echo ""
log_info "Verifying critical imports..."
python -c "
import torch
import mlflow
import pandas
import flask
print(f'PyTorch: {torch.__version__}')
print(f'MLflow: {mlflow.__version__}')
print(f'Flask: {flask.__version__}')
if '$gpu_choice' == '2':
    if torch.cuda.is_available():
        print(f'CUDA: Available ({torch.cuda.get_device_name(0)})')
    else:
        print('WARNING: CUDA requested but not available')
else:
    print('Mode: CPU')
"

if [ $? -ne 0 ]; then
    log_error "Import verification failed!"
    exit 1
fi

# Success summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  1. Activate venv:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "     source .venv/Scripts/activate"
else
    echo "     source .venv/bin/activate"
fi
echo "  2. Run tests: pytest tests/unit/ -v"
echo "  3. Start backend: ./scripts/init_backend.sh"
echo "  4. Train model: stock-predict train --ticker PETR4.SA"
echo ""
echo -e "${YELLOW}Available commands:${NC}"
echo "  - stock-predict --help          # CLI help"
echo "  - pytest tests/ -v --cov        # Run tests with coverage"
echo "  - ./scripts/init_backend.sh     # Start Flask API"
echo ""
echo -e "${YELLOW}Tips:${NC}"
echo "  - Recreate venv: rm -rf .venv && ./scripts/setup.sh"
echo "  - Check GPU: python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
