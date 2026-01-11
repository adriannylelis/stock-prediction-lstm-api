#!/bin/bash
# Init Backend - Start Flask API server
# Requires: ./scripts/setup.sh already executed

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}   Starting Flask Backend${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: No .venv found!${NC}"
    echo "Run ./scripts/setup.sh first to create environment."
    exit 1
fi

# Activate venv
echo -e "${YELLOW}[1/2] Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Verify Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo -e "${RED}Error: Flask not installed!${NC}"
    echo "Run ./scripts/setup.sh to install dependencies."
    exit 1
fi

# Start backend
echo ""
echo -e "${YELLOW}[2/2] Starting Flask development server...${NC}"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Backend URL:  ${NC}http://localhost:5000"
echo -e "${GREEN}  Health Check: ${NC}http://localhost:5000/health"
echo -e "${GREEN}  Endpoints:    ${NC}/predict, /model-info"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Set Flask environment and run
export FLASK_APP=src.api.main:create_app
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000 --reload
