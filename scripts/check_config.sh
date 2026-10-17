#!/bin/bash

# Check Paper2Slides Configuration
# This script verifies that all required dependencies and configurations are in place

echo "=============================================="
echo "Paper2Slides Configuration Checker"
echo "=============================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track errors
ERRORS=0
WARNINGS=0

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check 1: Python version
echo -n "Checking Python version... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python3 not found"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Node.js
echo -n "Checking Node.js... "
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓${NC} $NODE_VERSION"
else
    echo -e "${YELLOW}⚠${NC}  Node.js not found (required for frontend)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 3: .env file
echo -n "Checking .env file... "
ENV_FILE="paper2slides/.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}✓${NC} Found at $ENV_FILE"
    
    # Check 4: LLM provider credentials. Never print key material.
    source "$ENV_FILE"
    LLM_PROVIDER="${RAG_LLM_PROVIDER:-openai_compatible}"
    echo -n "Checking LLM provider ($LLM_PROVIDER)... "
    case "$LLM_PROVIDER" in
        openai_compatible)
            if [ -z "$RAG_LLM_API_KEY" ]; then
                echo -e "${RED}✗${NC} RAG_LLM_API_KEY is not set"
                ERRORS=$((ERRORS + 1))
            else
                echo -e "${GREEN}✓${NC} Text provider configured"
            fi
            ;;
        codex_gateway)
            if [ -z "$CODEX_API_KEY" ] || [ -z "$CODEX_BASE_URL" ] || [ -z "$CODEX_MODEL" ]; then
                echo -e "${RED}✗${NC} CODEX_API_KEY, CODEX_BASE_URL, and CODEX_MODEL are required"
                ERRORS=$((ERRORS + 1))
            else
                echo -e "${GREEN}✓${NC} External Responses provider configured"
            fi
            ;;
        *)
            echo -e "${RED}✗${NC} Unsupported provider"
            ERRORS=$((ERRORS + 1))
            ;;
    esac
else
    echo -e "${RED}✗${NC} Not found"
    echo "   Please copy env.example to paper2slides/.env"
    ERRORS=$((ERRORS + 1))
fi

# Check 5: Python dependencies
echo -n "Checking Python dependencies... "
if python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Backend dependencies installed"
else
    echo -e "${RED}✗${NC} Missing dependencies"
    echo "   Run: pip install -r requirements.txt"
    ERRORS=$((ERRORS + 1))
fi

# Check 6: paper2slides dependencies
echo -n "Checking paper2slides modules... "
cd "$PROJECT_ROOT"
if python3 -c "
import sys
import importlib.util
from pathlib import Path
PAPER2SLIDES_DIR = Path('.').resolve() / 'paper2slides'
sys.path.insert(0, str(PAPER2SLIDES_DIR))
spec = importlib.util.spec_from_file_location('main', str(PAPER2SLIDES_DIR / 'main.py'))
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} paper2slides modules loadable"
else
    echo -e "${RED}✗${NC} Cannot load paper2slides modules"
    echo "   Check: pip install -r requirements.txt"
    ERRORS=$((ERRORS + 1))
fi

# Check 7: Node modules
echo -n "Checking frontend dependencies... "
if [ -d "frontend/node_modules" ]; then
    echo -e "${GREEN}✓${NC} Frontend dependencies installed"
else
    echo -e "${YELLOW}⚠${NC}  node_modules not found"
    echo "   Run: cd frontend && npm install"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 8: Directories
echo -n "Checking required directories... "
if [ -d "data/uploads" ] && [ -d "outputs" ]; then
    echo -e "${GREEN}✓${NC} data/uploads/ and outputs/ exist"
else
    echo -e "${YELLOW}⚠${NC}  Will be created automatically"
fi

# Summary
echo ""
echo "=============================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ Configuration looks good!${NC}"
    echo ""
    echo "To start the servers:"
    echo "  All services: ./scripts/start.sh"
    echo "  Backend only: ./scripts/start_backend.sh [port]"
    echo "  Frontend only: ./scripts/start_frontend.sh"
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS error(s)${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ Found $WARNINGS warning(s)${NC}"
    fi
    echo ""
    echo "Please fix the errors above before starting the server."
    exit 1
fi
