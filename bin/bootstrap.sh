#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting bootstrap process with uv...${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv is not installed. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Check again after installation
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Failed to install uv. Please install it manually: https://docs.astral.sh/uv/getting-started/installation/${NC}"
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${GREEN}Creating new virtual environment with Python 3.12...${NC}"
    uv venv --python 3.12
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}Installing requirements with uv...${NC}"
    uv pip install -r requirements.txt
else
    echo -e "${RED}requirements.txt not found. Skipping requirements installation.${NC}"
fi

echo -e "${GREEN}Bootstrap completed successfully!${NC}"
echo -e "${GREEN}To activate the virtual environment, run: source .venv/bin/activate${NC}"
echo -e "${YELLOW}Note: uv uses Python 3.12+ by default for optimal performance${NC}"
