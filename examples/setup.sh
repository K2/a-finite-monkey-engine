#!/bin/bash
# Setup script for the Finite Monkey Engine

set -e  # Exit immediately if any command fails

# Print banner
echo "========================================================"
echo "Finite Monkey Engine - Setup"
echo "Smart Contract Audit & Analysis Framework"
echo "========================================================"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Please install uv from https://github.com/astral-sh/uv"
    echo "Recommended: pip install uv"
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment..."
uv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install the package in development mode
echo "Installing dependencies..."
uv pip install -e .

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p reports
mkdir -p lancedb
mkdir -p db

echo "========================================================"
echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the CLI:"
echo "  python -m finite_monkey analyze -f <solidity_file>"
echo ""
echo "To run the web interface:"
echo "  python -m finite_monkey web"
echo "========================================================"