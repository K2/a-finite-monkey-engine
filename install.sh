#!/bin/bash
# Installation script for Finite Monkey framework

set -e  # Exit on error

# Print banner
echo "=============================================="
echo "Finite Monkey Installation"
echo "Smart Contract Audit & Analysis Framework"
echo "=============================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Detected Python version: $python_version"

required_version="3.10.0"
if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Error: Python 3.10.0 or higher is required"
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .

# Setup success
echo ""
echo "=============================================="
echo "Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the framework, run:"
echo "  ./run.py <solidity_file> [options]"
echo "=============================================="