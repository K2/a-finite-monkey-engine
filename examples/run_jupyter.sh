#!/bin/bash
# Script to run the Finite Monkey Engine FastHTML web interface with Jupyter integration

# Set environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONIOENCODING=utf-8
export TERM=xterm-256color
export FORCE_COLOR=1

# Make script directory the working directory
cd "$(dirname "$0")"

# Detect virtual environment
if [ -d ".venv" ]; then
    echo "Using existing virtual environment..."
    source .venv/bin/activate
else
    echo "No virtual environment found, running in current Python environment."
fi

# Install dependencies if needed
if [ -f "requirements-web.txt" ]; then
    echo "Checking for required dependencies..."
    pip install -r requirements-web.txt --quiet
fi

# Install FastHTML if present
if [ -d "web/fasthtml" ]; then
    echo "Using local FastHTML..."
    pip install -e web/fasthtml/ --quiet
fi

# Install ipykernel if not already installed
pip install ipykernel --quiet

echo "Starting FastHTML web interface with Jupyter integration..."
echo "Open http://localhost:8000 in your browser"
echo ""
echo "- Use the Jupyter terminal for interactive coding"
echo "- Initialize agents using the provided code snippet"
echo "- Ctrl+C to exit"
echo ""

python run_jupyter_iframe.py