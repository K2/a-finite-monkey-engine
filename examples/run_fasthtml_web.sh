#!/bin/bash
# Script to run the FastHTML-based web interface

# Set environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Activate the virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-web.txt
pip install -e .

# Check if local FastHTML repo exists
if [ -d "web/fasthtml" ]; then
    echo "Installing FastHTML from local repo..."
    pip install -e web/fasthtml/
else
    echo "FastHTML repository not found locally."
    exit 1
fi

# Run the web interface with Uvicorn for better performance and hot reload
echo "Starting FastHTML web interface..."
python -m uvicorn finite_monkey.web.fasthtml_app:asgi_app --host 0.0.0.0 --port 8000 --reload