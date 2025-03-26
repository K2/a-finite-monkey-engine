#!/bin/bash
# Run the full-featured Engine Terminal for the Finite Monkey Engine

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
echo "Checking for required dependencies..."
pip install fastapi uvicorn websockets jupyter_client ipykernel jinja2 python-multipart pydantic

# Create directories if they don't exist
mkdir -p templates static logs uploads

# Start the engine terminal
echo "Starting the Engine Terminal..."
echo "This will open your browser when the server is ready."
echo "Press Ctrl+C to exit."

# Run the terminal
python engine_terminal.py