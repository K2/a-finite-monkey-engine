#!/bin/bash
# Run the full integrated UI for the Finite Monkey Engine
# This script starts both the FastHTML web interface and the enhanced terminal

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
pip install -r requirements-web.txt
# Install specific terminal dependencies
pip install jupyter_client ipykernel markdown python-multipart jinja2

# Install FastHTML if present
if [ -d "web/fasthtml" ]; then
    echo "Installing FastHTML from local repo..."
    pip install -e web/fasthtml/
fi

# Create needed directories
mkdir -p logs uploads templates static

# Start the enhanced terminal in the background
echo "Starting the enhanced terminal on port 8888..."
python enhanced_terminal.py &
TERM_PID=$!

# Wait for the terminal to initialize
sleep 3

# Start the FastHTML web interface
echo "Starting the FastHTML web interface on port 8000..."
echo "Open http://localhost:8000 in your browser"
echo "The terminal dashboard is available at http://localhost:8888"
echo "Press Ctrl+C to exit both services"

python -m uvicorn finite_monkey.web.fasthtml_app:asgi_app --host 0.0.0.0 --port 8887 --reload

# If the FastHTML app exits, also kill the terminal process
kill $TERM_PID
