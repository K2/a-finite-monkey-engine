#!/bin/bash
# Simple script to run just the FastHTML web app

# Set environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONIOENCODING=utf-8
export TERM=xterm-256color
export TERMINFO=/etc/terminfo

# Detect virtual environment
if [ -d ".venv" ]; then
    echo "Using existing virtual environment..."
    source .venv/bin/activate
else
    echo "No virtual environment found, running in current Python environment."
fi

# Make sure script directory is the working directory
cd "$(dirname "$0")"

# Install FastHTML if present
if [ -d "web/fasthtml" ]; then
    echo "Installing FastHTML from local repo..."
    pip install -e web/fasthtml/
fi

# Run the Python script with better debug output
echo "Starting FastHTML web interface with Jupyter Terminal integration..."
echo "Open http://localhost:8000 in your browser"

python -c "
import asyncio
import uvicorn
from finite_monkey.web.fasthtml_app import asgi_app

async def main():
    config = uvicorn.Config(
        app=asgi_app, 
        host='0.0.0.0', 
        port=8888, 
        log_level='info'
    )
    server = uvicorn.Server(config)
    await server.serve()

asyncio.run(main())
"
