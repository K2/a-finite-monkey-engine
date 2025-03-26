#!/bin/bash

# Guidance Integration Installation Script with UV
# This script sets up the Python environment for guidance integration using UV package manager

echo "Setting up Guidance integration for A Finite Monkey Engine..."

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV package manager..."
    curl -sSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | bash
    
    # Add UV to PATH for the current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo "UV installed successfully!"
fi

# Create a virtual environment (optional)
echo "Creating a Python virtual environment with UV..."
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install requirements using UV
echo "Installing Python dependencies with UV..."
uv pip install -r requirements.txt

echo "Testing Guidance installation..."
python -c "import guidance; print(f'Guidance {guidance.__version__} installed successfully!')"

echo "Setup complete!"
echo "To use the virtual environment, run: source .venv/bin/activate"
