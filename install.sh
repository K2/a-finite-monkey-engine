#!/bin/bash
set -e

echo "Installing A Finite Monkey Engine using uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Please install uv from https://github.com/astral-sh/uv"
    echo "Install command: curl -sSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install the package in development mode
uv pip install -e .

# Install dependencies from requirements.txt
uv pip install -r requirements.txt

# Make sure the script is executable
chmod +x start.py

echo "Installation complete!"
echo "Run the engine with: ./start.py [input_directory] --output [output_directory]"
