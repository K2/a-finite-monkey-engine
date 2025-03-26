#!/bin/bash

# Install the project using uv
echo "Installing A Finite Monkey Engine using uv..."
uv pip install -e .

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Setup complete! You can now run the project with: python start.py"
