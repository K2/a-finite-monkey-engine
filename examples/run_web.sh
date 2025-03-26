#!/bin/bash
# Script to run the web interface

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the web interface
python -m finite_monkey web "$@"