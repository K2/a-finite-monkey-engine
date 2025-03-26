#!/bin/bash
#
# Convenience script to run the Finite Monkey Engine analysis pipeline
# with proper environment setup.
#

# Ensure script fails on error
set -eo pipefail

# Check for virtual environment and activate it if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "No virtual environment found, checking for uv..."
    if command -v uv >/dev/null 2>&1; then
        echo "Setting up virtual environment with uv..."
        uv venv
        source .venv/bin/activate
        uv pip install -e .
        mkdir -p reports lancedb db
    else
        echo "Please set up your environment first using setup.sh or manually."
        echo "You can install uv with: pip install uv"
        exit 1
    fi
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the analysis pipeline with all arguments passed
python "$SCRIPT_DIR/run_analysis_pipeline.py" "$@"