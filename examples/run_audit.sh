#!/bin/bash
# Script to run an audit on a smart contract

# Check if file argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <solidity_file_or_directory> [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL          Specify the LLM model to use"
    echo "  -q, --query QUERY          Specify the audit query"
    echo "  -o, --output OUTPUT        Specify the output file"
    echo ""
    echo "Examples:"
    echo "  $0 examples/Vault.sol"
    echo "  $0 examples/Vault.sol -m llama3:8b-instruct"
    echo "  $0 examples/defi_project/ -q \"Check for reentrancy vulnerabilities\""
    exit 1
fi

# First argument is the file or directory to analyze
FILE_OR_DIR="$1"
shift  # Remove first argument from $@

# Determine if it's a file or directory
if [ -f "$FILE_OR_DIR" ]; then
    ANALYZE_ARG="-f $FILE_OR_DIR"
elif [ -d "$FILE_OR_DIR" ]; then
    ANALYZE_ARG="-d $FILE_OR_DIR"
else
    echo "Error: File or directory not found: $FILE_OR_DIR"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    # Make sure we have required dependencies
    pip install aiosqlite > /dev/null
fi

# Run the analysis
python -m finite_monkey analyze $ANALYZE_ARG "$@"