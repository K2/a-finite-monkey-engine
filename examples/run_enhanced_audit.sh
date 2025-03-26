#!/bin/bash
# Enhanced Audit Runner for Finite Monkey Engine
# Provides a convenient way to run the enhanced chunking audit with smart defaults

set -e  # Exit on error

# Print banner
echo "=============================================="
echo "Finite Monkey Engine - Enhanced Audit Runner"
echo "=============================================="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Create virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    if command -v uv &> /dev/null; then
        uv venv
    else
        python3 -m venv .venv
    fi
fi

# Activate virtual environment
source .venv/bin/activate

# Ensure dependencies are installed
echo "Ensuring dependencies are installed..."
if command -v uv &> /dev/null; then
    uv pip install -e .
else
    pip install -e .
fi

# Ensure required directories exist
mkdir -p reports lancedb db

# Parse arguments
POSITIONAL_ARGS=()
MODEL="llama3:8b-instruct-q6_K"
VALIDATOR_MODEL=""
QUERY="Perform a comprehensive security audit focusing on reentrancy, access control, and data validation issues"
OUTPUT_DIR="reports"
USE_DB=true
INCLUDE_CALL_GRAPH=true
CHUNK_SIZE=4000
FILE_LIMIT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--file)
      FILE="$2"
      shift
      shift
      ;;
    -d|--directory)
      DIRECTORY="$2"
      shift
      shift
      ;;
    -m|--model)
      MODEL="$2"
      shift
      shift
      ;;
    -v|--validator-model)
      VALIDATOR_MODEL="$2"
      shift
      shift
      ;;
    -q|--query)
      QUERY="$2"
      shift
      shift
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --no-db)
      USE_DB=false
      shift
      ;;
    --no-call-graph)
      INCLUDE_CALL_GRAPH=false
      shift
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift
      shift
      ;;
    --limit)
      FILE_LIMIT="$2"
      shift
      shift
      ;;
    --help)
      echo "Usage: ./run_enhanced_audit.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -f, --file FILE                  Path to a Solidity file to analyze"
      echo "  -d, --directory DIR              Path to a directory containing Solidity files"
      echo "  -m, --model MODEL                Model to use for analysis (default: llama3:8b-instruct-q6_K)"
      echo "  -v, --validator-model MODEL      Model to use for validation (default: same as --model)"
      echo "  -q, --query QUERY                Analysis query"
      echo "  -o, --output-dir DIR             Output directory (default: reports)"
      echo "  --no-db                          Disable database persistence"
      echo "  --no-call-graph                  Disable call graph integration"
      echo "  --chunk-size SIZE                Maximum chunk size in characters (default: 4000)"
      echo "  --limit N                        Limit number of files to analyze"
      echo "  --help                           Show this help message"
      echo ""
      echo "Examples:"
      echo "  ./run_enhanced_audit.sh -f examples/SimpleVault.sol"
      echo "  ./run_enhanced_audit.sh -d examples/defi_project -m llama3:70b"
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

# Build command
CMD="python3 run_enhanced_chunking_audit.py"

if [ -n "$FILE" ]; then
  CMD="$CMD -f $FILE"
elif [ -n "$DIRECTORY" ]; then
  CMD="$CMD -d $DIRECTORY"
else
  echo "Error: Either --file or --directory must be specified"
  echo "Use --help for usage information"
  exit 1
fi

CMD="$CMD -m $MODEL"

if [ -n "$VALIDATOR_MODEL" ]; then
  CMD="$CMD -v $VALIDATOR_MODEL"
fi

CMD="$CMD -q \"$QUERY\" -o $OUTPUT_DIR --chunk-size $CHUNK_SIZE"

if [ "$USE_DB" = false ]; then
  CMD="$CMD --no-db"
fi

if [ "$INCLUDE_CALL_GRAPH" = false ]; then
  CMD="$CMD --no-call-graph"
fi

if [ -n "$FILE_LIMIT" ]; then
  CMD="$CMD --limit $FILE_LIMIT"
fi

# Execute the command
echo "Running command: $CMD"
echo "=============================================="
eval "$CMD"

# Success
exit 0