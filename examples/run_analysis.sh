#!/bin/bash
# Run the async analyzer for Finite Monkey Engine

# Set environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONIOENCODING=utf-8

# Detect virtual environment
if [ -d ".venv" ]; then
    echo "Using existing virtual environment..."
    source .venv/bin/activate
else
    echo "No virtual environment found, running in current Python environment."
fi

# Default values
FILE=""
DIR=""
OUTPUT="reports"
MODEL=""
VALIDATOR_MODEL=""
QUERY="Perform a comprehensive security audit"
PROJECT_NAME=""
NO_DB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--file)
      FILE="$2"
      shift 2
      ;;
    -d|--directory)
      DIR="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -v|--validator-model)
      VALIDATOR_MODEL="$2"
      shift 2
      ;;
    -q|--query)
      QUERY="$2"
      shift 2
      ;;
    -n|--project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --no-db)
      NO_DB=true
      shift
      ;;
    -h|--help)
      echo "Usage: run_analysis.sh [options]"
      echo ""
      echo "Options:"
      echo "  -f, --file FILE            Path to a Solidity file to analyze"
      echo "  -d, --directory DIR        Path to a directory containing Solidity files"
      echo "  -o, --output DIR           Directory to store reports (default: reports)"
      echo "  -m, --model MODEL          LLM model to use (default: from config)"
      echo "  -v, --validator-model MODEL Model for validation (default: from config)"
      echo "  -q, --query QUERY          Analysis query (default: general security audit)"
      echo "  -n, --project-name NAME    Project name (default: derived from file/dir)"
      echo "  --no-db                    Disable database integration (uses PostgreSQL by default)"
      echo "  -h, --help                 Show this help message"
      echo ""
      echo "Examples:"
      echo "  ./run_analysis.sh -f examples/SimpleVault.sol"
      echo "  ./run_analysis.sh -d examples/defi_project -m llama3:70b"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate input
if [[ -z "$FILE" && -z "$DIR" ]]; then
    echo "Error: Either --file or --directory must be specified"
    echo "Use --help for usage information"
    exit 1
fi

# Build command
CMD="python3 run_async_analyzer.py"

if [[ -n "$FILE" ]]; then
    CMD="$CMD -f $FILE"
elif [[ -n "$DIR" ]]; then
    CMD="$CMD -d $DIR"
fi

CMD="$CMD -o $OUTPUT"

if [[ -n "$MODEL" ]]; then
    CMD="$CMD -m \"$MODEL\""
fi

if [[ -n "$VALIDATOR_MODEL" ]]; then
    CMD="$CMD -v \"$VALIDATOR_MODEL\""
fi

if [[ -n "$PROJECT_NAME" ]]; then
    CMD="$CMD -n \"$PROJECT_NAME\""
fi

if [[ "$NO_DB" = true ]]; then
    CMD="$CMD --no-db"
fi

CMD="$CMD -q \"$QUERY\""

# Run the command
echo "Running command: $CMD"
eval $CMD

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo "Analysis completed successfully"
    echo "Report saved to: $OUTPUT"
else
    echo "Analysis failed with exit code $exit_code"
fi

exit $exit_code