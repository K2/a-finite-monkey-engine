# Finite Monkey Engine - Quickstart Guide

Welcome to Finite Monkey Engine! This guide will help you get started quickly with the basic features and workflows.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Smart Contract Chunking](#smart-contract-chunking)
4. [Web Interface](#web-interface)
5. [Python API](#python-api)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

## Installation

The fastest way to get started is to use the setup script:

```bash
# Clone the repository
git clone https://github.com/yourusername/a-finite-monkey-engine.git
cd a-finite-monkey-engine

# Make setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

This will create a virtual environment, install all dependencies, and set up the necessary directories.

If you prefer manual installation:

```bash
# Clone the repository
git clone https://github.com/yourusername/a-finite-monkey-engine.git
cd a-finite-monkey-engine

# Create and activate virtual environment using uv (recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Create necessary directories
mkdir -p reports lancedb db
```

## Basic Usage

### Quick Start with Helper Scripts

The simplest way to use Finite Monkey Engine is with the included helper scripts:

```bash
# Make the scripts executable
chmod +x run.py run_audit.sh run_web.sh

# Analyze a contract (automatically activates virtual environment)
./run_audit.sh examples/Vault.sol

# Start the web interface
./run_web.sh
```

### Zero-Configuration Mode

For a completely zero-configuration experience:

```bash
# Analyze a single Solidity contract with zero configuration
./run.py

# This will:
# 1. Use the default example contract if no file is specified
# 2. Run a comprehensive security audit
# 3. Save the report to reports/<filename>_report_<timestamp>.md
```

### Command Line Interface

For more control, use the full command-line interface:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Analyze a specific contract
python -m finite_monkey analyze -f examples/Vault.sol

# Analyze multiple contracts
python -m finite_monkey analyze --files examples/Vault.sol examples/Token.sol

# Analyze all contracts in a directory
python -m finite_monkey analyze -d examples/ --pattern "*.sol"

# Custom analysis query
python -m finite_monkey analyze -f examples/Vault.sol -q "Check for reentrancy vulnerabilities"

# Use a specific model
python -m finite_monkey analyze -f examples/Vault.sol -m "llama3:8b-instruct"

# Analyze a GitHub repository
python -m finite_monkey github https://github.com/username/repo
```

## Web Interface

The Finite Monkey Engine includes two web interfaces for analysis, visualization, and management:

### FastAPI Web Interface

The original web interface built with FastAPI:

```bash
# Start the FastAPI web interface with default settings
./run_web.sh

# Or using the module directly
python -m finite_monkey web

# Customize host and port
python -m finite_monkey web --host 127.0.0.1 --port 8080

# Enable development mode
python -m finite_monkey web --reload --debug

# Enable IPython terminal for interactive debugging
python -m finite_monkey web --enable-ipython
```

### FastHTML Web Interface (Recommended)

The newer, more responsive web interface built with FastHTML:

```bash
# Start the FastHTML web interface with default settings
./run_fasthtml_web.sh

# This will automatically:
# - Create/activate a virtual environment if needed
# - Install required dependencies
# - Start the server with hot reload enabled
```

## Smart Contract Chunking

Finite Monkey Engine automatically handles large contracts using semantic chunking. Large contracts often exceed the context window size of LLMs, which would normally prevent comprehensive analysis. Our chunking system solves this by:

1. Detecting if a contract exceeds the context window size (default: 8000 characters)
2. Splitting the contract into semantically meaningful chunks:
   - By contract boundaries (separating multiple contracts in a file)
   - By function boundaries (analyzing functions separately)
   - Preserving imports and contract structure
3. Analyzing each chunk separately
4. Intelligently combining the results to provide a comprehensive analysis

This works transparently without any configuration, but you can customize the chunking behavior in your code:

```python
from finite_monkey.utils.chunking import chunk_solidity_file, ContractChunker

# Simple usage with default parameters
chunks = chunk_solidity_file("path/to/large/contract.sol")

# Customize chunking parameters
chunker = ContractChunker(
    max_chunk_size=12000,       # Increase for models with larger context windows
    overlap_size=500,           # Overlap between chunks for context continuity
    preserve_imports=True,      # Include imports in each chunk
    chunk_by_contract=True,     # Split by contract boundaries
    chunk_by_function=True      # Further split by function if needed
)

# Process a file or code snippet
file_chunks = chunker.chunk_file("path/to/large/contract.sol")
code_chunks = chunker.chunk_code(contract_code, name="MyContract")

# Each chunk contains detailed metadata
for chunk in file_chunks:
    print(f"Processing {chunk['chunk_id']} (type: {chunk['chunk_type']})")
    print(f"Contains {len(chunk['content'])} characters")
    
    # The chunk includes metadata like:
    # - chunk_id: Unique identifier
    # - chunk_type: 'contract', 'function', 'size_chunk', or 'complete_file'
    # - contract_name: Name of the contract (if applicable)
    # - function_name: Name of the function (if applicable)
    # - imports: List of import statements
    # - start_char/end_char: Original position in the file
```

You can also combine analysis results from multiple chunks:

```python
from finite_monkey.utils.chunking import ContractChunker

# Combine results from analyzing multiple chunks
combined_results = ContractChunker.combine_analysis_results([
    chunk1_analysis, 
    chunk2_analysis,
    chunk3_analysis
])

# The combined results will:
# - Deduplicate findings
# - Merge recommendations
# - Summarize insights
```

Both interfaces will start at http://localhost:8000 by default. From there, you can:

1. **Dashboard**: View project metrics and recent activities
2. **Terminal**: Interactive IPython terminal with framework objects pre-loaded
3. **Reports**: Browse and view security audit reports with markdown rendering
4. **Visualizations**: Interactive charts and graphs for security analysis insights

The web interface provides several key features:
- Complete smart contract security audit workflow
- Real-time terminal with WebSocket updates
- Markdown report rendering with code highlighting
- Interactive visualizations for vulnerability analysis
- Project management and configuration
- Agent telemetry monitoring

The web interface is particularly useful for:
- Managing complex projects with multiple contracts
- Long-running audits that need monitoring
- Detailed configuration adjustments
- Interactive exploration of results
- Visualizing contract relationships and vulnerabilities

## Python API

For programmatic usage, Finite Monkey Engine provides a simple Python API:

```python
from finite_monkey.agents import WorkflowOrchestrator

# Create an orchestrator with default settings (just works!)
orchestrator = WorkflowOrchestrator()

# Run a complete audit with default parameters
async def run_simple_audit():
    # With no arguments, it uses the example contract
    report = await orchestrator.run_audit()
    await report.save()
    print(f"Report saved to {report.report_path}")

# For more control:
async def run_custom_audit():
    report = await orchestrator.run_audit(
        solidity_path="examples/Vault.sol",
        query="Check for reentrancy vulnerabilities",
        project_name="MyProject",
        max_chunk_size=8000  # Control chunking behavior
    )
    await report.save("custom_report.md")

# Run with asyncio
import asyncio
asyncio.run(run_simple_audit())
```

## Configuration

Finite Monkey Engine can be configured in several ways:

1. **Environment Variables**: Set variables like `WORKFLOW_MODEL` or `CLAUDE_API_KEY`
2. **.env File**: Create a `.env` file in the project root
3. **Command-line Arguments**: Pass options to the CLI commands

Example `.env` file:

```
WORKFLOW_MODEL=claude-3-5-sonnet
CLAUDE_API_KEY=your_api_key_here
WEB_PORT=8080
```

## Troubleshooting

If you encounter issues:

1. **Dependency Problems**:
   ```bash
   # Verify all dependencies are installed
   uv pip install -e .
   ```

2. **Model Issues**:
   - Check that you have the appropriate API keys set
   - For local models, ensure Ollama is running (`ollama serve`)

3. **Database Errors**:
   - The default SQLite database should work out of the box
   - For PostgreSQL, make sure the database exists and is accessible

4. **Debug Mode**:
   ```bash
   # Run with debug logging
   DEBUG=1 ./run.py
   ```

5. **Interactive Debugging**:
   ```python
   # In the web interface console
   orchestrator.researcher.debug_mode = True
   await orchestrator.researcher.analyze_code_async(code_snippet="your code here")
   ```

## Advanced Features

For more advanced usage, check out:

- [Agent Architecture](./AGENT_ARCHITECTURE.md)
- [Web Interface Guide](./WEB_INTERFACE.md)
- [Async Workflow](./ASYNC_WORKFLOW.md)
- [LLM Model Evaluation](../LLM_MODEL_EVALUATION.md)