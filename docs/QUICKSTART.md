# Finite Monkey Engine - Quickstart Guide

Welcome to Finite Monkey Engine! This guide will help you get started quickly with the basic features and workflows.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Web Interface](#web-interface)
4. [Python API](#python-api)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/a-finite-monkey-engine.git
cd a-finite-monkey-engine

# Install dependencies with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Basic Usage

The simplest way to use Finite Monkey Engine is through the command-line interface:

```bash
# Analyze a single Solidity contract with zero configuration
./run.py

# This will:
# 1. Use the default example contract if no file is specified
# 2. Run a comprehensive security audit
# 3. Save the report to reports/<filename>_report_<timestamp>.md
```

For more specific analysis:

```bash
# Analyze a specific contract
./run.py analyze -f examples/Vault.sol

# Analyze multiple contracts
./run.py analyze --files examples/Vault.sol examples/Token.sol

# Analyze all contracts in a directory
./run.py analyze -d examples/

# Custom analysis query
./run.py analyze -f examples/Vault.sol -q "Check for reentrancy vulnerabilities"
```

## Web Interface

The Finite Monkey Engine includes a web interface for easier analysis and visualization:

```bash
# Start the web interface with default settings
./run_web_interface.py

# Or using the module directly
python -m finite_monkey web
```

This will start the web server at http://localhost:8000 by default. From there, you can:

1. Upload and analyze smart contracts
2. View analysis results and reports
3. Use the interactive IPython console for debugging
4. Visualize contract relationships and vulnerabilities

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
        project_name="MyProject"
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