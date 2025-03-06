# Finite Monkey Engine

A comprehensive smart contract security analysis framework with multi-agent architecture and asynchronous processing.
[![CI](https://github.com/K2/a-finite-monkey-engine/actions/workflows/python-publish.yml/badge.svg?branch=main)](https://github.com/K2/a-finite-monkey-engine/actions/workflows/python-publish.yml)
## Features

- **Asynchronous Architecture**: Fully asynchronous workflow processing for handling large codebases efficiently
- **Multi-LLM Support**: Supports multiple LLM providers (Ollama, Claude, OpenAI) for different analysis tasks
- **Specialized Agents**: Dedicated agents for research, validation, and documentation
- **Vector Store Integration**: LlamaIndex-based vector store for efficient code analysis
- **TreeSitter Analysis**: Semantic code analysis using TreeSitter
- **Web Interface**: Interactive web interface for project management and result visualization
- **Cognitive Bias Analysis**: Identifies cognitive biases in code that may lead to vulnerabilities
- **Interactive Console**: Built-in IPython console with syntax highlighting for debugging

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

## Configuration

Finite Monkey Engine is configured through environment variables, `.env` file, or command-line arguments. Key configurations:

- `WORKFLOW_MODEL`: Default LLM model for workflow
- `CLAUDE_API_KEY`: API key for Claude integration
- `OPENAI_API_BASE`: Base URL for OpenAI/Ollama API
- `VECTOR_STORE_PATH`: Path to LanceDB vector store

## Usage

### Command Line Interface

```bash
# Analyze a single contract
./run.py analyze -f examples/Vault.sol

# Analyze all contracts in a directory
./run.py analyze -d examples/

# Generate visualization for a contract
./run.py visualize examples/Vault.sol

# Analyze contracts and generate visualization
./run.py full-audit -f examples/Vault.sol
```

### Python API

```python
from finite_monkey.agents import WorkflowOrchestrator

async def main():
    orchestrator = WorkflowOrchestrator()
    
    # Run a complete audit
    report = await orchestrator.run_audit(
        solidity_path="examples/Vault.sol",
        query="Check for reentrancy vulnerabilities"
    )
    
    # Save the report
    await report.save("report.md")

# Run with asyncio
import asyncio
asyncio.run(main())
```

## Web Interface

```bash
# Start the web interface
python -m finite_monkey web
```

## Development

See `NEXT_STEPS.md` for current roadmap and priorities.

## License

MIT
