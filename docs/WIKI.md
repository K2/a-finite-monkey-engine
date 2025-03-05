# Finite Monkey Engine Wiki

Welcome to the Finite Monkey Engine Wiki! This comprehensive guide will help you understand the framework, its architecture, and how to use it effectively for smart contract security analysis.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Workflows](#workflows)
4. [Agents](#agents)
5. [Configuration](#configuration)
6. [Web Interface](#web-interface)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Overview

Finite Monkey Engine is a comprehensive smart contract security analysis framework with a multi-agent architecture and asynchronous processing capabilities. It leverages the power of Large Language Models (LLMs) to analyze smart contracts for security vulnerabilities, with a special focus on cognitive biases that can lead to security issues.

### Key Features

- **Multi-agent Architecture**: Specialized agents for different aspects of security analysis
- **Asynchronous Processing**: Efficient handling of large codebases
- **Multiple LLM Support**: Works with Claude, Ollama, and OpenAI models
- **Vector Store Integration**: Efficient code analysis with LlamaIndex
- **TreeSitter Analysis**: Semantic code parsing
- **Web Interface**: Interactive visualization and management
- **Cognitive Bias Analysis**: Identifies human biases in code that may lead to vulnerabilities

## Architecture

The framework follows a modular architecture centered around specialized agents that work together to analyze smart contracts.

### High-Level Components

```
                    ┌──────────────┐
                    │              │
                    │ Orchestrator │
                    │              │
                    └──────┬───────┘
                           │
      ┌──────────┬────────┼────────┬──────────┐
      │          │        │        │          │
┌─────▼─────┐┌───▼───┐┌───▼───┐┌───▼───┐┌─────▼─────┐
│           ││       ││       ││       ││           │
│Researcher ││Validator││Documentor││ Bias  ││Counterfactual│
│  Agent   ││ Agent  ││ Agent  ││Analyzer││  Generator │
│           ││       ││       ││       ││           │
└─────┬─────┘└───┬───┘└───┬───┘└───┬───┘└─────┬─────┘
      │          │        │        │          │
      └──────────┴────────┼────────┴──────────┘
                          │
                  ┌───────▼──────┐
                  │              │
                  │ Final Report │
                  │              │
                  └──────────────┘
```

### Database Structure

- **Projects**: Top-level container for audits
- **Files**: Code files being analyzed
- **Audits**: Analysis sessions
- **Findings**: Discovered vulnerabilities

## Workflows

The framework supports multiple workflow types to accommodate different use cases:

### 1. Simple Workflow

```bash
./run.py analyze -f examples/Vault.sol --simple
```

The simple workflow uses a serial execution model where each agent processes in turn:
1. Researcher analyzes code
2. Validator checks results
3. Documentor generates report

### 2. Atomic Agent Workflow (Default)

```bash
./run.py analyze -f examples/Vault.sol
```

The atomic agent workflow uses a more complex monitoring system where agents monitor each other's outputs:
1. Researcher analyzes code
2. Validator checks and provides feedback
3. Orchestrator monitors and coordinates
4. Documentor generates final report

### 3. Asynchronous Workflow

```bash
./run.py full-audit -d examples/
```

For larger projects, the asynchronous workflow handles multiple files in parallel with database persistence.

### Zero-Configuration Mode

For quick testing and debugging, you can run with no arguments:

```bash
./run.py
```

This will:
1. Use the default example contract (or find a .sol file in the current directory)
2. Run a comprehensive security audit with default settings
3. Save the report to the reports directory

## Agents

The framework contains several specialized agents:

### Researcher Agent

Responsible for the initial code analysis and vulnerability detection. Uses a combination of static pattern matching and semantic understanding to find issues in the code.

### Validator Agent

Verifies findings from the Researcher, eliminating false positives and providing confidence scores for each vulnerability. Uses TreeSitter for semantic code analysis.

### Documentor Agent

Generates comprehensive security reports based on validated findings, including explanations, impact assessments, and remediation recommendations.

### Cognitive Bias Analyzer

Identifies cognitive biases in code that may lead to security vulnerabilities, such as:
- Normalcy Bias: Assuming normal conditions will persist
- Authority Bias: Over-reliance on trusted roles
- Confirmation Bias: Focusing on evidence that supports assumptions
- Curse of Knowledge: Inability to imagine how others might misuse contracts
- Hyperbolic Discounting: Prioritizing immediate benefits over long-term security

### Counterfactual Generator

Generates alternative code paths and scenarios to test contract behavior in edge cases.

## Configuration

Configuration can be set in multiple ways:

1. **Environment Variables**:
   ```bash
   WORKFLOW_MODEL=claude-3-5-sonnet ./run.py
   ```

2. **.env File**:
   ```
   WORKFLOW_MODEL=claude-3-5-sonnet
   CLAUDE_API_KEY=your_api_key_here
   ```

3. **pyproject.toml**:
   ```toml
   [tool.finite-monkey-engine]
   WORKFLOW_MODEL="claude-3-5-sonnet"
   ```

4. **Command Line Arguments**:
   ```bash
   ./run.py analyze -f examples/Vault.sol -m claude-3-5-sonnet
   ```

### Key Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| WORKFLOW_MODEL | Default LLM model | llama3 |
| SCAN_MODEL | Model for researcher agent | WORKFLOW_MODEL |
| CONFIRMATION_MODEL | Model for validator agent | claude-3-5-sonnet |
| RELATION_MODEL | Model for documentor agent | WORKFLOW_MODEL |
| COGNITIVE_BIAS_MODEL | Model for bias analysis | claude-3-5-sonnet |
| CLAUDE_API_KEY | API key for Claude | None |
| OPENAI_API_BASE | Base URL for OpenAI/Ollama API | http://localhost:11434/v1 |
| WEB_HOST | Host for web interface | 0.0.0.0 |
| WEB_PORT | Port for web interface | 8000 |

## Web Interface

The web interface provides a visual way to interact with the framework.

### Starting the Web Interface

```bash
./run_web_interface.py
# OR
python -m finite_monkey web
```

### Features

- Upload and analyze contracts
- View analysis results and reports
- Interactive IPython console with syntax highlighting
- Visualize contract relationships and vulnerabilities
- Track audit history

## API Reference

### Python API

```python
from finite_monkey.agents import WorkflowOrchestrator

# Create orchestrator
orchestrator = WorkflowOrchestrator()

# Run audit with zero-configuration (uses defaults)
report = await orchestrator.run_audit()

# Run audit with specific parameters
report = await orchestrator.run_audit(
    solidity_path="examples/Vault.sol",
    query="Check for reentrancy vulnerabilities",
    project_name="MyProject"
)

# Save report
await report.save("report.md")
```

### Web API

The framework includes a RESTful API:

```
POST /api/analyze
GET /api/projects
GET /api/projects/{project_id}/audits
GET /api/audits/{audit_id}/findings
```

## Troubleshooting

### Common Issues

1. **Model Unavailable**:
   - Ensure Ollama is running with `ollama serve`
   - Check API keys for Claude/OpenAI

2. **Database Errors**:
   - Default SQLite database should work out of the box
   - For PostgreSQL, ensure connection string is correct

3. **Vector Store Issues**:
   - Check LanceDB installation and permissions

4. **TreeSitter Parser Loading**:
   - Verify language libraries are compiled and accessible

### Debugging

Enable debug mode:

```bash
DEBUG=1 ./run.py
```

Use the IPython console for interactive debugging:

```python
# In the web interface console
orchestrator.researcher.debug_mode = True
await orchestrator.researcher.analyze_code_async(code_snippet="your code here")
```

## Best Practices

1. **Use Specific Queries**:
   - Instead of "Check for vulnerabilities"
   - Try "Check for reentrancy, front-running, and integer overflow"

2. **Multiple Model Approach**:
   - Use Claude for validation (more nuanced)
   - Use local models for research (faster)

3. **Iterative Analysis**:
   - Start with a general audit
   - Follow up with targeted queries on specific areas

4. **Include Tests and Documentation**:
   - Analyze tests alongside contracts
   - Include documentation for context

## Examples

### Basic Usage

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

### Python API Examples

```python
import asyncio
from finite_monkey.agents import WorkflowOrchestrator

async def audit_contracts():
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        model_name="claude-3-5-sonnet",
        researcher_model="ollama/llama3",
        validator_model="claude-3-5-sonnet"
    )
    
    # Run audit
    report = await orchestrator.run_audit(
        solidity_path="examples/Vault.sol",
        query="Check for reentrancy vulnerabilities",
    )
    
    # Save report
    await report.save("report.md")
    
    # Print findings
    for finding in report.findings:
        print(f"{finding['severity']}: {finding['title']}")

# Run with asyncio
asyncio.run(audit_contracts())
```

For more examples, see the [examples directory](../examples/) and the [tests](../tests/).