# Async Workflow Architecture for Finite Monkey Engine

The async workflow architecture provides a fully asynchronous implementation of the Finite Monkey Engine's smart contract analysis pipeline with comprehensive flow analysis. This document outlines the components, flow, and technical design of the system.

## Core Components

The async architecture consists of several key components:

1. **ContractParser** - Parses Solidity contracts using Tree-Sitter for AST-based analysis
2. **ControlFlowAnalyzer** - Extracts detailed control flow information with line numbers and context
3. **ExpressionGenerator** - Creates and manages test expressions for detecting vulnerabilities
4. **AsyncAnalyzer** - Coordinates the full analysis pipeline with asynchronous processing
5. **DatabaseManager** - Provides async database access for persistent storage
6. **SQLAlchemy TaskEngine** - Database-driven vulnerability analysis with targeted expressions
7. **run_async_analyzer.py** - Command-line interface for the async analyzer

## Analysis Flow

The enhanced async workflow follows these steps:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │     │             │
│  Contract   │────▶│  Flow       │────▶│  Expression │────▶│  Primary    │────▶│  Secondary  │
│  Parsing    │     │  Analysis   │     │  Generation │     │  Analysis   │     │  Validation │
│             │     │             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                                       │
                                                                                       ▼
                                                                              ┌─────────────┐
                                                                              │             │
                                                                              │  Report     │
                                                                              │  Generation │
                                                                              │             │
                                                                              └─────────────┘
```

1. **Contract Parsing** - Parse and structure the Solidity contracts using Tree-Sitter
2. **Flow Analysis** - Extract detailed control flow, state changes, and path conditions
3. **Expression Generation** - Generate test expressions based on contract structure and flow
4. **Primary Analysis** - Perform initial analysis using LLM with flow-enhanced context
5. **Secondary Validation** - Validate findings with secondary LLM and targeted flow examination
6. **Report Generation** - Generate comprehensive security reports with flow context

## Concurrency Model

The async architecture takes advantage of Python's asyncio framework to support:

1. **File-level Parallelization** - Multiple files are analyzed concurrently
2. **Controlled Concurrency** - Configurable concurrency limits to prevent overloading
3. **Semaphore Control** - Ensures limited concurrent access to resources

Concurrency limits are defined in nodes_config and can be tuned based on available hardware:

```python
# Example concurrency control
semaphore = asyncio.Semaphore(config.MAX_THREADS_OF_SCAN)
async with semaphore:
    # This code runs with controlled concurrency
```

## Configuration System

The architecture integrates with the nodes_config system for flexible configuration:

- **Model Selection** - Configure which models to use for analysis and validation
- **Database Settings** - Database URL and connection parameters
- **Concurrency Settings** - Control parallel execution limits
- **File Filtering** - Configure which files to analyze or ignore

## Database Integration

The system uses PostgreSQL with async database access through SQLAlchemy:

- **PostgreSQL** - Primary database for robustness and performance
- **AsyncSession** - Fully async database sessions with asyncpg driver
- **Task Tracking** - Persistent tracking of analysis tasks and progress
- **Result Storage** - Storage of analysis results and findings
- **Schema Compatibility** - Uses the same database schema as the synchronous version

## Tree-Sitter Integration

For advanced code parsing:

- **AST Analysis** - Abstract Syntax Tree based code analysis
- **Pattern Matching** - Identifying common vulnerability patterns
- **Function Relationships** - Tracking call graphs and data flows

## Usage Examples

### Basic Analysis

```bash
# Analyze a single contract
./run_analysis.sh -f examples/SimpleVault.sol

# Analyze a project directory
./run_analysis.sh -d examples/defi_project
```

### Advanced Options

```bash
# Use specific models and output directory
./run_analysis.sh -f examples/SimpleVault.sol -m llama3:70b -v claude-3-sonnet-20240229 -o custom_reports

# Run a targeted analysis query
./run_analysis.sh -f examples/SimpleVault.sol -q "Check for reentrancy vulnerabilities"
```

### Python API

```python
# Basic usage
from finite_monkey.core_async_analyzer import AsyncAnalyzer

async def analyze_contract():
    analyzer = AsyncAnalyzer()
    results = await analyzer.analyze_contract_file("examples/SimpleVault.sol")
    print(f"Risk assessment: {results['final_report']['risk_assessment']}")
```

## Package Management

The system uses uv for efficient package management:

```bash
# Install dependencies with uv
uv pip install tree-sitter fastapi uvicorn sqlalchemy[asyncio] asyncpg psycopg2-binary
```

## Future Extensions

1. **Distributed Analysis** - Support for distributed analysis across multiple nodes
2. **Incremental Analysis** - Only analyze changed files in successive runs
3. **Custom Rules Engine** - Allow for user-defined analysis rules
4. **Integration with CI/CD** - Automated analysis in continuous integration pipelines