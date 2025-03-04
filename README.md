# Finite Monkey: Smart Contract Audit & Analysis Framework

## Overview

Finite Monkey is an advanced framework for comprehensive smart contract analysis, audit, and verification. It employs a fully asynchronous architecture with atomic agents orchestrating workflows powered by LlamaIndex and Ollama LLM integration.
I'll bw working on re-merging our other enhancements ASSAP back into @BradMoonUESTC ASAP it's been busy and it took me a minute to decide on an appropiate relativly future proof framework.  

More or less we have 2 differing agentic frameworks, the inner set is derived from llama-index and it is used to form the innner set of well-defined, yet feature rich interactions, likely where the 
majority of execution flow will be naturally. The outer set of agent's are the managment layer, which encourages, evaluates, reports and composes more elaborate interactions with the world that is less
structured.  You could also say there is a 3'rd executive agent framework (human in the looop!) so that would be were moving the code-query and otehr tooling into you're hands for extensive
prototyping, research and developkent.   Looking forward to expanding functionality.

### Solidity Analysis Agent System (asyc finite monkey engine)

```mermaid
classDiagram
    %% ========== Core Components ==========
    class WorkflowOrchestrator {
        -llama_index: LlamaIndex
        -ollama: Ollama
        -researcher: Researcher
        -validator: Validator
        -documentor: Documentor
        +async run_audit(solidity_path, query) AuditReport
        +async run_atomic_agent_workflow(solidity_path, query) AuditReport
    }

    class LlamaIndex {
        -index: VectorStoreIndex
        +async aquery(query: str) ContextData
        +async load_and_index(file_paths) void
    }

    class Ollama {
        -model: str
        +async acomplete(prompt: str) str
        +async achat(messages: list) str
    }

    %% ========== Atomic Agents ==========
    class Researcher {
        -query_engine: LlamaIndex
        +async analyze_code_async(query: str, code_snippet: str) CodeAnalysis
        <<interface>> Retrieves code context via LlamaIndex + generates initial analysis
    }

    class Validator {
        -tree_sitter_analyzer: TreeSitterAnalyzer
        +async validate_analysis(code: str, analysis: CodeAnalysis) ValidationResult
        <<interface>> Cross-checks results via tree-sitter analysis and LLM critique
    }

    class Documentor {
        +async generate_report_async(analysis: dict, validation: dict) MarkdownReport
        <<interface>> Formats outputs into structured reports/docs
    }

    %% ========== Visualization ==========
    class GraphFactory {
        +analyze_solidity_file(file_path: str) CytoscapeGraph
        <<interface>> Generates contract visualizations
    }

    class CytoscapeGraph {
        +add_node(id, label, type, properties) void
        +add_edge(source, target, label, type) void
        +export_html(output_path) void
        <<interface>> Creates interactive HTML visualizations
    }

    %% ========== Data Flow ==========
    WorkflowOrchestrator --> Researcher : delegates analysis
    WorkflowOrchestrator --> Validator : delegates validation
    WorkflowOrchestrator --> Documentor : delegates reporting
    Researcher --> LlamaIndex : queries context
    Validator --> Ollama : validates via LLM
    WorkflowOrchestrator --> GraphFactory : generates visualization
```

The system enables:
- Deep code analysis of smart contracts
- Vulnerability detection and validation
- Comprehensive report generation
- Interactive visualization of contract relationships
- Multi-file analysis for complex projects
- Human-in-the-loop review facilitation# Finite Monkey: Smart Contract Audit & Analysis Framework

## Overview

Finite Monkey is an advanced framework for comprehensive smart contract analysis, audit, and verification. It employs a fully asynchronous architecture with atomic agents orchestrating workflows powered by LlamaIndex and Ollama LLM integration.

### Solidity Analysis Agent System (asyc finite monkey engine)

```mermaid
classDiagram
    %% ========== Core Components ==========
    class WorkflowOrchestrator {
        -llama_index: LlamaIndex
        -ollama: Ollama
        -researcher: Researcher
        -validator: Validator
        -documentor: Documentor
        +async run_audit(solidity_path, query) AuditReport
        +async run_atomic_agent_workflow(solidity_path, query) AuditReport
    }

    class LlamaIndex {
        -index: VectorStoreIndex
        +async aquery(query: str) ContextData
        +async load_and_index(file_paths) void
    }

    class Ollama {
        -model: str
        +async acomplete(prompt: str) str
        +async achat(messages: list) str
    }

    %% ========== Atomic Agents ==========
    class Researcher {
        -query_engine: LlamaIndex
        +async analyze_code_async(query: str, code_snippet: str) CodeAnalysis
        <<interface>> Retrieves code context via LlamaIndex + generates initial analysis
    }

    class Validator {
        -tree_sitter_analyzer: TreeSitterAnalyzer
        +async validate_analysis(code: str, analysis: CodeAnalysis) ValidationResult
        <<interface>> Cross-checks results via tree-sitter analysis and LLM critique
    }

    class Documentor {
        +async generate_report_async(analysis: dict, validation: dict) MarkdownReport
        <<interface>> Formats outputs into structured reports/docs
    }

    %% ========== Visualization ==========
    class GraphFactory {
        +analyze_solidity_file(file_path: str) CytoscapeGraph
        <<interface>> Generates contract visualizations
    }

    class CytoscapeGraph {
        +add_node(id, label, type, properties) void
        +add_edge(source, target, label, type) void
        +export_html(output_path) void
        <<interface>> Creates interactive HTML visualizations
    }

    %% ========== Data Flow ==========
    WorkflowOrchestrator --> Researcher : delegates analysis
    WorkflowOrchestrator --> Validator : delegates validation
    WorkflowOrchestrator --> Documentor : delegates reporting
    Researcher --> LlamaIndex : queries context
    Validator --> Ollama : validates via LLM
    WorkflowOrchestrator --> GraphFactory : generates visualization
```
# Finite Monkey: Smart Contract Audit & Analysis Framework

## Overview

Finite Monkey is an advanced framework for comprehensive smart contract analysis, audit, and verification. It employs a fully asynchronous architecture with atomic agents orchestrating workflows powered by LlamaIndex and Ollama LLM integration.

### Solidity Analysis Agent System (asyc finite monkey engine)

```mermaid
classDiagram
    %% ========== Core Components ==========
    class WorkflowOrchestrator {
        -llama_index: LlamaIndex
        -ollama: Ollama
        -researcher: Researcher
        -validator: Validator
        -documentor: Documentor
        +async run_audit(solidity_path, query) AuditReport
        +async run_atomic_agent_workflow(solidity_path, query) AuditReport
    }

    class LlamaIndex {
        -index: VectorStoreIndex
        +async aquery(query: str) ContextData
        +async load_and_index(file_paths) void
    }

    class Ollama {
        -model: str
        +async acomplete(prompt: str) str
        +async achat(messages: list) str
    }

    %% ========== Atomic Agents ==========
    class Researcher {
        -query_engine: LlamaIndex
        +async analyze_code_async(query: str, code_snippet: str) CodeAnalysis
        <<interface>> Retrieves code context via LlamaIndex + generates initial analysis
    }

    class Validator {
        -tree_sitter_analyzer: TreeSitterAnalyzer
        +async validate_analysis(code: str, analysis: CodeAnalysis) ValidationResult
        <<interface>> Cross-checks results via tree-sitter analysis and LLM critique
    }

    class Documentor {
        +async generate_report_async(analysis: dict, validation: dict) MarkdownReport
        <<interface>> Formats outputs into structured reports/docs
    }

    %% ========== Visualization ==========
    class GraphFactory {
        +analyze_solidity_file(file_path: str) CytoscapeGraph
        <<interface>> Generates contract visualizations
    }

    class CytoscapeGraph {
        +add_node(id, label, type, properties) void
        +add_edge(source, target, label, type) void
        +export_html(output_path) void
        <<interface>> Creates interactive HTML visualizations
    }

    %% ========== Data Flow ==========
    WorkflowOrchestrator --> Researcher : delegates analysis
    WorkflowOrchestrator --> Validator : delegates validation
    WorkflowOrchestrator --> Documentor : delegates reporting
    Researcher --> LlamaIndex : queries context
    Validator --> Ollama : validates via LLM
    WorkflowOrchestrator --> GraphFactory : generates visualization
```

The system enables:
- Deep code analysis of smart contracts
- Vulnerability detection and validation
- Comprehensive report generation
- Interactive visualization of contract relationships
- Multi-file analysis for complex projects
- Human-in-the-loop review facilitation

## Architecture

The framework follows these key architectural principles:

### Fully Asynchronous Execution
- All operations are non-blocking
- Parallel processing where possible
- Efficient handling of complex analysis tasks
   - Atomic agents (Researcher, Validator, Documentor)
   - WorkflowOrchestrator for coordinating the workflow
   - LlamaIndex integration for efficient code retrieval
   - Ollama adapter for LLM integration
   - SQLAlchemy database for persistence

### Atomic Agent Design
- Independent, specialized agents for specific tasks
- Coordinated through a central orchestrator
- Prompt-driven interactions
   - Integration with tree-sitter for static analysis
   - Interactive visualization of contracts and their relationships
   - Multi-file analysis support
   - Comprehensive reporting system
   - Unified command-line interface

### Vector Database Integration
- LlamaIndex for efficient code context retrieval
- Semantic search for related code sections
- Function call tree generation

### LLM Integration
- Ollama for local LLM inference
- Contextual prompting for specialized analysis
- Cross-validation of findings
   - Clean separation of concerns with atomic agents
   - Modular design for easy extension
   - Prompt-driven interactions with LLMs
   - Database persistence for tracking state

## Key Components


## Workflow Sequence

```mermaid
sequenceDiagram
    participant User as User
    participant Orchestrator as WorkflowOrchestrator
    participant Researcher
    participant Validator
    participant Documentor
    participant Visualization as GraphFactory

    User ->> Orchestrator: run_audit("Vault.sol", "Check reentrancy")
    activate Orchestrator

    Orchestrator ->> Researcher: analyze_code_async()
    activate Researcher
    Researcher ->> LlamaIndex: aquery()
    LlamaIndex -->> Researcher: ContextData
    Researcher -->> Orchestrator: CodeAnalysis
    deactivate Researcher

    Orchestrator ->> Validator: validate_analysis()
    activate Validator
    Validator ->> TreeSitter: analyze_code() (async)
    Validator ->> Ollama: acomplete() (async)
    TreeSitter -->> Validator: AnalysisReport
    Ollama -->> Validator: LLMValidation
    Validator -->> Orchestrator: ValidationResult
    deactivate Validator

    Orchestrator ->> Documentor: generate_report_async()
    activate Documentor
    Documentor -->> Orchestrator: MarkdownReport
    deactivate Documentor

    Orchestrator ->> Visualization: analyze_solidity_file()
    activate Visualization
    Visualization -->> Orchestrator: CytoscapeGraph
    deactivate Visualization

    Orchestrator -->> User: AuditReport and Visualization
    deactivate Orchestrator
```

## Recent Enhancements

In the latest updates, we've made significant improvements to the framework:

1. **Enhanced Visualization**:
   - Detailed interactive visualizations of contract structure
   - Graph shows functions, state variables, events, and their relationships
   - Hover tooltips show detailed information about each component
   - Filtering options to focus on specific aspects of the contract

2. **Multi-File Analysis**:
   - Support for analyzing multiple Solidity files in one run
   - Directory scanning with glob pattern matching
   - Contextual relationships between contracts in different files

3. **Unified Command-Line Interface**:
   - Comprehensive CLI with subcommands for different operations
   - `analyze` - Run security analysis on contracts
   - `visualize` - Generate visualizations of contracts
   - `full-audit` - Perform analysis and visualization in one step

4. **Improved Text Processing**:
   - Better parsing of analysis results
   - More accurate extraction of findings and recommendations
   - Structured output for better integration with other tools

## Usage

### Basic Usage

```bash
# Analyze a single contract
./run.py analyze -f examples/Vault.sol

# Analyze all contracts in a directory
./run.py analyze -d examples/

# Generate visualization for a contract
./run.py visualize examples/Vault.sol

# Analyze contracts and generate visualization in one step
./run.py full-audit -f examples/Vault.sol
```

### API Usage

```python
# Example usage in Python code
async def main():
    orchestrator = WorkflowOrchestrator()
    
    # Single file analysis
    report = await orchestrator.run_audit(
        solidity_path="contracts/Vault.sol",
        query="Check for reentrancy vulnerabilities"
    )
    
    # Multi-file analysis
    report = await orchestrator.run_atomic_agent_workflow(
        solidity_path=["contracts/Vault.sol", "contracts/Token.sol"],
        query="Perform a comprehensive security audit",
        project_name="DeFi-Project"
    )
    
    print(report.summary)
    
    # Save detailed report
    await report.save("audit_report.md")
    
    # Generate visualization
    from finite_monkey.visualization import GraphFactory
    graph = GraphFactory.analyze_solidity_file("contracts/Vault.sol")
    graph.export_html("contract_visualization.html")
```

## Installation

```bash
#!/bin/bash
# Installation script for Async Finite Monkey framework

set -e  # Exit on error

# Print banner
echo "=============================================="
echo "Async Finite Monkey Installation"
echo "Smart Contract Audit & Analysis Framework"
echo "=============================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Detected Python version: $python_version"

required_version="3.10.0"
if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Error: Python 3.10.0 or higher is required"
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .

# Setup success
echo ""
echo "=============================================="
echo "Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the framework, run:"
echo "  ./run.py [command] [options]"
echo "  ./run.py --help for more information"
echo "=============================================="
```

The system enables:
- Deep code analysis of smart contracts
- Vulnerability detection and validation
- Comprehensive report generation
- Interactive visualization of contract relationships
- Multi-file analysis for complex projects
- Human-in-the-loop review facilitation

## Architecture

The framework follows these key architectural principles:

### Fully Asynchronous Execution
- All operations are non-blocking
- Parallel processing where possible
- Efficient handling of complex analysis tasks
   - Atomic agents (Researcher, Validator, Documentor)
   - WorkflowOrchestrator for coordinating the workflow
   - LlamaIndex integration for efficient code retrieval
   - Ollama adapter for LLM integration
   - SQLAlchemy database for persistence

### Atomic Agent Design
- Independent, specialized agents for specific tasks
- Coordinated through a central orchestrator
- Prompt-driven interactions
   - Integration with tree-sitter for static analysis
   - Interactive visualization of contracts and their relationships
   - Multi-file analysis support
   - Comprehensive reporting system
   - Unified command-line interface

### Vector Database Integration
- LlamaIndex for efficient code context retrieval
- Semantic search for related code sections
- Function call tree generation

### LLM Integration
- Ollama for local LLM inference
- Contextual prompting for specialized analysis
- Cross-validation of findings
   - Clean separation of concerns with atomic agents
   - Modular design for easy extension
   - Prompt-driven interactions with LLMs
   - Database persistence for tracking state

## Key Components


## Workflow Sequence

```mermaid
sequenceDiagram
    participant User as User
    participant Orchestrator as WorkflowOrchestrator
    participant Researcher
    participant Validator
    participant Documentor
    participant Visualization as GraphFactory

    User ->> Orchestrator: run_audit("Vault.sol", "Check reentrancy")
    activate Orchestrator

    Orchestrator ->> Researcher: analyze_code_async()
    activate Researcher
    Researcher ->> LlamaIndex: aquery()
    LlamaIndex -->> Researcher: ContextData
    Researcher -->> Orchestrator: CodeAnalysis
    deactivate Researcher

    Orchestrator ->> Validator: validate_analysis()
    activate Validator
    Validator ->> TreeSitter: analyze_code() (async)
    Validator ->> Ollama: acomplete() (async)
    TreeSitter -->> Validator: AnalysisReport
    Ollama -->> Validator: LLMValidation
    Validator -->> Orchestrator: ValidationResult
    deactivate Validator

    Orchestrator ->> Documentor: generate_report_async()
    activate Documentor
    Documentor -->> Orchestrator: MarkdownReport
    deactivate Documentor

    Orchestrator ->> Visualization: analyze_solidity_file()
    activate Visualization
    Visualization -->> Orchestrator: CytoscapeGraph
    deactivate Visualization

    Orchestrator -->> User: AuditReport and Visualization
    deactivate Orchestrator
```

## Recent Enhancements

In the latest updates, we've made significant improvements to the framework:

1. **Enhanced Visualization**:
   - Detailed interactive visualizations of contract structure
   - Graph shows functions, state variables, events, and their relationships
   - Hover tooltips show detailed information about each component
   - Filtering options to focus on specific aspects of the contract

2. **Multi-File Analysis**:
   - Support for analyzing multiple Solidity files in one run
   - Directory scanning with glob pattern matching
   - Contextual relationships between contracts in different files

3. **Unified Command-Line Interface**:
   - Comprehensive CLI with subcommands for different operations
   - `analyze` - Run security analysis on contracts
   - `visualize` - Generate visualizations of contracts
   - `full-audit` - Perform analysis and visualization in one step

4. **Improved Text Processing**:
   - Better parsing of analysis results
   - More accurate extraction of findings and recommendations
   - Structured output for better integration with other tools

## Usage

### Basic Usage

```bash
# Analyze a single contract
./run.py analyze -f examples/Vault.sol

# Analyze all contracts in a directory
./run.py analyze -d examples/

# Generate visualization for a contract
./run.py visualize examples/Vault.sol

# Analyze contracts and generate visualization in one step
./run.py full-audit -f examples/Vault.sol
```

### API Usage

```python
# Example usage in Python code
async def main():
    orchestrator = WorkflowOrchestrator()
    
    # Single file analysis
    report = await orchestrator.run_audit(
        solidity_path="contracts/Vault.sol",
        query="Check for reentrancy vulnerabilities"
    )
    
    # Multi-file analysis
    report = await orchestrator.run_atomic_agent_workflow(
        solidity_path=["contracts/Vault.sol", "contracts/Token.sol"],
        query="Perform a comprehensive security audit",
        project_name="DeFi-Project"
    )
    
    print(report.summary)
    
    # Save detailed report
    await report.save("audit_report.md")
    
    # Generate visualization
    from finite_monkey.visualization import GraphFactory
    graph = GraphFactory.analyze_solidity_file("contracts/Vault.sol")
    graph.export_html("contract_visualization.html")
```

## Installation

```bash
#!/bin/bash
# Installation script for Async Finite Monkey framework

set -e  # Exit on error

# Print banner
echo "=============================================="
echo "Async Finite Monkey Installation"
echo "Smart Contract Audit & Analysis Framework"
echo "=============================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Detected Python version: $python_version"

required_version="3.10.0"
if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Error: Python 3.10.0 or higher is required"
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .

# Setup success
echo ""
echo "=============================================="
echo "Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the framework, run:"
echo "  ./run.py [command] [options]"
echo "  ./run.py --help for more information"
echo "=============================================="
```


## Architecture

The framework follows these key architectural principles:

### Fully Asynchronous Execution
- All operations are non-blocking
- Parallel processing where possible
- Efficient handling of complex analysis tasks
   - Atomic agents (Researcher, Validator, Documentor)
   - WorkflowOrchestrator for coordinating the workflow
   - LlamaIndex integration for efficient code retrieval
   - Ollama adapter for LLM integration
   - SQLAlchemy database for persistence

### Atomic Agent Design
- Independent, specialized agents for specific tasks
- Coordinated through a central orchestrator
- Prompt-driven interactions
   - Integration with tree-sitter for static analysis
   - Interactive visualization of contracts and their relationships
   - Multi-file analysis support
   - Comprehensive reporting system
   - Unified command-line interface

### Vector Database Integration
- LlamaIndex for efficient code context retrieval
- Semantic search for related code sections
- Function call tree generation

### LLM Integration
- Ollama for local LLM inference
- Contextual prompting for specialized analysis
- Cross-validation of findings
   - Clean separation of concerns with atomic agents
   - Modular design for easy extension
   - Prompt-driven interactions with LLMs
   - Database persistence for tracking state

## Key Components


## Workflow Sequence

```mermaid
sequenceDiagram
    participant User as User
    participant Orchestrator as WorkflowOrchestrator
    participant Researcher
    participant Validator
    participant Documentor
    participant Visualization as GraphFactory

    User ->> Orchestrator: run_audit("Vault.sol", "Check reentrancy")
    activate Orchestrator

    Orchestrator ->> Researcher: analyze_code_async()
    activate Researcher
    Researcher ->> LlamaIndex: aquery()
    LlamaIndex -->> Researcher: ContextData
    Researcher -->> Orchestrator: CodeAnalysis
    deactivate Researcher

    Orchestrator ->> Validator: validate_analysis()
    activate Validator
    Validator ->> TreeSitter: analyze_code() (async)
    Validator ->> Ollama: acomplete() (async)
    TreeSitter -->> Validator: AnalysisReport
    Ollama -->> Validator: LLMValidation
    Validator -->> Orchestrator: ValidationResult
    deactivate Validator

    Orchestrator ->> Documentor: generate_report_async()
    activate Documentor
    Documentor -->> Orchestrator: MarkdownReport
    deactivate Documentor

    Orchestrator ->> Visualization: analyze_solidity_file()
    activate Visualization
    Visualization -->> Orchestrator: CytoscapeGraph
    deactivate Visualization

    Orchestrator -->> User: AuditReport and Visualization
    deactivate Orchestrator
```

## Recent Enhancements

In the latest updates, we've made significant improvements to the framework:

1. **Enhanced Visualization**:
   - Detailed interactive visualizations of contract structure
   - Graph shows functions, state variables, events, and their relationships
   - Hover tooltips show detailed information about each component
   - Filtering options to focus on specific aspects of the contract

2. **Multi-File Analysis**:
   - Support for analyzing multiple Solidity files in one run
   - Directory scanning with glob pattern matching
   - Contextual relationships between contracts in different files

3. **Unified Command-Line Interface**:
   - Comprehensive CLI with subcommands for different operations
   - `analyze` - Run security analysis on contracts
   - `visualize` - Generate visualizations of contracts
   - `full-audit` - Perform analysis and visualization in one step

4. **Improved Text Processing**:
   - Better parsing of analysis results
   - More accurate extraction of findings and recommendations
   - Structured output for better integration with other tools

## Usage
uv should work out of the box, however if you are hyaving troubole, **see below**;


### Basic Usage

```bash
# Analyze a single contract
./run.py analyze -f examples/Vault.sol

# Analyze all contracts in a directory
./run.py analyze -d examples/

# Generate visualization for a contract
./run.py visualize examples/Vault.sol

# Analyze contracts and generate visualization in one step
./run.py full-audit -f examples/Vault.sol
```

### API Usage

```python
# Example usage in Python code
async def main():
    orchestrator = WorkflowOrchestrator()
    
    # Single file analysis
    report = await orchestrator.run_audit(
        solidity_path="contracts/Vault.sol",
        query="Check for reentrancy vulnerabilities"
    )
    
    # Multi-file analysis
    report = await orchestrator.run_atomic_agent_workflow(
        solidity_path=["contracts/Vault.sol", "contracts/Token.sol"],
        query="Perform a comprehensive security audit",
        project_name="DeFi-Project"
    )
    
    print(report.summary)
    
    # Save detailed report
    await report.save("audit_report.md")
    
    # Generate visualization
    from finite_monkey.visualization import GraphFactory
    graph = GraphFactory.analyze_solidity_file("contracts/Vault.sol")
    graph.export_html("contract_visualization.html")
```

## Installation

```bash
#!/bin/bash
# Installation script for Async Finite Monkey framework

set -e  # Exit on error

# Print banner
echo "=============================================="
echo "Async Finite Monkey Installation"
echo "Smart Contract Audit & Analysis Framework"
echo "=============================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Detected Python version: $python_version"

required_version="3.10.0"
if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Error: Python 3.10.0 or higher is required"
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .

# Setup success
echo ""
echo "=============================================="
echo "Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the framework, run:"
echo "  ./run.py [command] [options]"
echo "  ./run.py --help for more information"
echo "=============================================="
```
