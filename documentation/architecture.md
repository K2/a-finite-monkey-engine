# Architecture Overview

## Core Components

- **Pipeline**: The main processing pipeline that orchestrates the analysis process
- **Chunker**: Handles contract chunking for better LLM processing
- **Analyzers**: Various analysis components for comprehensive contract evaluation:
  - **Function Extractor**: Identifies functions and their relationships
  - **Business Flow Analyzer**: Maps business logic flows
  - **Data Flow Analyzer**: Identifies exploitable source-to-sink paths
  - **Vulnerability Scanner**: Detects common security issues
  - **Cognitive Bias Analyzer**: Identifies cognitive biases in code
  - **Counterfactual Analyzer**: Generates "what if" security scenarios
  - **Documentation Analyzer**: Evaluates documentation quality
  - **Documentation Inconsistency Analyzer**: Finds security-critical inconsistencies between code and comments
- **LLM Adapter**: Integrates with LlamaIndex for LLM interactions

## LLM Integration

The system uses LlamaIndex for LLM integration with the following configuration:

- **Settings-based configuration**: Global LlamaIndex settings for LLM and embedding models
- **Cross-analyzer context sharing**: Information discovered by one analyzer is fed to others
- **Default models**: 
  - qwen2.5-coder:7b-instruct-q8_0 for analysis tasks
  - BAAI/bge-small-en-v1.5 for embeddings

## Processing Flow

1. Document loading from local files or GitHub
2. Contract chunking for better LLM analysis
3. Function extraction from contracts
4. Business flow analysis 
5. Data flow analysis (source-to-sink paths)
6. Cognitive bias identification
7. Documentation quality assessment
8. Documentation inconsistency detection
9. Counterfactual scenario analysis
10. Cross-cutting correlation of findings
11. Report generation

## Command-line Interface

The system provides a command-line interface built with standard Python `argparse`:

```bash
# Full pipeline analysis
python start.py analyze --input ./contracts --output ./results/analysis.md

# Test specific components
python start.py test-chunking contracts/Token.sol
python start.py chunk-directory ./contracts --output ./output
```

## Analyzer Components Interaction

```mermaid
graph TD
    A[Document Loading] --> B[Contract Chunking]
    B --> C[Function Extraction]
    C --> D[Business Flow Analysis]
    D --> E[Data Flow Analysis]
    E --> G[Vulnerability Scanner]
    G --> H[Cognitive Bias Analysis]
    H --> I[Documentation Analysis]
    I --> J[Documentation Inconsistency Analysis]
    J --> K[Counterfactual Analysis]
    K --> L[Report Generation]
    
    %% Cross-cutting interactions
    D -.-> E[Data Flow Analysis]
    E -.-> K[Counterfactual Analysis]
    H -.-> K[Counterfactual Analysis]
    G -.-> I[Documentation Analysis]
    G -.-> J[Doc Inconsistency Analysis]
    D -.-> K[Counterfactual Analysis]
