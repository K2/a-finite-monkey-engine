# Finite Monkey Engine - Usage Guide

## Command-line Interface

Finite Monkey Engine provides a command-line interface for all core functionalities:

### Analyze Smart Contracts

Run the full analysis pipeline on a directory or GitHub repository:

```bash
# Analyze local directory
python start.py analyze --input ./contracts --output ./results/analysis.md

# Analyze GitHub repository
python start.py analyze --input https://github.com/username/repo --github --branch main --subdirectory src --output ./results/analysis.md
```

Options:
- `--input`: Directory path or GitHub URL
- `--output`: Path for results file (default: output/analysis.html)
- `--github`: Flag to indicate GitHub repository URL
- `--branch`: Branch to use (default: main)
- `--subdirectory`: Subdirectory within repository (default: /src)

### Test Chunking

Test the chunking functionality on a single Solidity file:

```bash
python start.py test-chunking path/to/contract.sol --verbose
```

Options:
- `file`: Path to Solidity file
- `--verbose` or `-v`: Enable verbose output

### Chunk Directory

Process all Solidity files in a directory:

```bash
python start.py chunk-directory ./contracts --output ./results --max-concurrent 10 --verbose
```

Options:
- `directory`: Directory containing Solidity files
- `--output` or `-o`: Output directory for results (default: ./output)
- `--max-concurrent` or `-m`: Maximum concurrent files (default: 5)
- `--verbose` or `-v`: Enable verbose output

### Process Project

Load and chunk an entire project:

```bash
python start.py process-project ./project --output ./results --chunk-size 4000
```

Options:
- `directory`: Directory to process
- `--output` or `-o`: Output directory for results (default: ./output)
- `--max-concurrent` or `-m`: Maximum concurrent files (default: 5)
- `--chunk-size` or `-c`: Maximum chunk size (default: 8000)
- `--verbose` or `-v`: Enable verbose output

## Analysis Components

The Finite Monkey Engine includes several powerful analysis components:

### Data Flow Analyzer
- Identifies exploitable paths from sources (user inputs) to sinks (vulnerable operations)
- Analyzes taint propagation through function calls
- Integrates with business flow information

### Cognitive Bias Analyzer
- Detects developer cognitive biases such as optimism and anchoring biases
- Identifies assumptions that could lead to security vulnerabilities
- Cross-correlates with business flows and data paths

### Documentation Analyzer
- Evaluates documentation quality and completeness
- Checks NatSpec compliance
- Provides recommendations for improvement

### Documentation Inconsistency Analyzer
- Detects mismatches between code comments and actual behavior
- Identifies security-critical documentation inconsistencies
- Focuses on misleading security guarantees

### Counterfactual Analyzer
- Generates "what if" scenarios to identify edge cases
- Analyzes parameter extremes, state divergence, and external failures
- Provides detailed exploit paths and conditions

## Output Files

The system generates various output files:

- `analysis.html`: Main analysis report with findings
- `chunking_summary.json`: Summary of chunking results
- `chunking_details.json`: Detailed information about chunks
- `project_summary.json`: Project processing summary
