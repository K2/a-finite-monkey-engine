# Finite Monkey Engine Examples

This directory contains example code, deprecated components, reference implementations, and development utilities that are not part of the core codebase.

## Directory Structure

- **tests/**: Test files and test utilities moved out of the main codebase
- **debug/**: Debugging utilities and development tools
- **dev/**: Development configuration and setup files
- **notebooks/**: Jupyter notebooks for experimentation
- **starlight-prototype/**: Early iterations of Starlight UI components before they were finalized

## Purpose

This directory serves to keep the main codebase clean while preserving useful reference implementations. Files here:

1. Are not required for core functionality
2. May serve as examples or templates for users
3. May be used for development or debugging but aren't part of production
4. May contain deprecated code that we want to preserve for reference

## Usage

The examples and utilities in this directory are provided for reference only and may require additional setup to run.

## Contents

### Starlight Prototype
Early iterations and experimental components from the Starlight UI development:

- **components/**: Early component implementations before security and internationalization improvements
- **pages/**: Prototype pages that were used during development
- **utils/**: Utility modules replaced by more secure or specialized versions
- **api/**: API endpoints that have been superseded by more specialized implementations

## Usage

These examples should not be used in production. They are kept for:
1. Reference purposes
2. Documentation of the development process
3. Educational value for understanding the evolution of the codebase

## Relationship to Main Codebase

Files in this directory have been moved from the main codebase as they were replaced by improved implementations. The current working versions are in their respective locations in the main project structure.

## DeFi Project

A comprehensive set of example DeFi contracts with intentional security vulnerabilities for testing:

- **Token.sol**: An ERC20-like token with various security issues
- **Vault.sol**: A token vault for storing tokens with access control issues
- **StakingPool.sol**: A staking pool with reward calculation flaws

### Vulnerabilities

The DeFi project contains various vulnerability types:

1. **Reentrancy**: State changes after external calls
2. **Access Control**: Missing authorization checks
3. **Integer Arithmetic**: Precision loss and overflow/underflow issues
4. **Input Validation**: Missing checks for critical parameters
5. **Logic Errors**: Flawed business logic in stake/unstake mechanics
6. **Timestamp Dependence**: Reliance on block timestamps for security decisions

### Usage

To analyze these contracts, run:

```bash
# Standard analysis
./run.py analyze -d examples/defi_project/contracts

# Async workflow analysis
./run_async_workflow.py -d examples/defi_project/contracts

# Analysis with specific query
./run.py analyze -d examples/defi_project/contracts -q "Check for reentrancy vulnerabilities"
```

## Vault.sol

A simpler standalone example of a vulnerable vault contract:

```bash
./run.py analyze -f examples/Vault.sol
```

This contract contains issues with withdrawal logic and access control.

## Using Examples in Web Interface

You can also use these examples with the web interface:

1. Start the web interface: `./run_web_interface.py`
2. Navigate to `http://localhost:8000/` in your browser
3. Use the IPython Terminal to run analysis:

```python
# Analyze defi project
files = ["examples/defi_project/contracts/Token.sol", 
         "examples/defi_project/contracts/Vault.sol",
         "examples/defi_project/contracts/StakingPool.sol"]

# Start analysis
task_ids = await orchestrator.run_audit_workflow(
    solidity_paths=files,
    query="Perform a comprehensive security audit",
    wait_for_completion=False
)

# Check task status
for file, tasks in task_ids.items():
    status = await task_manager.get_task_status(tasks["analysis"])
    print(f"{file}: {status['status']}")
```