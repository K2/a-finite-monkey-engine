# TreeSitterAdapter API Notes

## Overview

The `TreeSitterAdapter` provides a higher-level interface to tree-sitter's parsing capabilities, focusing on analyzing smart contract code structure and identifying data flows, particularly source/sink relationships.

## Key Capabilities

1. **Code Structure Analysis**: Parse code to identify functions, blocks, and expressions
2. **Source/Sink Identification**: Identify potential input sources and output sinks in code
3. **Control Flow Analysis**: Build control flow graphs for functions
4. **Data Flow Tracking**: Track how data flows between sources and sinks

## Source/Sink Analysis

The adapter identifies:

- **Sources**: Points where data enters the system, such as:
  - User inputs (msg.sender, msg.value, calldata)
  - External state (block.timestamp, tx.origin)
  - Storage reads

- **Sinks**: Points where data is used or leaves the system, such as:
  - External calls (transfer, send, call)
  - Storage writes (state variables, assignments)
  - Self-destruction (selfdestruct)

## Integration with Guidance

This adapter works seamlessly with the `GuidanceFlowAnalyzer` to:

1. First extract syntactic structure using tree-sitter
2. Then enhance the analysis with semantic understanding using LLMs via Guidance
3. Produce comprehensive flow graphs with vulnerability information

## Usage Example

```python
# Create adapter
adapter = TreeSitterAdapter()

# Parse contract code
tree = adapter.parse_code(contract_code)

# Extract functions
functions = adapter.extract_functions(tree)

# Analyze sources and sinks with custom patterns
flows = adapter.identify_sources_and_sinks(
    tree,
    source_patterns=["msg.sender", "block.timestamp"],
    sink_patterns=["transfer", "call"]
)

# Analyze full contract
contract_flows = await adapter.analyze_contract_flows(
    contract_code,
    file_path="Contract.sol"
)
```

## Future Enhancements

1. More sophisticated pattern matching using tree-sitter queries
2. Cross-function and cross-contract flow analysis
3. Integration with symbolic execution for deeper analysis
