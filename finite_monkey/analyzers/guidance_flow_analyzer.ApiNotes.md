# Guidance Flow Analyzer API Notes

## Overview

The `GuidanceFlowAnalyzer` combines tree-sitter's syntactic parsing with Guidance-based LLM semantic analysis to provide comprehensive flow analysis for smart contracts. It focuses on identifying source/sink relationships and potential vulnerabilities.

## Hybrid Analysis Approach

This analyzer uses a two-phase approach:

1. **Syntactic Phase (Tree-Sitter)**:
   - Parse code structure
   - Identify basic sources and sinks based on patterns
   - Build initial flow graph

2. **Semantic Phase (Guidance LLM)**:
   - Enhance the initial flow graph
   - Identify additional sources and sinks
   - Connect sources to sinks with data flows
   - Identify potential vulnerabilities

This hybrid approach combines the speed and precision of static analysis with the semantic understanding of LLMs.

## Key Features

1. **Custom Source/Sink Specification**: Analyze specific relationships
2. **Vulnerability Detection**: Identify security issues in the code
3. **Flow Visualization**: Generate graphs of data flow through contract
4. **Semantic Understanding**: Leverage LLMs to understand code intent

## Integration Points

- **Business Flow Extractor**: Complements business flow analysis
- **Vulnerability Scanner**: Feeds into vulnerability analysis
- **FLARE Query Engine**: Can be used to answer queries about data flows

## Usage Example

```python
# Create analyzer
analyzer = GuidanceFlowAnalyzer()

# Analyze specific function with custom sources/sinks
flow_graph = await analyzer.analyze_function(
    contract_code,
    "transfer",
    custom_sources=["msg.sender", "balances"],
    custom_sinks=["transfer", "balances"]
)

# Extract vulnerabilities
vulnerabilities = flow_graph.metadata.get('vulnerabilities', [])

# Generate visualization of flow graph
visualization = visualize_flow_graph(flow_graph)

# Add results to analysis report
report.add_section("Data Flow Analysis", visualization)
```

## Flow Graph Structure

The flow graphs created by the analyzer follow this structure:

1. **Nodes**: Represent sources, sinks, and intermediate values
2. **Edges**: Represent data flows between nodes
3. **Metadata**: Contains additional information like vulnerabilities

## Customization Options

### Custom Source Patterns

Commonly used source patterns:

- `msg.sender`, `msg.value`, `msg.data` - Transaction context
- `tx.origin`, `block.timestamp` - Blockchain context
- State variable reads - Contract storage

### Custom Sink Patterns

Commonly used sink patterns:

- `transfer`, `send`, `call` - External calls
- `delegatecall`, `staticcall` - Potentially dangerous calls
- State variable writes - Contract storage changes

## Performance Considerations

The analyzer operates in two phases:

1. **Fast Path**: Tree-sitter analysis only (when Guidance is unavailable)
2. **Full Analysis**: Tree-sitter + Guidance (when available)

You can control this with the `guidance_adapter` parameter.
