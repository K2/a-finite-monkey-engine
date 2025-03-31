# Derived vs. Acquired: A Unified Taxonomy

## Overview

A-Finite-Monkey-Engine uses a consistent taxonomy throughout the system to classify information sources and processing approaches:

**Derived**: Information extracted algorithmically from source code through analysis
**Acquired**: Information gathered from human descriptions, issues, and discussions

This distinction is applied across multiple components of the system to provide clarity about how information was obtained and processed.

## Application Areas

### Vulnerability Analysis
- **Derived vulnerabilities**: Found through code path traversal and static analysis
- **Acquired vulnerabilities**: Found in issue text, comments, or discussions
- **Hybrid vulnerabilities**: Identified through both approaches

### Embeddings
- **Derived embeddings**: Generated locally on your infrastructure (IPEX, HuggingFace)
- **Acquired embeddings**: Obtained from external services (Ollama, OpenAI)

### Business Flows
- **Derived flows**: Code paths detected through static analysis
- **Acquired flows**: Process flows described in natural language text

## Benefits of the Taxonomy

This taxonomy provides several advantages:

1. **Source transparency**: Always know where information originated
2. **Confidence assessment**: Different sources have different reliability characteristics
3. **Comprehensive analysis**: Combining derived and acquired approaches yields more thorough results
4. **Privacy control**: Choose derived approaches for sensitive code that shouldn't be shared

## Implementation Details

Each component in the system tags data with its source:

```python
# Example of tagging a vulnerability
vulnerability = Vulnerability(
    name="Reentrancy Vulnerability",
    description="Function vulnerable to reentrancy attacks",
    source=VulnerabilitySource.DERIVED  # Explicitly mark the source
)

# Example of working with business flows
if business_flow.metadata.get("flow_source") == "acquired":
    # Handle acquired flow specially
    pass
```

## Extending the Taxonomy

When developing new components, consider:

1. Always track the source of information (derived/acquired/hybrid)
2. Consider implementing both derived and acquired approaches for comprehensive analysis
3. Document the strengths and limitations of each approach
4. Provide unified results that leverage both sources when possible
