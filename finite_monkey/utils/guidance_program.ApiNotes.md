# Guidance Program API Notes

## Overview

The `GuidancePydanticProgram` provides a high-level interface for generating structured outputs using Guidance. It manages the complexity of working with Guidance and provides fallback mechanisms for reliability.

## Architecture Design

The program follows a layered approach:
1. **High-level Interface**: Simple callable that handles all the complexity
2. **Guidance Integration**: Primary method using Guidance for structured output
3. **Fallback Mechanism**: Alternative methods if Guidance fails or is unavailable
4. **Result Handling**: Consistent return types and error handling

## Key Features

1. **Graceful Degradation**: Falls back to alternative methods if Guidance fails
2. **Consistent Return Types**: Always returns either a Pydantic model or a compatible dict
3. **Verbose Logging**: Optional detailed logging for debugging
4. **Custom Fallbacks**: Support for custom fallback functions

## Circular Import Resolution

To avoid circular imports between `guidance_integration.core` and `finite_monkey.utils.guidance_program`, 
both modules must implement their own version of `GuidancePydanticProgram`. 

### Implementation Strategy

1. Each module defines its own version of `GuidancePydanticProgram` with identical interfaces
2. No cross-imports occur between these modules
3. Future updates must maintain this separation or restructure the code architecture

The preferred long-term solution would be to:
1. Extract common components to a separate base module
2. Have both modules extend/import from this base module
3. Remove the duplication

## Configuration Management

All configuration should come from `finite_monkey.nodes_config.config` and not from arbitrary 
configuration dictionaries. This ensures:

1. Centralized configuration management
2. Consistent access to settings across the codebase
3. Predictable behavior for all components

Example of correct configuration usage:
```python
from finite_monkey.nodes_config import config

# Get configuration values
default_model = getattr(config, "DEFAULT_MODEL", "fallback-value")
temperature = getattr(config, "TEMPERATURE", 0.1)
```

## Usage Example

```python
# Create a program
program = GuidancePydanticProgram(
    output_cls=BusinessFlowAnalysisResult,
    prompt_template_str="""
    Analyze this contract:
    {{contract_code}}
    {{#schema}}
    {
      "flows": [
        {
          "name": "Flow name",
          "description": "Flow description",
          "steps": ["Step 1", "Step 2"],
          "functions": ["function1", "function2"]
        }
      ],
      "contract_summary": "Summary"
    }
    {{/schema}}
    """,
    fallback_fn=my_fallback_function,
    verbose=True
)

# Use the program
result = await program(contract_code="contract Example { ... }")
```

## Error Handling Strategy

The program implements a three-tier error handling strategy:
1. Try Guidance first (if available)
2. Fall back to alternative method (if provided)
3. Return empty result as last resort

This ensures that the pipeline continues to function even when components fail.
