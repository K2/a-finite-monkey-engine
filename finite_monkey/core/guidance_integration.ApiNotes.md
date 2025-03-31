# Guidance Integration API Notes

## LlamaIndex API Evolution

The LlamaIndex API for Guidance integration has evolved over time, making it challenging to maintain compatible code. This module addresses those challenges with a unified interface.

### API Changes Timeline:

1. **Legacy API (v0.9.x)**:
   - Import path: `from llama_index.prompts.guidance import GuidancePydanticProgram`
   - Used `guidance.llms.OpenAI` directly
   - Required separate setup for OpenAI

2. **Intermediate API (v0.10.x)**:
   - Import path: `from llama_index.core.program.guidance import GuidancePydanticProgram`
   - Started using LlamaIndex's own LLM interfaces

3. **Current API (v0.12+)**:
   - Import path: `from llama_index.program.guidance import GuidancePydanticProgram`
   - Uses LlamaIndex LLM factory with a unified interface

## Implementation Strategy

The `GuidanceManager` class provides a version-agnostic way to work with Guidance:

1. **Import Detection**: Automatically detects which API version is available
2. **Unified Interface**: Presents a consistent interface regardless of underlying API
3. **Error Handling**: Robust fallback mechanisms if components are missing
4. **Async Support**: Properly handles both sync and async underlying implementations

## Usage Example

```python
# Create manager
guidance_mgr = GuidanceManager(
    model="gpt-3.5-turbo",
    provider="openai"
)

# Create a structured program
program = await guidance_mgr.create_structured_program(
    output_cls=MyOutputClass,
    prompt_template="""
    Generate an analysis of {{data}}
    
    {{#schema}}
    {
      "score": 5,
      "findings": ["finding1", "finding2"]
    }
    {{/schema}}
    """
)

# Use the program
if program:
    result = await program(data="Sample data to analyze")
```

## Fallback Strategy

The implementation uses multiple fallback levels:

1. Try the latest API first
2. Fall back to intermediate API if latest is unavailable
3. Try legacy API as last resort
4. Provide clear warning if no integration is available

This ensures maximal compatibility across different LlamaIndex versions.
