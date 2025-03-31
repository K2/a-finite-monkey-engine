# Guidance Integration API Notes

## Overview

The Guidance integration package provides a robust, version-aware interface to Microsoft's Guidance library for structured output generation with LlamaIndex. It ensures compatibility with different versions of the LlamaIndex API, which has evolved significantly over time.

## Key Components

1. **GuidanceManager**: Core manager that handles LLM initialization and configuration
2. **StructuredProgram**: Wrapper for Guidance programs with consistent interface
3. **create_structured_program**: Helper function to create structured output programs
4. **GuidanceQuestionGenerator**: Specialized generator for FLARE query decomposition

## Usage Patterns

### Basic Structured Output Generation

```python
from finite_monkey.guidance import create_structured_program
from pydantic import BaseModel, Field

# Define output schema
class Analysis(BaseModel):
    score: int = Field(..., description="Score from 1-10")
    findings: list[str] = Field(default_factory=list, description="Key findings")

# Create prompt template
prompt = """
Analyze this code: {{code}}

{{#schema}}
{
  "score": 7,
  "findings": ["Finding 1", "Finding 2"]
}
{{/schema}}
"""

# Create program
program = await create_structured_program(
    output_cls=Analysis,
    prompt_template=prompt,
    model="gpt-3.5-turbo"
)

# Use the program
result = await program(code="function example() { ... }")
print(f"Score: {result.score}")
print(f"Findings: {result.findings}")
```

### Using the Question Generator with FLARE

```python
from finite_monkey.guidance import GuidanceQuestionGenerator

# Create generator
generator = GuidanceQuestionGenerator(
    model="gpt-4",
    verbose=True
)

# Define tools
tools = [
    {"name": "code_analyzer", "description": "Analyzes code structure"},
    {"name": "security_scanner", "description": "Finds security vulnerabilities"}
]

# Generate sub-questions
questions = await generator.generate(
    "What vulnerabilities exist in this smart contract?",
    tools
)

# Use the questions
for q in questions:
    print(f"Q: {q.text} (Tool: {q.tool_name})")
```

## API Evolution in LlamaIndex

The LlamaIndex Guidance API has gone through several iterations:

1. **v0.8-0.9 (Legacy)**:
   - Import path: `from llama_index.prompts.guidance import GuidancePydanticProgram`
   - Constructor: `GuidancePydanticProgram(output_cls, prompt_template_str, guidance_llm)`
   - Required separate Guidance LLM creation

2. **v0.10-0.11 (Core)**:
   - Import path: `from llama_index.core.program.guidance import GuidancePydanticProgram`
   - Factory method: `GuidancePydanticProgram.from_defaults(output_cls, prompt_template_str, llm)`
   - Used LlamaIndex LLM interface

3. **v0.12+ (Current)**:
   - Import path: `from llama_index.program.guidance import GuidancePydanticProgram`
   - Factory method: `GuidancePydanticProgram.from_defaults(output_cls, prompt_template_str, llm)`
   - LLM might need conversion via `as_guidance_llm()`

## Implementation Strategy

Our implementation uses a version-detection approach:

1. **Check Available Components**: Detect which LlamaIndex version is available
2. **Try Latest API First**: Always attempt to use the most recent API pattern
3. **Fall Back Gracefully**: If latest API is not available, try older versions
4. **Raw Guidance Fallback**: Use direct Guidance library as last resort
5. **Consistent Interface**: Wrap all implementations in a unified async interface

## Using the Guidance Integration

The main entry point is the `create_program` function, which handles all the complexity:

```python
from finite_monkey.guidance import create_program
from pydantic import BaseModel, Field

# Define output schema
class Analysis(BaseModel):
    summary: str = Field(..., description="Summary")
    points: List[str] = Field(default_factory=list)

# Create program
program = await create_program(
    output_cls=Analysis,
    prompt_template="Analyze {{text}} {{#schema}}...",
    model="gpt-3.5-turbo",
    provider="openai"
)

# Use program
result = await program(text="Sample text")
print(f"Summary: {result.summary}")
```

## Handlebars Templates

Guidance uses handlebars-style templates (`{{variable}}`) rather than Python's f-string style (`{variable}`). Our implementation handles both:

1. Automatically detects template format
2. Converts Python style to handlebars if needed
3. Uses LlamaIndex's converter if available
4. Falls back to regex replacement if needed

## Error Handling

The implementation includes multiple layers of error handling:

1. **Version Compatibility**: Gracefully handles different LlamaIndex versions
2. **LLM Creation**: Adapts to different LLM interfaces
3. **Program Execution**: Handles both sync and async execution patterns
4. **Result Processing**: Converts different result formats to the desired schema
5. **Fallback Functions**: Supports custom fallback functions if Guidance fails

This ensures that the system continues to work even when some components are unavailable or fail.
