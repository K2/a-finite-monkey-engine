# Guidance Integration API Notes

## Overview

The `finite_monkey.guidance` package provides a robust, version-aware interface to Microsoft's Guidance library for structured output generation. This implementation handles the complexity of different LlamaIndex API versions and provides consistent behavior regardless of the underlying implementation.

## Key Components

1. **Core Module** (`core.py`): Contains the foundation classes and functions for working with Guidance, including:
   - `GuidanceManager`: Manages the LLM setup for different API versions
   - `StructuredProgram`: Provides a consistent interface for structured output programs
   - `create_structured_program`: Factory function to create structured output programs

2. **Models** (`models.py`): Contains the Pydantic models for structured outputs:
   - `SubQuestion`: Represents a decomposed sub-question
   - `QuestionDecompositionResult`: Contains a set of sub-questions with reasoning
   - `BusinessFlow`: Represents a business flow in a smart contract
   - `BusinessFlowAnalysisResult`: Contains a set of business flows with context

3. **Question Generator** (`question_gen.py`): Specialized component for query decomposition:
   - `GuidanceQuestionGenerator`: Decomposes complex queries into sub-questions

## API Evolution Handling

LlamaIndex's Guidance integration has evolved through several incompatible API versions:

1. **v0.8-0.9**: Used `llama_index.prompts.guidance` module with a direct `guidance_llm` parameter
2. **v0.10-0.11**: Used `llama_index.core.program.guidance` with `from_defaults` factory
3. **v0.12+**: Uses `llama_index.program.guidance` and requires LLM conversion with `as_guidance_llm()`

Our implementation automatically detects which API version is available and adapts accordingly, providing a consistent interface regardless of the underlying implementation.

## Usage Examples

### Basic Structured Output

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
```

### Query Decomposition

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
```

## Error Handling Strategy

The implementation uses a multi-tier fallback approach:

1. Try to use the latest LlamaIndex Guidance integration
2. Fall back to older API versions if the latest isn't available
3. Try direct Guidance integration if LlamaIndex's isn't available
4. Fall back to custom implementations if all else fails
5. Return empty/default objects as a last resort

This ensures that the pipeline can continue functioning even if specific components fail.

## Performance Considerations

- Structured output generation requires more processing than standard LLM calls
- The LLM must follow the schema constraints, which requires additional tokens
- Performance is generally better with models that have strong JSON generation capabilities (GPT-4, Claude)
