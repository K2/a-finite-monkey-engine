# Guidance Question Generator API Notes

## Overview

The `GuidanceQuestionGenerator` provides a robust way to decompose complex queries into structured sub-questions for the FLARE query engine. It uses Microsoft's Guidance library to enforce structured output formatting and eliminate parsing errors.

## API Compatibility

This implementation is designed to work with different versions of LlamaIndex's Guidance API:

1. **Latest API (v0.12+)**: Uses `llama_index.program.guidance`
2. **Core API (v0.10-0.11)**: Uses `llama_index.core.program.guidance`
3. **Legacy API (v0.9 and earlier)**: Uses `llama_index.prompts.guidance`
4. **Raw Guidance**: Falls back to direct Guidance library use if no LlamaIndex integration is available

The version detection happens automatically, making the implementation resilient to API changes.

## Usage in FLARE Engine

FLARE uses this generator to reliably break down complex queries:

```python
# Create FLARE engine with Guidance
flare_engine = FlareQueryEngine(
    underlying_engine=vector_engine,
    use_guidance=True  # Enable Guidance integration
)

# Execute query
result = await flare_engine.query("What are the security risks in this smart contract?")
```

## Error Handling Strategy

The implementation has multiple fallback layers:
1. Try Guidance-based generation first
2. Fall back to provided fallback function if Guidance fails
3. Use standard LLM-based generation as last resort
4. Create generic sub-questions if all else fails

This ensures that query decomposition remains functional even when components fail.

## SubQuestion Schema

The SubQuestion model provides a structured format:

```python
class SubQuestion(BaseModel):
    text: str        # The actual sub-question text
    tool_name: str   # Tool to use for answering
    reasoning: str   # Optional reasoning for this question
```

## Performance Considerations

Guidance adds slight overhead but significantly improves reliability:
- Eliminates JSON parsing errors completely
- Guarantees proper schema compliance
- More consistent decomposition quality
- Works better with weaker LLMs
