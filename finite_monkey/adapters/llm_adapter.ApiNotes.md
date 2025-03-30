# LLM Adapter API Notes

## Serializable Message Format

To ensure compatibility across the pipeline, we use a serializable dictionary format for messages:

```python
messages = [
    {"role": "system", "content": "You are an expert analyst..."},
    {"role": "user", "content": "Analyze this contract..."}
]
```

This format can be:
1. Directly serialized to JSON for API calls
2. Converted to ChatMessage objects when needed
3. Joined into a single prompt for completion-only LLMs

## Message Conversion Patterns

The adapters provide methods to handle different LLM interfaces:

1. `achat_dict()` - Takes dictionary messages and handles conversion internally
2. `acomplete()` - Takes a single prompt string for completion-only LLMs 
3. `achat()` - Takes ChatMessage objects (less recommended due to serialization issues)

## Serialization Issues

Some components like ChatMessage are not directly JSON serializable. When working with APIs that require JSON (like Ollama), always use the dictionary format instead of ChatMessage objects directly.

## Example Usage with Different Adapters

```python
# For any adapter type
messages = [
    {"role": "system", "content": "You are an expert..."},
    {"role": "user", "content": "Analyze this..."}
]

# Use the adapter's methods that accept dictionary messages
response = await llm_adapter.achat_dict(messages)
```

By following this pattern throughout the pipeline, we ensure consistent serialization and compatibility with different LLM backends.
