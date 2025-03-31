# AsyncOllamaClient API Notes

## HTTP Client Lifecycle Management

The AsyncOllamaClient uses httpx for asynchronous HTTP requests. This requires careful management of the client lifecycle to avoid issues:

1. **Initialization**: The HTTP client is created during object initialization
2. **Cleanup**: The client must be properly closed to release resources

The client implements three methods for lifecycle management:

- `__init__`: Creates the HTTP client with appropriate timeout settings
- `aclose()`: Async method for properly closing the client
- `__del__`: Standard destructor that attempts to schedule client closure

### Best Practices

When using AsyncOllamaClient:

```python
# Create the client
client = AsyncOllamaClient(model="llama2")

try:
    # Use the client
    response = await client.acomplete("Hello, world!")
finally:
    # Ensure the client is properly closed
    await client.aclose()
```

## Error Handling

The `acomplete` method implements several layers of error handling:

1. HTTP errors (non-200 status codes)
2. JSON parsing errors
3. Network/connection errors

All errors are converted to a standardized response format with an `error` key:

```python
{"error": "Error message details"}
```

This ensures that error responses can be consistently handled by calling code.

## Stream Control

Ollama supports streaming responses, but this can cause issues with JSON parsing. The client explicitly disables streaming with:

```python
"stream": False
```

This ensures that complete responses are returned as a single JSON object.
