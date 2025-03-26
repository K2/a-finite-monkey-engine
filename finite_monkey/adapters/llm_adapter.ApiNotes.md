# LLM Adapter

## Overview

The `LLMAdapter` class provides a unified interface for interacting with different Large Language Model providers (OpenAI, Ollama, etc.). It abstracts away the provider-specific implementation details, allowing the rest of the system to use a consistent API regardless of the underlying LLM service.

## Core Functionality

### Initialization and Configuration

The adapter is initialized with:
- `model`: The LLM model to use (e.g., "gpt-4", "llama2")
- `provider`: The LLM provider (e.g., "openai", "ollama")
- `base_url`: An optional custom API endpoint

If these values aren't provided, they're pulled from the global configuration.

### Client Management

The `_ensure_client()` method handles lazy initialization of the appropriate client:
- For Ollama, it creates an `AsyncOllamaClient`
- For OpenAI, it creates an `AsyncOpenAIClient`
- For other providers, it falls back to a default client

### Key Methods

#### generate()

The primary method for generating text. It:
1. Ensures the client is initialized
2. Passes the prompt to the appropriate backend
3. Handles errors gracefully

#### llm()

This is an alias for `generate()` that exists specifically to maintain compatibility with components that expect an `llm()` method. The cognitive_bias_analyzer.py and other analyzers expect this method, and without it, they show warnings like:

#### structured_generate()

Generates responses in structured formats (like JSON). It:
1. Uses native structured generation if available in the client
2. Falls back to generating text with a JSON formatting request
3. Attempts to parse the response as JSON
4. Uses regex to extract JSON from code blocks if needed

## Integration Points

The adapter is used by:
- Analyzers (CognitiveBiasAnalyzer, etc.)
- FLARE query engine for reasoning steps
- Script generation components

## Error Handling

Errors are:
- Logged using the logger
- Wrapped in a response indicating the error
- Never allowed to propagate up to calling code

## Design Considerations

1. **Asynchronous API**: All methods are async to avoid blocking during API calls
2. **Lazy Initialization**: Clients are only created when needed
3. **Graceful Fallbacks**: Each method has fallbacks for unsupported features
4. **Provider Abstraction**: Callers never need to know which provider is being used

## Common Issues

If you encounter the warning about `llm()` not being callable, it means:
1. The component is trying to call `llm()` but it doesn't exist on the adapter
2. Check the LLMAdapter implementation to ensure it has the `llm()` method
3. Verify the method signature matches what the component expects
