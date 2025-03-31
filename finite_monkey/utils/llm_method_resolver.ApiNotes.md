# LLM Method Resolver API Notes

## Unified LLM Interface

Different LLM frameworks and libraries use various method names and conventions:

1. **OpenAI-style**: Uses `chat`, `achat`, `complete`, `acomplete`
2. **HuggingFace-style**: Uses `generate`, `__call__`
3. **LlamaIndex-style**: Uses `as_structured_llm`, `complete`, `acomplete`, plus response objects
4. **Custom implementations**: May use any combination of methods

The `llm_method_resolver.py` provides a unified interface to handle all these variations:

```python
from finite_monkey.utils.llm_method_resolver import call_llm_async, extract_content_from_response

# Call any LLM with consistent interface
response = await call_llm_async(
    llm=any_llm_instance,
    input_text="Your prompt here",
    as_chat=True,  # Format as chat message
    system_prompt="Optional system prompt"
)

# Extract content from any type of response
content = extract_content_from_response(response)
```

## Method Resolution Strategy

The resolver attempts methods in this priority order:

1. **Structured methods** (LlamaIndex): `as_structured_llm()`
2. **Async chat methods**: `achat(messages=...)`
3. **Async completion methods**: `acomplete()`, `agenerate()`
4. **Direct callable**: If the object itself is callable
5. **Standard methods**: `chat()`, `complete()`, `generate()`

Each method is tried in a try-except block, so if one fails, we continue to the next.

## Response Extraction

The `extract_content_from_response` function handles:

1. String responses
2. Dictionary responses with common content keys
3. OpenAI-style response objects with message.content
4. LlamaIndex-style response objects 
5. Hugging Face style generation objects

This ensures consistent handling of response content regardless of LLM implementation.
