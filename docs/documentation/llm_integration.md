# LLM Integration Guide

## Overview

Finite Monkey Engine uses LlamaIndex for LLM integration. The system now uses the Settings-based approach (replacing the deprecated ServiceContext) for configuring LLM models and embeddings.

## Configuration

LLM settings are configured in the `setup_llama_service()` function in `start.py`:

```python
def setup_llama_service():
    """Initialize LlamaIndex settings with appropriate LLM and embedding models"""
    try:
        # Configure Ollama LLM
        Settings.llm = Ollama(
            model="qwen2.5-coder:7b-instruct-q8_0",
            temperature=0.1,
            request_timeout=config.REQUEST_TIMEOUT,
        )
        
        # Configure embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
    except Exception as e:
        logger.error(f"Error configuring LlamaIndex settings: {str(e)}")
```

## LLM Adapter

The `LlamaIndexAdapter` class in `finite_monkey/llm/llama_index_adapter.py` provides a unified interface for LLM interactions:

```python
# Initialize adapter
adapter = LlamaIndexAdapter(
    provider="ollama",
    model_name="qwen2.5-coder:7b-instruct-q8_0",
    temperature=0.1
)

# Submit JSON prompt
response_future = await adapter.submit_json_prompt(
    prompt="Analyze this contract for vulnerabilities...",
    schema={"type": "object", "properties": {...}},
    system_prompt="You are a smart contract security expert..."
)
```

## Prompt Engineering

For optimal results, prompts should:

1. Use clear instructions
2. Specify JSON output format
3. Provide specific examples when needed
4. Explicitly mention proper formatting (e.g., commas between array items)

Example vulnerability scanning prompt:

