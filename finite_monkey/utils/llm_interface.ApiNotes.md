# LLM Interface Utility API Notes

## Problem: Inconsistent LLM Interfaces

Different LLM libraries and wrappers use inconsistent methods for generating text:

- **OpenAI-style**: Uses `chat()`, `achat()`, `complete()`, `acomplete()`
- **HuggingFace**: Uses `generate()`, `__call__()`
- **LlamaIndex**: Uses `as_structured_llm()`, `complete()`

This leads to errors like:
