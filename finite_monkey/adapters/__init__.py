"""
Adapters for integrating various components into the Finite Monkey Engine.
"""

from .ollama import AsyncOllamaClient as Ollama
from .agent_adapter import DocumentationInconsistencyAdapter

# Importing Claude conditionally to avoid requiring the API key
try:
    from .claude import Claude
    from .llm_factory import LLMFactory
    __all__ = ["Ollama", "Claude", "LLMFactory", "DocumentationInconsistencyAdapter"]
except ImportError:
    try:
        from .llm_factory import LLMFactory
        __all__ = ["Ollama", "LLMFactory", "DocumentationInconsistencyAdapter"]
    except ImportError:
        __all__ = ["Ollama", "DocumentationInconsistencyAdapter"]