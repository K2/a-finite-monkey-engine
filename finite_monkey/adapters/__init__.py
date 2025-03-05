"""
Adapters for external services and libraries
"""

from .ollama import Ollama
# Importing Claude conditionally to avoid requiring the API key
try:
    from .claude import Claude
    __all__ = ["Ollama", "Claude"]
except ImportError:
    __all__ = ["Ollama"]