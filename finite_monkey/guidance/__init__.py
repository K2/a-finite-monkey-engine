"""
Guidance integration for structured outputs with LlamaIndex.

This package provides a clean interface to Microsoft's Guidance library
for generating structured outputs using LLMs.
"""

# Check for guidance availability
try:
    import guidance
    GUIDANCE_AVAILABLE = True
except ImportError:
    GUIDANCE_AVAILABLE = False

# Export version-agnostic interfaces
from .program import create_program, GuidanceProgramWrapper
from .utils import ensure_handlebars_format

__all__ = [
    "create_program",
    "GuidanceProgramWrapper",
    "ensure_handlebars_format",
    "GUIDANCE_AVAILABLE"
]
