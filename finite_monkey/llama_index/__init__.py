"""
LlamaIndex integration for the Finite Monkey framework

This package provides integration with LlamaIndex for semantic search,
agent-based analysis, and other advanced capabilities.

This forms the inner agent layer in the Finite Monkey architecture,
providing structured analysis capabilities that are orchestrated by the
outer atomic agent layer.

IMPORTANT: The agent implementations that were previously in this module
have been moved to finite_monkey/agents/ for better separation of concerns
and architecture clarity. This module now primarily provides compatibility
imports for backward compatibility.
"""

import warnings

# Show a deprecation warning but only once
warnings.warn(
    "The LlamaIndex agent implementations are now deprecated and have been "
    "moved to finite_monkey.agents. Please update your imports.",
    DeprecationWarning,
    stacklevel=2
)

from .processor import AsyncIndexProcessor


def get_agent_classes():
        from ..agents.researcher import Researcher as ResearchAgent
        from ..agents.validator import Validator as ValidatorAgent
        from ..agents.documentor import Documentor as DocumentorAgent
        return ResearchAgent, ValidatorAgent, DocumentorAgent
ResearchAgent, ValidatorAgent, DocumentorAgent = get_agent_classes()

__all__ = [
    "AsyncIndexProcessor",
    "ResearchAgent",
    "ValidatorAgent",
    "DocumentorAgent",
]