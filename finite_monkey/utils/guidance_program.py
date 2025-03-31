"""
Guidance program utilities for structured output generation.
This module provides wrappers around guidance library for Pydantic integration.
"""
import inspect
import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union, List
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type variable for generic structured output
T = TypeVar('T', bound=BaseModel)

# Import guidance only if available
try:
    import guidance
    GUIDANCE_AVAILABLE = True
except ImportError:
    GUIDANCE_AVAILABLE = False
    logger.warning("Guidance library not available. Install with 'pip install guidance'")

class GuidancePydanticProgram:
    """
    Pydantic-enabled guidance program for structured output.
    Re-implemented here to avoid circular imports with guidance_integration.core.
    """
    
    def __init__(self, llm, system_prompt=None):
        """Initialize with LLM and optional system prompt"""
        self.llm = llm
        self.messages = []
        if system_prompt:
            self.system(system_prompt)
    
    def system(self, content):
        """Add a system message"""
        self.messages.append({"role": "system", "content": content})
        return self
    
    def user(self, content):
        """Add a user message"""
        self.messages.append({"role": "user", "content": content})
        return self
    
    def assistant(self, content):
        """Add an assistant message"""
        self.messages.append({"role": "assistant", "content": content})
        return self
    
    async def agenerate(self, variables=None):
        """Generate response asynchronously"""
        variables = variables or {}
        # Use the LLM to generate a response
        try:
            if hasattr(self.llm, "achat") and inspect.iscoroutinefunction(self.llm.achat):
                response = await self.llm.achat(messages=self.messages)
                return {"response": response.message.content, "messages": self.messages}
            else:
                # Fallback to non-async method if needed
                logger.warning("Using non-async chat method - consider using async version")
                if hasattr(self.llm, "chat") and callable(self.llm.chat):
                    response = self.llm.chat(messages=self.messages)
                    return {"response": response.message.content, "messages": self.messages}
                else:
                    raise ValueError("LLM does not support chat methods")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e), "messages": self.messages}
