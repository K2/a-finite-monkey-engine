"""
Core Guidance integration manager that handles different LlamaIndex API versions.
"""
from typing import Any, Dict, Optional, Union
import asyncio
import json
from loguru import logger

try:
    import guidance
    GUIDANCE_AVAILABLE = True
    
    # Check for different LlamaIndex Guidance import paths
    try:
        # First try the latest import path (v0.12+)
        from llama_index.program.guidance import GuidancePydanticProgram
        LEGACY_API = False
        LLAMAINDEX_GUIDANCE = True
        logger.info("Using latest LlamaIndex Guidance integration (program.guidance)")
    except ImportError:
        try:
            # Try the core import path (v0.10+)
            from llama_index.core.program.guidance import GuidancePydanticProgram
            LEGACY_API = False
            LLAMAINDEX_GUIDANCE = True
            logger.info("Using core LlamaIndex Guidance integration (core.program.guidance)")
        except ImportError:
            try:
                # Try the legacy import path (v0.9)
                from llama_index.prompts.guidance import GuidancePydanticProgram
                LEGACY_API = True
                LLAMAINDEX_GUIDANCE = True
                logger.info("Using legacy LlamaIndex Guidance integration (prompts.guidance)")
            except ImportError:
                logger.warning("No LlamaIndex Guidance integration found, using raw guidance")
                LEGACY_API = False
                LLAMAINDEX_GUIDANCE = False
except ImportError:
    GUIDANCE_AVAILABLE = False
    LEGACY_API = False
    LLAMAINDEX_GUIDANCE = False
    logger.warning("Guidance library not available. Install with 'pip install guidance'")

from ..nodes_config import config


class GuidanceManager:
    """
    Manager for Guidance integration with support for different LlamaIndex API versions.
    
    This class provides a consistent interface for working with Guidance regardless
    of which version of LlamaIndex is installed.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize the Guidance manager.
        
        Args:
            model: Model name to use
            provider: Model provider (openai, anthropic, etc.)
            temperature: Temperature parameter for generation
            verbose: Whether to enable verbose logging
        """
        if not GUIDANCE_AVAILABLE:
            raise ImportError("Guidance library is required but not installed")
            
        self.model = model or getattr(config, "ANALYSIS_MODEL", config.DEFAULT_MODEL)
        self.provider = provider or getattr(config, "ANALYSIS_MODEL_PROVIDER", config.DEFAULT_PROVIDER)
        self.temperature = temperature
        self.verbose = verbose
        self.guidance_llm = None
        
        # Set up the LLM for Guidance
        self._setup_llm()
    
    def _setup_llm(self):
        """Set up the appropriate LLM for Guidance based on API version"""
        if self.provider.lower() == "openai":
            try:
                if LEGACY_API:
                    # Legacy API used separate OpenAI class
                    from guidance.llms import OpenAI as GuidanceOpenAI
                    self.guidance_llm = GuidanceOpenAI(
                        self.model,
                        temperature=self.temperature
                    )
                else:
                    # New API uses LlamaIndex's LLM interface
                    from llama_index.llms import OpenAI
                    self.guidance_llm = OpenAI(
                        model=self.model,
                        temperature=self.temperature
                    )
                logger.info(f"Initialized OpenAI Guidance LLM with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI Guidance LLM: {e}")
        elif self.provider.lower() == "anthropic":
            try:
                from llama_index.llms import Anthropic
                self.guidance_llm = Anthropic(
                    model=self.model,
                    temperature=self.temperature
                )
                logger.info(f"Initialized Anthropic Guidance LLM with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic Guidance LLM: {e}")
        else:
            try:
                # Try to use a generic LLM interface
                from llama_index.llms import LLM
                self.guidance_llm = LLM(
                    model=self.model,
                    temperature=self.temperature
                )
                logger.info(f"Initialized generic Guidance LLM with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize generic Guidance LLM: {e}")
                # Fall back to OpenAI as a last resort
                try:
                    from llama_index.llms import OpenAI
                    self.guidance_llm = OpenAI(temperature=self.temperature)
                    logger.warning(f"Falling back to default OpenAI model")
                except Exception as fb_e:
                    logger.error(f"Could not initialize any Guidance LLM: {fb_e}")
    
    def _ensure_handlebars_format(self, prompt: str) -> str:
        """
        Ensure the prompt is in handlebars format for Guidance.
        
        Args:
            prompt: Template string, either Python format style or handlebars
            
        Returns:
            Prompt in handlebars format
        """
        if "{{" in prompt and "}}" in prompt:
            return prompt
            
        try:
            # Try to use the official converter
            try:
                from llama_index.core.prompts.guidance_utils import convert_to_handlebars
                return convert_to_handlebars(prompt)
            except ImportError:
                try:
                    from llama_index.prompts.guidance_utils import convert_to_handlebars
                    return convert_to_handlebars(prompt)
                except ImportError:
                    pass
        except Exception as e:
            logger.warning(f"Failed to convert prompt to handlebars: {e}")
        
        # Simple fallback conversion
        import re
        return re.sub(r'\{([^{}]*)\}', r'{{\1}}', prompt)
