"""
Core Guidance integration functionality.

This module provides the foundation for working with Microsoft's Guidance
library, handling API version differences and providing a consistent interface.
"""
from typing import Any, Dict, List, Type, Optional, Union, Callable
import asyncio
import json
from loguru import logger
from pydantic import BaseModel

# Check if guidance is available
try:
    import guidance
    GUIDANCE_AVAILABLE = True
    logger.info("Microsoft Guidance library is available")
except ImportError:
    GUIDANCE_AVAILABLE = False
    logger.warning("Microsoft Guidance library not installed. Install with: pip install guidance")

# Check which LlamaIndex Guidance API is available
if GUIDANCE_AVAILABLE:
    LLAMAINDEX_API_AVAILABLE = False
    LEGACY_API = False
    
    # Try the latest API first (v0.12+)
    try:
        from llama_index.program.guidance import GuidancePydanticProgram
        LLAMAINDEX_API_AVAILABLE = True
        LEGACY_API = False
        logger.info("Using latest LlamaIndex Guidance API (program.guidance)")
    except ImportError:
        # Try the core API (v0.10+)
        try:
            from llama_index.core.program.guidance import GuidancePydanticProgram
            LLAMAINDEX_API_AVAILABLE = True
            LEGACY_API = False
            logger.info("Using core LlamaIndex Guidance API (core.program.guidance)")
        except ImportError:
            # Try the legacy API (v0.9)
            try:
                from llama_index.prompts.guidance import GuidancePydanticProgram
                LLAMAINDEX_API_AVAILABLE = True
                LEGACY_API = True
                logger.info("Using legacy LlamaIndex Guidance API (prompts.guidance)")
            except ImportError:
                logger.warning("No LlamaIndex Guidance API found")

from ..nodes_config import config

class GuidanceManager:
    """
    Manager for Guidance integration that handles different API versions.
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
            provider: Model provider
            temperature: Temperature for generation
            verbose: Whether to enable verbose logging
        """
        if not GUIDANCE_AVAILABLE:
            raise ImportError("Guidance library is required but not installed")
            
        self.model = model or getattr(config, "ANALYSIS_MODEL", config.DEFAULT_MODEL)
        self.provider = provider or getattr(config, "ANALYSIS_MODEL_PROVIDER", config.DEFAULT_PROVIDER)
        self.temperature = temperature
        self.verbose = verbose
        self.guidance_llm = None
        
        # Initialize the LLM
        self._setup_llm()
    
    def _setup_llm(self):
        """Set up the appropriate LLM for Guidance based on API version"""
        try:
            if LEGACY_API:
                # Legacy API used separate OpenAI class
                from guidance.llms import OpenAI as GuidanceOpenAI
                self.guidance_llm = GuidanceOpenAI(
                    self.model,
                    temperature=self.temperature
                )
                logger.info(f"Initialized Guidance LLM (legacy) with model: {self.model}")
            else:
                # Try to use LlamaIndex's LLM
                try:
                    if self.provider.lower() == "openai":
                        from llama_index.llms import OpenAI
                        self.guidance_llm = OpenAI(
                            model=self.model,
                            temperature=self.temperature
                        )
                    elif self.provider.lower() == "anthropic":
                        from llama_index.llms import Anthropic
                        self.guidance_llm = Anthropic(
                            model=self.model,
                            temperature=self.temperature
                        )
                    else:
                        # Fall back to OpenAI
                        from llama_index.llms import OpenAI
                        self.guidance_llm = OpenAI(
                            model=self.model,
                            temperature=self.temperature
                        )
                        logger.warning(f"Provider {self.provider} not directly supported, using OpenAI")
                        
                    logger.info(f"Initialized Guidance LLM with model: {self.model}")
                except Exception as e:
                    logger.error(f"Error initializing LlamaIndex LLM: {e}")
                    # Fall back to raw guidance
                    from guidance.llms import OpenAI as GuidanceOpenAI
                    self.guidance_llm = GuidanceOpenAI(
                        self.model,
                        temperature=self.temperature
                    )
                    logger.info(f"Falling back to raw Guidance LLM with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize any Guidance LLM: {e}")
            self.guidance_llm = None
    
    def _ensure_handlebars_format(self, prompt: str) -> str:
        """
        Ensure the prompt is in handlebars format for Guidance.
        
        Args:
            prompt: Input prompt, either Python format string or handlebars
            
        Returns:
            Prompt in handlebars format
        """
        # Simple heuristic: if we see {{variable}} style, assume it's already handlebars
        if "{{" in prompt and "}}" in prompt:
            return prompt
            
        # Otherwise try to convert from Python style to handlebars
        try:
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
            logger.warning(f"Failed to convert prompt to handlebars using LlamaIndex: {e}")
        
        # Simple naive conversion as fallback
        import re
        return re.sub(r'\{([^{}]*)\}', r'{{\1}}', prompt)


class StructuredProgram:
    """
    Wrapper for Guidance programs with a consistent interface.
    """
    
    def __init__(
        self, 
        program: Any, 
        output_cls: Type[BaseModel], 
        fallback_fn: Optional[Callable] = None,
        verbose: bool = False
    ):
        """
        Initialize with a Guidance program.
        
        Args:
            program: The underlying Guidance program
            output_cls: The Pydantic model class for the output
            fallback_fn: Optional fallback function if the program fails
            verbose: Whether to enable verbose logging
        """
        self.program = program
        self.output_cls = output_cls
        self.fallback_fn = fallback_fn
        self.verbose = verbose
    
    async def __call__(self, **kwargs) -> Union[BaseModel, Dict[str, Any]]:
        """
        Execute the program with the given parameters.
        
        Args:
            **kwargs: Parameters to pass to the program
            
        Returns:
            A structured output object or dictionary
        """
        if self.verbose:
            logger.debug(f"StructuredProgram executing with: {kwargs}")
            
        try:
            # Execute the program
            if asyncio.iscoroutinefunction(self.program.__call__):
                # If it's already an async method
                result = await self.program(**kwargs)
            else:
                # If it's synchronous, run in executor
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.program(**kwargs)
                )
                
            if self.verbose:
                logger.debug(f"StructuredProgram result: {result}")
                
            # For debugging:
            # logger.debug(f"Result type: {type(result)}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error executing Guidance program: {e}")
            
            # Try fallback if available
            if self.fallback_fn:
                try:
                    logger.info("Using fallback function for structured output")
                    return await self.fallback_fn(**kwargs)
                except Exception as fallback_e:
                    logger.error(f"Fallback function also failed: {fallback_e}")
            
            # Return empty instance as last resort
            logger.warning(f"Returning empty {self.output_cls.__name__} instance due to failures")
            return self.output_cls()


async def create_structured_program(
    output_cls: Type[BaseModel],
    prompt_template: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    fallback_fn: Optional[Callable] = None,
    verbose: bool = False
) -> Optional[StructuredProgram]:
    """
    Create a structured output program using Guidance.
    
    Args:
        output_cls: Pydantic model class defining the output schema
        prompt_template: Template for the prompt
        model: Model name to use
        provider: Model provider
        fallback_fn: Optional fallback function if Guidance fails
        verbose: Whether to enable verbose logging
        
    Returns:
        A callable StructuredProgram or None if creation fails
    """
    if not GUIDANCE_AVAILABLE:
        logger.warning("Guidance library not available, cannot create structured program")
        return None
        
    try:
        # Create Guidance manager
        manager = GuidanceManager(
            model=model,
            provider=provider,
            verbose=verbose
        )
        
        if not manager.guidance_llm:
            logger.error("Failed to initialize Guidance LLM")
            return None
            
        # Ensure the prompt is in handlebars format
        handlebars_prompt = manager._ensure_handlebars_format(prompt_template)
        
        if LLAMAINDEX_API_AVAILABLE:
            if LEGACY_API:
                # Legacy LlamaIndex Guidance
                from llama_index.prompts.guidance import GuidancePydanticProgram
                program = GuidancePydanticProgram(
                    output_cls=output_cls,
                    prompt_template_str=handlebars_prompt,
                    guidance_llm=manager.guidance_llm,
                    verbose=verbose
                )
            else:
                # Modern LlamaIndex Guidance
                if hasattr(manager.guidance_llm, 'as_guidance_llm'):
                    # Latest API requiring conversion
                    guidance_llm = manager.guidance_llm.as_guidance_llm()
                else:
                    # Direct use
                    guidance_llm = manager.guidance_llm
                    
                try:
                    # Try newer API first
                    from llama_index.program.guidance import GuidancePydanticProgram
                except ImportError:
                    # Fall back to core API
                    from llama_index.core.program.guidance import GuidancePydanticProgram
                    
                program = GuidancePydanticProgram.from_defaults(
                    output_cls=output_cls,
                    prompt_template_str=handlebars_prompt,
                    llm=guidance_llm,
                    verbose=verbose
                )
        else:
            # Raw guidance without LlamaIndex
            program = guidance.Program(
                guidance_llm=manager.guidance_llm,
                prompt=handlebars_prompt
            )
            
        # Wrap in our structured program interface
        return StructuredProgram(
            program=program,
            output_cls=output_cls,
            fallback_fn=fallback_fn,
            verbose=verbose
        )
            
    except Exception as e:
        logger.error(f"Error creating structured program: {e}")
        return None
