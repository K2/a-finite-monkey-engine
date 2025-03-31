"""
Structured output generation with Guidance.

This module provides the implementation for generating structured outputs
using Guidance, with robust error handling and fallbacks.
"""
from typing import Any, Dict, List, Type, Optional, Union, Callable
import asyncio
from pydantic import BaseModel
from loguru import logger

from .manager import GuidanceManager, GUIDANCE_AVAILABLE, LLAMAINDEX_GUIDANCE, LEGACY_API
from ..nodes_config import config


class StructuredProgram:
    """
    Wrapper for Guidance programs that ensures consistent behavior.
    """
    
    def __init__(
        self, 
        program: Any, 
        output_cls: Type[BaseModel], 
        fallback_fn: Optional[Callable] = None,
        verbose: bool = False
    ):
        """
        Initialize with a Guidance program and output class.
        
        Args:
            program: The underlying Guidance program
            output_cls: Pydantic model class for the structured output
            fallback_fn: Optional fallback function if the program fails
            verbose: Whether to enable verbose logging
        """
        self.program = program
        self.output_cls = output_cls
        self.fallback_fn = fallback_fn
        self.verbose = verbose
    
    async def __call__(self, **kwargs) -> Union[BaseModel, Dict[str, Any]]:
        """
        Call the program with the given parameters.
        
        Args:
            **kwargs: Parameters to pass to the program
            
        Returns:
            Structured output as a Pydantic model or dictionary
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
                
            # Handle result based on type
            if isinstance(result, self.output_cls):
                # Already a Pydantic model, just return it
                return result
            elif isinstance(result, dict):
                # Convert dict to Pydantic model
                try:
                    return self.output_cls(**result)
                except Exception as e:
                    logger.error(f"Failed to convert dict to {self.output_cls.__name__}: {e}")
                    return result
            elif isinstance(result, str):
                # Try to parse as JSON
                try:
                    data = json.loads(result)
                    return self.output_cls(**data)
                except Exception as e:
                    logger.error(f"Failed to parse string as JSON: {e}")
                    # Return as is if we can't convert
                    return result
            else:
                # Return whatever we got
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
    Create a structured output program using the latest available Guidance integration.
    
    Args:
        output_cls: Pydantic model class for the output schema
        prompt_template: Handlebars template for the prompt
        model: Model name to use
        provider: Model provider
        fallback_fn: Optional fallback function
        verbose: Whether to enable verbose logging
        
    Returns:
        A callable StructuredProgram or None if creation fails
    """
    if not GUIDANCE_AVAILABLE:
        logger.warning("Guidance not available, cannot create structured program")
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
        
        # Create program based on what's available
        if LLAMAINDEX_GUIDANCE:
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
