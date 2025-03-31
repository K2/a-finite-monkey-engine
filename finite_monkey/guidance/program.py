"""
Guidance program creation and management with LlamaIndex compatibility.

This module handles the creation of Guidance programs with proper version detection
and compatibility with different LlamaIndex API versions.
"""
import asyncio
import json
from typing import Any, Dict, List, Type, Optional, Union, Callable
from pydantic import BaseModel
from loguru import logger

# Import check utilities
from .utils import is_guidance_available, get_llamaindex_version
from ..nodes_config import config


class GuidanceProgramWrapper:
    """
    A wrapper around Guidance programs that provides a unified interface.
    
    This class handles the different implementations of Guidance programs
    across different LlamaIndex versions and provides a consistent async interface.
    """
    
    def __init__(
        self, 
        program: Any, 
        output_cls: Type[BaseModel],
        verbose: bool = False,
        fallback_fn: Optional[Callable] = None
    ):
        """
        Initialize with a Guidance program.
        
        Args:
            program: The underlying Guidance program
            output_cls: The Pydantic class for structuring the output
            verbose: Whether to enable verbose logging
            fallback_fn: Optional fallback function if program execution fails
        """
        self.program = program
        self.output_cls = output_cls
        self.verbose = verbose
        self.fallback_fn = fallback_fn
    
    async def __call__(self, **kwargs) -> Union[BaseModel, Dict[str, Any]]:
        """
        Call the Guidance program with arguments.
        
        Args:
            **kwargs: Arguments to pass to the program
            
        Returns:
            A structured output as a Pydantic model or dictionary
        """
        if self.verbose:
            logger.debug(f"Executing Guidance program with parameters: {kwargs}")
            
        try:
            # Handle different program execution patterns
            if hasattr(self.program, "__call__"):
                if asyncio.iscoroutinefunction(self.program.__call__):
                    # Program is already async
                    result = await self.program(**kwargs)
                else:
                    # Run sync program in executor to not block
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, lambda: self.program(**kwargs)
                    )
            else:
                # Fallback for legacy implementations
                logger.warning("Program doesn't have a __call__ method, attempting direct execution")
                result = self.program  # Assume program is the result itself
            
            # Process the result
            if self.verbose:
                logger.debug(f"Raw program result: {result}")
                
            return self._process_result(result)
                
        except Exception as e:
            logger.error(f"Error executing Guidance program: {e}")
            
            # Try fallback if available
            if self.fallback_fn:
                try:
                    logger.info("Using fallback function")
                    fallback_result = await self.fallback_fn(**kwargs)
                    return self._process_result(fallback_result)
                except Exception as fe:
                    logger.error(f"Fallback function also failed: {fe}")
            
            # Return empty instance as last resort
            logger.warning(f"Returning empty {self.output_cls.__name__} due to execution failure")
            return self.output_cls()
            
    def _process_result(self, result: Any) -> Union[BaseModel, Dict[str, Any]]:
        """
        Process the result from program execution into the desired output type.
        
        Args:
            result: The raw result from program execution
            
        Returns:
            Processed result as a Pydantic model or dictionary
        """
        # Handle different result types
        if isinstance(result, self.output_cls):
            # Already the right type
            return result
        elif isinstance(result, dict):
            # Convert dict to output class
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
                logger.error(f"Failed to parse string result as JSON: {e}")
                return result
        elif hasattr(result, "get") and callable(result.get):
            # Might be a structured_output from guidance
            try:
                structured_output = result.get("structured_output", "{}")
                if isinstance(structured_output, str):
                    data = json.loads(structured_output)
                    return self.output_cls(**data)
                elif isinstance(structured_output, dict):
                    return self.output_cls(**structured_output)
                else:
                    return structured_output
            except Exception as e:
                logger.error(f"Failed to process structured_output: {e}")
                return result
        else:
            # Return as-is if we can't process it
            logger.warning(f"Unexpected result type: {type(result)}")
            return result


async def create_program(
    output_cls: Type[BaseModel],
    prompt_template: str,
    llm: Optional[Any] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    fallback_fn: Optional[Callable] = None,
    verbose: bool = False
) -> Optional[GuidanceProgramWrapper]:
    """
    Create a Guidance program with version detection.
    
    This function detects the available LlamaIndex Guidance implementation
    and creates an appropriate program with it.
    
    Args:
        output_cls: Pydantic model class for the output schema
        prompt_template: Prompt template in handlebars format or Python f-string format
        llm: Optional LLM to use (if provided, model/provider are ignored)
        model: Model name to use if llm not provided
        provider: Provider to use if llm not provided
        fallback_fn: Optional fallback function if program execution fails
        verbose: Whether to enable verbose logging
        
    Returns:
        A wrapped Guidance program or None if creation fails
    """
    from .utils import ensure_handlebars_format
    
    if not is_guidance_available():
        logger.warning("Guidance library not available")
        return None
        
    # Convert template to handlebars format if needed
    handlebars_template = ensure_handlebars_format(prompt_template)
    
    # Get LlamaIndex version to determine API pattern
    version = get_llamaindex_version()
    
    try:
        # Set up LLM if not provided
        if llm is None:
            llm = await _create_llm(model, provider)
            if llm is None:
                logger.error("Failed to create LLM")
                return None
    
        # Create the appropriate program based on version
        program = None
        
        if version >= (0, 12, 0):  # Current API (0.12+)
            try:
                from llama_index.program.guidance import GuidancePydanticProgram
                
                # Check if we need to convert LLM to guidance format
                if hasattr(llm, "as_guidance_llm"):
                    guidance_llm = llm.as_guidance_llm()
                else:
                    guidance_llm = llm
                    
                program = GuidancePydanticProgram.from_defaults(
                    output_cls=output_cls,
                    prompt_template_str=handlebars_template,
                    llm=guidance_llm,
                    verbose=verbose
                )
                logger.info("Created Guidance program with current API (program.guidance)")
            except ImportError:
                logger.warning("Could not import from llama_index.program.guidance")
                
        if program is None and version >= (0, 10, 0):  # Try core API (0.10-0.11)
            try:
                from llama_index.core.program.guidance import GuidancePydanticProgram
                
                # Check if we need to convert LLM to guidance format
                if hasattr(llm, "as_guidance_llm"):
                    guidance_llm = llm.as_guidance_llm()
                else:
                    guidance_llm = llm
                    
                program = GuidancePydanticProgram.from_defaults(
                    output_cls=output_cls,
                    prompt_template_str=handlebars_template,
                    llm=guidance_llm,
                    verbose=verbose
                )
                logger.info("Created Guidance program with core API (core.program.guidance)")
            except ImportError:
                logger.warning("Could not import from llama_index.core.program.guidance")
                
        if program is None:  # Try legacy API (0.9)
            try:
                from llama_index.prompts.guidance import GuidancePydanticProgram
                
                # Legacy API used different LLM structure
                if isinstance(llm, str):
                    # Try to create OpenAI-style LLM for legacy
                    from guidance.llms import OpenAI as GuidanceOpenAI
                    guidance_llm = GuidanceOpenAI(llm)
                else:
                    guidance_llm = llm
                    
                program = GuidancePydanticProgram(
                    output_cls=output_cls,
                    prompt_template_str=handlebars_template,
                    guidance_llm=guidance_llm,
                    verbose=verbose
                )
                logger.info("Created Guidance program with legacy API (prompts.guidance)")
            except ImportError:
                logger.warning("Could not import from llama_index.prompts.guidance")
            
        if program is None:  # Last resort: raw guidance
            try:
                import guidance
                
                # Create raw guidance program
                program = guidance.Program(
                    guidance_llm=llm,
                    prompt=handlebars_template
                )
                logger.info("Created raw Guidance program (no LlamaIndex integration)")
            except Exception as e:
                logger.error(f"Failed to create raw Guidance program: {e}")
                return None
        
        # Wrap the program
        return GuidanceProgramWrapper(
            program=program,
            output_cls=output_cls,
            verbose=verbose,
            fallback_fn=fallback_fn
        )
        
    except Exception as e:
        logger.error(f"Failed to create Guidance program: {e}")
        return None


async def _create_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None
) -> Optional[Any]:
    """
    Create an LLM instance for Guidance.
    
    Args:
        model: Model name
        provider: Provider name
        
    Returns:
        An LLM instance compatible with Guidance
    """
    # Use defaults if not provided
    model = model or getattr(config, "ANALYSIS_MODEL", config.DEFAULT_MODEL)
    provider = provider or getattr(config, "ANALYSIS_MODEL_PROVIDER", config.DEFAULT_PROVIDER)
    
    try:
        # Try to use LlamaIndex LLM factory
        try:
            # Latest API
            from llama_index.llms import LLM, OpenAI, Anthropic
        except ImportError:
            try:
                # Core API
                from llama_index.core.llms import LLM, OpenAI, Anthropic
            except ImportError:
                # Legacy - just use raw guidance
                from guidance.llms import OpenAI as GuidanceOpenAI
                return GuidanceOpenAI(model)
        
        # Create provider-specific LLM
        provider = provider.lower()
        if provider == "openai":
            llm = OpenAI(model=model, temperature=0.1)
        elif provider == "anthropic":
            llm = Anthropic(model=model, temperature=0.1)
        else:
            # Generic LLM
            llm = LLM(model=model)
            
        return llm
        
    except Exception as e:
        logger.error(f"Failed to create LLM: {e}")
        return None
