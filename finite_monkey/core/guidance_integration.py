"""
Core integration with Microsoft's Guidance library for structured output generation.

This module provides a simplified wrapper around the latest LlamaIndex Guidance
integration, handling the complexities of different API versions and providing
a consistent interface.
"""
from typing import Any, Dict, List, Type, Optional, Union, Callable
import asyncio
from pydantic import BaseModel
from loguru import logger

# Conditionally import guidance components
try:
    import guidance
    # Import the latest LlamaIndex guidance integration
    try:
        # First try the newer module path
        from llama_index.program.guidance import GuidancePydanticProgram
        GUIDANCE_AVAILABLE = True
        LEGACY_API = False
        logger.info("Using latest LlamaIndex Guidance integration")
    except ImportError:
        # Fall back to the older module path
        try:
            from llama_index.core.program.guidance import GuidancePydanticProgram
            GUIDANCE_AVAILABLE = True
            LEGACY_API = False
            logger.info("Using core LlamaIndex Guidance integration")
        except ImportError:
            # Try the legacy path as last resort
            try:
                from llama_index.prompts.guidance import GuidancePydanticProgram
                GUIDANCE_AVAILABLE = True
                LEGACY_API = True
                logger.info("Using legacy LlamaIndex Guidance integration")
            except ImportError:
                logger.warning("No LlamaIndex Guidance integration found")
                GUIDANCE_AVAILABLE = False
                LEGACY_API = False
except ImportError:
    logger.warning("Guidance library not available. Install with: pip install guidance")
    GUIDANCE_AVAILABLE = False
    LEGACY_API = False

from ..nodes_config import config


class GuidanceManager:
    """
    Unified interface for working with Guidance across different LlamaIndex API versions.
    
    This class handles the complexity of different Guidance integration paths
    and provides a consistent interface for structured output generation.
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
            verbose: Whether to enable verbose output
        """
        self.model = model or getattr(config, "ANALYSIS_MODEL", config.DEFAULT_MODEL)
        self.provider = provider or getattr(config, "ANALYSIS_MODEL_PROVIDER", config.DEFAULT_PROVIDER)
        self.temperature = temperature
        self.verbose = verbose
        
        # Check if Guidance is available
        if not GUIDANCE_AVAILABLE:
            logger.warning("Guidance functionality is not available")
            return
        
        # Set up appropriate LLM based on API version
        self._setup_llm()
    
    def _setup_llm(self):
        """Set up the appropriate LLM for Guidance based on API version"""
        if not GUIDANCE_AVAILABLE:
            return
            
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
            
            logger.info(f"Initialized Guidance LLM with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Guidance LLM: {e}")
            self.guidance_llm = None
    
    async def create_structured_program(
        self, 
        output_cls: Type[BaseModel],
        prompt_template: str,
        fallback_fn: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Create a Guidance program for structured output generation.
        
        Args:
            output_cls: Pydantic model class defining the output schema
            prompt_template: Handlebars template for the prompt
            fallback_fn: Optional fallback function if Guidance fails
            
        Returns:
            A callable program or None if Guidance isn't available
        """
        if not GUIDANCE_AVAILABLE or not self.guidance_llm:
            logger.warning("Guidance not available, can't create structured program")
            return None
            
        try:
            # Create the program using appropriate API
            if LEGACY_API:
                program = GuidancePydanticProgram(
                    output_cls=output_cls,
                    prompt_template_str=prompt_template,
                    guidance_llm=self.guidance_llm,
                    verbose=self.verbose
                )
            else:
                program = GuidancePydanticProgram.from_defaults(
                    output_cls=output_cls,
                    prompt_template_str=prompt_template,
                    llm=self.guidance_llm,
                    verbose=self.verbose
                )
                
            # Wrap the program in our unified async interface
            return StructuredProgram(program, fallback_fn)
            
        except Exception as e:
            logger.error(f"Error creating Guidance program: {e}")
            return None


class StructuredProgram:
    """
    Wrapper for Guidance program that provides a consistent async interface.
    """
    
    def __init__(
        self, 
        program: Any, 
        fallback_fn: Optional[Callable] = None
    ):
        """
        Initialize with a Guidance program.
        
        Args:
            program: The underlying Guidance program
            fallback_fn: Optional fallback function if the program fails
        """
        self.program = program
        self.fallback_fn = fallback_fn
    
    async def __call__(self, **kwargs) -> Union[BaseModel, Dict[str, Any]]:
        """
        Execute the program with the given parameters.
        
        Args:
            **kwargs: Parameters to pass to the program
            
        Returns:
            A structured output object or dictionary
        """
        try:
            # Execute the program
            if asyncio.iscoroutinefunction(self.program.__call__):
                # If it's already async
                result = await self.program(**kwargs)
            else:
                # If it's synchronous, run in executor
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.program(**kwargs)
                )
                
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
            logger.warning("Returning empty result due to failures")
            output_cls = getattr(self.program, "output_cls", None)
            if output_cls:
                return output_cls()
            return {}
