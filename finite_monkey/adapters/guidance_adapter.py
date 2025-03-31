"""
Guidance adapter for structured output generation.
"""
from typing import Any, Dict, List, Type, Optional, Union
import json
import asyncio
from pydantic import BaseModel
from loguru import logger
import tiktoken

from guidance_integration.core import GuidanceManager

try:
    import guidance
    from guidance.models import OpenAI as GuidanceOpenAI
    from guidance.models.llama_cpp import LlamaCpp
    GUIDANCE_AVAILABLE = True
    from llama_index.program.openai import OpenAIPydanticProgram 
    from llama_index.program.guidance import GuidancePydanticProgram
    from llama_index.question_gen.guidance import GuidanceQuestionGenerator
    
except ImportError:
    logger.warning("Guidance library not installed, structured output features will be limited")
    GUIDANCE_AVAILABLE = False

from ..nodes_config import config


class GuidanceAdapter:
    """
    Adapter for Microsoft's Guidance library to generate structured outputs.
    
    Guidance forces LLMs to follow specific output schemas, dramatically
    improving the reliability of structured outputs like JSON.
    """
    
    def __init__(
        self,
        model: str = None,
        provider: str = None,
        temperature: float = 0.1,
        timeout: Optional[int] = None
    ):
        """
        Initialize the Guidance adapter.
        
        Args:
            model: Model name to use
            provider: Model provider
            temperature: Temperature for generation
            timeout: Request timeout in seconds
        """
        if not GUIDANCE_AVAILABLE:
            raise ImportError("Guidance library is required but not installed. Install with 'pip install guidance'")
            
        self.model = model or config.ANALYSIS_MODEL
        self.provider = provider or config.ANALYSIS_MODEL_PROVIDER
        self.temperature = temperature
        self.timeout = timeout or getattr(config, "REQUEST_TIMEOUT", 60)
        self.encoding =tiktoken.get_encoding( "cl100k_base")
        # Initialize guidance LLM
        self._init_guidance_llm()
        
    def _init_guidance_llm(self):
        """Initialize the appropriate guidance LLM based on provider"""
        if self.provider.lower() == "openai":
            self.guidance_llm = GuidanceOpenAI(
                self.model,tokenizer=self.encoding,
                base_url=config.DEFAULT_BASE_URL,
                temperature=self.temperature,
                timeout=self.timeout
            )
        else:
            # Default to OpenAI if provider not supported yet
            logger.warning(f"Provider {self.provider} not directly supported by Guidance adapter, attempting LlamaCpp")
            self.guidance_llm = LlamaCpp(
                self.model, echo=LlamaCpp, tokenizer=self.encoding,
                base_url=config.DEFAULT_BASE_URL,
                timeout=self.timeout
            )
    
    async def generate_structured_output(
        self, 
        output_cls: Type[BaseModel], 
        prompt: str,
        **kwargs
    ) -> Union[BaseModel, Dict[str, Any]]:
        """
        Generate a structured output object using guidance.
        
        Args:
            output_cls: Pydantic model class to use as output schema
            prompt: Prompt template to use (in handlebars format)
            **kwargs: Additional variables for the prompt template
            
        Returns:
            Instantiated Pydantic model with the LLM's structured output
        """
        try:
            # Convert Python format strings to handlebars if needed
            prompt = self._ensure_handlebars_format(prompt)
            
            # Create the guidance program
            program = OpenAIPydanticProgram.from_defaults(output_cls=output_cls,prompt_template_str=prompt,
                guidance_llm=self.guidance_llm, llm = self.guidance_llm, base_api=config.DEFAULT_BASE_URL,
            )
            
            # Run the program to get structured output
            # This needs to be run in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: program(**kwargs))
            
            # Extract the JSON output
            json_str = result.get("structured_output", "{}")
            
            # Parse into the output class
            parsed_obj = output_cls.parse_raw(json_str)
            return parsed_obj
            
        except Exception as e:
            logger.error(f"Error generating structured output with guidance: {e}")
            # Return an empty instance of the output class as fallback
            return output_cls()
    
    def _ensure_handlebars_format(self, prompt: str) -> str:
        """
        Ensure prompt is in handlebars format for guidance.
        
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
            from llama_index.core.prompts.guidance_utils import convert_to_handlebars
            return convert_to_handlebars(prompt)
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to convert prompt to handlebars: {e}")
            # Simple naive conversion as fallback
            import re
            return re.sub(r'\{([^{}]*)\}', r'{{\1}}', prompt)
