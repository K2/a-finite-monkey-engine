"""
Extension module for guidance to better support structured output using Pydantic models
"""

import json
import logging
import asyncio
from typing import Type, TypeVar, Any, Dict, List, Optional
import guidance
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def patch_guidance_llm():
    """
    Patch the guidance LLM class to add structured output capabilities
    """
    original_as_structured_llm = getattr(guidance.llms.LLM, 'as_structured_llm', None)
    
    if original_as_structured_llm:
        logger.info("Guidance LLM already has as_structured_llm method")
        return
    
    # Add the method to the LLM class if it doesn't exist
    def as_structured_llm(self, output_class: Type[T]) -> 'StructuredLLM':
        """
        Convert the LLM to a structured LLM that returns instances of the given class
        
        Args:
            output_class: Pydantic model class for the output
            
        Returns:
            StructuredLLM instance configured for the output class
        """
        return StructuredLLM(self, output_class)
    
    # Patch the method
    setattr(guidance.llms.LLM, 'as_structured_llm', as_structured_llm)
    logger.info("Added as_structured_llm method to guidance LLM class")

class StructuredLLM:
    """
    Wrapper around a guidance LLM that returns structured outputs
    """
    
    def __init__(self, llm: guidance.llms.LLM, output_class: Type[T]):
        """
        Initialize the structured LLM
        
        Args:
            llm: The underlying LLM
            output_class: Pydantic model class for the output
        """
        self.llm = llm
        self.output_class = output_class
        self.schema = output_class.model_json_schema()
        logger.info(f"Created StructuredLLM for {output_class.__name__} with schema")
    
    def complete(self, messages: List[Dict[str, Any]]) -> T:
        """
        Generate a structured completion from the messages
        
        Args:
            messages: List of chat messages (system, user, assistant)
            
        Returns:
            Instance of the output class
        """
        # Create a guidance program with the messages
        program = guidance.Program(self.llm)
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                program = program.system(content)
            elif role == 'user':
                program = program.user(content)
            elif role == 'assistant':
                program = program.assistant(content)
        
        # Add assistant prompt with JSON schema constraint
        program = program.assistant(f'{{% gen "response" json_schema={json.dumps(self.schema)} %}}')
        
        # Generate response
        result = program.generate()
        response = result.get("response")
        
        # Parse response into Pydantic model
        return self.output_class.model_validate(response)
    
    async def acomplete(self, messages: List[Dict[str, Any]]) -> T:
        """
        Generate a structured completion from the messages asynchronously
        
        Args:
            messages: List of chat messages (system, user, assistant)
            
        Returns:
            Instance of the output class
        """
        # Run in thread pool to avoid blocking
        return await asyncio.to_thread(self.complete, messages)
