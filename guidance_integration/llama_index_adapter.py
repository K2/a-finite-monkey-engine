"""
Adapter for LlamaIndex integration with structured output support
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Type, TypeVar, List, Union
from pydantic import BaseModel
import json
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms import LLM

from finite_monkey.models.security import SecurityAnalysisResult

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class LlamaIndexAdapter:
    """
    Adapter for LlamaIndex that simplifies interaction with structured output
    """
    
    def __init__(self, 
                 model_name: str = "ollama:llama3", 
                 provider: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs):
        """
        Initialize the LlamaIndex adapter
        
        Args:
            model_name: Model identifier
            provider: LLM provider (optional)
            base_url: Base URL for API (optional)
            **kwargs: Additional configuration options
        """
        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url
        self.kwargs = kwargs
        self._llm = None
    
    @property
    def llm(self) -> LLM:
        """
        Get or create the LlamaIndex LLM instance
        
        Returns:
            LlamaIndex LLM instance
        """
        if self._llm is None:
            self._create_llm()
        return self._llm
    
    def _create_llm(self) -> None:
        """Create the appropriate LLM based on provider and model name"""
        provider = self.provider.lower() if self.provider else self._detect_provider()
        
        try:
            if provider == "ollama":
                from llama_index.llms.ollama import Ollama
                self._llm = Ollama(
                    model=self.model_name.split(':', 1)[-1],
                    base_url=self.base_url or "http://localhost:11434",
                    json_mode=True,
                    **self.kwargs
                )
            elif provider == "openai":
                from llama_index.llms.openai import OpenAI
                self._llm = OpenAI(
                    model=self.model_name.split(':', 1)[-1],
                    **self.kwargs
                )
            elif provider == "anthropic":
                from llama_index.llms.anthropic import Anthropic
                self._llm = Anthropic(
                    model=self.model_name.split(':', 1)[-1],
                    **self.kwargs
                )
            else:
                logger.warning(f"Unsupported provider: {provider}. Falling back to OpenAI.")
                from llama_index.llms.openai import OpenAI
                self._llm = OpenAI(
                    model="gpt-4o",
                    **self.kwargs
                )
                
            logger.info(f"Created LLM of type {type(self._llm)} for model {self.model_name}")
        except Exception as e:
            logger.error(f"Error creating LLM: {e}", exc_info=True)
            raise
    
    def _detect_provider(self) -> str:
        """Detect provider from model name"""
        if ":" in self.model_name:
            return self.model_name.split(":", 1)[0].lower()
        elif "gpt" in self.model_name.lower():
            return "openai"
        elif "claude" in self.model_name.lower():
            return "anthropic"
        else:
            return "ollama"
    
    def as_structured_llm(self, output_class: Type[T]) -> "StructuredLLM":
        """
        Convert the LLM to a structured LLM that returns instances of a Pydantic class
        
        Args:
            output_class: Pydantic model class for output
            
        Returns:
            StructuredLLM instance for generating structured outputs
        """
        return StructuredLLM(self.llm, output_class)
    
    async def chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """
        Send a chat request to the LLM
        
        Args:
            messages: List of chat messages
            **kwargs: Additional LLM options
            
        Returns:
            LLM response text
        """
        response = await self.llm.achat(messages, **kwargs)
        return response.message.content
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """
        Send a completion request to the LLM
        
        Args:
            prompt: Prompt text
            **kwargs: Additional LLM options
            
        Returns:
            LLM response text
        """
        response = await self.llm.acomplete(prompt, **kwargs)
        return response.text


class StructuredLLM:
    """
    Wrapper around a LlamaIndex LLM that returns structured outputs
    """
    
    def __init__(self, llm: LLM, output_class: Type[T]):
        """
        Initialize the structured LLM
        
        Args:
            llm: LlamaIndex LLM instance
            output_class: Pydantic model class for output
        """
        self.llm = llm
        self.output_class = output_class
        logger.info(f"Created StructuredLLM for {output_class.__name__}")
    
    async def achat(self, messages: List[ChatMessage]) -> T:
        """
        Send a chat request and return structured output
        
        Args:
            messages: List of chat messages
            
        Returns:
            Instance of the output class
        """
        try:
            # Convert to LlamaIndex structured output response
            structured_llm = self.llm.as_structured_output(self.output_class)
            
            # Generate the structured output
            result = json.loads((await structured_llm.achat(messages)).message.content)
            logger.info(f"Successfully generated structured output of type {self.output_class.__name__}")
            
            return result
        except Exception as e:
            logger.error(f"Error generating structured output: {e}", exc_info=True)
            # Create a minimal valid instance as fallback
            return self.output_class()
    
    async def acomplete(self, prompt: str) -> T:
        """
        Send a completion request and return structured output
        
        Args:
            prompt: Prompt text
            
        Returns:
            Instance of the output class
        """
        # Create a chat message from the prompt
        message = ChatMessage(role=MessageRole.USER, content=prompt)
        
        # Use the chat API which is more reliable for structured output
        return await self.achat([message])
    
    async def analyze_security(self, system_prompt: str, prompt: str, file_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform security analysis on the given file data
        
        Args:
            system_prompt: System prompt for the LLM
            prompt: User prompt for the LLM
            file_data: File data for analysis
            
        Returns:
            List of security findings
        """
        # Create chat message
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]
        
        # Generate the structured output
        try:
            structured_llm = self.llm.as_structured_output(SecurityAnalysisResult)
            result: SecurityAnalysisResult = json.loads((await structured_llm.achat(messages)).message.content)  
            logger.info(f"Successfully generated structured security analysis with {len(result.findings)} findings")
            
            # Add file path to each finding
            for finding in result.findings:
                finding.location = f"{file_data.get('path', 'unknown')}: {finding.location}"
            
            return result.findings
            
        except asyncio.TimeoutError:
            logger.error(f"LLM analysis timed out for file {file_data.get('path', 'unknown')}")
            return []
            
        except Exception as e:
            logger.error(f"Error in vulnerability analysis: {str(e)}")
            return []

system_prompt = """You are an expert smart contract security auditor with deep knowledge of
"""
