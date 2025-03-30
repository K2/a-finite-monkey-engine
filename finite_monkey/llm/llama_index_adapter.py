from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Literal
import json
import asyncio
import httpx
from concurrent.futures import Future
from loguru import logger

# Updated imports for the new Settings approach
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
#from llama_index.llms.openai import    
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.callbacks import CallbackManager

from finite_monkey.nodes_config import config

"""
Adapter for LlamaIndex LLM integrations
"""

import asyncio
import time
from typing import Dict, Any, Optional
from concurrent.futures import Future
from loguru import logger

# Import our logging middleware
from .logging_middleware import log_llm_call, LLMLogger

# Initialize LLM logger
LLMLogger.setup()

class LlamaIndexAdapter:
    """
    Adapter for LlamaIndex LLM integrations
    
    This adapter provides a unified interface for working with different LLM providers.
    """
    
    def __init__(
        self,
        model_name: str = None,
        provider: str = None,
        base_url: Optional[str] = None,
        request_timeout: Optional[int] = 60,  # Changed from 'timeout' to 'request_timeout'
        temperature: float = 0.1,
        additional_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initialize the LlamaIndex adapter with model settings"""
        self._model_name = model_name
        self._provider = provider
        self._base_url = base_url
        self.request_timeout = request_timeout  # Make sure to use consistent naming
        self._kwargs = additional_kwargs or {}
        self._llm = None
        
        # Try to initialize LLM at creation time
        try:
            self._init_llm()
        except Exception as e:
            logger.warning(f"Deferred LLM initialization - will retry at first use: {e}")
    
    @property
    def model_name(self) -> str:
        """Get the model name"""
        return self._model_name
    
    @property
    def provider(self) -> str:
        """Get the provider name"""
        return self._provider
    
    @property
    def llm(self):
        """Get the LLM instance, initializing if needed"""
        if self._llm is None:
            self._init_llm()
        return self._llm
    
    @llm.setter
    def llm(self, value):
        """Set the LLM instance"""
        self._llm = value
    
    def _init_llm(self):
        """Initialize the LLM based on provider"""
        try:
            if self._provider.lower() == "ollama":
                from llama_index.llms.ollama import Ollama
                self._llm = Ollama(
                    model=self._model_name,
                    base_url=self._base_url,
                    request_timeout=self.request_timeout,  # Use the consistent parameter name
                    **self._kwargs
                )
                logger.debug(f"Initialized Ollama LLM with model {self._model_name}")
                
            elif self._provider.lower() == "openai":
                from llama_index.llms.openai import OpenAI
                self._llm = OpenAI(
                    model=self._model_name,
                    request_timeout=self.request_timeout,  # Use the consistent parameter name
                    **self._kwargs
                )
                logger.debug(f"Initialized OpenAI LLM with model {self._model_name}")
                
            else:
                raise ValueError(f"Unsupported LLM provider: {self._provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    @log_llm_call
    async def submit_prompt(self, prompt: str) -> str:
        """Submit a prompt to the LLM and get a response"""
        if self._llm is None:
            self._init_llm()
            
        try:
            # Log the start of the request
            start_time = time.time()
            
            # Process with appropriate method based on provider
            response = await self._llm.acomplete(prompt)
            response_text = response.text
            
            # Log completion time
            duration = time.time() - start_time
            logger.debug(f"LLM response received in {duration:.2f}s")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in LLM request: {e}")
            raise
    
    @log_llm_call
    async def submit_json_prompt(self, prompt: str, schema: Dict[str, Any]) -> Future:
        """Submit a prompt to the LLM and extract structured JSON"""
        if self._llm is None:
            self._init_llm()
        
        try:
            # Import here to avoid circular imports
            from finite_monkey.utils.json_extractor import extract_json_with_schema
            
            # Create a future to hold the result
            future = Future()
            
            # Run the extraction in a task
            async def run_extraction():
                try:
                    
                    
                    result = await extract_json_with_schema(
                        llm=self._llm,
                        prompt=prompt,
                        schema=schema
                    )
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
            
            # Schedule the task
            asyncio.create_task(run_extraction())
            
            return future
            
        except Exception as e:
            logger.error(f"Error in JSON extraction: {e}")
            raise