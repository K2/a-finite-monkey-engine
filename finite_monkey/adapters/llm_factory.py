"""
LLM factory for the Finite Monkey framework

This module provides a factory for creating LLM clients based on configuration
settings, with appropriate abstractions for the dual-layer agent architecture.
"""

from typing import Optional, Dict, Any, Union
import os

from llama_index.llms.ollama import Ollama as LlamaOllama
#from llama_index.llms.openai import OpenAI as LlamaOpenAI

from .ollama import AsyncOllamaClient
from .claude import Claude
from finite_monkey.nodes_config import config


class LLMFactory:
    """
    Factory for creating LLM clients
    
    This class provides methods for creating the appropriate LLM client
    based on configuration settings. It supports both outer atomic agents
    (using AsyncOllamaClient/Claude) and inner llama-index agents
    (using LlamaOllama/OpenAI wrappers).
    """
    
    @classmethod
    def create_atomic_client(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 300,
    ) -> Union[AsyncOllamaClient, Claude]:
        """
        Create an LLM client for atomic agents
        
        Args:
            provider: Provider name (ollama, claude, etc.)
            model: Model name
            timeout: Request timeout in seconds
            
        Returns:
            LLM client for atomic agents
        """
        # No need to call nodes_config() anymore, use config directly
        
        # Determine provider based on config and available API keys
        provider = provider or cls._get_default_provider(config)
        
        # Create the appropriate client
        if provider.lower() == "claude":
            return Claude(
                model=model or config.CLAUDE_MODEL,
                timeout=timeout,
            )
        else:  # Default to ollama
            return AsyncOllamaClient(
                model=model or config.WORKFLOW_MODEL,
                timeout=timeout,
            )
    
    @classmethod
    def create_llama_index_client(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 300,
    ) -> Union[LlamaOllama,LlamaOllama]:
        """
        Create an LLM client for llama-index agents
        
        Args:
            provider: Provider name (ollama, openai, etc.)
            model: Model name
            timeout: Request timeout in seconds
            
        Returns:
            LLM client for llama-index agents
        """
        # No need to call nodes_config() anymore, use config directly
        
        # Determine provider based on config and available API keys
        provider = provider or cls._get_default_provider(config)
        
        # Create the appropriate client
        # if provider.lower() == "openai":
        #     return LlamaOpenAI(
        #         model=model or config.OPENAI_MODEL,
        #         api_key=config.OPENAI_API_KEY,
        #         timeout=timeout,
        #     )
        # else:  # Default to ollama
        return LlamaOllama(
            model=model or config.WORKFLOW_MODEL,
            request_timeout=timeout,
        )
    
    @classmethod
    def _get_default_provider(cls, config) -> str:
        """
        Get the default provider based on configuration
        
        Args:
            config: Configuration object
            
        Returns:
            Default provider name
        """
        # Check for API keys to determine available providers
        # if config.CLAUDE_API_KEY or os.environ.get("CLAUDE_API_KEY"):
        #     return "claude"
        # elif config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"):
        #     return "openai"
        # else:
        
        return "ollama"  # Default to ollama for local deployment