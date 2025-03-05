"""
Claude adapter for the Finite Monkey framework

This module provides an adapter for the Claude API, allowing the framework
to use Anthropic's Claude models for large language model functions.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple

import httpx


class Claude:
    """
    Adapter for the Claude API
    
    This class provides an interface to the Claude API for LLM functions,
    with the same interface as the Ollama adapter to allow for interchangeability.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet",
        api_base: str = "https://api.anthropic.com/v1",
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        """
        Initialize the Claude adapter
        
        Args:
            api_key: Claude API key (defaults to CLAUDE_API_KEY environment variable)
            model: Claude model name to use
            api_base: Base URL for the Claude API
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (higher = more creative)
        """
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("Claude API key not provided and CLAUDE_API_KEY environment variable not set")
        
        self.model = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set up httpx client
        self.client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        )
    
    async def complete(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Generate completion for a prompt (synchronous wrapper)
        
        Args:
            prompt: Prompt to complete
            model: Model to use (defaults to self.model)
            
        Returns:
            Generated text
        """
        return await self.acomplete(prompt, model)
    
    async def acomplete(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Generate completion for a prompt asynchronously
        
        Args:
            prompt: Prompt to complete
            model: Model to use (defaults to self.model)
            
        Returns:
            Generated text
        """
        # Determine model to use
        model_name = model or self.model
        
        try:
            # Create request
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            
            # Send request
            response = await self.client.post(
                "/messages",
                json=payload,
                timeout=600.0  # 10 minute timeout
            )
            
            # Parse response
            response.raise_for_status()
            result = response.json()
            
            # Extract text
            if "content" in result and len(result["content"]) > 0:
                return result["content"][0]["text"]
            else:
                return ""
        
        except Exception as e:
            print(f"Error in Claude API call: {str(e)}")
            return f"Error: {str(e)}"
    
    async def close(self):
        """
        Close the client session
        """
        await self.client.aclose()
    
    def __del__(self):
        """
        Clean up resources on deletion
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except Exception:
            pass