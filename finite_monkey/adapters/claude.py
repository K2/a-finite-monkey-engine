"""
Claude adapter for the Finite Monkey framework

This module provides an adapter for the Claude API, allowing the framework
to use Anthropic's Claude models for large language model functions.
Implements llama-index's LLM interface for seamless integration.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple, Sequence, Callable, Generator, AsyncGenerator

import httpx

try:
    # Import llama-index classes for proper integration
    from llama_index.core.llms import (
        CompletionResponse, 
        ChatMessage, 
        MessageRole, 
        ChatResponse, 
        LLM,
    )
    from llama_index.core.callbacks import CallbackManager
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    # Fallback if llama-index is not available
    print("Warning: llama-index not available for Claude adapter, using base implementation")
    CompletionResponse = Any
    ChatMessage = Any
    MessageRole = Any
    ChatResponse = Any
    LLM = object
    CallbackManager = Any
    LLAMA_INDEX_AVAILABLE = False


class Claude(LLM):
    """
    Adapter for the Claude API
    
    This class provides an interface to the Claude API for LLM functions,
    with the same interface as the Ollama adapter to allow for interchangeability.
    Implements the llama-index LLM interface for seamless integration.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet",
        api_base: str = "https://api.anthropic.com/v1",
        max_tokens: int = 4096,
        temperature: float = 0.2,
        callback_manager: Optional[CallbackManager] = None,
        context_window: int = 200000,
        additional_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the Claude adapter
        
        Args:
            api_key: Claude API key (defaults to CLAUDE_API_KEY environment variable)
            model: Claude model name to use
            api_base: Base URL for the Claude API
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (higher = more creative)
            callback_manager: LlamaIndex callback manager for logging/tracing
            context_window: Maximum context window size
            additional_kwargs: Additional keyword arguments for API calls
        """
        # Call parent constructor if available for proper LLM inheritance
        if LLM is not object:
            super().__init__(callback_manager=callback_manager)
        
        # Get settings from config
        from ..nodes_config import nodes_config
        config = nodes_config()
        
        # Use API key from config or environment
        self.api_key = api_key or config.CLAUDE_API_KEY or os.environ.get("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("Claude API key not provided and CLAUDE_API_KEY environment variable not set")
        
        # Use model from params if provided, otherwise from config
        self.model = model or config.CLAUDE_MODEL or "claude-3-5-sonnet"
        
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_window = context_window
        self.additional_kwargs = additional_kwargs or {}
        
        # Set up httpx client
        self.client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        )
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """LLM metadata for llama-index integration"""
        return {
            "model_name": self.model,
            "base_url": self.api_base,
            "context_window": self.context_window,
            "is_async": True,
        }
    
    # LlamaIndex interface methods
    async def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        LlamaIndex LLM interface method for completion
        
        Args:
            prompt: Prompt to complete
            **kwargs: Additional parameters
            
        Returns:
            CompletionResponse object
        """
        # Extract parameters from kwargs or use defaults
        model = kwargs.get("model", self.model)
        
        # Get text response using legacy method
        text = await self.acomplete(prompt, model)
        
        # Convert to llama-index CompletionResponse
        if LLAMA_INDEX_AVAILABLE:
            return CompletionResponse(
                text=text,
                raw={
                    "model": model,
                    "text": text,
                },
                additional_kwargs=kwargs,
            )
        else:
            # Simple fallback when llama-index is not available
            return {"text": text, "model": model}
    
    async def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        """
        LlamaIndex LLM interface method for chat
        
        Args:
            messages: List of ChatMessage objects
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        # Convert llama-index ChatMessage objects to Claude format
        claude_messages = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            # Map llama-index roles to Claude roles
            if role == "system":
                role = "system"
            elif role == "assistant":
                role = "assistant"
            else:
                role = "user"
                
            claude_messages.append({
                "role": role,
                "content": msg.content,
            })
        
        # Extract parameters from kwargs or use defaults
        model = kwargs.get("model", self.model)
        
        # Call the Claude API
        try:
            # Create request
            payload = {
                "model": model,
                "messages": claude_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **self.additional_kwargs,
                **kwargs.get("additional_kwargs", {})
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
            response_text = ""
            if "content" in result and len(result["content"]) > 0:
                response_text = result["content"][0]["text"]
            
            # Convert to llama-index ChatResponse
            if LLAMA_INDEX_AVAILABLE:
                response_message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response_text,
                )
                
                return ChatResponse(
                    message=response_message,
                    raw=result,
                    additional_kwargs=kwargs,
                )
            else:
                # Simple fallback when llama-index is not available
                return {
                    "message": {"role": "assistant", "content": response_text},
                    "model": model,
                }
                
        except Exception as e:
            print(f"Error in Claude API call: {str(e)}")
            error_msg = f"Error: {str(e)}"
            
            # Return error in appropriate format
            if LLAMA_INDEX_AVAILABLE:
                response_message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=error_msg,
                )
                return ChatResponse(
                    message=response_message,
                    raw={"error": str(e)},
                    additional_kwargs=kwargs,
                )
            else:
                return {
                    "message": {"role": "assistant", "content": error_msg},
                    "model": model,
                    "error": str(e)
                }
    
    # Legacy methods for backward compatibility
    async def acomplete(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Generate completion for a prompt asynchronously (legacy method)
        
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
    
    # Implement the required streaming methods for LlamaIndex integration
    async def astream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[CompletionResponse, None]:
        """
        Stream completion for the given prompt (async version)
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for completion
            
        Yields:
            CompletionResponse objects with partial completion chunks
        """
        # For now, this is a simplified implementation that doesn't actually stream
        # It just returns the full completion in one response
        text = await self.acomplete(
            prompt=prompt, 
            model=kwargs.get("model", self.model),
        )
        
        if LLAMA_INDEX_AVAILABLE:
            yield CompletionResponse(
                text=text,
                raw={
                    "model": self.model,
                    "text": text,
                },
                additional_kwargs=kwargs,
            )
        else:
            yield {"text": text, "model": self.model}
    
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        Stream chat response for the given messages (async version)
        
        Args:
            messages: List of ChatMessage objects
            **kwargs: Additional parameters for chat
            
        Yields:
            ChatResponse objects with partial response chunks
        """
        # Convert llama-index ChatMessage objects to Claude format
        claude_messages = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            # Map llama-index roles to Claude roles
            if role == "system":
                role = "system"
            elif role == "assistant":
                role = "assistant"
            else:
                role = "user"
                
            claude_messages.append({
                "role": role,
                "content": msg.content,
            })
        
        # Call acomplete with the first user message for simplicity
        user_content = ""
        for msg in claude_messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break
                
        if not user_content:
            user_content = "Please respond to my request."
            
        response_text = await self.acomplete(user_content)
        
        if LLAMA_INDEX_AVAILABLE:
            response_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_text,
            )
            
            yield ChatResponse(
                message=response_message,
                raw={
                    "model": self.model,
                    "messages": claude_messages,
                    "response": response_text,
                },
                additional_kwargs=kwargs,
            )
        else:
            yield {
                "message": {"role": "assistant", "content": response_text},
                "model": self.model,
            }
    
    def stream_complete(self, prompt: str, **kwargs) -> Generator[CompletionResponse, None, None]:
        """
        Stream completion for the given prompt (sync version)
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional parameters for completion
            
        Yields:
            CompletionResponse objects with partial completion chunks
        """
        # Run the async function in a new event loop
        async def run_async():
            async for response in self.astream_complete(prompt, **kwargs):
                yield response
                
        # Create a loop and run the async generator
        loop = asyncio.new_event_loop()
        try:
            for response in loop.run_until_complete(self._collect_async_generator(run_async())):
                yield response
        finally:
            loop.close()
    
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> Generator[ChatResponse, None, None]:
        """
        Stream chat response for the given messages (sync version)
        
        Args:
            messages: List of ChatMessage objects
            **kwargs: Additional parameters for chat
            
        Yields:
            ChatResponse objects with partial response chunks
        """
        # Run the async function in a new event loop
        async def run_async():
            async for response in self.astream_chat(messages, **kwargs):
                yield response
                
        # Create a loop and run the async generator
        loop = asyncio.new_event_loop()
        try:
            for response in loop.run_until_complete(self._collect_async_generator(run_async())):
                yield response
        finally:
            loop.close()
    
    async def _collect_async_generator(self, agen):
        """Helper function to collect async generator results"""
        results = []
        async for item in agen:
            results.append(item)
        return results
    
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