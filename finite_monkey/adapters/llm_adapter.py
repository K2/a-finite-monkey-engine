"""
LLM adapter for interacting with various LLM backends.
Provides a unified interface for making LLM calls.
"""
from typing import Dict, List, Any, Optional, Union, Callable
import asyncio
from loguru import logger

from ..nodes_config import config
from ..utils.llm_monitor import LLMInteractionTracker

class LLMAdapter:
    """
    Adapter for LLM interactions.
    Provides a unified interface for different LLM backends.
    """
    
    def __init__(self, model: str = None, provider: str = None, base_url: str = None):
        """
        Initialize the LLM adapter.
        
        Args:
            model: Model name to use
            provider: Provider name (e.g., 'openai', 'ollama')
            base_url: Base URL for the provider's API
        """
        self.model = model or config.DEFAULT_MODEL
        self.provider = provider or config.DEFAULT_PROVIDER
        self.base_url = base_url or config.get('PROVIDER_URLS', {}).get(self.provider)
        self.logger = logger
        self._client = None
        self._tracker = None
    
    def set_tracker(self, tracker: LLMInteractionTracker) -> None:
        """
        Set the LLM interaction tracker for monitoring calls.
        
        Args:
            tracker: The tracker instance
        """
        self._tracker = tracker
    
    async def _ensure_client(self):
        """Ensure client is initialized"""
        if self._client is None:
            if self.provider == 'ollama':
                from .ollama import AsyncOllamaClient
                self._client = AsyncOllamaClient(model=self.model, base_url=self.base_url)
            elif self.provider == 'openai':
                from .openai import AsyncOpenAIClient
                self._client = AsyncOpenAIClient(model=self.model, base_url=self.base_url)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def acomplete(self, prompt: str, stage_name: str = None) -> Any:
        """
        Complete a prompt asynchronously with tracking.
        
        Args:
            prompt: Prompt to complete
            stage_name: Optional name of the calling stage
            
        Returns:
            LLM response
        """
        await self._ensure_client()
        
        if self._tracker and stage_name:
            return await self._tracker.track_interaction(
                stage_name, 
                prompt, 
                lambda p: self._client.acomplete(p)
            )
        else:
            return await self._client.acomplete(prompt)
    
    async def llm(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Core LLM method that cognitive_bias_analyzer.py and other components expect.
        This was missing and causing the warning.
        
        Args:
            prompt: The prompt text
            options: Additional options for the generation
            
        Returns:
            Generated response
        """
        # Track this call if needed
        stage_name = options.get("stage_name") if options else None
        result = await self.acomplete(prompt, stage_name)
        
        # Extract text from the response based on type
        if isinstance(result, str):
            return result
        elif hasattr(result, "choices") and hasattr(result.choices[0], "message"):
            return result.choices[0].message.content
        elif hasattr(result, "content"):
            return result.content
        elif hasattr(result, "text"):
            return result.text
        else:
            # Fallback to string representation
            return str(result)
    
    async def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The prompt text
            options: Additional options for the generation
            
        Returns:
            Generated response
        """
        return await self.llm(prompt, options)
    
    async def structured_generate(self, prompt: str, response_format: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a structured response according to the specified format.
        
        Args:
            prompt: The prompt text
            response_format: Format specification for the response
            options: Additional options for the generation
            
        Returns:
            Structured response
        """
        await self._ensure_client()
        try:
            # Combine options with format requirements
            combined_options = options.copy() if options else {}
            combined_options["response_format"] = response_format
            
            # Check if client supports structured generation
            if hasattr(self._client, 'structured_generate'):
                return await self._client.structured_generate(prompt, response_format, combined_options)
            else:
                # Fallback to regular generation and attempt to parse
                import json
                response_text = await self.generate(
                    f"{prompt}\n\nRespond with a JSON object following this format: {json.dumps(response_format)}",
                    options
                )
                try:
                    return json.loads(response_text)
                except:
                    # If JSON parsing fails, try to extract JSON from the response
                    import re
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                    if json_match:
                        try:
                            return json.loads(json_match.group(1))
                        except:
                            pass
                    return {"error": "Failed to parse structured response", "raw_response": response_text}
        except Exception as e:
            self.logger.error(f"Error in structured LLM generation: {e}")
            return {"error": str(e)}
    
    async def achat_dict(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send a chat request with dictionary-based messages for better serialization
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters for the request
            
        Returns:
            The response text from the LLM
        """
        # Convert dict messages to ChatMessage objects if needed by the underlying LLM
        try:
            from llama_index.core.llms import ChatMessage, MessageRole
            
            chat_messages = []
            for msg in messages:
                role_str = msg["role"].upper()
                role = MessageRole.USER  # Default
                if hasattr(MessageRole, role_str):
                    role = getattr(MessageRole, role_str)
                chat_messages.append(ChatMessage(role=role, content=msg["content"]))
                
            response = await self.llm(chat_messages, **kwargs) 
            return response.message.content
            
        except Exception as e:
            logger.error(f"Error in achat_dict: {e}")
            # Fallback to basic completion if chat fails
            combined_prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            return await self.acomplete(combined_prompt, **kwargs)
