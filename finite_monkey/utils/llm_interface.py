"""
Universal LLM interface adapter that works with different LLM implementations.
Provides consistent methods regardless of the underlying library.
"""
import inspect
from typing import Any, Dict, List, Optional, Union
import asyncio
from loguru import logger

async def call_llm(
    llm: Any, 
    prompt: str, 
    as_chat: bool = True,
    system_message: Optional[str] = None,
    timeout: Optional[float] = None
) -> Any:
    """
    Call an LLM with automatic method detection and fallbacks.
    
    Args:
        llm: The LLM instance or adapter
        prompt: The input prompt
        as_chat: Whether to use chat format
        system_message: Optional system message for chat format
        timeout: Optional timeout in seconds
        
    Returns:
        LLM response in whatever format the underlying LLM provides
    """
    # Extract the actual LLM object if we have an adapter
    if hasattr(llm, 'llm'):
        llm_obj = llm.llm
    else:
        llm_obj = llm
        
    logger.debug(f"Using LLM type: {type(llm_obj)}")
    
    # Prepare chat messages if using chat format
    messages = None
    if as_chat:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
    
    # Try structured output first
    if hasattr(llm_obj, 'as_structured_llm') and callable(llm_obj.as_structured_llm):
        try:
            logger.debug("Attempting structured LLM call")
            structured_llm = llm_obj.as_structured_llm()
            return await structured_llm(prompt)
        except Exception as e:
            logger.warning(f"Structured LLM call failed: {e}")
    
    # Try async methods (with proper timeout handling)
    async def with_timeout(coro, timeout_seconds):
        if timeout_seconds:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        return await coro
    
    # Try async chat methods
    if messages and hasattr(llm_obj, 'achat') and callable(llm_obj.achat):
        try:
            logger.debug("Using achat method")
            return await with_timeout(llm_obj.achat(messages=messages), timeout)
        except Exception as e:
            logger.debug(f"achat method failed: {e}")
    
    # Try async completion methods
    if hasattr(llm_obj, 'acomplete') and callable(llm_obj.acomplete):
        try:
            logger.debug("Using acomplete method")
            return await with_timeout(llm_obj.acomplete(prompt), timeout)
        except Exception as e:
            logger.debug(f"acomplete method failed: {e}")
    
    # Try direct callable as coroutine
    if callable(llm_obj) and inspect.iscoroutinefunction(llm_obj):
        try:
            logger.debug("Using direct async callable")
            return await with_timeout(llm_obj(prompt), timeout)
        except Exception as e:
            logger.debug(f"Direct async call failed: {e}")
    
    # Try sync methods running in thread pool
    loop = asyncio.get_event_loop()
    
    # Try chat method
    if messages and hasattr(llm_obj, 'chat') and callable(llm_obj.chat):
        try:
            logger.debug("Using chat method (sync)")
            return await loop.run_in_executor(
                None, lambda: llm_obj.chat(messages=messages)
            )
        except Exception as e:
            logger.debug(f"chat method failed: {e}")
    
    # Try completion method
    if hasattr(llm_obj, 'complete') and callable(llm_obj.complete):
        try:
            logger.debug("Using complete method (sync)")
            return await loop.run_in_executor(
                None, lambda: llm_obj.complete(prompt)
            )
        except Exception as e:
            logger.debug(f"complete method failed: {e}")
    
    # Try direct callable
    if callable(llm_obj) and not inspect.iscoroutinefunction(llm_obj):
        try:
            logger.debug("Using direct callable (sync)")
            return await loop.run_in_executor(
                None, lambda: llm_obj(prompt)
            )
        except Exception as e:
            logger.debug(f"Direct sync call failed: {e}")
    
    raise ValueError(f"No compatible method found for LLM type: {type(llm_obj)}")

def extract_text_from_response(response: Any) -> str:
    """
    Extract text content from various LLM response formats.
    
    Args:
        response: LLM response in any format
        
    Returns:
        Extracted text as string
    """
    if response is None:
        return ""
    
    # Handle string response
    if isinstance(response, str):
        return response
    
    # Handle dictionary response
    if isinstance(response, dict):
        # Check common keys in order of likelihood
        for key in ["content", "text", "message", "completion", "response", "output"]:
            if key in response:
                content = response[key]
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict) and "content" in content:
                    return content["content"]
    
    # Handle object with message attribute (OpenAI/LlamaIndex style)
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return response.message.content
    
    # Handle object with direct text attributes
    for attr in ["text", "content", "completion", "output", "response"]:
        if hasattr(response, attr):
            content = getattr(response, attr)
            if isinstance(content, str):
                return content
    
    # Handle generations style response (HuggingFace)
    if hasattr(response, "generations") and len(response.generations) > 0:
        gen = response.generations[0]
        if hasattr(gen, "text"):
            return gen.text
    
    # Last resort: convert to string
    return str(response)
