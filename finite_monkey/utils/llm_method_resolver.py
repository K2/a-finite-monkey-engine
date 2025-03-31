"""
Utility for resolving methods on different LLM implementations.
Provides a consistent interface regardless of the underlying LLM library.
"""
import inspect
from typing import Any, Dict, List, Callable, Optional, Union
import asyncio
from loguru import logger

async def call_llm_async(
    llm: Any, 
    input_text: str, 
    as_chat: bool = True,
    system_prompt: Optional[str] = None
) -> Any:
    """
    Call an LLM with dynamic method resolution, handling different LLM interfaces.
    
    Args:
        llm: The LLM instance to call
        input_text: The input text or prompt
        as_chat: Whether to format as chat messages
        system_prompt: Optional system prompt for chat format
        
    Returns:
        The LLM response
    """
    # If this is a wrapped LLM adapter, try to get the actual LLM
    if hasattr(llm, 'llm'):
        llm_obj = llm.llm
    else:
        llm_obj = llm
    
    # Prepare chat messages if using chat format
    messages = None
    if as_chat:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input_text})
    
    # Try each method in sequence
    try:
        # 1. Try explicitly named async methods first
        
        # 1.1. Try structured methods if available (like LlamaIndex structured LLMs)
        if hasattr(llm_obj, 'as_structured_llm'):
            try:
                structured_llm = llm_obj.as_structured_llm()
                result = await structured_llm(input_text)
                return result
            except (AttributeError, TypeError) as e:
                logger.warning(f"Structured LLM call failed: {e}")
                # Continue to other methods
        
        # 1.2. Try async chat methods
        if messages and hasattr(llm_obj, 'achat') and callable(llm_obj.achat):
            try:
                return await llm_obj.achat(messages=messages)
            except Exception as e:
                logger.debug(f"achat method failed: {e}")
                # Continue to other methods
        
        # 1.3. Try async completion methods
        if hasattr(llm_obj, 'acomplete') and callable(llm_obj.acomplete):
            try:
                return await llm_obj.acomplete(input_text)
            except Exception as e:
                logger.debug(f"acomplete method failed: {e}")
                # Continue to other methods
        
        if hasattr(llm_obj, 'agenerate') and callable(llm_obj.agenerate):
            try:
                return await llm_obj.agenerate(input_text)
            except Exception as e:
                logger.debug(f"agenerate method failed: {e}")
                # Continue to other methods
        
        # 2. Check if the object is callable directly
        if callable(llm_obj):
            if inspect.iscoroutinefunction(llm_obj):
                return await llm_obj(input_text)
            else:
                # Run synchronous function in executor to avoid blocking
                return await asyncio.get_event_loop().run_in_executor(
                    None, llm_obj, input_text
                )
        
        # 3. Try standard methods as fallbacks
        if messages and hasattr(llm_obj, 'chat') and callable(llm_obj.chat):
            chat_fn = llm_obj.chat
            if inspect.iscoroutinefunction(chat_fn):
                return await chat_fn(messages=messages)
            else:
                # Run synchronous function in executor
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: chat_fn(messages=messages)
                )
        
        if hasattr(llm_obj, 'complete') and callable(llm_obj.complete):
            complete_fn = llm_obj.complete
            if inspect.iscoroutinefunction(complete_fn):
                return await complete_fn(input_text)
            else:
                # Run synchronous function in executor
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: complete_fn(input_text)
                )
        
        if hasattr(llm_obj, 'generate') and callable(llm_obj.generate):
            generate_fn = llm_obj.generate
            if inspect.iscoroutinefunction(generate_fn):
                return await generate_fn(input_text)
            else:
                # Run synchronous function in executor
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: generate_fn(input_text)
                )
        
        # If we got here, no suitable method was found
        raise AttributeError(f"No suitable method found on LLM: {type(llm_obj)}")
    
    except Exception as e:
        logger.error(f"All LLM call methods failed: {e}")
        raise

def extract_content_from_response(response: Any) -> str:
    """
    Extract text content from various LLM response formats.
    
    Args:
        response: The LLM response object
        
    Returns:
        The extracted text content as a string
    """
    if response is None:
        return ""
    
    # String response
    if isinstance(response, str):
        return response
    
    # Dictionary response
    if isinstance(response, dict):
        for key in ["text", "content", "response", "result", "output", "message", "completion"]:
            if key in response:
                content = response[key]
                if isinstance(content, str):
                    return content
        
        # If response has a 'message' key with a dict value that has a 'content' key
        if "message" in response and isinstance(response["message"], dict):
            if "content" in response["message"]:
                return response["message"]["content"]
    
    # LlamaIndex / OpenAI response objects
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return response.message.content
    
    # Handle completion response objects
    if hasattr(response, "text"):
        return response.text
    
    if hasattr(response, "content"):
        return response.content
    
    if hasattr(response, "completion"):
        return response.completion
    
    # Handle generation response objects
    if hasattr(response, "generations"):
        generations = response.generations
        if generations and len(generations) > 0:
            gen = generations[0]
            if hasattr(gen, "text"):
                return gen.text
            if isinstance(gen, str):
                return gen
            if hasattr(gen, "message") and hasattr(gen.message, "content"):
                return gen.message.content
    
    # Last resort - convert to string
    return str(response)
