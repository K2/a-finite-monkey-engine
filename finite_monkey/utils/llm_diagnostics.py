"""
Utility functions for diagnosing and troubleshooting LLM-related issues.
"""
import asyncio
import time
import json
from typing import Any, Dict, List, Optional
from loguru import logger

async def test_llm_connection(llm_adapter, message: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Test the connection to an LLM service with diagnostic information.
    
    Args:
        llm_adapter: The LLM adapter to test
        message: A simple message to send to the LLM
        timeout: Timeout in seconds
        
    Returns:
        Dictionary with diagnostic information
    """
    if not llm_adapter or not hasattr(llm_adapter, 'llm'):
        return {
            "success": False,
            "error": "No valid LLM adapter provided",
            "timestamp": time.time(),
            "elapsed_time": 0
        }
    
    start_time = time.time()
    results = {
        "success": False,
        "timestamp": start_time,
        "elapsed_time": 0,
        "provider": getattr(llm_adapter.llm, "provider", "unknown"),
        "model": getattr(llm_adapter.llm, "model", getattr(llm_adapter.llm, "model_name", "unknown")),
        "timeout_setting": getattr(llm_adapter.llm, "timeout", timeout)
    }
    
    try:
        # Simple test message
        from llama_index.core.llms import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content=message)
        ]
        
        # Create a timeout for the entire operation
        response_task = asyncio.create_task(llm_adapter.achat(messages=messages))
        response = await asyncio.wait_for(response_task, timeout=timeout)
        
        # Calculate timing
        end_time = time.time()
        results["elapsed_time"] = end_time - start_time
        
        # Extract response content
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            content = response.message.content
        else:
            content = str(response)
            
        results["success"] = True
        results["response"] = content[:500]  # First 500 chars
        
        return results
        
    except asyncio.TimeoutError:
        end_time = time.time()
        results["elapsed_time"] = end_time - start_time
        results["error"] = "Request timed out"
        return results
        
    except Exception as e:
        end_time = time.time()
        results["elapsed_time"] = end_time - start_time
        results["error"] = f"Error: {str(e)}"
        return results

async def diagnose_llm_issues(llm_adapter, detailed: bool = False) -> Dict[str, Any]:
    """
    Run a series of diagnostic tests on an LLM adapter to identify issues.
    
    Args:
        llm_adapter: The LLM adapter to test
        detailed: Whether to run detailed tests
        
    Returns:
        Dictionary with detailed diagnostic information
    """
    logger.info("Starting LLM diagnostics")
    
    # Basic connectivity test
    basic_test = await test_llm_connection(
        llm_adapter, 
        "Hello, this is a basic connectivity test. Please respond with 'Connection successful'.",
        timeout=30
    )
    
    diagnostics = {
        "basic_connectivity": basic_test,
        "timestamp": time.time(),
        "adapter_type": type(llm_adapter).__name__,
        "configurations": {
            "request_timeout": getattr(llm_adapter.llm, "request_timeout", "unknown"),
            "api_key_set": bool(getattr(llm_adapter.llm, "api_key", None)) 
                           or "API_KEY" in os.environ
                           or "OPENAI_API_KEY" in os.environ
        }
    }
    
    # If basic test failed, don't run more tests
    if not basic_test["success"]:
        logger.warning("Basic connectivity test failed, skipping detailed tests")
        return diagnostics
    
    # Only run detailed tests if requested
    if detailed:
        # Test structured output
        structured_test = await test_structured_output(llm_adapter)
        diagnostics["structured_output"] = structured_test
        
        # Test response time with larger input
        large_input = "A" * 5000  # 5000 character input
        large_input_test = await test_llm_connection(
            llm_adapter,
            f"Summarize this text in one sentence: {large_input}",
            timeout=60
        )
        diagnostics["large_input_handling"] = large_input_test
    
    logger.info(f"LLM diagnostics complete. Basic connectivity: {basic_test['success']}")
    return diagnostics

async def test_structured_output(llm_adapter) -> Dict[str, Any]:
    """Test the LLM's ability to produce structured JSON output"""
    from llama_index.core.llms import ChatMessage, MessageRole
    
    start_time = time.time()
    results = {
        "success": False,
        "timestamp": start_time,
        "elapsed_time": 0
    }
    
    try:
        # Test message requesting JSON
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant that always responds with valid JSON."),
            ChatMessage(role=MessageRole.USER, content="Return a JSON object with keys 'status' set to 'success' and 'message' set to 'JSON generation is working'.")
        ]
        
        # Set timeout
        response_task = asyncio.create_task(llm_adapter.achat(messages=messages))
        response = await asyncio.wait_for(response_task, timeout=30)
        
        # Calculate timing
        end_time = time.time()
        results["elapsed_time"] = end_time - start_time
        
        # Extract response content
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            content = response.message.content
        else:
            content = str(response)
        
        # Try to parse as JSON
        try:
            json_data = json.loads(content)
            results["success"] = True
            results["json_data"] = json_data
        except json.JSONDecodeError:
            results["success"] = False
            results["error"] = "Response is not valid JSON"
            results["raw_response"] = content[:500]  # First 500 chars
        
        return results
        
    except Exception as e:
        end_time = time.time()
        results["elapsed_time"] = end_time - start_time
        results["error"] = f"Error: {str(e)}"
        return results
