"""
Logging middleware for LLM interactions

This module provides comprehensive logging capabilities for all interactions
with LLM services, capturing requests, responses, timing, and errors.
"""

import time
import json
import uuid
import asyncio
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional, Callable
from loguru import logger

class LLMLogger:
    """Logger specifically for LLM interactions"""
    
    @staticmethod
    def setup():
        """Set up LLM-specific logger"""
        # Add a specialized sink for LLM interactions
        logger.add(
            "logs/llm_interactions.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention=10,
            filter=lambda record: record["extra"].get("llm_log", False)
        )
        
        # Add a JSON sink for structured logging
        logger.add(
            "logs/llm_interactions.jsonl",
            serialize=True,  # Output as JSON
            format="{message}",
            level="DEBUG",
            rotation="10 MB",
            retention=10,
            filter=lambda record: record["extra"].get("llm_log", False)
        )
    
    @staticmethod
    def log_interaction(
        request_id: str,
        provider: str,
        model: str,
        endpoint: str,
        request_data: Dict[str, Any],
        response_data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a complete LLM interaction"""
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "provider": provider,
            "model": model,
            "endpoint": endpoint,
            "request": request_data,
            "duration_ms": duration_ms,
            "metadata": metadata or {}
        }
        
        if response_data:
            log_entry["response"] = response_data
            
        if error:
            log_entry["error"] = str(error)
            log_entry["error_type"] = type(error).__name__
        
        # Log structured entry
        logger.bind(llm_log=True).info(json.dumps(log_entry))
        
        # Also log a human-readable summary
        if error:
            logger.bind(llm_log=True).error(
                f"LLM {provider}/{model} request {request_id} failed after {duration_ms:.2f}ms: {str(error)}"
            )
        else:
            logger.bind(llm_log=True).success(
                f"LLM {provider}/{model} request {request_id} completed in {duration_ms:.2f}ms"
            )

def log_llm_call(func):
    """Decorator to log LLM API calls"""
    @wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        provider = getattr(self, "provider", "unknown")
        model = getattr(self, "model_name", "unknown")
        
        # Extract function name as endpoint
        endpoint = func.__name__
        
        # Log the request
        request_data = {
            "args": args,
            "kwargs": {k: v for k, v in kwargs.items() if k != "api_key"}  # Don't log API keys
        }
        
        logger.bind(llm_log=True).debug(
            f"LLM request {request_id} to {provider}/{model}: {func.__name__}"
        )
        
        # Call the function
        try:
            response = await func(self, *args, **kwargs)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log success
            response_data = {
                "type": type(response).__name__,
                # Don't log the complete response as it might be very large
                "summary": str(response)[:200] + ("..." if len(str(response)) > 200 else "")
            }
            
            LLMLogger.log_interaction(
                request_id=request_id,
                provider=provider,
                model=model,
                endpoint=endpoint,
                request_data=request_data,
                response_data=response_data,
                duration_ms=duration_ms
            )
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            LLMLogger.log_interaction(
                request_id=request_id,
                provider=provider,
                model=model,
                endpoint=endpoint,
                request_data=request_data,
                error=e,
                duration_ms=duration_ms
            )
            
            # Re-raise the exception
            raise
    
    return async_wrapper