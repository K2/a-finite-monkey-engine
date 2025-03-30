import asyncio
from typing import Optional, Dict, Any
import httpx
import logging

from loguru import logger
from finite_monkey.nodes_config import config

class AsyncOllamaClient:
    """Client for the Ollama API."""
    
    def __init__(
        self, 
        model: str, 
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        # Make sure model is never empty
        if not model or model.strip() == '':
            model = config.DEFAULT_MODEL
            logging.warning(f"Empty model provided, using default model: {model}")
        
        self.model = model
        self.base_url = base_url if base_url is not None else config.OPENAI_API_BASE
        self.timeout = timeout if timeout is not None else config.REQUEST_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else config.MAX_RETRIES
        
        # Initialize the HTTP client session with explicit timeout
        timeout_obj = httpx.Timeout(timeout or config.REQUEST_TIMEOUT)
        self.session = httpx.AsyncClient(timeout=timeout_obj)
        logger.debug(f"Initialized AsyncOllamaClient with model: {self.model}")
    
    async def aclose(self):
        """Properly close the async HTTP client"""
        if hasattr(self, 'session') and self.session:
            await self.session.aclose()
            logger.debug("Closed AsyncOllamaClient session")

    def __del__(self):
        """Standard destructor - cannot be async but will try to clean up"""
        if hasattr(self, 'session') and self.session:
            import asyncio
            try:
                # Try to get an event loop and run the close task
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.aclose())
                else:
                    # If no loop is running, we can't do much here
                    pass
            except Exception:
                # Just pass if we can't access the event loop
                pass
        
    async def acomplete(self, prompt: str) -> Dict[str, Any]:
        """
        Asynchronously complete a prompt using Ollama.
        
        Args:
            prompt: The prompt to complete
            
        Returns:
            The completion response
        """
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False  # Explicitly disable streaming to ensure proper JSON response
        }
        
        try:
            # Don't use async with since some httpx versions might not support it properly
            response = await self.session.post(url, json=data)
            
            if response.status_code != 200:
                error_text = await response.aread()
                error_text = error_text.decode('utf-8') if isinstance(error_text, bytes) else str(error_text)
                logger.error(f"Ollama API error: {response.status_code} - {error_text}")
                return {"error": error_text}
                
            # Parse the response with explicit error handling
            try:
                result = response.json()
                return {
                    "text": result.get("response", ""),
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    }
                }
            except Exception as e:
                logger.error(f"Failed to parse Ollama response: {e}")
                return {"error": f"Failed to parse response: {str(e)}"}
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return {"error": str(e)}