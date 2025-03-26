import asyncio
from typing import Optional, Dict, Any
import httpx
import logging

class AsyncOllamaClient:
    """Client for the Ollama API."""
    
    def __init__(
        self, 
        model: str, 
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        # Get config for defaults
        from ..nodes_config import nodes_config
        config = nodes_config()
        
        # Make sure model is never empty
        if not model or model.strip() == '':
            model = config.DEFAULT_MODEL
            logging.warning(f"Empty model provided, using default model: {model}")
        
        self.model = model
        self.base_url = base_url if base_url is not None else config.OPENAI_API_BASE
        self.timeout = timeout if timeout is not None else config.REQUEST_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else config.MAX_RETRIES
        
    async def acomplete(self, prompt: str, **kwargs) -> str:
        """
        Send an asynchronous completion request to Ollama.
        
        Args:
            prompt: The prompt to complete
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The generated completion text
        """
        # Verify model is not empty before sending request
        if not self.model or self.model.strip() == '':
            from ..nodes_config import nodes_config
            self.model = nodes_config().DEFAULT_MODEL
            logging.warning(f"Empty model detected during request, using default: {self.model}")
            
        url = f"{self.base_url}/api/generate"
        
        # Set up default parameters
        params = {
            "model": self.model,
            "prompt": prompt,
            **kwargs
        }
        
        # Set up client with appropriate timeout
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(url, json=params)
                    response.raise_for_status()
                    result = response.json()
                    return result.get("response", "")
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    if attempt == self.max_retries - 1:
                        # Last attempt failed
                        raise e
                    # Wait before retrying (exponential backoff)
                    await asyncio.sleep(2 ** attempt)