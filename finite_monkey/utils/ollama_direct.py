"""
Direct Ollama API utilities that bypass LlamaIndex for diagnostics and testing.
This helps identify if the issue is with LlamaIndex or the Ollama connection itself.
"""
import httpx
import json
from typing import Dict, Any, Optional, List
from loguru import logger

class DirectOllamaClient:
    """
    A minimal client for direct Ollama API calls, bypassing LlamaIndex.
    Used for diagnosing connectivity issues.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for Ollama API, defaults to localhost:11434
        """
        self.base_url = base_url
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"Initialized direct Ollama client with base_url={base_url}")
    
    async def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Send a direct generation request to Ollama API.
        
        Args:
            model: Model name (e.g., "llama2", "dolphin3:8b-llama3.1-q8_0")
            prompt: Text prompt to send
            **kwargs: Additional parameters for Ollama
            
        Returns:
            Dict containing the Ollama response
        """
        logger.debug(f"Making direct Ollama request to model={model}")
        
        # Modify the prompt to explicitly request JSON
        json_prompt = f"""
You are a helpful assistant that always responds with valid JSON.

{prompt}

Response (in JSON format only):"""
        
        # Build the request payload
        payload = {
            "model": model,
            "prompt": json_prompt,
            "stream": False,
            "format": "json",  # Request JSON format if supported by the Ollama version
            **kwargs
        }
        
        try:
            # Log the actual network request for debugging
            logger.debug(f"Sending HTTP POST to {self.base_url}/api/generate")
            
            # Make the API call
            response = await self.http_client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
            
            # Get the response text and parse it as JSON
            try:
                # Parse raw response text
                result = json.loads(response.text)
                # Log raw response for debugging double encoding
                logger.debug(f"Raw Ollama response type: {type(result.get('response'))}")
                logger.debug(f"Raw response sample: {result.get('response', '')[:50]}")
                
                # Check if response is already a JSON string that needs parsing
                response_text = result.get('response', '')
                if isinstance(response_text, str) and response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                    try:
                        # Try to parse it as JSON to see if it's already JSON
                        maybe_json = json.loads(response_text)
                        # If successful, flag this so caller knows response is pre-parsed
                        if isinstance(maybe_json, dict):
                            result['response_is_json'] = True
                            logger.debug("Detected JSON in response text")
                    except json.JSONDecodeError:
                        # Not valid JSON, just a string that happens to start/end with braces
                        pass
                    
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Ollama response as JSON: {e}")
                return {"error": f"JSON parsing error: {e}", "raw_response": response.text[:200]}
            
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Ollama: {e}")
            return {"error": f"Connection error: {str(e)}", "status": "connection_failed"}
            
        except Exception as e:
            logger.exception(f"Error in direct Ollama call: {e}")
            return {"error": str(e), "status": "error"}
    
    async def generate_json(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Send a direct generation request to Ollama API optimized for JSON responses.
        
        Args:
            model: Model name (e.g., "llama2", "dolphin3:8b-llama3.1-q8_0")
            prompt: Text prompt to send
            **kwargs: Additional parameters for Ollama
            
        Returns:
            Dict containing the Ollama response
        """
        logger.debug(f"Making direct Ollama JSON request to model={model}")
        
        # Modify the prompt to explicitly request JSON
        json_prompt = f"""
You are a helpful assistant that always responds with valid JSON.

{prompt}

Response (in JSON format only):"""
        
        # Build the request payload
        payload = {
            "model": model,
            "prompt": json_prompt,
            "stream": False,
            "format": "json",  # Request JSON format if supported by the Ollama version
            **kwargs
        }
        
        try:
            # Log the actual network request for debugging
            logger.debug(f"Sending HTTP POST to {self.base_url}/api/generate")
            
            # Make the API call
            response = await self.http_client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
            
            # Get the response text and parse it as JSON
            try:
                # Parse raw response text
                result = json.loads(response.text)
                logger.debug(f"Received Ollama response: {result.get('response', '')[:50]}...")
                
                # Try to parse the response as JSON if it looks like JSON
                response_text = result.get('response', '')
                if isinstance(response_text, str) and response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                    try:
                        parsed_json = json.loads(response_text)
                        result['parsed_json'] = parsed_json
                        result['response_is_json'] = True
                    except json.JSONDecodeError:
                        pass
                    
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Ollama response as JSON: {e}")
                return {"error": f"JSON parsing error: {e}", "raw_response": response.text[:200]}
                
        except Exception as e:
            logger.error(f"Error in direct Ollama call: {e}")
            return {"error": str(e), "status": "error"}
    
    async def list_models(self) -> Dict[str, Any]:
        """
        List available models in the Ollama server.
        
        Returns:
            Dict containing model information
        """
        logger.debug("Listing available Ollama models")
        
        try:
            response = await self.http_client.get(f"{self.base_url}/api/tags")
            
            # Check for HTTP errors
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
            
            # Parse the response as JSON directly from text
            try:
                result = json.loads(response.text)
                logger.debug(f"Found {len(result.get('models', []))} Ollama models")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse model list response as JSON: {e}")
                return {"error": f"JSON parsing error: {e}", "raw_response": response.text[:200]}
            
        except Exception as e:
            logger.exception(f"Error listing Ollama models: {e}")
            return {"error": str(e), "status": "error"}
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check if Ollama server is running and responsive.
        
        Returns:
            Dict containing health status
        """
        logger.debug("Checking Ollama server health")
        
        try:
            response = await self.http_client.get(f"{self.base_url}/api/version")
            
            # Check for HTTP errors
            if response.status_code != 200:
                logger.error(f"Ollama health check failed: {response.status_code} - {response.text}")
                return {
                    "status": "unhealthy", 
                    "error": response.text, 
                    "status_code": response.status_code
                }
            
            # Parse the response as JSON directly from text
            try:
                result = json.loads(response.text)
                logger.debug(f"Ollama health check successful. Version: {result.get('version')}")
                return {"status": "healthy", "version": result.get("version")}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse health check response as JSON: {e}")
                return {"status": "unhealthy", "error": f"JSON parsing error: {e}"}
            
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama server: {e}")
            return {
                "status": "unreachable",
                "error": f"Connection error: {str(e)}"
            }
            
        except Exception as e:
            logger.exception(f"Error checking Ollama health: {e}")
            return {"status": "error", "error": str(e)}
    
    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()
