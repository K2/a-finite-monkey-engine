"""
Utility for monitoring and diagnosing Ollama LLM integrations.
"""
import time
import asyncio
import os
from typing import Dict, Any, List, Optional
import httpx
from loguru import logger

class OllamaMonitor:
    """
    Monitor and diagnose Ollama LLM integrations.
    
    This class provides utilities for checking Ollama server status,
    monitoring model availability, and diagnosing common issues.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama monitor.
        
        Args:
            base_url: Base URL for Ollama API
        """
        self.base_url = base_url
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
    async def check_server_status(self) -> Dict[str, Any]:
        """
        Check if Ollama server is running and accessible.
        
        Returns:
            Dict containing status information
        """
        try:
            response = await self.http_client.get(f"{self.base_url}/api/version")
            if response.status_code == 200:
                return {
                    "status": "online",
                    "version": response.json().get("version", "unknown"),
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "error",
                    "code": response.status_code,
                    "error": response.text,
                    "timestamp": time.time()
                }
        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in the Ollama server.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = await self.http_client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return response.json().get("models", [])
            else:
                logger.error(f"Failed to list models: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def check_model_availability(self, model_name: str) -> Dict[str, Any]:
        """
        Check if a specific model is available in Ollama.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Dict containing model status information
        """
        models = await self.list_available_models()
        
        # Find the model
        for model in models:
            if model.get("name") == model_name:
                return {
                    "available": True,
                    "name": model_name,
                    "details": model,
                    "timestamp": time.time()
                }
        
        return {
            "available": False,
            "name": model_name,
            "timestamp": time.time()
        }
    
    async def perform_simple_query(self, model_name: str, prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """
        Perform a simple query to test model responsiveness.
        
        Args:
            model_name: Name of the model to test
            prompt: Simple prompt to test with
            
        Returns:
            Dict containing query results and timing information
        """
        start_time = time.time()
        
        try:
            response = await self.http_client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "elapsed_time": elapsed,
                    "response": result.get("response", ""),
                    "timestamp": time.time()
                }
            else:
                return {
                    "success": False,
                    "elapsed_time": elapsed,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "timestamp": time.time()
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "elapsed_time": end_time - start_time,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def diagnose_ollama_integration(self, model_name: str) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics on Ollama integration.
        
        Args:
            model_name: Name of the model to diagnose
            
        Returns:
            Dict containing diagnostic information
        """
        diagnostics = {
            "timestamp": time.time(),
            "model": model_name
        }
        
        # Check server status
        server_status = await self.check_server_status()
        diagnostics["server_status"] = server_status
        
        if server_status["status"] != "online":
            diagnostics["overall_status"] = "failed"
            diagnostics["failure_reason"] = "Ollama server is not accessible"
            return diagnostics
        
        # Check model availability
        model_status = await self.check_model_availability(model_name)
        diagnostics["model_status"] = model_status
        
        if not model_status["available"]:
            diagnostics["overall_status"] = "failed"
            diagnostics["failure_reason"] = f"Model {model_name} is not available"
            return diagnostics
        
        # Test simple query
        query_result = await self.perform_simple_query(model_name)
        diagnostics["query_test"] = query_result
        
        if not query_result["success"]:
            diagnostics["overall_status"] = "failed"
            diagnostics["failure_reason"] = "Failed to execute simple query"
            return diagnostics
        
        # Test structured output
        structured_prompt = """
        Return a JSON object with the following structure:
        {
          "greeting": "Hello there",
          "items": ["item1", "item2", "item3"],
          "count": 3
        }
        """
        
        structured_result = await self.perform_simple_query(model_name, structured_prompt)
        diagnostics["structured_test"] = structured_result
        
        # Overall assessment
        diagnostics["overall_status"] = "operational" if structured_result["success"] else "degraded"
        
        return diagnostics
    
    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()
