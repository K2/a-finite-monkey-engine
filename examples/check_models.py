#\!/usr/bin/env python3
"""
Simple script to check available Ollama models
"""

import os
import sys
import asyncio
import httpx

async def check_ollama_models():
    """Check which models are available in Ollama"""
    print("Checking available Ollama models...")
    
    # Use a short timeout to avoid hanging
    timeout = httpx.Timeout(5.0, connect=2.0)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            if "models" in data:
                models = data["models"]
                if models:
                    print(f"Found {len(models)} models:")
                    for model in models:
                        print(f"  - {model.get('name')}")
                else:
                    print("No models are available.")
            else:
                print("Invalid response format from Ollama API")
    except httpx.TimeoutException:
        print("Connection to Ollama timed out - make sure Ollama is running")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code}")
    except httpx.RequestError:
        print("Could not connect to Ollama - make sure it's running")
    except Exception as e:
        print(f"Error checking Ollama models: {str(e)}")

if __name__ == "__main__":
    asyncio.run(check_ollama_models())
