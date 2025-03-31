#!/usr/bin/env python3
"""
Test script to verify Ollama connectivity.
"""
import os
import sys
import asyncio
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

async def test_direct_api():
    """Test direct API connection to Ollama."""
    import aiohttp
    
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    logger.info(f"Testing Ollama API connection to {ollama_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # First, test the version endpoint
            async with session.get(f"{ollama_url}/api/version") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.success(f"✅ Connected to Ollama version: {data.get('version')}")
                else:
                    logger.error(f"❌ Failed to connect to Ollama: {response.status}")
                    return False
            
            # Now test a simple chat completion
            model = os.environ.get("PROMPT_MODEL", "gemma:2b")
            logger.info(f"Testing chat completion with model: {model}")
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, are you working?"}
                ]
            }
            
            async with session.post(
                f"{ollama_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "message" in result and "content" in result["message"]:
                        logger.success(f"✅ Ollama chat completion successful")
                        logger.info(f"Response: {result['message']['content']}")
                        return True
                    else:
                        logger.error(f"❌ Unexpected response format: {result}")
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Chat completion failed: {response.status} - {error_text}")
                    return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Ollama: {e}")
        return False

async def test_python_client():
    """Test the official Python client."""
    try:
        import ollama
        
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        if ollama_url != "http://localhost:11434":
            ollama.set_host(ollama_url)
        
        logger.info(f"Testing Ollama Python client connecting to {ollama_url}")
        
        # Get models list
        try:
            models = ollama.list()
            logger.success(f"✅ Connected to Ollama, available models: {[m['name'] for m in models['models']]}")
        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            return False
        
        # Test chat completion
        model = os.environ.get("PROMPT_MODEL", "gemma:2b")
        logger.info(f"Testing chat completion with model: {model}")
        
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, are you working?"}
                ]
            )
            
            if response and "message" in response and "content" in response["message"]:
                logger.success(f"✅ Ollama Python client chat completion successful")
                logger.info(f"Response: {response['message']['content']}")
                return True
            else:
                logger.error(f"❌ Unexpected response format: {response}")
                return False
        except Exception as e:
            logger.error(f"❌ Chat completion failed: {e}")
            return False
    except ImportError:
        logger.warning("⚠️ Ollama Python client not installed, skipping test")
        return False
    except Exception as e:
        logger.error(f"❌ Error with Ollama Python client: {e}")
        return False

async def main():
    """Run all tests."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("Testing Ollama connectivity...")
    
    # Test both methods
    client_result = await test_python_client()
    api_result = await test_direct_api()
    
    if client_result:
        logger.info("Python client test: ✅ PASSED")
    else:
        logger.info("Python client test: ❌ FAILED")
    
    if api_result:
        logger.info("Direct API test: ✅ PASSED")
    else:
        logger.info("Direct API test: ❌ FAILED")
    
    if client_result or api_result:
        logger.info("✅ At least one method of connecting to Ollama works")
        return 0
    else:
        logger.error("❌ All methods of connecting to Ollama failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
