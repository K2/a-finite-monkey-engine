#!/usr/bin/env python
"""
Simple diagnostic tool to check Ollama timeout behavior.
"""
import sys
import os
import asyncio
import time
from loguru import logger

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finite_monkey.nodes_config import config
from finite_monkey.llm.llama_index_adapter import LlamaIndexAdapter
from llama_index.core.llms import ChatMessage, MessageRole

async def main():
    """Test Ollama with different timeout values"""
    model = config.ANALYSIS_MODEL
    provider = config.ANALYSIS_MODEL_PROVIDER
    
    if provider.lower() != "ollama":
        print(f"Current provider is {provider}, not Ollama. Exiting.")
        return
    
    print(f"Testing Ollama model: {model}")
    print(f"Current configured timeout: {config.REQUEST_TIMEOUT}s")
    
    # Test with different timeout values
    timeout_values = [10, 30, 60, 120]
    
    for timeout in timeout_values:
        print(f"\n\nTesting with timeout={timeout}s")
        
        # Create adapter with specific timeout
        adapter = LlamaIndexAdapter(
            model_name=model,
            provider=provider,
            request_timeout=timeout
        )
        
        # Create a simple message
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content="Say hello in exactly 5 words.")
        ]
        
        # Time the request
        start_time = time.time()
        try:
            print(f"Sending request to Ollama...")
            response = await adapter.achat(messages=messages)
            elapsed = time.time() - start_time
            
            print(f"✅ Success in {elapsed:.2f}s")
            if hasattr(response, 'message'):
                print(f"Response: {response.message.content}")
            else:
                print(f"Response: {response}")
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Timeout after {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Error after {elapsed:.2f}s: {str(e)}")
    
    print("\nTesting complete.")
    print("Recommendation: Set REQUEST_TIMEOUT in nodes_config.py to the minimum value that works reliably.")

if __name__ == "__main__":
    asyncio.run(main())
