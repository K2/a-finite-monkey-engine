#!/usr/bin/env python
"""
Direct test script for Ollama API connectivity.
This bypasses LlamaIndex and Guidance to test if Ollama is reachable.
"""
import sys
import os
import asyncio
import argparse
from loguru import logger

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finite_monkey.utils.ollama_direct import DirectOllamaClient
from finite_monkey.nodes_config import config

async def main():
    parser = argparse.ArgumentParser(description="Test Ollama connectivity directly")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--model", default=None, help="Model to test (default: from config)")
    args = parser.parse_args()
    
    # Use configured model if not specified
    model = args.model or config.ANALYSIS_MODEL
    
    print(f"=== Ollama Direct API Test ===")
    print(f"URL: {args.url}")
    print(f"Model: {model}")
    print("=============================\n")
    
    # Create a direct client
    client = DirectOllamaClient(base_url=args.url)
    
    try:
        # Step 1: Check server health
        print("Testing Ollama server health...")
        health = await client.check_health()
        
        if health.get("status") == "healthy":
            print(f"✅ Ollama server is healthy (version: {health.get('version')})")
        else:
            print(f"❌ Ollama server is not healthy: {health}")
            print("\nPossible fixes:")
            print("- Ensure Ollama is running with: ollama serve")
            print("- Check network connectivity to the Ollama server")
            print("- Verify the URL is correct (e.g., http://localhost:11434)")
            return
        
        # Step 2: List available models
        print("\nListing available Ollama models...")
        models_result = await client.list_models()
        
        if "error" in models_result:
            print(f"❌ Failed to list models: {models_result['error']}")
            return
        
        models = models_result.get("models", [])
        print(f"Found {len(models)} models:")
        for i, model_info in enumerate(models, 1):
            model_name = model_info.get("name")
            print(f"{i}. {model_name}")
        
        # Step 3: Check if our target model exists
        model_exists = any(m.get("name") == model for m in models)
        if not model_exists:
            print(f"\n❌ Model '{model}' not found in available models!")
            print("\nPossible fixes:")
            print(f"- Pull the model with: ollama pull {model}")
            print("- Check for typos in model name")
            print(f"- Choose from available models listed above")
            return
        else:
            print(f"\n✅ Model '{model}' is available")
        
        # Step 4: Test generation
        print("\nTesting generation with a simple prompt...")
        result = await client.generate(
            model=model,
            prompt="Say hello in 5 words or less."
        )
        
        if "error" in result:
            print(f"❌ Generation failed: {result['error']}")
            return
        
        response = result.get("response", "")
        print(f"\nResponse: {response}")
        print("\n✅ Generation successful!")
        
        # Step 5: Test JSON generation (similar to what we need for analysis)
        print("\nTesting structured JSON generation...")
        json_result = await client.generate(
            model=model,
            prompt="""Return a valid JSON object with the following structure:
{
  "flows": [
    {
      "name": "Example flow",
      "description": "A sample flow for testing",
      "steps": ["Step 1", "Step 2"]
    }
  ],
  "summary": "This is a test"
}

Response:"""
        )
        
        if "error" in json_result:
            print(f"❌ JSON generation failed: {json_result['error']}")
            return
        
        json_response = json_result.get("response", "")
        print(f"\nJSON Response: {json_response[:200]}...")
        
        # Try to parse as JSON
        import re
        import json as json_module
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'(\{.*\})', json_response, re.DOTALL)
            if json_match:
                parsed = json_module.loads(json_match.group(1))
                print("\n✅ Successfully parsed response as JSON!")
                print(f"Found {len(parsed.get('flows', []))} flows in response")
            else:
                print("\n❌ Could not find JSON in response!")
        except json_module.JSONDecodeError as e:
            print(f"\n❌ Failed to parse response as JSON: {e}")
        
        print("\n=== Test Summary ===")
        print("✅ Ollama server is healthy")
        print(f"✅ Model '{model}' is available")
        print("✅ Basic generation works")
        
        # Final advice
        print("\nIf your application isn't communicating with Ollama but this test works:")
        print("1. Check for networking or proxy issues in your application")
        print("2. Verify the Ollama URL in your application config")
        print("3. Look for logger.error messages in your application logs")
        
    finally:
        # Clean up
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
