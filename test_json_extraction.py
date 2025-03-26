#!/usr/bin/env python3
"""
Test script for JSON extraction from LLM responses
"""

import asyncio
import json
import sys
from loguru import logger
from finite_monkey.llm.llama_index_adapter import LlamaIndexAdapter
from finite_monkey.nodes_config import config

# Configure verbose logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")

async def test_json_extraction(model_name=None, provider=None):
    """Test JSON extraction with a simple schema"""
    
    # Create adapter with provided or default model
    adapter = LlamaIndexAdapter(
        provider=provider or "ollama",
        model_name=model_name or config.WORKFLOW_MODEL,
    )
    
    # Simple test schema
    test_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "number"},
            "items": {"type": "array", "items": {"type": "string"}}
        }
    }
    
    # Test prompt
    test_prompt = "Generate a fictional inventory with a name, count, and list of items."
    
    print(f"Testing JSON extraction with model: {adapter.model_name}")
    print(f"Provider: {adapter.provider}")
    print(f"Schema: {json.dumps(test_schema, indent=2)}")
    
    # Submit prompt and get future
    future = await adapter.submit_json_prompt(test_prompt, test_schema)
    
    try:
        # Wait for result with timeout
        result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=300.0)
        
        print("\nResult:")
        print(json.dumps(result, indent=2))
        
        # Check if error occurred
        if "error" in result:
            print("\nJSON extraction failed!")
        else:
            print("\nJSON extraction succeeded!")
            
    except asyncio.TimeoutError:
        print("\nRequest timed out!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    # Get model name from command line if provided
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    provider = sys.argv[2] if len(sys.argv) > 2 else None
    
    asyncio.run(test_json_extraction(model_name, provider))
