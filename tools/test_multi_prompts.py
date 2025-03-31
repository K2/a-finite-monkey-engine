#!/usr/bin/env python3
"""
Test script to verify multi-LLM prompt generation.
"""
import asyncio
import sys
from loguru import logger
from vector_store_util import SimpleVectorStore
from vector_store_prompts import PromptGenerator

async def test_multi_prompts():
    """Test multi-LLM prompt generation."""
    # Sample document with code
    test_doc = {
        "text": """
# Integer Overflow Example

```solidity
function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    return c;
}
```

This function doesn't check for overflow conditions.
        """,
        "metadata": {
            "title": "Integer Overflow Example",
            "category": "security",
            "source": "test"
        }
    }
    
    # Create prompt generator with multi-LLM prompts enabled
    generator = PromptGenerator(
        generate_prompts=True,
        use_ollama_for_prompts=True,
        prompt_model="gemma:2b",  # Use available model
        multi_llm_prompts=True
    )
    
    # Generate multi-LLM prompts
    logger.info("Generating multi-LLM prompts...")
    prompts = await generator.generate_multi_llm_prompts(test_doc)
    
    # Print the different prompt types
    print("\n=== MULTI-LLM PROMPTS ===")
    for prompt_type, prompt in prompts.items():
        # Skip list items like additional_security
        if isinstance(prompt, str):
            print(f"\n--- {prompt_type.upper()} ---")
            print(prompt)
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_multi_prompts())
        print("\nMulti-LLM prompt test completed successfully!")
    except Exception as e:
        logger.error(f"Error testing multi-LLM prompts: {e}")
        sys.exit(1)
