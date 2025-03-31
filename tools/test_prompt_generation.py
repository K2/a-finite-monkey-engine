#!/usr/bin/env python3
"""
Test script for the prompt generation functionality.

This script tests the PromptGenerator with sample GitHub issue data to ensure
proper prompt generation for security vulnerability analysis.
"""
import asyncio
import sys
import json
from pathlib import Path
from loguru import logger
from vector_store_prompts import PromptGenerator
from vector_store_util import SimpleVectorStore

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Sample GitHub issue data for testing
SAMPLE_ISSUE = {
    "text": """
# Integer Overflow in SafeMath Contract

The contract has a potential issue with integer arithmetic.

```solidity
function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    require(c >= a, "SafeMath: addition overflow");
    return c;
}

function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b <= a, "SafeMath: subtraction overflow");
    uint256 c = a - b;
    return c;
}
```

When large values are used, this could lead to unexpected behavior.
""",
    "metadata": {
        "title": "Integer Overflow in SafeMath Contract",
        "category": "security",
        "source": "github",
        "url": "https://github.com/example/contracts/issues/123"
    }
}

async def test_prompt_generation():
    """Test the prompt generator with sample issue data."""
    logger.info("Testing basic prompt generation...")
    
    generator = PromptGenerator(
        generate_prompts=True,
        use_ollama_for_prompts=True,
        prompt_model="gemma:2b",
        ollama_url="http://localhost:11434"
    )
    
    # Generate basic prompt
    prompt = await generator.generate_prompt(SAMPLE_ISSUE)
    logger.info(f"Generated prompt: {prompt}")
    
    # Test non-code text extraction
    code_segments = generator._extract_code_segments(SAMPLE_ISSUE["text"])
    non_code = generator._extract_non_code_text(SAMPLE_ISSUE["text"], code_segments)
    logger.info(f"Extracted non-code text: {non_code}")
    
    # Test issue context extraction
    context = generator._extract_issue_context(
        non_code, 
        SAMPLE_ISSUE["metadata"]["title"]
    )
    logger.info(f"Extracted issue context: {context}")
    
    # Test multi-LLM prompts
    generator.multi_llm_prompts = True
    multi_prompts = await generator.generate_multi_llm_prompts(SAMPLE_ISSUE)
    
    print("\nMulti-LLM Prompts:")
    for llm_type, prompt in multi_prompts.items():
        if isinstance(prompt, str):
            print(f"- {llm_type}: {prompt}")
    
    # Create prompt generator directly
    prompt_gen = PromptGenerator(
        generate_prompts=True,
        use_ollama_for_prompts=True,
        prompt_model="gemma:2b",  # Use default or specify a different model
        ollama_url="http://localhost:11434",
        multi_llm_prompts=False
    )
    
    # Create a test document
    test_doc = {
        "text": "function transfer(address to, uint256 amount) external {\n  balances[msg.sender] -= amount;\n  balances[to] += amount;\n}",
        "metadata": {
            "title": "Test Smart Contract",
            "category": "solidity",
            "source": "test"
        }
    }
    
    # Try to generate a prompt
    try:
        prompt = await prompt_gen.generate_prompt(test_doc)
        logger.info(f"Generated prompt: {prompt}")
        assert prompt, "Prompt should not be empty"
        logger.success("✅ Prompt generation successful")
    except Exception as e:
        logger.error(f"❌ Error generating prompt: {e}")
    
    # Test through vector store
    try:
        store = SimpleVectorStore(
            collection_name="test_prompt_gen",
            embedding_model="local"
        )
        
        # Add the test document
        success = await store.add_documents([test_doc])
        
        # Check if prompt was added to metadata
        if success and test_doc.get('metadata', {}).get('prompt'):
            logger.success("✅ Prompt successfully added to document metadata")
            logger.info(f"Prompt: {test_doc['metadata']['prompt']}")
        else:
            logger.error("❌ Prompt was not added to document metadata")
    except Exception as e:
        logger.error(f"❌ Error testing vector store prompt generation: {e}")

    return True

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    try:
        asyncio.run(test_prompt_generation())
        logger.success("Prompt generation test completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Prompt generation test failed: {e}")
        sys.exit(1)
