"""
Test the integration with different versions of Guidance.
"""
import asyncio
import pytest
from loguru import logger
from typing import Dict, Any
from pydantic import BaseModel

from finite_monkey.utils.guidance_version_utils import (
    create_guidance_program,
    GuidanceProgramWrapper,
    GUIDANCE_VERSION,
    GUIDANCE_AVAILABLE
)

class SimpleOutput(BaseModel):
    """Simple output model for testing guidance integration."""
    response: str

@pytest.mark.asyncio
async def test_guidance_integration():
    """Test creating and executing a Guidance program."""
    # Skip if Guidance is not available
    if not GUIDANCE_AVAILABLE:
        pytest.skip("Guidance library not available")
    
    # Create a simple template
    template = """
    You are a helpful assistant.
    {{#if question}}
    Question: {{question}}
    {{/if}}
    {{#schema}}
    {
        "response": "Your response to the question"
    }
    {{/schema}}
    """
    
    # Create the program
    program = await create_guidance_program(
        output_cls=SimpleOutput,
        prompt_template=template,
        model="dolphin3:8b-llama3.1-q8_0",  # Use a model that's available locally
        provider="ollama",
        verbose=True
    )
    
    # Check if program was created
    assert program is not None, f"Failed to create Guidance program with version {GUIDANCE_VERSION}"
    
    # Execute the program
    result = await program(question="What is the capital of France?")
    
    # Validate the result
    assert isinstance(result, (SimpleOutput, dict)), f"Expected SimpleOutput or dict, got {type(result)}"
    
    if isinstance(result, SimpleOutput):
        assert result.response, "Response should not be empty"
        logger.info(f"Got response: {result.response}")
    else:
        assert "response" in result, "Response key should be in result dictionary"
        logger.info(f"Got response dictionary: {result}")
    
    logger.info("Guidance integration test completed successfully")

if __name__ == "__main__":
    asyncio.run(test_guidance_integration())
