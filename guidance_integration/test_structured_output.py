#!/usr/bin/env python
"""
Test script to verify the structured output functionality
"""

import asyncio
import logging
import sys
from typing import List, Optional
from pydantic import BaseModel, Field

from .core import GuidanceManager
from .models import Node, Link, BusinessFlowData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTest(BaseModel):
    """Simple test class for structured output"""
    message: str = Field(..., description="A simple message")
    items: List[str] = Field(default_factory=list, description="List of items")
    count: int = Field(0, description="A counter")

async def test_structured_output():
    """
    Test the structured output functionality
    """
    logger.info("Testing structured output")
    
    # Create a guidance manager
    manager = GuidanceManager()
    
    # Test with a simple class
    try:
        logger.info("Testing with SimpleTest class")
        result = await manager.execute_structured_prompt(
            prompt_text="Return a simple message saying 'Hello, world!' with 3 items ['apple', 'banana', 'cherry'] and count set to 42",
            output_class=SimpleTest,
            system_prompt="You are a helpful assistant that returns structured data."
        )
        logger.info(f"Result: {result}")
        assert isinstance(result, SimpleTest)
        assert result.message == "Hello, world!"
        assert len(result.items) == 3
        assert result.count == 42
        logger.info("SimpleTest passed!")
    except Exception as e:
        logger.error(f"Error testing SimpleTest: {e}", exc_info=True)
        
    # Test with BusinessFlowData
    try:
        logger.info("Testing with BusinessFlowData class")
        result = await manager.execute_structured_prompt(
            prompt_text="""Generate a small business flow with 2 nodes and 1 link.
            First node should be a function named 'deposit' with type 'function'.
            Second node should be a state named 'balance' with type 'state'.
            Link should go from 'deposit' to 'balance'.""",
            output_class=BusinessFlowData,
            system_prompt="You are a helpful assistant that creates business flow diagrams."
        )
        logger.info(f"Result: {result}")
        assert isinstance(result, BusinessFlowData)
        assert len(result.nodes) == 2
        assert len(result.links) == 1
        logger.info("BusinessFlowData passed!")
    except Exception as e:
        logger.error(f"Error testing BusinessFlowData: {e}", exc_info=True)
    
    logger.info("All tests complete")

if __name__ == "__main__":
    asyncio.run(test_structured_output())
    sys.exit(0)
