"""
Comprehensive test for the Guidance integration.

This script tests the Guidance integration with different output schemas
and demonstrates how to use the create_program function.
"""
import asyncio
import sys
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger

# Add project root to path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from finite_monkey.guidance import create_program, GUIDANCE_AVAILABLE


class SimpleAnalysis(BaseModel):
    """Simple analysis result schema"""
    summary: str = Field(..., description="Summary of analysis")
    points: List[str] = Field(default_factory=list, description="Key points")
    score: int = Field(1, description="Score from 1-10")


class BusinessFlow(BaseModel):
    """Business flow schema"""
    name: str = Field(..., description="Name of the flow")
    description: str = Field(..., description="Description of what the flow does")
    steps: List[str] = Field(..., description="Steps in the flow")
    importance: int = Field(1, description="Importance score (1-5)")


async def test_simple_analysis():
    """Test simple text analysis with Guidance"""
    if not GUIDANCE_AVAILABLE:
        logger.warning("Guidance not available, skipping test")
        return
        
    logger.info("Testing simple analysis with Guidance")
    
    # Create prompt template (using Python f-string style)
    prompt = """
Analyze the following text: {{text}}

{{#schema}}
{
  "summary": "Brief summary of the text",
  "points": ["Key point 1", "Key point 2"],
  "score": 7
}
{{/schema}}
    """
    
    # Create program
    program = await create_program(
        output_cls=SimpleAnalysis,
        prompt_template=prompt,
        model="gpt-3.5-turbo",
        provider="openai",
        verbose=True
    )
    
    if not program:
        logger.error("Failed to create Guidance program")
        return
        
    # Execute program
    result = await program(
        text="The Finite Monkey Engine analyzes smart contracts using LLMs to identify business flows, potential vulnerabilities, and other insights."
    )
    
    # Print results
    logger.info(f"Analysis result: {result}")
    if isinstance(result, SimpleAnalysis):
        logger.info(f"Summary: {result.summary}")
        logger.info(f"Points: {result.points}")
        logger.info(f"Score: {result.score}")
    else:
        logger.warning(f"Unexpected result type: {type(result)}")
        
    return result


async def test_business_flow_analysis():
    """Test business flow analysis with Guidance"""
    if not GUIDANCE_AVAILABLE:
        logger.warning("Guidance not available, skipping test")
        return
        
    logger.info("Testing business flow analysis with Guidance")
    
    # Create prompt template (using Python f-string style)
    prompt = """
Analyze the following smart contract function:

```solidity
{{code}}
```

Identify the main business flow in this function.

{{#schema}}
{
  "name": "Main business flow",
  "description": "Description of the flow",
  "steps": ["Step 1", "Step 2", "Step 3"],
  "importance": 4
}
{{/schema}}
    """
    
    # Create program
    program = await create_program(
        output_cls=BusinessFlow,
        prompt_template=prompt,
        model="gpt-3.5-turbo",
        provider="openai",
        verbose=True
    )
    
    if not program:
        logger.error("Failed to create Guidance program")
        return
        
    # Execute program with a sample contract function
    result = await program(
        code="""
function transfer(address to, uint256 amount) public returns (bool) {
    require(balanceOf[msg.sender] >= amount, "Insufficient balance");
    
    balanceOf[msg.sender] -= amount;
    balanceOf[to] += amount;
    
    emit Transfer(msg.sender, to, amount);
    return true;
}
        """
    )
    
    # Print results
    logger.info(f"Business flow analysis result: {result}")
    if isinstance(result, BusinessFlow):
        logger.info(f"Flow name: {result.name}")
        logger.info(f"Description: {result.description}")
        logger.info(f"Steps: {result.steps}")
        logger.info(f"Importance: {result.importance}")
    else:
        logger.warning(f"Unexpected result type: {type(result)}")
        
    return result


async def run_all_tests():
    """Run all Guidance integration tests"""
    logger.info("Starting Guidance integration tests")
    
    # Run simple analysis test
    analysis_result = await test_simple_analysis()
    
    # Run business flow analysis test
    flow_result = await test_business_flow_analysis()
    
    logger.info("All tests completed")
    return {
        "analysis": analysis_result,
        "business_flow": flow_result
    }


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run all tests
    asyncio.run(run_all_tests())
