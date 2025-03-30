"""
Test script for the Guidance integration.

This script verifies that the Guidance integration is functioning properly
with the correct handling of different API versions and fallbacks.
"""
import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add root directory to path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from pydantic import BaseModel, Field
from typing import List, Optional

from finite_monkey.guidance import (
    GUIDANCE_AVAILABLE, 
    create_structured_program, 
    GuidanceQuestionGenerator
)


class TestOutput(BaseModel):
    """Test output schema for Guidance structured generation"""
    summary: str = Field(..., description="Summary of the analysis")
    points: List[str] = Field(default_factory=list, description="Key points")
    score: int = Field(1, description="Score from 1-10")


async def test_structured_output():
    """Test the structured output generation"""
    if not GUIDANCE_AVAILABLE:
        logger.warning("Guidance not available, skipping test")
        return None
    
    logger.info("Testing structured output generation")
    
    # Create prompt template
    prompt = """
Analyze the following text: {{text}}

{{#schema}}
{
  "summary": "Brief summary",
  "points": ["Point 1", "Point 2"],
  "score": 7
}
{{/schema}}
    """
    
    # Create program
    program = await create_structured_program(
        output_cls=TestOutput,
        prompt_template=prompt,
        model="gpt-3.5-turbo",
        provider="openai",
        verbose=True
    )
    
    if not program:
        logger.error("Failed to create structured program")
        return None
    
    # Test the program
    result = await program(text="This is a test text that should be analyzed for key points and given a score.")
    
    # Print result
    logger.info(f"Generated result: {result}")
    if isinstance(result, TestOutput):
        logger.info(f"Summary: {result.summary}")
        logger.info(f"Points: {', '.join(result.points)}")
        logger.info(f"Score: {result.score}")
    else:
        logger.warning(f"Unexpected result type: {type(result)}")
    
    return result


async def test_question_generator():
    """Test the question generator"""
    if not GUIDANCE_AVAILABLE:
        logger.warning("Guidance not available, skipping test")
        return None
    
    logger.info("Testing question generator")
    
    # Create question generator
    generator = GuidanceQuestionGenerator(
        model="gpt-3.5-turbo",
        provider="openai",
        verbose=True
    )
    
    # Define test tools
    tools = [
        {
            "name": "solidity_info",
            "description": "Provides information about Solidity programming language"
        },
        {
            "name": "security_scanner",
            "description": "Identifies security vulnerabilities in smart contracts"
        },
        {
            "name": "gas_optimizer",
            "description": "Provides gas optimization recommendations"
        }
    ]
    
    # Test query
    query = "What are important security considerations and gas optimization techniques in Solidity smart contracts?"
    
    # Generate sub-questions
    sub_questions = await generator.generate(query, tools)
    
    # Print results
    logger.info(f"Generated {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions):
        logger.info(f"{i+1}. {sq.text} (Tool: {sq.tool_name})")
        if sq.reasoning:
            logger.info(f"   Reasoning: {sq.reasoning}")
    
    return sub_questions


async def run_all_tests():
    """Run all tests"""
    logger.info("Starting Guidance integration tests")
    
    # Test structured output
    output_result = await test_structured_output()
    
    # Test question generator
    question_result = await test_question_generator()
    
    logger.info("All tests completed")
    return {
        "output": output_result,
        "questions": question_result
    }


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run tests
    asyncio.run(run_all_tests())
