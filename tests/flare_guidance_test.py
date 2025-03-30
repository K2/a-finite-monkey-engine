"""
Test script for the FLARE engine with Guidance integration.

This script validates that the FLARE engine correctly decomposes
queries using the Guidance-based question generator.
"""
import asyncio
import os
import sys
from pathlib import Path
from loguru import logger

# Add root directory to path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from finite_monkey.query_engine.flare_engine import FlareQueryEngine
from finite_monkey.query_engine.guidance_question_gen import GUIDANCE_AVAILABLE
from finite_monkey.llm.llama_index_adapter import LlamaIndexAdapter


async def test_flare_engine():
    """Test the FLARE engine with Guidance question generator"""
    logger.info("Testing FLARE engine with Guidance integration")
    
    # Set up a simple vector engine for testing
    from llama_index.core import VectorStoreIndex, Document
    
    # Create some sample documents
    docs = [
        Document(text="Solidity is a programming language used for Ethereum smart contracts."),
        Document(text="Smart contracts can have security vulnerabilities like reentrancy."),
        Document(text="Gas optimization is important for efficient Ethereum smart contracts.")
    ]
    
    # Create vector index
    vector_index = VectorStoreIndex.from_documents(docs)
    vector_engine = vector_index.as_query_engine()
    
    # Create a FLARE engine
    flare_engine = FlareQueryEngine(
        underlying_engine=vector_engine,
        max_iterations=2,
        verbose=True,
        use_guidance=True
    )
    
    # Test query
    query = "What are important security considerations when writing smart contracts?"
    
    # Execute query
    result = await flare_engine.query(query)
    
    # Print results
    logger.info(f"Query: {query}")
    logger.info(f"Response: {result.response}")
    
    # Check if it used sub-questions
    if hasattr(result, "sub_questions") and result.sub_questions:
        logger.info(f"Generated {len(result.sub_questions)} sub-questions:")
        for i, sq in enumerate(result.sub_questions, 1):
            logger.info(f"  {i}. {sq.get('text', '')} (Tool: {sq.get('tool_name', '')})")
    
    return result


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run test
    asyncio.run(test_flare_engine())
