"""
Test the Guidance integration with the FLARE query engine.
"""
import asyncio
import json
from loguru import logger

from finite_monkey.query_engine.flare_engine import FlareQueryEngine
from finite_monkey.query_engine.existing_engine import ExistingQueryEngine
from finite_monkey.adapters.guidance_adapter import GuidanceAdapter, GUIDANCE_AVAILABLE
from finite_monkey.nodes_config import config


async def test_guidance_flare():
    """Test the Guidance-based FLARE query engine"""
    
    if not GUIDANCE_AVAILABLE:
        logger.warning("Guidance not available, skipping test")
        return
        
    logger.info("Testing Guidance-based FLARE query engine")
    
    # Create a simple vector engine for testing
    from llama_index.core import VectorStoreIndex, Document
    docs = [
        Document(text="Solidity is a programming language for Ethereum smart contracts."),
        Document(text="Smart contracts can have security vulnerabilities like reentrancy."),
        Document(text="Gas optimization is important for Ethereum smart contracts.")
    ]
    vector_index = VectorStoreIndex.from_documents(docs)
    
    # Create the underlying query engine
    underlying_engine = ExistingQueryEngine(vector_index=vector_index)
    
    # Create FLARE engine with Guidance
    flare_engine = FlareQueryEngine(
        underlying_engine=underlying_engine,
        max_iterations=2,
        verbose=True,
        use_guidance=True
    )
    
    # Initialize the engine
    await flare_engine.initialize()
    
    # Test a complex query
    query = "What are important considerations when writing Ethereum smart contracts?"
    
    # Execute the query
    result = await flare_engine.query(query)
    
    # Print results
    logger.info(f"Query: {query}")
    logger.info(f"Response: {result.response}")
    logger.info(f"Sub-questions: {json.dumps(result.sub_questions, indent=2) if hasattr(result, 'sub_questions') else 'None'}")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_guidance_flare())
