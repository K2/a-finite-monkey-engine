#!/usr/bin/env python3
"""
Test the document processing functionality of the vector store.
"""
import asyncio
import sys
from pathlib import Path
from loguru import logger
from vector_store_util import SimpleVectorStore

async def test_document_processing():
    """Test document processing with sample data."""
    # Create a test vector store
    store = SimpleVectorStore(
        storage_dir="./test_vector_store",
        collection_name="test_processing",
        embedding_model="local"
    )
    
    # Create some test documents
    documents = [
        {
            "text": "# Integer Overflow Example\n\n```solidity\nfunction add(uint256 a, uint256 b) internal pure returns (uint256) {\n    uint256 c = a + b;\n    return c;\n}\n```",
            "metadata": {
                "title": "Integer Overflow Example",
                "category": "security"
            }
        },
        {
            "text": "# Reentrancy Example\n\n```solidity\nfunction withdraw() public {\n    uint amount = balances[msg.sender];\n    (bool success, ) = msg.sender.call{value: amount}(\"\");\n    balances[msg.sender] = 0;\n}\n```",
            "metadata": {
                "title": "Reentrancy Example",
                "category": "security"
            }
        }
    ]
    
    # Process documents
    logger.info(f"Adding {len(documents)} test documents...")
    result = await store.add_documents(documents)
    
    # Check result
    if result:
        logger.success("Successfully processed test documents")
        
        # Test fingerprinting
        fingerprints = set()
        for doc in documents:
            fingerprint = store._create_document_fingerprint(doc)
            logger.info(f"Document: {doc['metadata']['title']} -> Fingerprint: {fingerprint[:8]}...")
            fingerprints.add(fingerprint)
        
        logger.info(f"Generated {len(fingerprints)} unique fingerprints")
        
        # Clean up test directory
        # Path("./test_vector_store").unlink(missing_ok=True)
    else:
        logger.error("Failed to process test documents")
    
    return result

if __name__ == "__main__":
    try:
        asyncio.run(test_document_processing())
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
