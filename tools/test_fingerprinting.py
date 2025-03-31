#!/usr/bin/env python3
"""
Test the document fingerprinting functionality.
"""
import hashlib
import asyncio
from vector_store_util import SimpleVectorStore
from loguru import logger

async def test_fingerprinting():
    """Test document fingerprinting with different input types."""
    store = SimpleVectorStore(collection_name="test_fingerprinting")
    
    # Test with string input
    text_input = "This is a test document"
    str_fingerprint = store._create_document_fingerprint(text_input)
    logger.info(f"String fingerprint: {str_fingerprint}")
    
    # Test with dictionary input with text field
    dict_input = {"text": "This is a test document"}
    dict_fingerprint = store._create_document_fingerprint(dict_input)
    logger.info(f"Dict fingerprint: {dict_fingerprint}")
    
    # Test with dictionary input with text and metadata
    dict_meta_input = {
        "text": "This is a test document",
        "metadata": {
            "title": "Test Document",
            "source": "test"
        }
    }
    dict_meta_fingerprint = store._create_document_fingerprint(dict_meta_input)
    logger.info(f"Dict with metadata fingerprint: {dict_meta_fingerprint}")
    
    # Check that string and basic dict produce the same fingerprint
    assert str_fingerprint == dict_fingerprint, "String and dict fingerprints should match"
    
    # Check that adding metadata changes the fingerprint
    assert dict_fingerprint != dict_meta_fingerprint, "Adding metadata should change the fingerprint"
    
    logger.success("All fingerprinting tests passed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_fingerprinting())
