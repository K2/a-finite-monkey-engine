#!/usr/bin/env python3
"""
Test script to verify that prompts are being properly saved and loaded in the vector store.
"""
import os
import json
import asyncio
from pathlib import Path
from loguru import logger

from vector_store_util import SimpleVectorStore
from vector_store_prompts import PromptGenerator

async def test_prompt_saving():
    """Test if prompts are properly saved and loaded in the vector store."""
    # Create a test vector store
    test_dir = "./test_vector_store"
    test_collection = "test_prompts"
    
    # Clean up existing test data if it exists
    if os.path.exists(os.path.join(test_dir, test_collection)):
        import shutil
        shutil.rmtree(os.path.join(test_dir, test_collection))
    
    # Create vector store
    vector_store = SimpleVectorStore(
        storage_dir=test_dir,
        collection_name=test_collection,
        embedding_model="local"
    )
    
    # Create a test document with security-related content
    test_doc = {
        "text": """
        def authenticate(username, password):
            query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
            cursor.execute(query)
            user = cursor.fetchone()
            if user:
                return True
            return False
        """,
        "metadata": {
            "language": "python",
            "security_flaw": "sql_injection",
            "description": "SQL injection vulnerability in authentication function"
        }
    }
    
    # Add the document to the vector store
    logger.info("Adding test document to vector store...")
    success = await vector_store.add_documents([test_doc])
    
    if not success:
        logger.error("Failed to add document to vector store")
        return False
    
    # Check if prompts file was created
    prompts_path = os.path.join(test_dir, test_collection, "document_prompts.json")
    if not os.path.exists(prompts_path):
        logger.error(f"Prompts file not created at {prompts_path}")
        return False
    
    # Load prompts and check content
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)
    
    # Check if we have any prompts
    if not prompts_data:
        logger.error("No prompts saved to file")
        return False
    
    logger.info(f"Found {len(prompts_data)} documents with prompts")
    
    # Get the document ID (should be the first one)
    doc_id = next(iter(prompts_data))
    
    # Check what types of prompts we have
    prompt_types = prompts_data[doc_id].keys()
    logger.info(f"Prompt types for document {doc_id}: {', '.join(prompt_types)}")
    
    # Check specific prompt types
    if 'single' in prompt_types:
        logger.info("✅ Basic prompt saved successfully")
    else:
        logger.warning("❌ Basic prompt not saved")
    
    if 'invariant' in prompt_types:
        logger.info("✅ Invariant analysis saved successfully")
    else:
        logger.warning("❌ Invariant analysis not saved")
    
    if 'general_pattern' in prompt_types:
        logger.info("✅ General pattern saved successfully")
    else:
        logger.warning("❌ General pattern not saved")
    
    # Check for flaw patterns
    patterns_path = os.path.join(test_dir, test_collection, "flaw_patterns.json")
    if os.path.exists(patterns_path):
        with open(patterns_path, 'r') as f:
            patterns_data = json.load(f)
        
        if patterns_data:
            logger.info(f"✅ Flaw patterns saved successfully")
            # Get the document ID (should be the first one)
            doc_id = next(iter(patterns_data))
            pattern_types = patterns_data[doc_id].keys()
            logger.info(f"Pattern types for document {doc_id}: {', '.join(pattern_types)}")
        else:
            logger.warning("❌ No flaw patterns saved")
    else:
        logger.warning(f"❌ Flaw patterns file not created at {patterns_path}")
    
    # Try querying with prompts to verify loading
    logger.info("Testing prompt loading with query...")
    query_result = await vector_store.query_with_prompts(
        "sql injection authentication",
        include_prompts=True,
        include_patterns=True
    )
    
    # Check if prompts are in the results
    if 'prompts' in query_result and query_result['prompts']:
        logger.info("✅ Prompts loaded and included in query results")
        prompt_types = query_result['prompts'].keys()
        logger.info(f"Loaded prompt types: {', '.join(prompt_types)}")
    else:
        logger.warning("❌ Prompts not included in query results")
    
    # Check if patterns are in the results
    if 'patterns' in query_result and query_result['patterns']:
        logger.info("✅ Patterns loaded and included in query results")
        pattern_types = query_result['patterns'].keys()
        logger.info(f"Loaded pattern types: {', '.join(pattern_types)}")
    else:
        logger.warning("❌ Patterns not included in query results")
    
    logger.info("Test completed!")
    return True

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_prompt_saving())
