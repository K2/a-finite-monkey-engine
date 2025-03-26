#!/usr/bin/env python3
"""
Comprehensive test for LlamaIndex integration with the Finite Monkey Engine

This script tests the full LlamaIndex integration workflow:
1. Loading and processing Solidity files
2. Creating embeddings and storing them in the vector database
3. Performing semantic searches
4. Testing fallback mechanisms

Usage:
  python test_llama_index_integration.py
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("llama_index_test")

# Import the Finite Monkey components
from finite_monkey.llama_index.processor import AsyncIndexProcessor
from finite_monkey.llama_index.loaders import AsyncCodeLoader
from finite_monkey.llama_index.vector_store import VectorStoreManager
from finite_monkey.nodes_config import nodes_config

# Initialize config
config = nodes_config()

async def test_code_loading(file_paths: List[str]) -> None:
    """Test the code loading functionality"""
    logger.info("Testing code loading...")
    
    loader = AsyncCodeLoader()
    
    # Load individual files
    for i, file_path in enumerate(file_paths[:3]):  # Test first 3 files
        logger.info(f"Loading file {i+1}/{len(file_paths[:3])}: {file_path}")
        docs = await loader.load_data(file_path=file_path)
        logger.info(f"Loaded {len(docs)} documents from {file_path}")
        
        # Verify document content
        if docs:
            logger.info(f"Document metadata: {docs[0].metadata}")
            logger.info(f"Document content length: {len(docs[0].text)} characters")
    
    # Test directory loading
    dir_path = os.path.dirname(file_paths[0])
    logger.info(f"Loading all Solidity files from directory: {dir_path}")
    docs = await loader.load_data(dir_path=dir_path, extensions=[".sol"])
    logger.info(f"Loaded {len(docs)} documents from directory")

async def test_index_processor(file_paths: List[str], project_id: str) -> None:
    """Test the AsyncIndexProcessor"""
    logger.info("Testing AsyncIndexProcessor...")
    
    # Initialize the processor
    processor = AsyncIndexProcessor(
        project_id=project_id,
        embed_dim=384,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        embedding_model_name=config.EMBEDDING_MODEL_NAME
    )
    
    # Test loading and indexing
    start_time = time.time()
    logger.info(f"Indexing {len(file_paths)} files...")
    
    try:
        node_ids = await processor.load_and_index(file_paths=file_paths)
        logger.info(f"Indexed {len(node_ids)} nodes in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise
    
    # Test queries
    test_queries = [
        "reentrancy vulnerability",
        "access control issues",
        "transfer ownership",
        "emergency withdrawal",
        "calculate fee with precision"
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: '{query}'")
        
        try:
            # Test basic query
            results = await processor.aquery(query)
            logger.info(f"Found {len(results.get('nodes', []))} results for query '{query}'")
            
            # Log first result
            if results.get('nodes'):
                first_result = results['nodes'][0]
                logger.info(f"Top result score: {first_result['score']}")
                logger.info(f"Top result metadata: {json.dumps(first_result['metadata'], indent=2)}")
                
                # Show snippet of content
                content = first_result['text']
                logger.info(f"Content snippet: {content[:300]}..." if len(content) > 300 else content)
        except Exception as e:
            logger.error(f"Error during query: {e}")
    
    # Test similarity search
    if file_paths:
        try:
            with open(file_paths[0], "r") as f:
                sample_code = f.read()
            
            # Get a function from the code (simplified approach)
            import re
            function_match = re.search(r'function\s+\w+\([^)]*\).*?{.*?}', sample_code, re.DOTALL)
            
            if function_match:
                function_code = function_match.group(0)
                logger.info(f"Testing similarity search with function:\n{function_code[:200]}...")
                
                similar_functions = await processor.get_related_functions(function_code, top_k=3)
                logger.info(f"Found {len(similar_functions)} similar functions")
                
                for i, func in enumerate(similar_functions):
                    logger.info(f"Similar function {i+1}: Score = {func['score']}")
                    logger.info(f"From: {func['metadata'].get('file_path', 'unknown')}")
            else:
                logger.warning("Could not extract a function for similarity testing")
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")

async def test_vector_store_manager() -> None:
    """Test the VectorStoreManager"""
    logger.info("Testing VectorStoreManager...")
    
    # Initialize the manager
    manager = VectorStoreManager(vector_store_path=config.VECTOR_STORE_PATH)
    
    # Test collection creation
    collection_name = f"test_collection_{int(time.time())}"
    logger.info(f"Creating collection: {collection_name}")
    
    try:
        # Create collection
        collection = await manager.get_or_create_collection(collection_name)
        logger.info(f"Collection created: {collection_name}")
        
        # Create test document
        from llama_index.core.schema import Document
        test_docs = [
            Document(
                text="This is a test document for vector store functionality",
                metadata={"test_id": "1", "type": "test"}
            ),
            Document(
                text="Another test document with different content",
                metadata={"test_id": "2", "type": "test"}
            )
        ]
        
        # Add documents
        logger.info("Adding test documents to collection")
        success = await manager.add_documents(collection, test_docs)
        logger.info(f"Documents added successfully: {success}")
        
        # Test search
        if success:
            logger.info("Testing search functionality")
            query = "test document"
            results = await manager.search(collection, query, limit=5)
            
            logger.info(f"Found {len(results)} results for query '{query}'")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: Score = {result['score']}")
                logger.info(f"Content: {result['text'][:100]}...")
    except Exception as e:
        logger.error(f"Error testing vector store manager: {e}")

async def main():
    """Main test function"""
    logger.info("Starting LlamaIndex integration tests")
    
    # Get project directories
    examples_dir = Path(os.getcwd()) / "examples"
    defi_project_dir = examples_dir / "defi_project" / "contracts"
    
    # Collect all Solidity files for testing
    solidity_files = []
    
    # Add example Vault.sol
    main_example = examples_dir / "Vault.sol"
    if main_example.exists():
        solidity_files.append(str(main_example))
    
    # Add DeFi project files
    if defi_project_dir.exists():
        for sol_file in defi_project_dir.glob("*.sol"):
            solidity_files.append(str(sol_file))
    
    # Log the files we're testing with
    logger.info(f"Found {len(solidity_files)} Solidity files for testing:")
    for file in solidity_files:
        logger.info(f"- {file}")
    
    project_id = f"test_{int(time.time())}"
    
    # Run the tests
    try:
        # Test code loading
        await test_code_loading(solidity_files)
        
        # Test index processor
        await test_index_processor(solidity_files, project_id)
        
        # Test vector store manager
        await test_vector_store_manager()
        
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Set up asyncio policy for Windows if needed
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the tests
    exitcode = asyncio.run(main())
    sys.exit(exitcode)