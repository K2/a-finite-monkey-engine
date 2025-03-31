#!/usr/bin/env python3
"""
Test suite for the vector store functionality.

This script provides a unified test suite for all vector store components,
including embedding models, checkpoint handling, and document processing.
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import components to test
from vector_store_util import SimpleVectorStore
from embedding_models import IPEXEmbedding, OllamaEmbedding, create_local_embedding_model
from checkpoint_utils import save_checkpoint, load_checkpoint, sanitize_for_pickle


async def test_embedding_models():
    """Test the embedding models."""
    logger.info("Testing embedding models...")
    
    # Test local embedding model
    logger.info("Testing local embedding model...")
    local_model = await create_local_embedding_model()
    if local_model:
        test_text = "This is a test sentence for embedding."
        embedding = await local_model._aget_text_embedding(test_text)
        logger.info(f"Local embedding generated with length: {len(embedding)}")
        assert len(embedding) > 0, "Local embedding should have positive length"
    
    # Test IPEX embedding model
    try:
        logger.info("Testing IPEX embedding model...")
        ipex_model = IPEXEmbedding(model_name="BAAI/bge-small-en-v1.5")
        test_text = "This is a test sentence for embedding."
        embedding = ipex_model._get_text_embedding(test_text)
        logger.info(f"IPEX embedding generated with length: {len(embedding)}")
        assert len(embedding) > 0, "IPEX embedding should have positive length"
        
        # Test the async methods too
        async_embedding = await ipex_model._aget_text_embedding(test_text)
        assert len(async_embedding) > 0, "Async IPEX embedding should have positive length"
    except Exception as e:
        logger.warning(f"IPEX embedding test failed: {e}")
    
    logger.info("Embedding model tests completed")


async def test_checkpoint_utils():
    """Test the checkpoint utilities."""
    logger.info("Testing checkpoint utilities...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pkl")
        
        # Create test data with nested structures
        test_data = {
            "completed_fingerprints": ["hash1", "hash2", "hash3"],
            "pending_nodes": [
                {"text": "Document 1", "metadata": {"source": "test"}},
                {"text": "Document 2", "metadata": {"source": "test"}}
            ],
            "pending_docs": [
                {"id": "doc1", "metadata": {"title": "Test 1"}},
                {"id": "doc2", "metadata": {"title": "Test 2"}}
            ]
        }
        
        # Add a coroutine to test sanitization
        async def test_coroutine():
            return "This should be removed"
        
        test_data["coroutine"] = test_coroutine()
        
        # Save the checkpoint
        result = await save_checkpoint(checkpoint_path, test_data)
        assert result, "Checkpoint save should succeed"
        
        # Load the checkpoint
        loaded_data = await load_checkpoint(checkpoint_path)
        
        # Verify the data
        assert len(loaded_data["completed_fingerprints"]) == 3, "Should have 3 fingerprints"
        assert len(loaded_data["pending_nodes"]) == 2, "Should have 2 nodes"
        assert len(loaded_data["pending_docs"]) == 2, "Should have 2 docs"
        assert "coroutine" not in loaded_data, "Coroutine should be removed"
        
        logger.info("Checkpoint utils tests passed")


async def test_vector_store():
    """Test the vector store functionality."""
    logger.info("Testing vector store...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test vector store
        class SyncVectorStoreCreator:
            """Helper class to create vector store in sync context."""
            
            def __init__(self, storage_dir, collection_name, embedding_model):
                self.storage_dir = storage_dir
                self.collection_name = collection_name
                self.embedding_model = embedding_model
                
            def create(self):
                """Create the vector store synchronously."""
                from vector_store_util import SimpleVectorStore
                
                # Override the _create_local_embedding_model to work in sync context
                original_method = SimpleVectorStore._create_local_embedding_model
                
                def sync_create_embedding(self):
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                
                # Temporarily replace the method
                SimpleVectorStore._create_local_embedding_model = sync_create_embedding
                
                # Create the store
                store = SimpleVectorStore(
                    storage_dir=self.storage_dir,
                    collection_name=self.collection_name,
                    embedding_model=self.embedding_model
                )
                
                # Restore original method
                SimpleVectorStore._create_local_embedding_model = original_method
                
                return store

        # Create store using the helper
        store_creator = SyncVectorStoreCreator(
            storage_dir=tmpdir,
            collection_name="test_collection",
            embedding_model="local"
        )
        store = store_creator.create()
        
        # Create some test documents
        test_docs = [
            "This is a test document about artificial intelligence.",
            "Python is a programming language known for its readability.",
            "Vector stores are used for semantic search applications."
        ]
        
        # Add documents
        result = await store.add_documents(test_docs)
        assert result, "Document addition should succeed"
        
        # Search for documents
        results = await store.search("semantic search", top_k=1)
        assert len(results) > 0, "Search should return results"
        logger.info(f"Search result: {results[0]['text']}")
        
        logger.info("Vector store tests passed")


async def main():
    """Run all tests."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("Starting vector store test suite...")
    
    try:
        # Run tests
        await test_embedding_models()
        await test_checkpoint_utils()
        await test_vector_store()
        
        logger.info("All tests passed!")
        return 0
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
