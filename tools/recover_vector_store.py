#!/usr/bin/env python3
"""
Recovery utility for vector store after a crash.
This script attempts to recover any pending documents from the checkpoint
and complete the document addition process.
"""
import os
import sys
import json
import pickle
import asyncio
from pathlib import Path
from loguru import logger

# Add parent directory to path so we can import the vector store
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from tools.vector_store_util import SimpleVectorStore

async def recover_vector_store(collection_name="default", storage_dir="./vector_store", delete_checkpoint=False):
    """
    Attempt to recover a vector store after a crash by loading the checkpoint
    and completing document insertion.
    
    Args:
        collection_name: Name of the collection to recover
        storage_dir: Directory of the vector store
        delete_checkpoint: Whether to delete the checkpoint after recovery
    """
    logger.info(f"Attempting to recover vector store collection: {collection_name}")
    
    # Verify checkpoint exists
    checkpoint_path = os.path.join(storage_dir, collection_name, "checkpoint.pkl")
    if not os.path.exists(checkpoint_path):
        logger.error(f"No checkpoint found at {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        completed = len(checkpoint.get('completed_fingerprints', []))
        pending_nodes = len(checkpoint.get('pending_nodes', []))
        pending_docs = len(checkpoint.get('pending_docs', []))
        
        logger.info(f"Loaded checkpoint with {completed} completed fingerprints, "
                  f"{pending_nodes} pending nodes, and {pending_docs} pending docs")
        
        if pending_nodes == 0 and pending_docs == 0:
            logger.info("No pending documents to process. Nothing to recover.")
            return True
        
        # Initialize vector store
        store = SimpleVectorStore(
            storage_dir=storage_dir,
            collection_name=collection_name
        )
        
        # Insert pending nodes
        if pending_nodes > 0 and store._index:
            logger.info(f"Inserting {pending_nodes} pending nodes from checkpoint")
            
            pending_node_objs = checkpoint.get('pending_nodes', [])
            store._index.insert_nodes(pending_node_objs)
            
            # Update document list
            store._documents.extend(checkpoint.get('pending_docs', []))
            
            # Save index and metadata
            index_dir = os.path.join(storage_dir, collection_name)
            logger.info("Saving recovered index to disk...")
            store._index.storage_context.persist(persist_dir=index_dir)
            await store._save_document_metadata(index_dir)
            
            # Delete checkpoint if requested
            if delete_checkpoint and os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info("Deleted checkpoint after successful recovery")
            
            logger.info(f"Successfully recovered {pending_nodes} documents")
            return True
        
        logger.warning("Unable to recover: No pending nodes or index not initialized")
        return False
    
    except Exception as e:
        logger.error(f"Error recovering vector store: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recover vector store after crash")
    parser.add_argument("--collection", "-c", default="default", help="Collection name to recover")
    parser.add_argument("--dir", "-d", default="./vector_store", help="Vector store directory")
    parser.add_argument("--delete-checkpoint", "-D", action="store_true", help="Delete checkpoint after recovery")
    args = parser.parse_args()
    
    asyncio.run(recover_vector_store(args.collection, args.dir, args.delete_checkpoint))
