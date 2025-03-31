"""
Checkpoint utilities for the vector store.

This module provides functions for saving and loading checkpoints,
allowing interrupted operations to be resumed.
"""

import os
import pickle
import json
import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger
from datetime import datetime
from pathlib import Path


async def save_checkpoint(checkpoint_path: str, data: Dict[str, Any]) -> bool:
    """
    Save checkpoint data to resume from interruptions, using async file I/O.
    
    This function ensures that coroutine objects are not included in the data,
    which would cause pickle to fail.
    
    Args:
        checkpoint_path: Path to save the checkpoint
        data: Data to save (dictionary)
        
    Returns:
        Success status
    """
    try:
        import aiofiles
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Process data to remove unpickleable objects
        clean_data = await sanitize_for_pickle(data)
        
        # Save sanitized data
        async with aiofiles.open(checkpoint_path, 'wb') as f:
            serialized_data = pickle.dumps(clean_data)
            await f.write(serialized_data)
        
        return True
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        return False


async def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint data to resume processing, using async file I/O.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary with checkpoint data or empty values if not found
    """
    try:
        import aiofiles
        
        if not os.path.exists(checkpoint_path):
            return {'completed_fingerprints': [], 'pending_nodes': [], 'pending_docs': []}
        
        async with aiofiles.open(checkpoint_path, 'rb') as f:
            content = await f.read()
            data = pickle.loads(content)
        
        logger.info(f"Loaded checkpoint with {len(data.get('completed_fingerprints', []))} completed documents")
        return data
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return {'completed_fingerprints': [], 'pending_nodes': [], 'pending_docs': []}


async def sanitize_for_pickle(data: Any) -> Any:
    """
    Recursively sanitize data to ensure it can be pickled.
    
    Removes or replaces unpickleable objects like coroutines.
    
    Args:
        data: Data to sanitize
        
    Returns:
        Sanitized data that can be safely pickled
    """
    # Handle dictionaries
    if isinstance(data, dict):
        return {k: await sanitize_for_pickle(v) for k, v in data.items() 
                if not asyncio.iscoroutine(v)}
    
    # Handle lists
    elif isinstance(data, list):
        return [await sanitize_for_pickle(item) for item in data 
                if not asyncio.iscoroutine(item)]
    
    # Handle sets
    elif isinstance(data, set):
        return {await sanitize_for_pickle(item) for item in data 
                if not asyncio.iscoroutine(item)}
    
    # Handle tuples
    elif isinstance(data, tuple):
        return tuple(await sanitize_for_pickle(item) for item in data 
                     if not asyncio.iscoroutine(item))
    
    # Handle coroutines - we can't pickle these, so we'll return None
    elif asyncio.iscoroutine(data):
        return None
    
    # Return other types as-is
    return data


# Add specialized sanitization for node objects

async def sanitize_nodes_for_pickle(nodes: List[Any]) -> List[Any]:
    """
    Sanitize node objects for pickling.
    
    This handles special TextNode objects from llama_index which might
    contain unpickleable attributes.
    
    Args:
        nodes: List of TextNode objects
        
    Returns:
        List of sanitized node objects
    """
    try:
        sanitized_nodes = []
        for node in nodes:
            # Create a minimal version of the node with just the essential attributes
            sanitized_node = {
                'text': getattr(node, 'text', ''),
                'id_': getattr(node, 'id_', None),
                'metadata': getattr(node, 'metadata', {})
            }
            sanitized_nodes.append(sanitized_node)
        return sanitized_nodes
    except Exception as e:
        logger.error(f"Error sanitizing nodes: {e}")
        return []


async def save_document_metadata(storage_dir: str, collection_name: str, documents: List[Dict[str, Any]]) -> bool:
    """
    Save document metadata to disk using async I/O.
    
    Args:
        storage_dir: Base storage directory
        collection_name: Collection name for the vector store
        documents: List of document metadata dictionaries
        
    Returns:
        Success status
    """
    try:
        import aiofiles
        
        index_dir = os.path.join(storage_dir, collection_name)
        metadata_path = os.path.join(index_dir, "document_metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Sanitize documents for JSON serialization
        clean_documents = await sanitize_for_json(documents)
        
        json_data = json.dumps(clean_documents)
        
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json_data)
            
        logger.debug(f"Document metadata saved to {metadata_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving document metadata: {e}")
        return False


async def sanitize_for_json(data: Any) -> Any:
    """
    Recursively sanitize data to ensure it can be JSON serialized.
    
    Args:
        data: Data to sanitize
        
    Returns:
        Sanitized data that can be safely serialized to JSON
    """
    # Handle dictionaries
    if isinstance(data, dict):
        return {k: await sanitize_for_json(v) for k, v in data.items() 
                if not asyncio.iscoroutine(v)}
    
    # Handle lists
    elif isinstance(data, list):
        return [await sanitize_for_json(item) for item in data 
                if not asyncio.iscoroutine(item)]
    
    # Handle sets - convert to list for JSON
    elif isinstance(data, set):
        return [await sanitize_for_json(item) for item in data 
                if not asyncio.iscoroutine(item)]
    
    # Handle coroutines - we can't serialize these, so we'll return None
    elif asyncio.iscoroutine(data):
        return None
    
    # Handle datetime objects
    elif isinstance(data, datetime):
        return data.isoformat()
    
    # Handle Path objects
    elif isinstance(data, Path):
        return str(data)
    
    # Return other types as-is
    return data
