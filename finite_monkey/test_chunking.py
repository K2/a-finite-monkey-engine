"""
Test utility for validating chunking functionality
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from finite_monkey.utils.chunking import AsyncContractChunker
from finite_monkey.pipeline.core import Context

async def test_chunking(solidity_file: str) -> None:
    """Test chunking on a single Solidity file"""
    logger.info(f"Testing chunking on file: {solidity_file}")
    
    # Create chunker
    chunker = AsyncContractChunker(
        max_chunk_size=8000,
        include_call_graph=False  # Skip call graph for simple test
    )
    
    # Chunk the file
    try:
        chunks = await chunker.chunk_file(solidity_file)
        
        # Report results
        logger.info(f"Successfully chunked file into {len(chunks)} chunks")
        
        # Print chunk types and sizes
        for i, chunk in enumerate(chunks):
            chunk_type = chunk.get("chunk_type", "unknown")
            content = chunk.get("content", "")
            logger.info(f"  Chunk {i+1}: type={chunk_type}, size={len(content)}")
            
        return chunks
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise

async def test_chunking_in_context(solidity_file: str) -> Context:
    """Test chunking and adding to context"""
    logger.info(f"Testing context integration with file: {solidity_file}")
    
    # Create context
    context = Context()
    
    # Create chunker
    chunker = AsyncContractChunker(
        max_chunk_size=8000,
        include_call_graph=False
    )
    
    # Read file and add to context
    try:
        # Read file
        async with open(solidity_file, "r", encoding="utf-8", errors="ignore") as f:
            content = await f.read()
        
        # Add file to context
        file_id = solidity_file
        context.files[file_id] = {
            "path": solidity_file,
            "name": os.path.basename(solidity_file),
            "content": content,
            "is_solidity": True
        }
        
        # Process chunks and add to context
        updated_context = await chunker.process_file_chunks(context, context.files[file_id], file_id)
        
        # Report results
        logger.info(f"Successfully added chunks to context")
        logger.info(f"  Context now has {len(updated_context.chunks)} chunks")
        
        return updated_context
    except Exception as e:
        logger.error(f"Context integration failed: {e}")
        raise
