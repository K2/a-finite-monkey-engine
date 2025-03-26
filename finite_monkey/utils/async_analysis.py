"""
Async utilities for code analysis
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from .chunking import AsyncContractChunker

async def analyze_code(
    code: str,
    name: str = "Contract",
    file_path: Optional[str] = None,
    max_chunk_size: int = 8000,
    include_call_graph: bool = False,
    project_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze code asynchronously using chunking and LLM
    
    Args:
        code: Solidity code to analyze
        name: Name for the code (file name or contract name)
        file_path: Optional source file path
        max_chunk_size: Maximum chunk size in characters
        include_call_graph: Whether to include call graph information
        project_path: Optional project root path for call graph analysis
        
    Returns:
        Analysis result
    """
    # Create chunker
    chunker = AsyncContractChunker(
        max_chunk_size=max_chunk_size,
        chunk_by_contract=True,
        chunk_by_function=True,
        include_call_graph=include_call_graph
    )
    
    # Initialize call graph if needed
    if include_call_graph and project_path:
        await chunker.initialize_call_graph(project_path)
    
    # Chunk code
    chunks = await chunker.chunk_code(code, name, file_path)
    
    # Analyze chunks - this would involve your LLM logic
    # For now we'll just return the chunks
    return {
        "chunks": chunks,
        "num_chunks": len(chunks),
        "file_path": file_path,
        "name": name
    }

async def analyze_files(
    file_paths: List[str],
    max_chunk_size: int = 8000,
    include_call_graph: bool = True,
    project_path: Optional[str] = None,
    concurrency_limit: int = 10
) -> Dict[str, Any]:
    """
    Analyze multiple files asynchronously
    
    Args:
        file_paths: List of file paths to analyze
        max_chunk_size: Maximum chunk size in characters
        include_call_graph: Whether to include call graph information
        project_path: Optional project root path for call graph analysis
        concurrency_limit: Maximum number of concurrent file analyses
        
    Returns:
        Analysis results
    """
    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def process_file(file_path: str):
        async with semaphore:
            try:
                # Create chunker for this file
                chunker = AsyncContractChunker(
                    max_chunk_size=max_chunk_size,
                    include_call_graph=include_call_graph
                )
                
                # Get chunks
                chunks = await chunker.chunk_file(file_path)
                return file_path, chunks
                
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {str(e)}")
                return file_path, None
    
    # Process all files
    tasks = [process_file(file_path) for file_path in file_paths]
    results = await asyncio.gather(*tasks)
    
    # Combine results
    analysis_results = {}
    for file_path, chunks in results:
        if chunks:
            analysis_results[file_path] = {
                "chunks": chunks,
                "num_chunks": len(chunks)
            }
    
    return {
        "files_analyzed": len(analysis_results),
        "results": analysis_results
    }
