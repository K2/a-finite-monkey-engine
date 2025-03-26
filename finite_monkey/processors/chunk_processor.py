"""
Asynchronous Chunk Processor for handling Solidity files

This module implements a functional approach to processing chunks of Solidity files.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from loguru import logger

from ..utils.chunking import AsyncContractChunker
from ..utils.functional import AsyncPipeline, amap, afilter, areduce
from ..pipeline.core import Context

class AsyncChunkProcessor:
    """
    Asynchronous processor for Solidity contract chunks
    
    Implements a functional async processing pipeline for code chunks.
    """
    
    def __init__(
        self, 
        max_chunk_size: int = 8000,
        max_concurrency: int = 20,
        include_call_graph: bool = True
    ):
        """Initialize the chunk processor"""
        self.max_chunk_size = max_chunk_size
        self.max_concurrency = max_concurrency
        self.include_call_graph = include_call_graph
        self.chunker = AsyncContractChunker(
            max_chunk_size=max_chunk_size,
            include_call_graph=include_call_graph
        )
    
    async def process_project(
        self, 
        project_path: str,
        transformers: List[Callable[[Dict[str, Any]], Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process an entire project using functional patterns
        
        Args:
            project_path: Path to project
            transformers: List of transformation functions to apply to each chunk
            
        Returns:
            Dictionary of file paths to processed chunks
        """
        # Default transformers list
        transformers = transformers or []
        
        # Initialize call graph for project
        if self.include_call_graph:
            await self.chunker.initialize_call_graph(project_path)
        
        # Get all Solidity files
        sol_files = []
        async for file_path in self._find_sol_files(project_path):
            sol_files.append(file_path)
        
        # Process files concurrently with functional approach
        results = await amap(
            lambda file_path: AsyncPipeline.of(file_path)
                .map(self._read_file)
                .map(lambda content: self._process_content(content, file_path, transformers))
                .recover(lambda e: {"error": str(e), "file": file_path})
                .get_or_else(None),
            sol_files,
            max_concurrency=self.max_concurrency
        )
        
        # Filter and combine results
        return {
            str(file_path): result 
            for file_path, result in zip(sol_files, results)
            if result and "error" not in result  # Skip errors
        }
    
    async def _find_sol_files(self, project_path: str):
        """Find all Solidity files in project"""
        for root, _, files in await asyncio.to_thread(os.walk, project_path):
            for file in files:
                if file.endswith(".sol") and not file.endswith(".t.sol"):
                    yield Path(os.path.join(root, file))
    
    async def _read_file(self, file_path: Path) -> Tuple[str, Path]:
        """Read file content asynchronously"""
        import aiofiles
        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = await f.read()
        return content, file_path
    
    async def _process_content(
        self, 
        content_and_path: Tuple[str, Path],
        file_path: Path,
        transformers: List[Callable[[Dict[str, Any]], Any]]
    ) -> List[Dict[str, Any]]:
        """Process file content and apply transformations"""
        content, _ = content_and_path
        
        # Get chunks from content
        chunks = await self.chunker.chunk_code(
            code=content,
            name=file_path.name,
            file_path=str(file_path)
        )
        
        # Apply transformers to each chunk using functional pattern
        if transformers:
            processed_chunks = []
            for chunk in chunks:
                # Apply each transformer in sequence
                transformed_chunk = chunk
                for transform in transformers:
                    if asyncio.iscoroutinefunction(transform):
                        transformed_chunk = await transform(transformed_chunk)
                    else:
                        transformed_chunk = transform(transformed_chunk)
                processed_chunks.append(transformed_chunk)
            return processed_chunks
        
        return chunks
    
    async def process_context(
        self,
        context: Context,
        max_concurrency: int = 10
    ) -> Context:
        """
        Process files in a Context object
        
        Args:
            context: Pipeline context with files
            max_concurrency: Maximum concurrent file processing
            
        Returns:
            Updated context with chunks
        """
        # Get list of Solidity files from context
        sol_files = [
            (file_id, file_data) 
            for file_id, file_data in context.files.items()
            if file_data.get("is_solidity", False)
        ]
        
        # Process files concurrently
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_file(file_tuple):
            file_id, file_data = file_tuple
            async with semaphore:
                try:
                    # Use chunker to get chunks
                    chunks = await self.chunker.chunk_code(
                        code=file_data.get("content", ""),
                        name=file_data.get("name", ""),
                        file_path=file_data.get("path")
                    )
                    
                    # Update the file data and context
                    file_data["chunks"] = chunks
                    
                    # Also add to global chunks dictionary
                    for chunk in chunks:
                        chunk_id = chunk.get("id", f"{file_id}:chunk:{len(context.chunks)}")
                        context.chunks[chunk_id] = chunk
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_id}: {str(e)}")
                    context.add_error(
                        stage="chunk_processing", 
                        message=f"Failed to process file: {file_id}",
                        exception=e
                    )
                    return False
        
        # Process all files and count successes
        results = await asyncio.gather(*[process_file(file_tuple) for file_tuple in sol_files])
        success_count = sum(1 for result in results if result)
        
        logger.info(f"Processed {success_count}/{len(sol_files)} files successfully")
        
        return context
