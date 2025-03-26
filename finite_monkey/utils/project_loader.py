"""
Project loader utility for processing directory structures into context
"""

import os
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from loguru import logger

from ..pipeline.core import Context
from .chunking import AsyncContractChunker

class AsyncProjectLoader:
    """Asynchronous loader for smart contract projects"""
    
    def __init__(
        self,
        ignore_patterns: Optional[List[str]] = None,
        max_concurrency: int = 10
    ):
        """Initialize project loader"""
        self.ignore_patterns = ignore_patterns or ['node_modules', '.git', 'artifacts', 'cache']
        self.max_concurrency = max_concurrency
        
    async def load_project(self, project_path: str) -> Context:
        """
        Load a project directory into a context
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Context containing project files
        """
        context = Context(project_id=Path(project_path).name)
        
        # Find all Solidity files
        solidity_files = []
        for root, dirs, files in os.walk(project_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]
            
            for file in files:
                if file.endswith(".sol") and not file.endswith(".t.sol"):
                    solidity_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(solidity_files)} Solidity files in {project_path}")
        
        # Load files with bounded concurrency
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def load_file(file_path: str):
            async with semaphore:
                try:
                    # Read file content
                    async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = await f.read()
                    
                    # Create file info
                    file_id = file_path
                    file_info = {
                        "path": file_path,
                        "name": os.path.basename(file_path),
                        "content": content,
                        "size": len(content),
                        "extension": os.path.splitext(file_path)[1],
                        "is_solidity": True,
                        "relative_path": os.path.relpath(file_path, project_path)
                    }
                    
                    return file_id, file_info
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
                    return None, None
        
        # Load all files concurrently
        tasks = [load_file(file_path) for file_path in solidity_files]
        results = await asyncio.gather(*tasks)
        
        # Add successfully loaded files to context
        for file_id, file_info in results:
            if file_id is not None and file_info is not None:
                context.files[file_id] = file_info
        
        logger.info(f"Successfully loaded {len(context.files)} files into context")
        return context
        
    async def load_and_chunk_project(
        self, 
        project_path: str,
        chunk_size: int = 8000
    ) -> Context:
        """
        Load and chunk a project
        
        Args:
            project_path: Path to the project directory
            chunk_size: Maximum chunk size
            
        Returns:
            Context with loaded and chunked files
        """
        # Load project files
        context = await self.load_project(project_path)
        
        # Create chunker
        chunker = AsyncContractChunker(
            max_chunk_size=chunk_size,
            include_call_graph=True
        )
        
        # Initialize call graph
        await chunker.initialize_call_graph(project_path)
        
        # Chunk each file
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def process_file(file_id, file_data):
            async with semaphore:
                try:
                    # Process file chunks
                    return await chunker.process_file_chunks(context, file_data, file_id)
                except Exception as e:
                    logger.error(f"Error processing chunks for {file_id}: {e}")
                    return context
        
        # Process files concurrently
        tasks = []
        for file_id, file_data in list(context.files.items()):
            tasks.append(process_file(file_id, file_data))
        
        # Execute and gather results (context is modified in-place)
        await asyncio.gather(*tasks)
        
        logger.info(f"Chunking complete with {len(context.chunks)} total chunks")
        return context
