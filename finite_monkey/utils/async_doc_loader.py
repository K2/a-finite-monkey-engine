"""
Asynchronous document loading utilities
"""

import os
import asyncio
import aiofiles
import tempfile
import shutil
from typing import List, Dict, Any, Optional, AsyncIterator
from pathlib import Path
from loguru import logger

from llama_index.core.schema import Document

class AsyncDocumentLoader:
    """Asynchronous document loader for files and repositories"""
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize the document loader
        
        Args:
            max_workers: Maximum number of concurrent worker tasks
        """
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.batch_size = 50  # Default batch size for processing
    
    async def load_from_directory(self, directory_path: str, pattern: str = "**/*.sol") -> List[Document]:
        """
        Load documents from a directory asynchronously
        
        Args:
            directory_path: Path to directory
            pattern: Glob pattern for file selection
            
        Returns:
            List of documents
        """
        # Call our own implementation instead of importing from start.py
        return await self.load_directory(directory_path, pattern)
    
    async def load_from_github(self, repo_url: str, branch: str = "main", subdirectory: Optional[str] = None) -> List[Document]:
        """
        Load documents from a GitHub repository
        
        Args:
            repo_url: GitHub repository URL
            branch: Repository branch
            subdirectory: Optional subdirectory to focus on
            
        Returns:
            List of documents
        """
        # Call our own implementation instead of importing from start.py
        return await self.load_github_repo(repo_url, branch, subdirectory)
    
    async def load_directory(
        self, 
        directory_path: str, 
        pattern: str = "**/*.sol",
        include_non_solidity: bool = False
    ) -> List[Document]:
        """
        Load documents from a directory asynchronously
        
        Args:
            directory_path: Path to directory
            pattern: Glob pattern for matching files
            include_non_solidity: Whether to include non-Solidity files
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading documents from directory: {directory_path} with pattern {pattern}")
        
        # Convert to Path object
        directory = Path(directory_path)
        if not await asyncio.to_thread(directory.exists):
            raise ValueError(f"Directory {directory_path} does not exist")
        
        # Get matching files using thread pool
        matching_files = await asyncio.to_thread(
            lambda: list(directory.glob(pattern))
        )
        
        # Process files in batches
        documents = []
        for i in range(0, len(matching_files), self.batch_size):
            batch = matching_files[i:i+self.batch_size]
            batch_results = await asyncio.gather(*[
                self._process_file(file_path, include_non_solidity) 
                for file_path in batch
            ])
            documents.extend([doc for doc in batch_results if doc is not None])
            
            # Small delay between batches
            if i + self.batch_size < len(matching_files):
                await asyncio.sleep(0.1)
        
        # Count Solidity files
        solidity_count = sum(1 for doc in documents if doc.metadata.get("is_solidity"))
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path} ({solidity_count} Solidity files)")
        return documents
    
    async def load_github_repo(
        self,
        repo_url: str,
        branch: str = "main",
        subdirectory: Optional[str] = None,
        pattern: str = "**/*.sol",
        include_non_solidity: bool = False
    ) -> List[Document]:
        """
        Clone a GitHub repository and load its files asynchronously
        
        Args:
            repo_url: GitHub repository URL
            branch: Branch to clone
            subdirectory: Subdirectory within the repository to process
            pattern: Glob pattern for matching files
            include_non_solidity: Whether to include non-Solidity files
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading files from GitHub repo: {repo_url} (branch: {branch})")
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Clone the repository asynchronously
            logger.info(f"Cloning repository to {temp_dir}...")
            
            # Run git clone in subprocess
            process = await asyncio.create_subprocess_exec(
                "git", "clone", "--branch", branch, "--single-branch", "--depth=1", 
                repo_url, temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise ValueError(f"Failed to clone repository: {error_msg}")
            
            # Determine the directory to read from
            read_dir = temp_dir
            if subdirectory:
                read_dir = os.path.join(temp_dir, subdirectory)
                if not await asyncio.to_thread(os.path.exists, read_dir):
                    raise ValueError(f"Subdirectory '{subdirectory}' not found in repository")
            
            # Load documents from the directory
            documents = await self.load_directory(read_dir, pattern, include_non_solidity)
            return documents
            
        except Exception as e:
            logger.error(f"Error loading files from GitHub: {str(e)}")
            raise
            
        finally:
            # Clean up the temporary directory
            logger.info(f"Cleaning up temporary directory {temp_dir}")
            await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)
    
    async def _process_file(self, file_path: Path, include_non_solidity: bool = False) -> Optional[Document]:
        """Process a single file asynchronously"""
        async with self.semaphore:
            if not await asyncio.to_thread(file_path.is_file):
                return None
            
            is_solidity = file_path.suffix.lower() == ".sol"
            if not is_solidity and not include_non_solidity:
                return None
                
            try:
                # Read file content asynchronously
                async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = await f.read()
                
                # Create document with metadata
                return Document(
                    text=content,
                    metadata={
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix,
                        "is_solidity": is_solidity,
                        "loaded_at": (await asyncio.to_thread(os.path.getmtime, file_path)),
                        "size": len(content)
                    }
                )
                
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                return None
