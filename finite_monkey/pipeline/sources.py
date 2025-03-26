"""
Source handlers for loading data into pipeline contexts
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncIterator
import tempfile
import shutil

import aiofiles
from loguru import logger

from ..utils.functional import amap, afilter, AsyncPipeline
from .core import Context

class Source:
    """Base class for data sources"""
    
    async def load_into_context(self, context: Context) -> Context:
        """Load data from source into context"""
        raise NotImplementedError("Subclasses must implement load_into_context")

class FileSource(Source):
    """Source for loading a single file"""
    
    def __init__(self, file_path: Path):
        """Initialize with file path"""
        self.file_path = file_path
    
    async def load_into_context(self, context: Context) -> Context:
        """Load file into context"""
        try:
            # Use aiofiles for non-blocking file I/O
            async with aiofiles.open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()
                
            # Add file to context
            file_id = str(self.file_path)
            context.files[file_id] = {
                "path": str(self.file_path),
                "name": self.file_path.name,
                "content": content,
                "size": len(content),
                "extension": self.file_path.suffix,
                "is_solidity": self.file_path.suffix.lower() == ".sol"
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error loading file {self.file_path}: {str(e)}")
            context.add_error(
                stage="file_loading",
                message=f"Failed to load file: {self.file_path}",
                exception=e
            )
            return context

class DirectorySource(Source):
    """Source for loading files from a directory"""
    
    def __init__(self, directory_path: Path, file_pattern: str = "**/*.sol"):
        """Initialize with directory path and file pattern"""
        self.directory_path = directory_path
        self.file_pattern = file_pattern
    
    async def load_into_context(self, context: Context) -> Context:
        """Load directory contents into context"""
        try:
            # Get matching files asynchronously
            matching_files = await self._find_matching_files()
            
            # Load files concurrently with semaphore to limit open files
            semaphore = asyncio.Semaphore(20)  # Limit to 20 concurrent file operations
            
            async def process_file(file_path: Path) -> Optional[Dict[str, Any]]:
                async with semaphore:
                    try:
                        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = await f.read()
                            
                        return {
                            "path": str(file_path),
                            "id": str(file_path),
                            "name": file_path.name,
                            "content": content,
                            "size": len(content),
                            "extension": file_path.suffix,
                            "is_solidity": file_path.suffix.lower() == ".sol"
                        }
                        
                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {str(e)}")
                        context.add_error(
                            stage="file_loading",
                            message=f"Failed to load file: {file_path}",
                            exception=e
                        )
                        return None
            
            # Process files concurrently
            results = await amap(process_file, matching_files)
            
            # Add successfully loaded files to context
            for file_data in results:
                if file_data:
                    context.files[file_data["id"]] = file_data
            
            logger.info(f"Loaded {len(context.files)} files from directory {self.directory_path}")
            return context
            
        except Exception as e:
            logger.error(f"Error loading directory {self.directory_path}: {str(e)}")
            context.add_error(
                stage="directory_loading",
                message=f"Failed to load directory: {self.directory_path}",
                exception=e
            )
            return context
    
    async def _find_matching_files(self) -> List[Path]:
        """Find files matching the pattern asynchronously"""
        # Use thread pool for directory operations
        matching_files = await asyncio.to_thread(
            lambda: list(self.directory_path.glob(self.file_pattern))
        )
        return [f for f in matching_files if f.is_file()]

class GithubSource(Source):
    """Source for loading files from a GitHub repository"""
    
    def __init__(self, repo_url: str, branch: str = "main", file_pattern: str = "**/*.sol"):
        """Initialize with repository URL"""
        self.repo_url = repo_url
        self.branch = branch
        self.file_pattern = file_pattern
    
    async def load_into_context(self, context: Context) -> Context:
        """Load GitHub repository into context"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Clone repository asynchronously
            await self._clone_repository(temp_dir)
            
            # Create directory source from the cloned repository
            dir_source = DirectorySource(Path(temp_dir), self.file_pattern)
            
            # Load files from directory source
            context = await dir_source.load_into_context(context)
            
            return context
            
        except Exception as e:
            logger.error(f"Error loading repository {self.repo_url}: {str(e)}")
            context.add_error(
                stage="github_loading",
                message=f"Failed to load repository: {self.repo_url}",
                exception=e
            )
            return context
            
        finally:
            # Clean up temporary directory asynchronously
            await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)
    
    async def _clone_repository(self, target_dir: str) -> None:
        """Clone repository asynchronously"""
        # Use subprocess module for git operations
        import subprocess
        
        # Prepare git command
        cmd = ["git", "clone", "--depth", "1", "--branch", self.branch, self.repo_url, target_dir]
        
        # Run git command asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for process to complete
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Failed to clone repository: {stderr.decode()}")