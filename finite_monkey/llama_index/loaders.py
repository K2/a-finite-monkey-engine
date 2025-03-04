"""
Asynchronous document loaders for code files
"""

import os
import asyncio
import aiofiles
from typing import Dict, List, Optional, Union, Any

from llama_index.core.schema import Document

class AsyncCodeLoader:
    """Asynchronous loader for code files"""
    
    async def load_data(
        self, 
        file_path: Optional[str] = None, 
        dir_path: Optional[str] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load code files asynchronously
        
        Args:
            file_path: Path to a specific file to load
            dir_path: Path to a directory containing files to load
            recursive: Whether to search directories recursively
            extensions: List of file extensions to include (e.g., [".sol", ".py"])
            
        Returns:
            List of documents
        """
        # Set default file extensions
        if extensions is None:
            extensions = [".sol", ".js", ".ts", ".py", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp"]
        
        # Load file or directory
        if file_path is not None:
            return await self._load_file(file_path)
        elif dir_path is not None:
            return await self._load_directory(dir_path, recursive, extensions)
        else:
            raise ValueError("Either file_path or dir_path must be provided")
    
    async def _load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file asynchronously
        
        Args:
            file_path: Path to the file
            
        Returns:
            List containing a single document
        """
        try:
            # Check if the file exists
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read the file asynchronously
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()
            
            # Extract file metadata
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1]
            
            # Create a document
            metadata = {
                "file_path": file_path,
                "file_name": file_name,
                "file_type": file_ext.lstrip("."),
                "source_type": "file",
            }
            
            # Return the document
            return [Document(text=content, metadata=metadata)]
            
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return []
    
    async def _load_directory(
        self, 
        dir_path: str, 
        recursive: bool,
        extensions: List[str],
    ) -> List[Document]:
        """
        Load files from a directory asynchronously
        
        Args:
            dir_path: Path to the directory
            recursive: Whether to search subdirectories
            extensions: List of file extensions to include
            
        Returns:
            List of documents
        """
        try:
            # Check if the directory exists
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            
            # Walk through the directory
            file_paths = []
            for root, dirs, files in os.walk(dir_path):
                # Skip hidden directories if recursive
                if not recursive and root != dir_path:
                    continue
                
                # Add files with matching extensions
                for file_name in files:
                    file_ext = os.path.splitext(file_name)[1]
                    if file_ext in extensions:
                        file_path = os.path.join(root, file_name)
                        file_paths.append(file_path)
            
            # Load files concurrently
            tasks = [self._load_file(file_path) for file_path in file_paths]
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            return [doc for docs in results for doc in docs]
            
        except Exception as e:
            print(f"Error loading directory {dir_path}: {str(e)}")
            return []