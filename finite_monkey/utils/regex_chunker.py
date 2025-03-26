"""
Regex-based contract chunker for Solidity code

This module provides a lightweight, regex-based contract chunking implementation
that can be used as an alternative to tree-sitter based parsers.
"""

import re
import os
import uuid
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from loguru import logger
import concurrent.futures
import asyncio

class RegexContractChunker:
    """
    Regex-based contract chunker for Solidity code
    
    This class provides methods for chunking Solidity contracts into smaller
    parts using regular expressions. It's useful when Tree-Sitter is not
    available or when a lightweight solution is preferable.
    """
    
    # Regular expression patterns for identifying Solidity elements
    CONTRACT_PATTERN = re.compile(
        r'(contract|library|interface)\s+(\w+)(?:\s+is\s+([^{]+))?\s*{', 
        re.MULTILINE
    )
    
    FUNCTION_PATTERN = re.compile(
        r'function\s+(\w+)\s*\(([^)]*)\)\s*'
        r'(public|private|internal|external)?'
        r'\s*(pure|view|payable)?'
        r'\s*(returns\s*\([^)]*\))?\s*'
        r'({[^}]*})',
        re.DOTALL
    )
    
    STATE_VAR_PATTERN = re.compile(
        r'^\s*([\w\[\]]+)\s+(private|public|internal)?\s*(\w+)(?:\s*=\s*([^;]+))?;', 
        re.MULTILINE
    )
    
    def __init__(
        self, 
        max_chunk_size: int = 4000,
        chunk_by_contract: bool = True,
        chunk_by_function: bool = True,
        include_context: bool = True
    ):
        """
        Initialize the chunker with configuration options
        
        Args:
            max_chunk_size: Maximum size of chunks in characters
            chunk_by_contract: Whether to split by contract boundaries
            chunk_by_function: Whether to split by function boundaries
            include_context: Whether to include context in chunks
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_by_contract = chunk_by_contract
        self.chunk_by_function = chunk_by_function
        self.include_context = include_context
    
    def chunk_content(self, content: str, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chunk the code content
        
        Args:
            content: Solidity code content
            file_path: Optional file path for reference
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        if not content or not content.strip():
            return chunks
        
        # Use contract chunking if enabled
        if self.chunk_by_contract:
            contract_chunks = self._chunk_by_contracts(content)
            
            # If contract chunking produced chunks, process each contract
            if contract_chunks:
                for contract_chunk in contract_chunks:
                    # Further chunk by function if enabled and needed
                    if self.chunk_by_function and len(contract_chunk['content']) > self.max_chunk_size:
                        function_chunks = self._chunk_by_functions(
                            contract_chunk['content'],
                            contract_name=contract_chunk.get('contract_name')
                        )
                        if function_chunks:
                            # Add contract context to each function chunk if requested
                            if self.include_context:
                                for func_chunk in function_chunks:
                                    func_chunk['contract_name'] = contract_chunk.get('contract_name')
                                    func_chunk['contract_line'] = contract_chunk.get('start_line')
                            chunks.extend(function_chunks)
                        else:
                            # If no function chunks, add the contract as a chunk
                            chunks.append(contract_chunk)
                    else:
                        # Add the contract as a chunk
                        chunks.append(contract_chunk)
            else:
                # No contracts found, fall back to function chunking
                chunks = self._chunk_by_functions(content)
        else:
            # Skip contract chunking, go straight to function chunking
            chunks = self._chunk_by_functions(content)
        
        # If no chunks were created or chunks are still too large, fall back to line-based chunking
        if not chunks or any(len(chunk['content']) > self.max_chunk_size for chunk in chunks):
            line_chunks = self._chunk_by_lines(content, self.max_chunk_size)
            
            # If we have other chunks, only add line chunks for content not covered
            if chunks:
                # Find uncovered content and chunk it
                covered_lines = set()
                for chunk in chunks:
                    if 'start_line' in chunk and 'end_line' in chunk:
                        covered_lines.update(range(chunk['start_line'], chunk['end_line'] + 1))
                
                # Create line chunks only for uncovered content
                uncovered_line_chunks = []
                for line_chunk in line_chunks:
                    if 'start_line' in line_chunk and 'end_line' in line_chunk:
                        lines_range = set(range(line_chunk['start_line'], line_chunk['end_line'] + 1))
                        if not lines_range.issubset(covered_lines):
                            uncovered_line_chunks.append(line_chunk)
                
                chunks.extend(uncovered_line_chunks)
            else:
                chunks = line_chunks
        
        # Add file path if provided
        if file_path:
            for chunk in chunks:
                chunk['file_path'] = file_path
        
        # Assign unique IDs to chunks if they don't have one
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk:
                chunk['chunk_id'] = f"chunk-{i+1}-{uuid.uuid4().hex[:8]}"
        
        return chunks
    
    def chunk_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Chunk a Solidity file
        
        Args:
            file_path: Path to the Solidity file
            
        Returns:
            List of chunks with metadata
        """
        file_path_str = str(file_path)
        
        try:
            with open(file_path_str, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.chunk_content(content, file_path_str)
        except Exception as e:
            logger.error(f"Error chunking file {file_path_str}: {e}")
            return []
    
    def _chunk_by_contracts(self, content: str) -> List[Dict[str, Any]]:
        """
        Chunk code by contract boundaries
        
        Args:
            content: Solidity code content
            
        Returns:
            List of contract chunks
        """
        chunks = []
        
        # Find all contracts, libraries, and interfaces
        matches = list(self.CONTRACT_PATTERN.finditer(content))
        
        # If no matches, return empty list
        if not matches:
            return []
        
        # Calculate line numbers for each match
        lines = content.split('\n')
        line_indices = [0]
        for i in range(len(lines)):
            line_indices.append(line_indices[-1] + len(lines[i]) + 1)
        
        # For each match, find the contract body with balanced braces
        for i, match in enumerate(matches):
            contract_type = match.group(1)  # 'contract', 'library', or 'interface'
            contract_name = match.group(2)
            
            # Find opening brace position (end of match)
            start_pos = match.start()
            open_brace_pos = content.find('{', start_pos)
            
            if open_brace_pos == -1:
                continue  # Skip if no opening brace found
            
            # Find the end of contract with balanced braces
            brace_count = 1
            close_brace_pos = open_brace_pos + 1
            
            while brace_count > 0 and close_brace_pos < len(content):
                if content[close_brace_pos] == '{':
                    brace_count += 1
                elif content[close_brace_pos] == '}':
                    brace_count -= 1
                close_brace_pos += 1
            
            # Extract the contract including braces
            contract_content = content[start_pos:close_brace_pos]
            
            # Calculate start and end line numbers
            start_line = 0
            end_line = 0
            for j, index in enumerate(line_indices):
                if index > start_pos and start_line == 0:
                    start_line = j
                if index > close_brace_pos and end_line == 0:
                    end_line = j - 1
                    break
            
            # Create chunk
            chunk = {
                'chunk_id': f"contract-{i+1}-{uuid.uuid4().hex[:8]}",
                'chunk_type': 'contract',
                'contract_type': contract_type,
                'contract_name': contract_name,
                'content': contract_content,
                'start_pos': start_pos,
                'end_pos': close_brace_pos,
                'start_line': start_line,
                'end_line': end_line,
                'metadata': {
                    'contract_name': contract_name,
                    'inheritance': match.group(3).strip().split(',') if match.group(3) else []
                }
            }
            
            # Add chunk
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_functions(self, 
                          content: str, 
                          contract_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chunk code by function boundaries
        
        Args:
            content: Solidity code content
            contract_name: Optional contract name for context
            
        Returns:
            List of function chunks
        """
        chunks = []
        
        # Find all functions
        matches = list(self.FUNCTION_PATTERN.finditer(content))
        
        # If no matches, return empty list
        if not matches:
            return []
        
        # Calculate line numbers
        lines = content.split('\n')
        line_indices = [0]
        for i in range(len(lines)):
            line_indices.append(line_indices[-1] + len(lines[i]) + 1)
        
        # Process each function match
        for i, match in enumerate(matches):
            function_name = match.group(1)
            parameters = match.group(2)
            visibility = match.group(3) or "public"  # Default is public
            mutability = match.group(4) or ""
            returns = match.group(5) or ""
            body = match.group(6)
            
            # Get the full function text
            start_pos = match.start()
            end_pos = match.end()
            function_content = content[start_pos:end_pos]
            
            # Calculate start and end line numbers
            start_line = 0
            end_line = 0
            for j, index in enumerate(line_indices):
                if index > start_pos and start_line == 0:
                    start_line = j
                if index > end_pos and end_line == 0:
                    end_line = j - 1
                    break
            
            # Create chunk
            chunk = {
                'chunk_id': f"function-{i+1}-{uuid.uuid4().hex[:8]}",
                'chunk_type': 'function',
                'function_name': function_name,
                'content': function_content,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'start_line': start_line,
                'end_line': end_line,
                'metadata': {
                    'function_name': function_name,
                    'parameters': parameters,
                    'visibility': visibility,
                    'mutability': mutability,
                    'returns': returns
                }
            }
            
            # Add contract name if provided
            if contract_name:
                chunk['contract_name'] = contract_name
            
            # Add chunk
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_lines(self, content: str, max_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Chunk code by line boundaries when other methods fail
        
        Args:
            content: Solidity code content
            max_chunk_size: Maximum size of chunks in characters
            
        Returns:
            List of line-based chunks
        """
        chunks = []
        lines = content.split('\n')
        
        # Initialize variables
        current_chunk = []
        current_size = 0
        chunk_start_line = 0
        
        for i, line in enumerate(lines):
            line_with_newline = line + '\n'
            line_size = len(line_with_newline)
            
            # Check if adding this line exceeds chunk size
            if current_size + line_size > max_chunk_size and current_chunk:
                # Create a chunk from current lines
                chunk_content = ''.join(current_chunk)
                chunk = {
                    'chunk_id': f"line-chunk-{len(chunks)+1}-{uuid.uuid4().hex[:8]}",
                    'chunk_type': 'line_chunk',
                    'content': chunk_content,
                    'start_line': chunk_start_line,
                    'end_line': i - 1,
                    'metadata': {
                        'line_count': len(current_chunk)
                    }
                }
                chunks.append(chunk)
                
                # Reset for next chunk
                current_chunk = []
                current_size = 0
                chunk_start_line = i
            
            # Add this line to current chunk
            current_chunk.append(line_with_newline)
            current_size += line_size
        
        # Add the last chunk if there are remaining lines
        if current_chunk:
            chunk_content = ''.join(current_chunk)
            chunk = {
                'chunk_id': f"line-chunk-{len(chunks)+1}-{uuid.uuid4().hex[:8]}",
                'chunk_type': 'line_chunk',
                'content': chunk_content,
                'start_line': chunk_start_line,
                'end_line': len(lines) - 1,
                'metadata': {
                    'line_count': len(current_chunk)
                }
            }
            chunks.append(chunk)
        
        return chunks


class AsyncRegexChunker:
    """
    Asynchronous wrapper for the RegexContractChunker
    
    This class provides asynchronous methods for chunking Solidity code
    using a thread pool to avoid blocking the event loop.
    """
    
    def __init__(
        self, 
        max_chunk_size: int = 4000,
        chunk_by_contract: bool = True,
        chunk_by_function: bool = True,
        include_context: bool = True,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the async chunker with configuration options
        
        Args:
            max_chunk_size: Maximum size of chunks in characters
            chunk_by_contract: Whether to split by contract boundaries
            chunk_by_function: Whether to split by function boundaries
            include_context: Whether to include context in chunks
            max_workers: Maximum number of worker threads
        """
        self.chunker = RegexContractChunker(
            max_chunk_size=max_chunk_size,
            chunk_by_contract=chunk_by_contract,
            chunk_by_function=chunk_by_function,
            include_context=include_context
        )
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
    
    async def chunk_content(self, content: str, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chunk the code content asynchronously
        
        Args:
            content: Solidity code content
            file_path: Optional file path for reference
            
        Returns:
            List of chunks with metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.chunker.chunk_content,
            content,
            file_path
        )
    
    async def chunk_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Chunk a Solidity file asynchronously
        
        Args:
            file_path: Path to the Solidity file
            
        Returns:
            List of chunks with metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.chunker.chunk_file,
            file_path
        )
    
    async def chunk_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chunk multiple Solidity files concurrently
        
        Args:
            file_paths: List of paths to Solidity files
            
        Returns:
            Dictionary mapping file paths to lists of chunks
        """
        tasks = []
        for file_path in file_paths:
            task = self.chunk_file(file_path)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Map results to file paths
        return {str(path): result for path, result in zip(file_paths, results)}
    
    def close(self):
        """Close the executor to free resources"""
        self.executor.shutdown(wait=False)