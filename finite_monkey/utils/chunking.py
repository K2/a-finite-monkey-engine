"""
Chunking utilities for handling large contracts

This module provides utilities for breaking down large smart contracts
into manageable chunks to avoid context length limitations when analyzing
with LLMs.
"""

import concurrent
import re
import os
import json
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Iterator

from antlr4 import InputStream as ANTLRInputStream
from antlr4 import CommonTokenStream
from sgp.ast_node_types import SourceUnit
from sgp.parser.SolidityLexer import SolidityLexer
from sgp.parser.SolidityParser import SolidityParser
from sgp.sgp_error_listener import SGPErrorListener
from sgp.sgp_parser import ParserError
from sgp.sgp_visitor import SGPVisitor, SGPVisitorOptions
from sgp.tokens import build_token_list
from sgp.utils import string_from_snake_to_camel_case
import asyncio
from finite_monkey.utils.async_call_graph import AsyncCallGraph, AProjectAudit
from finite_monkey.nodes_config import config

from box import Box

import os
import re
import json
import asyncio
import aiofiles
from typing import Dict, List, Any, Tuple, Optional, Iterator, AsyncIterator
from pathlib import Path
from loguru import logger
from .async_call_graph import AsyncCallGraph
from .functional import amap, afilter, AsyncPipeline

class AsyncContractChunker:
    """
    Asynchronous chunker for Solidity smart contracts
    
    This class provides asynchronous methods to chunk large Solidity contracts
    into semantically meaningful segments.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 8000,
        overlap_size: int = 500,
        preserve_imports: bool = True,
        chunk_by_contract: bool = True,
        chunk_by_function: bool = True,
        include_call_graph: bool = True,
    ):
        """Initialize the async contract chunker"""
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.preserve_imports = preserve_imports
        self.chunk_by_contract = chunk_by_contract
        self.chunk_by_function = chunk_by_function
        self.include_call_graph = include_call_graph
        self.call_graph = None
        self._initialized = False
        
        # Lazy initialize the sync chunker - will be set when needed
        self._sync_chunker = None
    
    @property
    def sync_chunker(self):
        """Lazy-loaded synchronous chunker to avoid circular imports"""
        if self._sync_chunker is None:
            # This import is inside a method to avoid circular imports
            from .synchronous_chunking import ContractChunker
            self._sync_chunker = ContractChunker(
                max_chunk_size=self.max_chunk_size,
                overlap_size=self.overlap_size,
                preserve_imports=self.preserve_imports,
                chunk_by_contract=self.chunk_by_contract,
                chunk_by_function=self.chunk_by_function,
                include_call_graph=False  # We'll handle call graph separately
            )
        return self._sync_chunker

    
    async def initialize_call_graph(self, project_path: str) -> None:
        """Initialize the call graph analyzer asynchronously"""
        if self.include_call_graph and not self._initialized:
            # Use the async call graph implementation
            self.call_graph = await AsyncCallGraph.create(project_path)
            self._initialized = True
    
    async def chunk_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Chunk a Solidity file asynchronously"""
        try:
            # Read file asynchronously with aiofiles
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = await f.read()
            
            # Get file name
            file_name = os.path.basename(file_path)
            
            # Process code with chunking logic
            chunks = await self.chunk_code(code, file_name, file_path)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {str(e)}")
            # Return a minimal valid result even on error instead of raising
            return [{
                "chunk_id": f"{os.path.basename(file_path)}_error",
                "content": "// Error processing file",
                "start_char": 0,
                "end_char": 0,
                "source_file": file_path,
                "chunk_type": "error",
                "error": str(e)
            }]
        
        
        @staticmethod
    async def parse(
        input_string: str,
    ) -> SourceUnit:
        """Parse a Solidity source string into an AST.

        Parameters
        ----------
        input_string : str - The Solidity source string to parse.
        options : SGPVisitorOptions - Options to pass to the parser.
        dump_json : bool - Whether to dump the AST as a JSON file.
        dump_path : str - The path to dump the AST JSON file to.

        Returns
        -------
        SourceUnit - The root of an AST of the Solidity source string."""
        # TODO: Asyncify this deeper into the vistor and soforth
        def sync_parse(input_string: str):
            input_stream = ANTLRInputStream(input_string)
            lexer = SolidityLexer(input_stream)
            token_stream = CommonTokenStream(lexer)
            parser = SolidityParser(token_stream)

            listener = SGPErrorListener()
            lexer.removeErrorListeners()
            lexer.addErrorListener(listener)

            parser.removeErrorListeners()
            parser.addErrorListener(listener)
            source_unit = parser.sourceUnit()
            options = SGPVisitorOptions(tokens=False)
            ast_builder = SGPVisitor(options)
            try:
                source_unit: SourceUnit = ast_builder.visit(source_unit)
            except Exception as e:
                raise Exception("AST was not generated")
            else:
                if source_unit is None:
                    raise Exception("AST was not generated")

            # TODO: sort it out
            token_list = []
            if options.tokens:
                token_list = build_token_list(token_stream.getTokens(start=0, stop=len(input_string)), options)

            if not options.errors_tolerant and listener.has_errors():
                raise ParserError(errors=listener.get_errors())

            if options.errors_tolerant and listener.has_errors():
                source_unit.errors = listener.get_errors()

            # TODO: sort it out
            if options.tokens:
                source_unit["tokens"] = token_list
                
            return source_unit
            
        await asyncio.sleep(0)
        
        loop = asyncio.get_running_loop()
    
        # Use the default thread pool executor
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result1 = await loop.run_in_executor(pool, sync_parse, input_string)

        source_unit = result1
        return source_unit
    
        
    # # taken out of sync version of the code finite-monkey-engine
    # @staticmethod
    # async def parse(
    #     input_string: str,
    # ) -> SourceUnit:
    #     """Parse a Solidity source string into an AST.

    #     Parameters
    #     ----------
    #     input_string : str - The Solidity source string to parse.
    #     options : SGPVisitorOptions - Options to pass to the parser.
    #     dump_json : bool - Whether to dump the AST as a JSON file.
    #     dump_path : str - The path to dump the AST JSON file to.

    #     Returns
    #     -------
    #     SourceUnit - The root of an AST of the Solidity source string."""
    #     # TODO: Asyncify this deeper into the vistor and soforth
    #     def sync_parse(input_string: str):
    #         input_stream = ANTLRInputStream(input_string)
    #         lexer = SolidityLexer(input_stream)
    #         token_stream = CommonTokenStream(lexer)
    #         parser = SolidityParser(token_stream)

    #         listener = SGPErrorListener()
    #         lexer.removeErrorListeners()
    #         lexer.addErrorListener(listener)

    #         parser.removeErrorListeners()
    #         parser.addErrorListener(listener)
    #         source_unit = parser.sourceUnit()
    #         options = SGPVisitorOptions(tokens=False)
    #         ast_builder = SGPVisitor(options)
    #         try:
    #             source_unit: SourceUnit = ast_builder.visit(source_unit)
    #         except Exception as e:
    #             raise Exception("AST was not generated")
    #         else:
    #             if source_unit is None:
    #                 raise Exception("AST was not generated")

    #         # TODO: sort it out
    #         token_list = []
    #         if options.tokens:
    #             token_list = build_token_list(token_stream.getTokens(start=0, stop=len(input_string)), options)

    #         if not options.errors_tolerant and listener.has_errors():
    #             raise ParserError(errors=listener.get_errors())

    #         if options.errors_tolerant and listener.has_errors():
    #             source_unit.errors = listener.get_errors()

    #         # TODO: sort it out
    #         if options.tokens:
    #             source_unit["tokens"] = token_list
                
    #         return source_unit
            
    #     await asyncio.sleep(0)
        
    #     loop = asyncio.get_running_loop()
    
    #     # Use the default thread pool executor
    #     with concurrent.futures.ThreadPoolExecutor() as pool:
    #         result1 = await loop.run_in_executor(pool, sync_parse, input_string)

    #     source_unit = result1
    #     return source_unit
    
    
    async def chunk_code(
        self,
        code: str,
        name: str = "Contract",
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a Solidity code string into semantic segments
        Args:
            code: Solidity code string
            name: Name for the code (file name or contract name)
            file_path: Optional path to the source file
        Returns:
            List of chunks with metadata
        """
        try:
            # Parse the code into an AST
            results = await self.parse(code)
            absolute_path = os.path.abspath(file_path) if file_path else None
            all_results = []
            chunks = []
            
            # Define a recursive function to process all nodes
            def process_node(node, parent_contract=None):
                if not node:
                    return
                
                # Box the current node's data and add to results
                node_data = Box(vars(node), modify_tuples_box=True)
                
                # Set file path information
                if absolute_path:
                    node_data.relative_file_path = absolute_path
                    node_data.absolute_file_path = file_path
                
                # Set contract name from parent if available
                if parent_contract and not hasattr(node_data, 'contract_name'):
                    node_data.contract_name = parent_contract
                
                # Process node name if it exists
                if hasattr(node, 'name') and node.name:
                    # Handle constructor and normal functions differently
                    if hasattr(node, 'name') and node.name.startswith("SPECIAL_"):
                        if node.name[8:] != "tor":
                            node_data.name = node.name[8:]  # remove SPECIAL_ prefix
                        else:
                            node_data.name = "constructor"
                    
                    # Add full name if contract name exists
                    if hasattr(node_data, 'contract_name') and node_data.contract_name:
                        node_data.full_name = f"{node_data.contract_name}.{node_data.name}"
                
                # Store source location if available - explicitly add source to ensure it's always available
                if hasattr(node, 'src') and node.src:
                    node_data.src = node.src
                    
                    # Extract source code immediately if possible
                    try:
                        src_parts = str(node.src).split(':')
                        if len(src_parts) >= 2:
                            start_pos = int(src_parts[0])
                            length = int(src_parts[1])
                            node_data.source = code[start_pos:start_pos+length]
                            node_data.start_pos = start_pos
                            node_data.length = length
                    except (ValueError, IndexError):
                        pass
                
                # Add to results
                all_results.append(node_data)
                
                # Track current contract name for children
                current_contract = None
                if hasattr(node, 'type') and node.type == 'ContractDefinition' and hasattr(node, 'name'):
                    current_contract = node.name
                elif parent_contract:
                    current_contract = parent_contract
                    
                # Process children recursively
                if hasattr(node, 'children') and node.children:
                    for child in node.children:
                        process_node(child, current_contract)
            
            # Start recursive processing from the root
            if hasattr(results, 'children') and results.children:
                for child in results.children:
                    process_node(child)
            
            # Extract nodes by type
            functions = [result for result in all_results if getattr(result, 'type', None) == 'FunctionDefinition']
            contracts = [result for result in all_results if getattr(result, 'type', None) == 'ContractDefinition']
            imports = [result for result in all_results if getattr(result, 'type', None) == 'ImportDirective']
            
            # Default chunks list - ensure we always return something valid
            if not contracts and not functions:
                # Create a single chunk for the entire file if no contracts/functions found
                chunks.append({
                    "chunk_id": f"{name}_full",
                    "content": code,
                    "start_char": 0,
                    "end_char": len(code),
                    "source_file": file_path,
                    "chunk_type": "file",  # Mark as file type
                    "name": name,
                    "imports": []
                })
            
            # Create chunks for contracts
            for contract in contracts:
                node = Box(contract)
                node.chunk_id = f"{name}_{contract.name}"
                node.content = code[contract.range.offset_start:contract.range.offset_end]
                node.start_char = contract.range.offset_start
                node.end_char = contract.range.offset_end
                node.source_file = file_path
                node.chunk_type = "contract"
                node.contract_name = contract.name
                node.imports = [getattr(imp, 'path', '') for imp in imports]
                chunks.append(node)
            
            # Create chunks for functions with proper error handling
            for func in functions:
                try:
                    func_name = getattr(func, 'name', 'UnnamedFunction')
                    contract_name = getattr(func, 'contract_name', 'UnknownContract')
                    
                    # Get function source if available through multiple methods
                    func_text = None
                    
                    # Method 1: Use pre-extracted source if we stored it during node processing
                    if hasattr(func, 'source'):
                        func_text = func.source
                    
                    # Method 2: Try to get source from 'src' attribute (line:pos:length format)
                    elif hasattr(func, 'body') and func.body:
                        func_text= code[func.body.range.offset_start:func.body.range.offset_end]
                        
                    # Method 3: Try to get actual range from utility function
                    elif hasattr(func, 'type') and func.type == 'FunctionDefinition':
                        range_info = self._find_function_range(code, func_name)
                        if range_info:
                            start_pos, end_pos = range_info
                            func_text = code[start_pos:end_pos]
                    
                    # If we still don't have source, use a placeholder
                    if not func_text:
                        func_text = f"// Unable to extract source for function {func_name}"
                        logger.warning(f"Unable to extract source for function {func_name} in {contract_name}")
                    else:
                        node = Box(func)
                        node.chunk_id = f"{name}_{contract_name}_{func_name}"
                        node.content = func_text
                        node.start_char = 0
                        node.end_char = len(func_text)
                        node.source_file = file_path
                        node.chunk_type = "function"
                        node.contract_name = contract_name
                        node.function_name = func_name
                        node.imports = [getattr(imp, 'path', '') for imp in imports]
                        chunks.append(node)
                        
                except Exception as e:
                    # Log the error but continue processing other functions
                    logger.error(f"Error creating chunk for function: {e}")
                    continue
            
            return chunks
            
        except Exception as e:
            logger.error(f"Exception in chunk_code: {e}")
            # Return a minimal valid result even on error
            return [{
                "chunk_id": f"{name}_error",
                "content": "// Error processing code",
                "start_char": 0,
                "end_char": 0,
                "source_file": file_path,
                "chunk_type": "error",  # Mark as error type
                "error": str(e)
            }]

    def _find_function_range(self, code: str, function_name: str) -> Optional[Tuple[int, int]]:
        """
        Find the range of a function in the source code using regex
        
        Args:
            code: The full source code
            function_name: Name of the function to find
            
        Returns:
            Tuple of (start, end) positions or None if not found
        """
        # Pattern to match function definition with the specific name
        pattern = rf'function\s+{re.escape(function_name)}\s*\([^)]*\)(?:\s*(?:public|private|internal|external))?(?:\s*(?:view|pure|payable))?\s*(?:returns\s*\([^)]*\))?\s*\{{[^{{]*(?:\{{[^{{]*\}}[^{{]*)*\}}'
        
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.span()
        return None

    
    async def chunk_project(self, project_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Chunk all Solidity files in a project asynchronously using functional patterns"""
        try:
            # Initialize call graph if needed
            if self.include_call_graph:
                await self.initialize_call_graph(project_path)
            
            # Get list of Solidity files asynchronously
            sol_files = []
            async for file_path in self._find_solidity_files(project_path):
                sol_files.append(file_path)
            
            if not sol_files:
                logger.warning(f"No Solidity files found in {project_path}")
                return {}
                
            logger.info(f"Found {len(sol_files)} Solidity files in {project_path}")
            
            # Process files with bounded concurrency
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent files
            chunked_files = {}
            
            async def process_file_safely(file_path):
                try:
                    async with semaphore:
                        chunks = await self.chunk_file(file_path)
                        return file_path, chunks
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    # Return empty chunks rather than raising
                    return file_path, []
                    
            # Process all files concurrently with proper error handling
            tasks = [process_file_safely(file_path) for file_path in sol_files]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
            # Combine results
            for file_path, chunks in results:
                if chunks:  # Only add if we got valid chunks
                    chunked_files[file_path] = chunks
            
            return chunked_files
            
        except Exception as e:
            logger.error(f"Project chunking failed: {e}")
            # Return empty dict instead of raising
            return {}

    async def process_file_chunks(self, context, file_data, file_id):
        """Process file chunks using functional pipeline pattern"""
        try:
            # Extract content, name, and path from file data
            content = file_data.get("content", "")
            name = file_data.get("name", "")
            path = file_data.get("path", "")
            
            # Generate chunks
            chunks = await self.chunk_code(content, name, path)
            
            # Add chunks to context
            file_data["chunks"] = chunks
            
            # Also add to global chunks dictionary
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", f"{file_id}:chunk:{len(context.chunks)}")
                context.chunks[chunk_id] = chunk
            
            logger.debug(f"Added {len(chunks)} chunks for file: {file_id}")
            return context
            
        except Exception as e:
            logger.error(f"Error chunking file {file_id}: {str(e)}")
            context.add_error(
                stage="contract_chunking",
                message=f"Failed to chunk file: {file_id}",
                exception=e
            )
            return context

    async def _find_solidity_files(self, project_path: str) -> AsyncIterator[str]:
        """Find Solidity files in project directory asynchronously"""
        try:
            for root, dirs, files in await asyncio.to_thread(os.walk, project_path):
                # Skip node_modules and other problematic directories
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', 'artifacts', 'cache']]
                
                for file in files:
                    if file.endswith(".sol") and not file.endswith(".t.sol"):
                        yield os.path.join(root, file)
        except Exception as e:
            logger.error(f"Error finding Solidity files: {e}")
            # No yield - empty iterator

    async def _extract_contracts(self, content: str) -> List[Dict[str, Any]]:
        """Extract contract definitions asynchronously"""
        # Use thread pool for CPU-bound regex operations
        return await asyncio.to_thread(self.sync_chunker._extract_contracts, content)


# Helper function for easy chunking
def chunk_solidity_file(
    file_path: str,
    max_chunk_size: int = 8000,
    chunk_by_contract: bool = True,
    include_call_graph: bool = True,
    project_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Chunk a Solidity file into semantic segments (DEPRECATED: Use async_chunk_solidity_file instead)
    """
    import warnings
    warnings.warn(
        "chunk_solidity_file is deprecated. Use async_chunk_solidity_file instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Run the async function synchronously using asyncio.run
    import asyncio
    return asyncio.run(async_chunk_solidity_file(
        file_path=file_path,
        max_chunk_size=max_chunk_size,
        chunk_by_contract=chunk_by_contract,
        include_call_graph=include_call_graph,
        project_path=project_path
    ))


# Helper function for easy chunking of code
def chunk_solidity_code(
    code: str,
    name: str = "Contract",
    max_chunk_size: int = 8000,
    chunk_by_contract: bool = True,
    include_call_graph: bool = False,
) -> List[Dict[str, Any]]:
    """
    Chunk a Solidity code string into semantic segments (DEPRECATED: Use async_chunk_solidity_code instead)
    """
    import warnings
    warnings.warn(
        "chunk_solidity_code is deprecated. Use async_chunk_solidity_code instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Run the async function synchronously using asyncio.run
    import asyncio
    return asyncio.run(async_chunk_solidity_code(
        code=code,
        name=name,
        max_chunk_size=max_chunk_size,
        chunk_by_contract=chunk_by_contract,
        include_call_graph=include_call_graph
    ))


async def async_chunk_solidity_file(
    file_path: str,
    max_chunk_size: int = 8000,
    chunk_by_contract: bool = True,
    include_call_graph: bool = True,
    project_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Asynchronously chunk a Solidity file into semantic segments
    """
    chunker = AsyncContractChunker(
        max_chunk_size=max_chunk_size,
        chunk_by_contract=chunk_by_contract,
        include_call_graph=include_call_graph,
    )
    
    if include_call_graph:
        # If project_path is not provided, use the directory containing the file
        if project_path is None:
            project_path = os.path.dirname(file_path)
        await chunker.initialize_call_graph(project_path)
    
    return await chunker.chunk_file(file_path)


async def async_chunk_solidity_code(
    code: str,
    name: str = "Contract",
    max_chunk_size: int = 8000,
    chunk_by_contract: bool = True,
    include_call_graph: bool = False,
) -> List[Dict[str, Any]]:
    """
    Asynchronously chunk a Solidity code string into semantic segments
    """
    chunker = AsyncContractChunker(
        max_chunk_size=max_chunk_size,
        chunk_by_contract=chunk_by_contract,
        include_call_graph=include_call_graph,
    )
    
    return await chunker.chunk_code(code, name)


# Debugging helper
if __name__ == "__main__":
    import sys
    
    # Read the file and check for parenthesis balance
    with open(__file__, 'r') as f:
        content = f.readlines()
    
    # Track opening and closing parentheses
    stack = []
    for i, line in enumerate(content):
        for j, char in enumerate(line):
            if (char == '('):
                stack.append((i+1, j+1))  # Line and column numbers start at 1
            elif (char == ')'):
                if stack:
                    stack.pop()  # Matched with an opening parenthesis
                else:
                    # Found an unmatched closing parenthesis
                    print(f"Unmatched ')' at line {i+1}, column {j+1}")
                    sys.exit(1)
    
    # Check if we have any unmatched opening parentheses
    if stack:
        for line, col in stack:
            print(f"Unmatched '(' at line {line}, column {col}")
    else:
        print("Parentheses are balanced")
