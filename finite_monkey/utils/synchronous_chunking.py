"""
Synchronous chunking utilities for Solidity smart contracts

This module provides synchronous chunking capabilities that are used
by the asynchronous chunkers.
"""

import re
import os
from typing import List, Dict, Any, Optional

class ContractChunker:
    """
    Synchronous chunker for Solidity smart contracts
    
    This class provides methods to chunk large Solidity contracts
    into semantically meaningful segments.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 8000,
        overlap_size: int = 500,
        preserve_imports: bool = True,
        chunk_by_contract: bool = True,
        chunk_by_function: bool = True,
        include_call_graph: bool = False,
    ):
        """Initialize the contract chunker"""
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.preserve_imports = preserve_imports
        self.chunk_by_contract = chunk_by_contract
        self.chunk_by_function = chunk_by_function
        self.include_call_graph = include_call_graph
    
    def _extract_contracts(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract contract definitions using regex patterns
        
        Args:
            content: Solidity code string
            
        Returns:
            List of contract definitions
        """
        # Pattern to match contract, interface or library definitions
        pattern = r'(contract|interface|library)\s+(\w+)(?:\s+is\s+([^{]+))?\s*{([^}]*)}'
        
        # Find all matches
        contracts = []
        for match in re.finditer(pattern, content, re.DOTALL):
            contract_type = match.group(1)
            contract_name = match.group(2)
            inheritance = match.group(3)
            contract_body = match.group(4)
            
            # Add to results
            contracts.append({
                "name": contract_name,
                "type": contract_type,
                "inheritance": inheritance,
                "body": contract_body,
                "start": match.start(),
                "end": match.end()
            })
            
        return contracts

    def extract_functions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract function definitions using regex patterns
        
        Args:
            content: Solidity code string
            
        Returns:
            List of function definitions
        """
        # Pattern to match function definitions
        pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*(public|private|internal|external)?(\s+(view|pure|payable))?\s*(?:returns\s*\(([^)]*)\))?\s*\{([^}]*)\}'
        
        # Find all matches
        functions = []
        for match in re.finditer(pattern, content, re.DOTALL):
            func_name = match.group(1)
            params = match.group(2)
            visibility = match.group(3) or 'public'  # Default visibility is public
            modifier = match.group(5) or ''
            returns = match.group(6) or ''
            body = match.group(7)
            
            # Calculate line numbers
            start_line = content[:match.start()].count('\n') + 1
            end_line = start_line + match.group(0).count('\n')
            
            # Add to results
            functions.append({
                "name": func_name,
                "params": params.strip(),
                "visibility": visibility,
                "modifier": modifier,
                "returns": returns,
                "body": body,
                "full_text": match.group(0),
                "start_line": start_line,
                "end_line": end_line,
                "start_char": match.start(),
                "end_char": match.end()
            })
            
        return functions

    def extract_imports(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract import statements using regex patterns
        
        Args:
            content: Solidity code string
            
        Returns:
            List of import definitions
        """
        # Pattern to match import statements
        pattern = r'import\s+"([^"]+)"\s*;|import\s+\{([^}]+)\}\s+from\s+"([^"]+)"\s*;'
        
        # Find all matches
        imports = []
        for match in re.finditer(pattern, content):
            # Direct import
            if match.group(1):
                imports.append({
                    "path": match.group(1),
                    "symbols": [],
                    "full_text": match.group(0),
                    "start_char": match.start(),
                    "end_char": match.end()
                })
            # Named import
            else:
                symbols = [s.strip() for s in match.group(2).split(',')]
                imports.append({
                    "path": match.group(3),
                    "symbols": symbols,
                    "full_text": match.group(0),
                    "start_char": match.start(),
                    "end_char": match.end()
                })
        
        return imports
