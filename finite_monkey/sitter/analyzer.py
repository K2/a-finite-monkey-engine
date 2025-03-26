"""
Tree-Sitter based analyzer for Solidity code

This module provides a dedicated analyzer for Solidity code using Tree-Sitter.
It handles AST parsing, query execution, and vulnerability pattern detection.
"""

import re
import os
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from pathlib import Path
import asyncio
import traceback
from loguru import logger

# Import Tree-Sitter if available
try:
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("Tree-Sitter not available. Some advanced analysis features will be disabled.")

# Configuration - can be made more flexible through dependency injection
DEFAULT_QUERIES = {
    "contracts": """
    (contract_declaration
        name: (identifier) @contract_name
        body: (contract_body) @contract_body)
    """,
    
    "functions": """
    (function_definition
        name: (identifier) @function_name
        parameters: (parameter_list) @parameters
        [
            (visibility_specifier) @visibility
            (state_mutability_specifier) @mutability
        ]?
        return_parameters: (return_parameter_list)? @returns
        body: (function_body) @body)
    """,
    
    "state_vars": """
    (state_variable_declaration
        type: (type_name) @type
        (visibility_specifier)? @visibility
        name: (identifier) @name
        [
            (expression) @initial_value
        ]?)
    """,
    
    "external_calls": """
    (call_expression
        function: [
            (member_expression
                object: (_) @target
                property: (identifier) @method (#match? @method "^(call|transfer|send|delegatecall|staticcall)$"))
        ]
        arguments: (call_argument_list) @args)
    """,
    
    "assignments": """
    (assignment_expression
        left: [
            (identifier) @var_name
            (member_expression) @var_name
        ]
        right: (_) @value)
    """
}

# Vulnerability patterns represented as Tree-Sitter queries
VULNERABILITY_PATTERNS = {
    "reentrancy": """
    (call_expression
        function: (member_expression
            object: (_) @target
            property: (identifier) @method (#match? @method "^(call)$"))
        arguments: (call_argument_list
            (argument_list) @args))
    """,
    
    "unchecked_call": """
    (call_expression
        function: (member_expression
            object: (_) @target
            property: (identifier) @method (#match? @method "^(call|transfer|send)$"))
        arguments: (call_argument_list) @args)
    """,
    
    "timestamp_dependence": """
    (member_expression
        object: (identifier) @obj (#match? @obj "^(block)$")
        property: (identifier) @prop (#match? @prop "^(timestamp)$"))
    """
}

class TreeSitterAnalyzer:
    """
    Tree-Sitter based analyzer for Solidity code
    
    This class provides methods for parsing Solidity code into an AST,
    executing queries against the AST, and detecting common vulnerability patterns.
    """
    
    def __init__(self, language_path: Optional[str] = None):
        """
        Initialize the analyzer with Tree-Sitter if available
        
        Args:
            language_path: Path to the Solidity language grammar
        """
        self.available = TREE_SITTER_AVAILABLE
        self.parser = None
        self.language = None
        
        # Try to initialize parser
        if self.available:
            try:
                # Try to get language path from environment or use default
                if language_path is None:
                    for path in [
                        os.environ.get("TREE_SITTER_SOLIDITY_PATH"),
                        "./tree_sitter_languages/solidity.so",
                        "/usr/local/lib/tree-sitter-solidity.so",
                        str(Path.home() / ".tree-sitter" / "languages" / "solidity" / "solidity.so")
                    ]:
                        if path and os.path.exists(path):
                            language_path = path
                            break
                
                if language_path and os.path.exists(language_path):
                    self.language = Language(language_path, 'solidity')
                    self.parser = Parser()
                    self.parser.set_language(self.language)
                    logger.info(f"Tree-Sitter initialized with language path: {language_path}")
                else:
                    # Try to use already imported language if available
                    try:
                        from tree_sitter_solidity import language
                        self.language = language
                        self.parser = Parser()
                        self.parser.set_language(self.language)
                        logger.info("Tree-Sitter initialized with imported language")
                    except ImportError:
                        logger.error("Could not find Solidity language for Tree-Sitter")
                        self.available = False
                
            except Exception as e:
                logger.error(f"Error initializing Tree-Sitter: {e}")
                logger.error(traceback.format_exc())
                self.available = False
        
        # Initialize query cache
        self.query_cache = {}
        
        # Show availability status
        if self.available and self.parser:
            logger.info("Tree-Sitter analyzer is available")
        else:
            logger.warning("Tree-Sitter analyzer is not available")
    
    def is_available(self) -> bool:
        """Check if Tree-Sitter is available and properly initialized"""
        return self.available and self.parser is not None and self.language is not None
    
    def parse(self, code: str) -> Optional[Any]:
        """
        Parse source code into AST
        
        Args:
            code: Source code to parse
            
        Returns:
            Parsed AST tree or None if parsing failed
        """
        if not self.is_available():
            return None
        
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
            return None
    
    def compile_query(self, query_text: str) -> Optional[Any]:
        """
        Compile a Tree-Sitter query
        
        Args:
            query_text: The query to compile
            
        Returns:
            Compiled query or None if failed
        """
        if not self.is_available():
            return None
        
        if query_text in self.query_cache:
            return self.query_cache[query_text]
        
        try:
            query = self.language.query(query_text)
            self.query_cache[query_text] = query
            return query
        except Exception as e:
            logger.error(f"Error compiling query: {e}")
            return None
    
    def execute_query(self, tree: Any, query_text: str) -> List[Dict[str, Any]]:
        """
        Execute a Tree-Sitter query on the parsed AST
        
        Args:
            tree: Parsed Tree-Sitter tree
            query_text: Query text to execute
            
        Returns:
            List of matches with captured nodes
        """
        if not self.is_available():
            return []
        
        query = self.compile_query(query_text)
        if not query:
            return []
        
        try:
            captures = query.captures(tree.root_node)
            if not captures:
                return []
            
            # Convert captures to a more usable format
            results = []
            current_match = {}
            
            for node, capture_name in captures:
                # Get node text
                node_text = node.text.decode('utf8')
                
                # Check if this is a new match
                if capture_name.endswith('_name') or capture_name == 'contract_body':
                    if current_match:
                        results.append(current_match.copy())
                    current_match = {}
                
                # Add node to current match
                current_match[capture_name] = {
                    'text': node_text,
                    'start_point': node.start_point,
                    'end_point': node.end_point
                }
            
            # Add last match
            if current_match:
                results.append(current_match)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def extract_contracts(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract contracts from Solidity code
        
        Args:
            code: Solidity source code
            
        Returns:
            List of contracts with name and body
        """
        tree = self.parse(code)
        if not tree:
            return []
        
        query_text = DEFAULT_QUERIES["contracts"]
        matches = self.execute_query(tree, query_text)
        
        contracts = []
        for match in matches:
            if 'contract_name' in match:
                contract = {
                    'name': match['contract_name']['text'],
                    'start_line': match['contract_name']['start_point'][0],
                    'end_line': match['contract_body']['end_point'][0] if 'contract_body' in match else 0
                }
                contracts.append(contract)
        
        return contracts
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract functions from Solidity code
        
        Args:
            code: Solidity source code
            
        Returns:
            List of functions with details
        """
        tree = self.parse(code)
        if not tree:
            return []
        
        query_text = DEFAULT_QUERIES["functions"]
        matches = self.execute_query(tree, query_text)
        
        functions = []
        for match in matches:
            if 'function_name' in match:
                function = {
                    'name': match['function_name']['text'],
                    'start_line': match['function_name']['start_point'][0],
                    'end_line': match['body']['end_point'][0] if 'body' in match else 0,
                    'visibility': match['visibility']['text'] if 'visibility' in match else 'public',
                    'mutability': match['mutability']['text'] if 'mutability' in match else None
                }
                functions.append(function)
        
        return functions
    
    def detect_vulnerability_patterns(self, code: str) -> List[Dict[str, Any]]:
        """
        Detect vulnerability patterns in Solidity code
        
        Args:
            code: Solidity source code
            
        Returns:
            List of detected vulnerability patterns
        """
        tree = self.parse(code)
        if not tree:
            return []
        
        vulnerabilities = []
        
        # Detect each vulnerability pattern
        for pattern_name, query_text in VULNERABILITY_PATTERNS.items():
            matches = self.execute_query(tree, query_text)
            for i, match in enumerate(matches):
                # Get the main captured node
                main_node = None
                for node_name, node_info in match.items():
                    if node_name in ['target', 'var_name', 'obj']:
                        main_node = node_info
                        break
                
                if not main_node:
                    continue
                
                # Determine node location
                start_line = main_node['start_point'][0] + 1  # Convert to 1-based
                
                # Describe the vulnerability based on pattern
                description = ""
                severity = ""
                
                if pattern_name == "reentrancy":
                    description = "Potential reentrancy vulnerability. External call may allow callback before state is updated."
                    severity = "High"
                elif pattern_name == "unchecked_call":
                    description = "Unchecked return value from external call. Failure handling is missing."
                    severity = "Medium"
                elif pattern_name == "timestamp_dependence":
                    description = "Block timestamp dependence. Miners can slightly manipulate timestamps."
                    severity = "Low"
                
                vulnerabilities.append({
                    'name': pattern_name,
                    'line': start_line,
                    'description': description,
                    'severity': severity,
                    'confidence': 'Medium'  # Default confidence
                })
        
        return vulnerabilities
    
    def compute_complexity_metrics(self, code: str) -> Dict[str, Any]:
        """
        Compute code complexity metrics
        
        Args:
            code: Solidity source code
            
        Returns:
            Dictionary of complexity metrics
        """
        tree = self.parse(code)
        if not tree:
            return {}
        
        # Extract functions
        functions = self.extract_functions(code)
        
        # Compute cyclomatic complexity for each function
        function_complexity = {}
        
        for function in functions:
            name = function['name']
            start_line = function['start_line']
            end_line = function['end_line']
            
            # Count branches in function
            branches = 0
            
            # Simple heuristic for complexity: count branches like if, for, while
            lines = code.split('\n')[start_line:end_line+1]
            for line in lines:
                if re.search(r'\bif\s*\(', line):
                    branches += 1
                if re.search(r'\bfor\s*\(', line):
                    branches += 1
                if re.search(r'\bwhile\s*\(', line):
                    branches += 1
                if re.search(r'\bcase\s+', line):
                    branches += 1
                if re.search(r'\b\?\s*', line):  # Ternary operator
                    branches += 1
            
            # Cyclomatic complexity = branches + 1
            function_complexity[name] = branches + 1
        
        # Overall complexity metrics
        contract_count = len(self.extract_contracts(code))
        function_count = len(functions)
        lines = len(code.split('\n'))
        
        return {
            'function_complexity': function_complexity,
            'contract_count': contract_count,
            'function_count': function_count,
            'lines_of_code': lines,
            'average_complexity': sum(function_complexity.values()) / max(1, len(function_complexity))
        }
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze Solidity code and return comprehensive results
        
        Args:
            code: Solidity source code
            
        Returns:
            Comprehensive analysis results
        """
        if not self.is_available():
            return {
                'error': 'Tree-Sitter analyzer is not available',
                'available': False
            }
        
        try:
            # Parse code
            tree = self.parse(code)
            if not tree:
                return {
                    'error': 'Failed to parse code',
                    'available': True
                }
            
            # Get contracts
            contracts = self.extract_contracts(code)
            
            # Get functions for each contract
            functions = self.extract_functions(code)
            
            # Detect vulnerability patterns
            security_patterns = self.detect_vulnerability_patterns(code)
            
            # Compute complexity metrics
            complexity_metrics = self.compute_complexity_metrics(code)
            
            # Return comprehensive results
            return {
                'available': True,
                'contracts': contracts,
                'functions': functions,
                'security_patterns': security_patterns,
                'complexity_metrics': complexity_metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'available': True
            }

# Asynchronous wrapper for the analyzer
class AsyncTreeSitterAnalyzer:
    """
    Asynchronous wrapper for the TreeSitterAnalyzer
    
    This class provides async methods that run the synchronous TreeSitterAnalyzer
    methods in a thread pool to avoid blocking the event loop.
    """
    
    def __init__(self, language_path: Optional[str] = None):
        """Initialize with the same parameters as TreeSitterAnalyzer"""
        self.analyzer = TreeSitterAnalyzer(language_path)
    
    async def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze code asynchronously
        
        Args:
            code: Solidity source code
            
        Returns:
            Analysis results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.analyzer.analyze_code,
            code
        )
    
    async def extract_contracts(self, code: str) -> List[Dict[str, Any]]:
        """Extract contracts asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.analyzer.extract_contracts,
            code
        )
    
    async def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract functions asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.analyzer.extract_functions,
            code
        )
    
    async def detect_vulnerability_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Detect vulnerability patterns asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.analyzer.detect_vulnerability_patterns,
            code
        )
    
    async def compute_complexity_metrics(self, code: str) -> Dict[str, Any]:
        """Compute complexity metrics asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.analyzer.compute_complexity_metrics,
            code
        )