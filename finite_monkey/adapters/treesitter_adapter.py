"""
Adapter for tree-sitter functionality to analyze code structure and flow.
"""
from typing import List, Dict, Any, Optional, Tuple, Set, Iterator, Union
import os
from pathlib import Path
import asyncio
from loguru import logger

try:
    from tree_sitter import Language, Parser, Tree, Node
    TREESITTER_AVAILABLE = True
except ImportError:
    logger.warning("tree-sitter not available, code analysis capabilities will be limited")
    TREESITTER_AVAILABLE = False

from ..models.code_flow import (
    FlowNodeType, CodeLocation, FlowNode, FlowEdge, CodeFlowGraph
)


class TreeSitterAdapter:
    """
    Adapter for tree-sitter functionality to analyze code structure and flow.
    
    This adapter provides a higher-level interface to tree-sitter's parsing
    capabilities, with specific focus on identifying sources, sinks, and
    data flow relationships in smart contract code.
    """
    
    def __init__(self):
        """Initialize the TreeSitter adapter"""
        self.parser = None
        self.solidity_language = None
        
        if TREESITTER_AVAILABLE:
            try:
                # Initialize tree-sitter parser
                self.parser = Parser()
                
                # Try to load Solidity language if available
                # Path may need adjustment based on your setup
                repo_root = Path(__file__).parent.parent.parent
                language_path = repo_root / "build" / "languages.so"
                
                if language_path.exists():
                    self.solidity_language = Language(str(language_path), 'solidity')
                    self.parser.set_language(self.solidity_language)
                    logger.info("Initialized tree-sitter with Solidity language support")
                else:
                    logger.warning(f"Solidity language file not found at {language_path}")
            except Exception as e:
                logger.error(f"Failed to initialize tree-sitter: {e}")
                self.parser = None
                self.solidity_language = None
    
    def parse_code(self, code: str) -> Optional[Tree]:
        """
        Parse code using tree-sitter
        
        Args:
            code: Source code to parse
            
        Returns:
            Parsed tree or None if parsing failed
        """
        if not TREESITTER_AVAILABLE or not self.parser:
            logger.warning("tree-sitter not available for parsing")
            return None
            
        try:
            tree = self.parser.parse(bytes(code, 'utf8'))
            return tree
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
            return None
    
    def extract_functions(self, tree: Tree) -> List[Dict[str, Any]]:
        """
        Extract function declarations from the parse tree
        
        Args:
            tree: Parsed tree-sitter tree
            
        Returns:
            List of function information dictionaries
        """
        if not tree:
            return []
            
        functions = []
        
        # Query to find function definitions
        query_string = '(function_definition) @function'
        
        try:
            query = self.solidity_language.query(query_string)
            captures = query.captures(tree.root_node)
            
            for i, (node, tag) in enumerate(captures):
                if tag == 'function':
                    # Extract function name
                    function_name = None
                    for child in node.children:
                        if child.type == 'function_name':
                            function_name = child.text.decode('utf8')
                            break
                    
                    # Get function body if available
                    function_body = None
                    for child in node.children:
                        if child.type == 'function_body':
                            function_body = child
                            break
                    
                    # Function location
                    location = CodeLocation(
                        file_path="",  # This needs to be set by the caller
                        start_line=node.start_point[0] + 1,
                        start_column=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1],
                        code_snippet=node.text.decode('utf8')
                    )
                    
                    functions.append({
                        'name': function_name,
                        'node': node,
                        'body': function_body,
                        'location': location
                    })
        except Exception as e:
            logger.error(f"Error extracting functions: {e}")
        
        return functions
    
    def identify_sources_and_sinks(
        self, 
        tree: Tree, 
        source_patterns: List[str] = None,
        sink_patterns: List[str] = None
    ) -> CodeFlowGraph:
        """
        Identify potential sources and sinks in the code
        
        Args:
            tree: Parsed tree-sitter tree
            source_patterns: List of patterns identifying sources
            sink_patterns: List of patterns identifying sinks
            
        Returns:
            CodeFlowGraph with identified sources and sinks
        """
        if not tree:
            return CodeFlowGraph()
            
        # Default patterns if none provided
        if not source_patterns:
            source_patterns = [
                'msg.sender', 'msg.value', 'msg.data',  # Transaction context
                'tx.origin', 'block.',  # Blockchain context
                'calldata', 'external', 'public'  # External inputs
            ]
            
        if not sink_patterns:
            sink_patterns = [
                'transfer', 'send', 'call',  # External calls
                'delegatecall', 'staticcall',  # Dangerous calls
                'selfdestruct', 'suicide',  # Destruction
                '=', 'delete'  # State changes
            ]
        
        flow_graph = CodeFlowGraph()
        
        try:
            # Identify sources
            self._find_pattern_matches(tree, source_patterns, FlowNodeType.SOURCE, flow_graph)
            
            # Identify sinks
            self._find_pattern_matches(tree, sink_patterns, FlowNodeType.SINK, flow_graph)
            
            # TODO: Connect sources to sinks with control/data flow analysis
            
        except Exception as e:
            logger.error(f"Error identifying sources and sinks: {e}")
        
        return flow_graph
    
    def _find_pattern_matches(
        self,
        tree: Tree,
        patterns: List[str],
        node_type: FlowNodeType,
        flow_graph: CodeFlowGraph
    ) -> None:
        """
        Find matches for the given patterns and add them to the flow graph
        
        Args:
            tree: Parsed tree-sitter tree
            patterns: List of patterns to search for
            node_type: Type of flow node to create
            flow_graph: Flow graph to add nodes to
        """
        for pattern in patterns:
            # For simple string matching (a more sophisticated approach would use queries)
            query_string = f'(identifier) @id'
            try:
                query = self.solidity_language.query(query_string)
                captures = query.captures(tree.root_node)
                
                for i, (node, tag) in enumerate(captures):
                    if tag == 'id':
                        text = node.text.decode('utf8')
                        if pattern in text:
                            # Create a flow node
                            flow_node = FlowNode(
                                id=f"{node_type.value}_{i}",
                                name=text,
                                node_type=node_type,
                                location=CodeLocation(
                                    file_path="",  # Set by caller
                                    start_line=node.start_point[0] + 1,
                                    start_column=node.start_point[1],
                                    end_line=node.end_point[0] + 1,
                                    end_column=node.end_point[1],
                                    code_snippet=text
                                )
                            )
                            flow_graph.add_node(flow_node)
            except Exception as e:
                logger.error(f"Error matching pattern '{pattern}': {e}")
    
    def build_control_flow_graph(self, function_node: Node) -> CodeFlowGraph:
        """
        Build a control flow graph for a function
        
        Args:
            function_node: Tree-sitter node for the function
            
        Returns:
            Control flow graph
        """
        # This would be a complex implementation - simplified for now
        flow_graph = CodeFlowGraph(
            function_name=function_node.child_by_field_name('name').text.decode('utf8') 
            if function_node.child_by_field_name('name') else "unknown"
        )
        
        # For a real implementation, we would:
        # 1. Identify basic blocks
        # 2. Connect blocks with control flow edges
        # 3. Analyze conditions and loops
        
        return flow_graph
    
    async def analyze_contract_flows(
        self,
        contract_code: str,
        file_path: str = "",
        source_patterns: List[str] = None,
        sink_patterns: List[str] = None
    ) -> Dict[str, CodeFlowGraph]:
        """
        Analyze the flows in a contract
        
        Args:
            contract_code: Solidity code to analyze
            file_path: Path to the file (for reference)
            source_patterns: List of patterns identifying sources
            sink_patterns: List of patterns identifying sinks
            
        Returns:
            Dictionary mapping function names to their flow graphs
        """
        if not TREESITTER_AVAILABLE or not self.parser:
            logger.warning("tree-sitter not available for flow analysis")
            return {}
            
        flow_graphs = {}
        
        try:
            # Parse the contract
            tree = self.parse_code(contract_code)
            if not tree:
                return {}
                
            # Extract contract name
            contract_name = "UnknownContract"
            query_string = '(contract_declaration name: (identifier) @name)'
            try:
                query = self.solidity_language.query(query_string)
                captures = query.captures(tree.root_node)
                if captures:
                    for node, tag in captures:
                        if tag == 'name':
                            contract_name = node.text.decode('utf8')
                            break
            except Exception as e:
                logger.error(f"Error extracting contract name: {e}")
            
            # Extract functions
            functions = self.extract_functions(tree)
            
            # Analyze each function
            for func_info in functions:
                func_name = func_info.get('name', 'unknown')
                node = func_info.get('node')
                body = func_info.get('body')
                
                if body:
                    # Create flow graph for this function
                    flow_graph = self.identify_sources_and_sinks(
                        Tree(body), source_patterns, sink_patterns
                    )
                    flow_graph.function_name = func_name
                    flow_graph.contract_name = contract_name
                    
                    # Update location information
                    for flow_node in flow_graph.nodes:
                        if flow_node.location:
                            flow_node.location.file_path = file_path
                    
                    flow_graphs[func_name] = flow_graph
        
        except Exception as e:
            logger.error(f"Error analyzing contract flows: {e}")
        
        return flow_graphs
