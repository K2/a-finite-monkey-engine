"""
Data flow analyzer for Solidity contracts using tree-sitter.
Identifies exploitable source-to-sink paths that can be controlled by users.
"""

import asyncio
import os
import json
import re
from typing import Dict, List, Any, Set, Tuple, Optional
from pathlib import Path
from loguru import logger

from llama_index.core.settings import Settings
from ..pipeline.core import Context
from ..sitter.sitter import TreeSitterGraph
from ..sitter import sitterQL
from ..nodes_config import config
class DataFlowAnalyzer:
    """
    Analyzer for data flow paths in Solidity contracts.
    Identifies source-to-sink paths that can be exploited by users.
    """
    
    def __init__(self, llm_adapter=None):
        """
        Initialize the data flow analyzer
        
        Args:
            llm_adapter: LLM adapter for analysis assistance
        """
        if llm_adapter is None:
            try:
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                
                # Use scan model for data flow analysis
                self.llm_adapter = LlamaIndexAdapter(
                    model_name=config.SCAN_MODEL,
                    provider=config.SCAN_MODEL_PROVIDER,
                    base_url=config.SCAN_MODEL_BASE_URL
                )
                logger.info(f"Created dataflow analysis LLM adapter with model: {config.SCAN_MODEL}")
            except Exception as e:
                logger.error(f"Failed to create dataflow LLM adapter: {e}")
                self.llm_adapter = None
        else:
            self.llm_adapter = llm_adapter

        self.sources = set()  # Functions/variables that can be controlled by users
        self.sinks = set()    # Functions/operations that can be attacked
        
        # Initialize TreeSitterGraph for Solidity analysis
        try:
            self.tsg = TreeSitterGraph()
            
            # Initialize queries from sitterQL
            # We'll use taint queries as they already define sources and sinks
            self.taint_query = sitterQL.traceWithTaint()
            
            # Use the general function query to find functions
            self.function_query = sitterQL.getFunctionsQuery()
            
            self.tree_sitter_available = True
            logger.info("TreeSitterGraph initialized for Solidity analysis")
        except Exception as e:
            self.tree_sitter_available = False
            logger.warning(f"TreeSitterGraph initialization failed: {e}")
            logger.warning("Falling back to regex-based analysis")
    
    async def process(self, context: Context) -> Context:
        """
        Process the context to analyze data flows
        
        Args:
            context: Processing context with files and previous analysis
            
        Returns:
            Updated context with data flow analysis
        """
        logger.info("Starting data flow analysis for exploitable paths")
        
        # Skip if tree-sitter is not available
        if not self.tree_sitter_available:
            logger.warning("Skipping data flow analysis: TreeSitterGraph not available")
            context.add_error(
                stage="dataflow_analysis",
                message="TreeSitterGraph not available",
                exception=None
            )
            return context
        
        # Initialize data flow findings
        if not hasattr(context, "dataflows"):
            context.dataflows = {}
        
        # Get list of files to analyze
        solidity_files = [(file_id, file_data) for file_id, file_data in context.files.items() 
                       if file_data.get("is_solidity", False)]
        
        logger.info(f"Analyzing data flows in {len(solidity_files)} Solidity files")
        
        # Process files in chunks to manage resources
        chunk_size = 3  # Process fewer files concurrently as this is more intensive
        for i in range(0, len(solidity_files), chunk_size):
            chunk = solidity_files[i:i+chunk_size]
            
            # Process this chunk of files concurrently
            tasks = [self._analyze_file(context, file_id, file_data) for file_id, file_data in chunk]
            await asyncio.gather(*tasks)
            
            # Prevent resource exhaustion
            await asyncio.sleep(0.2)
        
        # After individual file analysis, perform cross-contract flow analysis
        await self._analyze_cross_contract_flows(context)
        
        # Generate summaries for LLM vulnerability analysis
        await self._generate_flow_summaries(context)
        
        total_flows = sum(len(flows) for flows in context.dataflows.values())
        logger.info(f"Data flow analysis complete. Found {total_flows} exploitable paths")
        return context
    
    async def _analyze_file(self, context: Context, file_id: str, file_data: Dict[str, Any]):
        """
        Analyze a file for data flows
        
        Args:
            context: Processing context
            file_id: ID of the file to analyze
            file_data: File data dictionary
        """
        try:
            # Parse the contract with TreeSitterGraph
            content = file_data["content"]
            
            # Load content into TreeSitterGraph
            self.tsg.lineArr = content.splitlines()
            
            # Parse using TreeSitterGraph's parser
            tree = self.tsg.parser.parse(bytes(content, "utf8"))
            
            # Initialize dataflows for this file
            context.dataflows[file_id] = []
            
            # Find sources (user-controllable inputs)
            sources = self._find_sources(tree, content)
            
            # Find sinks (vulnerable operations)
            sinks = self._find_sinks(tree, content)
            
            # Find variables influencing control flow
            control_vars = self._find_control_variables(tree, content)
            
            # Identify flows from sources to sinks
            flows = self._analyze_flows(tree, content, sources, sinks, control_vars)
            
            # Add business flow context from previous analysis
            if hasattr(context, "business_flows") and file_id in context.business_flows:
                flows = self._enrich_with_business_flows(flows, context.business_flows, file_id)
            
            # Store the flows in context
            context.dataflows[file_id] = flows
            
            logger.info(f"Found {len(flows)} exploitable data flows in {file_data.get('path', file_id)}")
            
        except Exception as e:
            logger.error(f"Error analyzing data flows in {file_id}: {str(e)}")
            context.add_error(
                stage="dataflow_analysis",
                message=f"Failed to analyze file: {file_id}",
                exception=e
            )
    
    def _find_sources(self, tree, content: str) -> List[Dict[str, Any]]:
        """
        Find sources (user-controllable inputs) in the code using sitterQL queries
        
        Args:
            tree: Tree-sitter parse tree
            content: Source code content
            
        Returns:
            List of identified sources
        """
        sources = []
        
        # Use the Taint query from sitterQL which already identifies sources
        # Run the query on the entire tree
        try:
            # Define a callback to process sources
            def source_callback(t, node, key):
                if key == "source" or key == "user_input":
                    # Extract the source information
                    source_type = "unknown"
                    source_name = "unknown"
                    
                    # Try to determine source type and name
                    if node.type == "identifier":
                        source_name = node.text.decode('utf8') if hasattr(node.text, 'decode') else str(node.text)
                        
                        # Check for msg.sender, msg.value, tx.origin
                        parent = node.parent
                        if parent and parent.type == "member_expression":
                            if source_name in ["sender", "value", "data"]:
                                object_node = parent.child_by_field_name('object')
                                if object_node and object_node.text == b"msg":
                                    source_type = "msg_property"
                                    source_name = f"msg.{source_name}"
                            elif source_name == "origin":
                                object_node = parent.child_by_field_name('object')
                                if object_node and object_node.text == b"tx":
                                    source_type = "tx_property"
                                    source_name = "tx.origin"
                    elif node.type == "function_definition":
                        source_type = "function"
                        name_node = node.child_by_field_name('name')
                        if name_node:
                            source_name = name_node.text.decode('utf8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                    
                    # Create source entry
                    sources.append({
                        "type": source_type,
                        "name": source_name,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "controllable": "user_input",
                        "text": TreeSitterGraph.read_node(self.tsg.lineArr, node)
                    })
            
            # Use the modified method to find sources
            self.tsg.parse_sol(content, self.tsg.taintQuery, False, source_callback)
            
            # If we didn't find any sources with taint query, fall back to function parameters
            if not sources:
                self._find_function_parameters(tree, content, sources)
            

            if not sources:
                self._find_tx_properties(tree, content, sources)
            
        except Exception as e:
            logger.error(f"Error in source finding: {e}")
            # Use a simpler approach
            self._find_function_parameters(tree, content, sources)
            self._find_tx_properties(tree, content, sources)
        
        return sources
    
    def _find_function_parameters(self, tree, content: str, sources: List[Dict[str, Any]]):
        """Find function parameters as sources"""
        try:
            # Use the function query from sitterQL
            def function_param_callback(t, node, key):
                if key == "function_definition" and node.type == "function_definition":
                    # Check if it's public or external
                    is_external = False
                    
                    # Check for visibility in children
                    for child in node.children:
                        if child.type == "visibility_qualifier" and child.text in [b"public", b"external"]:
                            is_external = True
                            break
                    
                    if is_external:
                        # Get function name
                        name_node = node.child_by_field_name('name')
                        if not name_node:
                            for child in node.children:
                                if child.type == "identifier":
                                    name_node = child
                                    break
                        
                        # Get parameters
                        param_list = []
                        param_node = None
                        for child in node.children:
                            if child.type == "parameter_list":
                                param_node = child
                                break
                        
                        if name_node and param_node:
                            function_name = name_node.text.decode('utf8') if hasattr(name_node.text, 'decode') else str(name_node.text)
                            
                            # Extract parameters
                            params = []
                            for param_child in param_node.children:
                                if param_child.type == "parameter":
                                    param_name = None
                                    param_type = None
                                    for detail in param_child.children:
                                        if detail.type == "identifier":
                                            param_name = detail.text.decode('utf8') if hasattr(detail.text, 'decode') else str(detail.text)
                                        elif detail.type in ["type_name", "elementary_type_name"]:
                                            param_type = detail.text.decode('utf8') if hasattr(detail.text, 'decode') else str(detail.text)
                                    
                                    if param_name and param_type:
                                        params.append({
                                            "name": param_name,
                                            "type": param_type
                                        })
                            
                            sources.append({
                                "type": "function",
                                "name": function_name,
                                "parameters": params,
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                                "start_byte": node.start_byte,
                                "end_byte": node.end_byte,
                                "controllable": "user_input",
                                "text": TreeSitterGraph.read_node(self.tsg.lineArr, node)
                            })
            
            # Apply the callback to the function query
            self.tsg.parse_sol(content, self.tsg.taintQuery, False, function_param_callback) 
        except Exception as e:
            logger.error(f"Error finding function parameters: {e}")
    
    def _find_tx_properties(self, tree, content: str, sources: List[Dict[str, Any]]):
        """Find msg.sender, msg.value and tx.origin as sources"""
        try:
            # Direct tree traversal for these specific properties
            def traverse_node(node):
                if node.type == "member_expression":
                    object_node = node.child_by_field_name('object')
                    property_node = node.child_by_field_name('property')
                    
                    if object_node and property_node and object_node.type == "identifier":
                        if object_node.text == b"msg" and property_node.text in [b"sender", b"value", b"data"]:
                            prop_text = property_node.text.decode('utf8') if hasattr(property_node.text, 'decode') else str(property_node.text)
                            sources.append({
                                "type": "msg_property",
                                "name": f"msg.{prop_text}",
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                                "start_byte": node.start_byte,
                                "end_byte": node.end_byte,
                                "controllable": "transaction_property",
                                "text": TreeSitterGraph.read_node(self.tsg.lineArr, node)
                            })
                        elif object_node.text == b"tx" and property_node.text == b"origin":
                            sources.append({
                                "type": "tx_property",
                                "name": "tx.origin",
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                                "start_byte": node.start_byte,
                                "end_byte": node.end_byte,
                                "controllable": "transaction_property",
                                "text": TreeSitterGraph.read_node(self.tsg.lineArr, node)
                            })
                
                # Recursively process children
                for child in node.children:
                    traverse_node(child)
            
            # Start traversal from root
            traverse_node(tree.root_node)
            
        except Exception as e:
            logger.error(f"Error finding tx properties: {e}")
    
    def _find_sinks(self, tree, content: str) -> List[Dict[str, Any]]:
        """
        Find sinks (vulnerable operations) in the code using sitterQL queries
        
        Args:
            tree: Tree-sitter parse tree
            content: Source code content
            
        Returns:
            List of identified sinks
        """
        sinks = []
        
        # Use the Taint query from sitterQL which already identifies sinks
        try:
            # Define a callback to process sinks
            def sink_callback(t, node, key):
                if key == "sink":
                    # Determine sink type and vulnerability category
                    sink_type = "unknown"
                    vulnerability = "unknown"
                    sink_name = "unknown"
                    
                    # Try to identify the sink type
                    if node.type == "call_expression":
                        function_node = node.child_by_field_name('function')
                        if function_node:
                            if function_node.type == "member_expression":
                                property_node = function_node.child_by_field_name('property')
                                if property_node and property_node.text:
                                    method = property_node.text.decode('utf8') if hasattr(property_node.text, 'decode') else str(property_node.text)
                                    if method in ["transfer", "send", "call"]:
                                        sink_type = "external_call"
                                        sink_name = method
                                        vulnerability = "reentrancy"
                                    elif method == "delegatecall":
                                        sink_type = "delegatecall"
                                        sink_name = "delegatecall"
                                        vulnerability = "code_injection"
                            elif function_node.type == "identifier" and function_node.text == b"selfdestruct":
                                sink_type = "selfdestruct"
                                sink_name = "selfdestruct"
                                vulnerability = "contract_destruction"
                    elif node.type in ["inline_assembly_statement", "assembly_block"]:
                        sink_type = "assembly"
                        sink_name = "assembly_block"
                        vulnerability = "low_level_operation"
                    elif node.type == "assignment_expression":
                        sink_type = "state_write"
                        sink_name = "storage_write"
                        vulnerability = "state_manipulation"
                    
                    # Add to sinks list
                    sinks.append({
                        "type": sink_type,
                        "name": sink_name,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "vulnerability": vulnerability,
                        "text": TreeSitterGraph.read_node(self.tsg.lineArr, node)
                    })
            
            # Run the query using sitterQL's existing taint query
            self.tsg.parse_sol(content, self.taint_query, False, sink_callback) 
            # If no sinks found, use fallback
            if not sinks:
                self._find_sinks_manually(tree, content, sinks)
                
        except Exception as e:
            logger.error(f"Error in sink finding: {e}")
            # Use fallback approach
            self._find_sinks_manually(tree, content, sinks) 
            logger.info("Fallback sink finding initiated.")
        
        return sinks
    
    def _find_sinks_manually(self, tree, content: str, sinks: List[Dict[str, Any]]):
        """Find sinks manually by traversing the tree"""
        def traverse_node(node):
            # Check for call expressions that may be sinks
            if node.type == "call_expression":
                # Check for member expressions (transfer, send, call, delegatecall)
                function_node = node.child_by_field_name('function')
                
                if function_node and function_node.type == "member_expression":
                    property_node = function_node.child_by_field_name('property')
                    
                    if property_node and property_node.type == "identifier":
                        method_name = property_node.text.decode('utf8') if hasattr(property_node.text, 'decode') else str(property_node.text)
                        
                        if method_name in ["transfer", "send", "call"]:
                            sinks.append({
                                "type": "external_call",
                                "name": method_name,
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                                "start_byte": node.start_byte,
                                "end_byte": node.end_byte,
                                "vulnerability": "reentrancy",
                                "text": TreeSitterGraph.read_node(self.tsg.lineArr, node)
                            })
                        elif method_name == "delegatecall":
                            sinks.append({
                                "type": "delegatecall",
                                "name": method_name,
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                                "start_byte": node.start_byte,
                                "end_byte": node.end_byte,
                                "vulnerability": "code_injection",
                                "text": TreeSitterGraph.read_node(self.tsg.lineArr, node)
                            })
                
                # Check for selfdestruct
                elif function_node and function_node.type == "identifier" and function_node.text == b"selfdestruct":
                    sinks.append({
                        "type": "selfdestruct",
                        "name": "selfdestruct",
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "vulnerability": "contract_destruction",
                        "text": TreeSitterGraph.read_node(self.tsg.lineArr, node)
                    })
            
            # Check for assembly blocks
            elif node.type in ["inline_assembly_statement", "assembly_block"]:
                sinks.append({
                    "type": "assembly",
                    "name": "assembly_block",
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                    "vulnerability": "low_level_operation",
                    "text": TreeSitterGraph.read_node(self.tsg.lineArr, node)
                })
            
            # Recursively process children
            for child in node.children:
                traverse_node(child)
        
        # Start traversal from root
        traverse_node(tree.root_node)
            
    def _find_control_variables(self, tree, content: str) -> List[Dict[str, Any]]:
        """Find variables that influence control flow"""
        control_vars = []
        var_dict = {}  # Track unique variables
        
        try:
            # Use sitterQL's existing queries or tree traversal to find control variables
            
            def traverse_node(node):
                # Check for conditional statements
                if node.type in ["if_statement", "while_statement", "for_statement"]:
                    condition_node = node.child_by_field_name('condition')
                    if condition_node:
                        extract_variables(condition_node, var_dict)
                
                # Check for require/assert statements
                elif node.type == "call_expression":
                    function_node = node.child_by_field_name('function')
                    if function_node and function_node.type == "identifier":
                        func_name = function_node.text.decode('utf8') if hasattr(function_node.text, 'decode') else str(function_node.text)
                        if func_name in ["require", "assert"]:
                            args_node = node.child_by_field_name('arguments')
                            if args_node and args_node.children:
                                # First argument is the condition
                                condition = args_node.children[0]
                                extract_variables(condition, var_dict)
                
                # Recursively process children
                for child in node.children:
                    traverse_node(child)
            
            def extract_variables(node, var_dict):
                """Extract variables from a node and add to the dictionary"""
                if node.type == "identifier":
                    var_name = node.text.decode('utf8') if hasattr(node.text, 'decode') else str(node.text)
                    if var_name not in var_dict:
                        var_dict[var_name] = {
                            "name": var_name,
                            "type": "control_variable",
                            "influences_flow": True
                        }
                
                # Process children
                for child in node.children:
                    extract_variables(child, var_dict)
            
            # Start traversal
            traverse_node(tree.root_node)
            
            # Convert dictionary to list
            control_vars = list(var_dict.values())
            
        except Exception as e:
            logger.error(f"Error finding control variables: {e}")
        
        return control_vars
    
    def _extract_variables_from_node(self, node) -> List[str]:
        """Extract variable names from an expression node"""
        variables = []
        
        def traverse(current_node):
            if current_node.type == "identifier":
                variables.append(current_node.text.decode('utf8'))
            for child in current_node.children:
                traverse(child)
                
        traverse(node)
        return variables
    
    def _analyze_flows(self, tree, content: str, sources: List[Dict[str, Any]], 
                      sinks: List[Dict[str, Any]], control_vars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze data flows from sources to sinks"""
        flows = []
        
        # For each source-sink pair, try to find a path
        for source in sources:
            for sink in sinks:
                # Try to find a data flow path from source to sink
                path = self._find_path(tree, content, source, sink)
                
                if path:
                    # Find control variables that affect this path
                    affecting_vars = []
                    for var in control_vars:
                        if self._variable_affects_path(var, path, content):
                            affecting_vars.append(var)
                    
                    # Only include flows with at least 2 affecting variables (as per requirement)
                    if len(affecting_vars) >= 2:
                        # Count shared nodes with other flows
                        shared_nodes = self._count_shared_nodes(path, flows)
                        
                        flows.append({
                            "source": source,
                            "sink": sink,
                            "path": path,
                            "affecting_variables": affecting_vars,
                            "shared_nodes": shared_nodes,
                            "exploitability": self._calculate_exploitability(source, sink, affecting_vars)
                        })
        
        # Sort flows by exploitability
        flows.sort(key=lambda x: x["exploitability"], reverse=True)
        return flows
    
    def _find_path(self, tree, content: str, source: Dict[str, Any], sink: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find a data flow path from source to sink"""
        # This is a simplified implementation - in practice, you would use a more
        # sophisticated approach like taint tracking or control flow analysis
        
        # For now, we'll use a very basic heuristic - check if the source and sink
        # are within the same function and the source appears before the sink
        if source["start_line"] <= sink["start_line"]:
            # Find the function containing both source and sink
            function = self._find_containing_function(tree, source["start_line"], sink["end_line"])
            
            if function:
                # Extract the function content
                func_content = content[function["start_byte"]:function["end_byte"]]
                
                # Create a path representation
                return [
                    {
                        "type": "source",
                        "name": source["name"],
                        "start_line": source["start_line"],
                        "end_line": source["end_line"],
                        "code": content[source["start_byte"]:source["end_byte"]]
                    },
                    {
                        "type": "function",
                        "name": function["name"],
                        "start_line": function["start_line"],
                        "end_line": function["end_line"],
                        "code": func_content
                    },
                    {
                        "type": "sink",
                        "name": sink["name"],
                        "start_line": sink["start_line"],
                        "end_line": sink["end_line"],
                        "code": content[sink["start_byte"]:sink["end_byte"]]
                    }
                ]
        
        return []  # No path found
    
    def _find_containing_function(self, tree, start_line: int, end_line: int) -> Dict[str, Any]:
        """Find function containing the specified line range"""
        # Query to find all functions
        query_string = """
        (function_definition
            name: (identifier) @function_name) @function
        """
        query = self.tsg.LANG.query(query_string)
        captures = query.captures(tree.root_node)
        
        # Find functions that contain the range
        for node, tag in captures:
            if tag == "function":
                func_start = node.start_point[0] + 1
                func_end = node.end_point[0] + 1
                
                if func_start <= start_line and func_end >= end_line:
                    # Find the function name
                    name = "unknown"
                    for name_node, name_tag in captures:
                        if name_tag == "function_name" and name_node.parent == node:
                            name = name_node.text.decode('utf8')
                            break
                    
                    return {
                        "name": name,
                        "start_line": func_start,
                        "end_line": func_end,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte
                    }
        
        return None
    
    def _variable_affects_path(self, var: Dict[str, Any], path: List[Dict[str, Any]], content: str) -> bool:
        """Check if a variable affects a path"""
        # Simple check - see if the variable name appears in the path code
        var_name = var["name"]
        for node in path:
            if "code" in node and var_name in node["code"]:
                return True
        return False
    
    def _count_shared_nodes(self, path: List[Dict[str, Any]], other_flows: List[Dict[str, Any]]) -> int:
        """Count shared nodes between this path and other flows"""
        # For simplicity, we'll count line overlaps as shared nodes
        path_lines = set()
        for node in path:
            path_lines.add(node["start_line"])
        
        shared = 0
        for flow in other_flows:
            for node in flow["path"]:
                if node["start_line"] in path_lines:
                    shared += 1
                    break  # Count each flow only once
        
        return shared
    
    def _calculate_exploitability(self, source: Dict[str, Any], sink: Dict[str, Any], affecting_vars: List[Dict[str, Any]]) -> float:
        """Calculate exploitability score"""
        # Basic scoring system - can be refined
        base_score = 5.0
        
        # Adjust based on sink types
        sink_scores = {
            "reentrancy": 3.0,
            "code_injection": 4.0,
            "contract_destruction": 5.0,
            "low_level_operation": 2.0,
            "state_manipulation": 2.5
        }
        base_score += sink_scores.get(sink.get("vulnerability"), 0.0)
        
        # Adjust based on source controllability
        if source["type"] == "function" and source.get("controllable") == "user_input":
            base_score += 1.0
        elif source["name"] in ["msg.sender", "msg.value", "tx.origin"]:
            base_score += 1.5
        
        # Adjust based on affecting variables
        base_score += min(len(affecting_vars) * 0.5, 2.0)
        
        # Cap at 10
        return min(base_score, 10.0)
    
    async def _analyze_cross_contract_flows(self, context: Context):
        """Analyze flows across multiple contracts"""
        # This would require more sophisticated analysis
        # For now, we'll just note this as a future enhancement
        pass
    
    def _enrich_with_business_flows(self, flows: List[Dict[str, Any]], 
                                  business_flows: Dict, file_id: str) -> List[Dict[str, Any]]:
        """Add business flow context to data flows"""
        if file_id not in business_flows:
            return flows
            
        file_business_flows = business_flows[file_id]
        for flow in flows:
            relevant_flows = []
            
            # Extract range of lines in the flow
            flow_lines = set()
            for node in flow["path"]:
                if "start_line" in node and "end_line" in node:
                    for line in range(node["start_line"], node["end_line"] + 1):
                        flow_lines.add(line)
            
            # Find business flows that overlap with this data flow
            for bflow in file_business_flows:
                if not isinstance(bflow, dict):
                    continue
                
                # Check if there's an overlap in lines
                if "start_line" in bflow and "end_line" in bflow:
                    bflow_lines = set(range(bflow["start_line"], bflow["end_line"] + 1))
                    if flow_lines.intersection(bflow_lines):
                        relevant_flows.append(bflow)
            
            # Add business context to the flow
            flow["business_context"] = relevant_flows
        
        return flows
    
    async def _generate_flow_summaries(self, context: Context):
        """Generate LLM-friendly summaries of exploitable flows"""
        if not self.llm_adapter or not hasattr(context, "dataflows"):
            logger.warning("No LLM available for flow summary generation")
            return
        
        llm = self.llm_adapter.llm if self.llm_adapter else Settings.llm
        if not llm:
            logger.warning("No LLM available for flow summary generation")
            return
        
        logger.info("Generating summaries for exploitable data flows")
        
        for file_id, flows in context.dataflows.items():
            if not flows:
                continue
            
            file_data = context.files.get(file_id, {})
            file_name = file_data.get("name", file_id)
            
            for i, flow in enumerate(flows):
                if "summary" in flow:
                    # Skip if already has a summary
                    continue
                
                try:
                    # Extract essential information for the prompt
                    source_info = f"{flow['source']['name']} (controllable by {flow['source'].get('controllable', 'unknown')})"
                    sink_info = f"{flow['sink']['name']} (vulnerability type: {flow['sink'].get('vulnerability', 'unknown')})"
                    
                    # Extract affected business flows
                    business_context = ""
                    if "business_context" in flow and flow["business_context"]:
                        business_context = "Affected business flows:\n"
                        business_flows = flow["business_context"]
                        for bf in business_flows[:3]:  # Limit to 3 for brevity
                            if "name" in bf:
                                business_context += f"- {bf['name']}\n"
                    
                    # Extract relevant code snippets
                    code_snippets = ""
                    for node in flow["path"]:
                        if "code" in node and len(node["code"]) < 500:  # Limit size
                            code_snippets += f"--- {node['type'].upper()} ({node['name']}) ---\n"
                            code_snippets += f"{node['code']}\n\n"
                    
                    # Generate the prompt for the LLM
                    prompt = f"""
                    You are a smart contract security analyst. Analyze this exploitable data flow and provide a concise summary:
                    
                    File: {file_name}
                    Data Flow: {source_info} → {sink_info}
                    Exploitability Score: {flow['exploitability']}/10
                    Affecting Variables: {', '.join(v['name'] for v in flow['affecting_variables'])}
                    Shared Nodes with Other Flows: {flow['shared_nodes']}
                    
                    {business_context}
                    
                    Relevant Code:
                    {code_snippets}
                    
                    Provide a concise summary of:
                    1. How this flow could be exploited
                    2. What the impact would be
                    3. What security checks should be implemented
                    
                    Format as JSON with the following structure:
                    {{
                        "attack_vector": "Brief description of how a user could exploit this flow",
                        "impact": "What would happen if exploited",
                        "mitigation": "How to protect against this",
                        "severity": "Critical|High|Medium|Low based on exploitability and impact"
                    }}
                    """
                    
                    # Get response from LLM
                    response = await llm.acomplete(prompt)
                    response_text = response.text
                    
                    # Extract and parse JSON response
                    summary = None
                    try:
                        # Try to parse as JSON directly
                        summary = json.loads(response_text)
                    except json.JSONDecodeError:
                        # Try to extract JSON from the response
                        try:
                            import re
                            json_pattern = r'\{[\s\S]*\}'
                            match = re.search(json_pattern, response_text)
                            if match:
                                json_str = match.group(0)
                                summary = json.loads(json_str)
                        except (json.JSONDecodeError, Exception):
                            logger.warning(f"Failed to parse flow summary as JSON: {response_text[:100]}...")
                            continue
                    
                    # Add summary to the flow
                    if summary:
                        flow["summary"] = summary
                        logger.debug(f"Generated summary for flow {i+1} in {file_name}")
                
                except Exception as e:
                    logger.error(f"Error generating summary for flow {i+1} in {file_id}: {str(e)}")
    
    async def process(self, context: Context) -> Context:
        """Process the context to analyze data flows"""
        logger.info("Analyzing data flows in contracts and correlating with attack surfaces")
        
        try:
            # Initialize data flow analysis if not exists
            if not hasattr(context, 'data_flow_analysis'):
                context.data_flow_analysis = {
                    'flows': [],
                    'variables': {},
                    'high_risk_flows': []
                }
                
            # Process flows
            await self._analyze_flows(context)
            
            # Correlate with attack surfaces if available
            if hasattr(context, 'attack_surfaces'):
                await self._correlate_with_attack_surfaces(context)
                
            logger.info("Data flow analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in data flow analysis: {str(e)}")
            context.add_error(
                stage="data_flow_analysis",
                message="Failed to analyze data flows",
                exception=e
            )
            
        return context
        
    async def _analyze_flows(self, context: Context):
        """Analyze data flows between functions"""
        # Existing data flow analysis implementation...
        pass
        
    async def _correlate_with_attack_surfaces(self, context: Context):
        """Correlate data flows with identified attack surfaces"""
        try:
            # Skip if no attack surfaces available
            if not hasattr(context, 'attack_surfaces') or not context.attack_surfaces:
                logger.debug("No attack surfaces available to correlate with data flows")
                return
                
            # Get all vulnerable variables from attack surfaces
            vulnerable_vars = set()
            for surface_id, surface in context.attack_surfaces.items():
                for var in surface.get('variables', []):
                    vulnerable_vars.add(var.get('name'))
            
            # Identify high-risk data flows (flows involving vulnerable variables)
            high_risk_flows = []
            for flow in context.data_flow_analysis.get('flows', []):
                # Check if flow involves any vulnerable variables
                involves_vulnerable_var = any(
                    var in vulnerable_vars for var in flow.get('variables', [])
                )
                
                if involves_vulnerable_var:
                    flow['high_risk'] = True
                    high_risk_flows.append(flow)
                    
            # Add high risk flows to analysis
            context.data_flow_analysis['high_risk_flows'] = high_risk_flows
            
            logger.info(f"Identified {len(high_risk_flows)} high-risk data flows involving potentially vulnerable variables")
            
            # Create a comprehensive attack path analysis
            context.data_flow_analysis['attack_paths'] = self._generate_attack_paths(context, high_risk_flows)
            
        except Exception as e:
            logger.error(f"Error correlating data flows with attack surfaces: {str(e)}")
            
    def _generate_attack_paths(self, context: Context, high_risk_flows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate potential attack paths based on high-risk data flows"""
        attack_paths = []
        
        try:
            # Group flows by entry points (external/public functions)
            entry_points = {}
            
            # Identify entry points from functions
            for func_id, func_data in context.functions.items() if hasattr(context, 'functions') else {}:
                visibility = func_data.get('visibility', '')
                if visibility in ['public', 'external']:
                    entry_points[func_data.get('name', '')] = {
                        'id': func_id,
                        'name': func_data.get('name', ''),
                        'contract': func_data.get('contract_name', ''),
                        'flows': []
                    }
            
            # Connect entry points to high-risk flows
            for flow in high_risk_flows:
                src_func = flow.get('source_function', '')
                if src_func in entry_points:
                    entry_points[src_func]['flows'].append(flow)
            
            # Build attack paths from entry points
            for entry_name, entry_data in entry_points.items():
                if entry_data['flows']:
                    attack_paths.append({
                        'entry_point': entry_name,
                        'contract': entry_data['contract'],
                        'flows': entry_data['flows'],
                        'variables_at_risk': self._extract_variables_at_risk(entry_data['flows']),
                        'severity': 'high' if len(entry_data['flows']) > 2 else 'medium'
                    })
            
        except Exception as e:
            logger.error(f"Error generating attack paths: {str(e)}")
        
        return attack_paths
        
    def _extract_variables_at_risk(self, flows: List[Dict[str, Any]]) -> List[str]:
        """Extract unique variables at risk from a set of flows"""
        variables = set()
        
        for flow in flows:
            for var in flow.get('variables', []):
                variables.add(var)
                
        return list(variables)
