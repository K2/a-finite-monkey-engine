"""
Transformers module for Finite Monkey Engine pipeline

This module provides transformers for the pipeline:
- ContractChunker: Split contracts into manageable chunks
- FunctionExtractor: Extract functions from contracts
- CallGraphBuilder: Build a call graph of contract functions
- ASTAnalyzer: Analyze contract AST for security patterns
"""

from contextvars import Context
import re
from loguru import logger
import streamlit as st
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import asyncio 

class AgentState(Enum):
    """Enum for agent workflow states"""
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    ERROR = "ERROR"
    FAILED = "FAILED"
    COMPLETED = "COMPLETE"

@dataclass
class WorkflowContext:
    """Context for workflow tracking"""
    state: AgentState
    metadata: dict = field(default_factory=dict)

def agent_workflow(cls: Optional[type] = None, *, initial_state: AgentState = AgentState.IDLE) -> Callable:
    """Class decorator to add workflow management to an agent class"""
    def wrap(cls):
        original_init = cls.__init__
        
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.state = initial_state
            self.workflow_context = WorkflowContext(state=initial_state)
            if not hasattr(self, 'name'):
                self.name = cls.__name__
                
        async def _set_state(self, state: AgentState):
            self.state = state
            self.workflow_context.state = state
            if st.session_state.get('debug'):
                st.write(f"{self.name}: {state.value}")
            
        cls.__init__ = __init__
        cls._set_state = _set_state
        return cls
    
    return wrap if cls is None else wrap(cls)

@agent_workflow
class AsyncContractChunker:
    """Split Solidity contracts into manageable chunks"""
    
    def __init__(self, name: str = "ContractChunker"):
        self.name = name
        self.state = AgentState.IDLE
        
    @staticmethod
    async def process(context: Context, *args, **kwargs) -> Context:
        if st.session_state.get('debug'):
            st.write(f"Processing {len(context.files)} files")
            
        # ... rest of process method ...
        return context


@agent_workflow
class FunctionExtractor:
    """Extract functions from Solidity contracts"""
    
    # Regex pattern to match Solidity functions
    FUNCTION_PATTERN = re.compile(
        r'(function\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*'
        r'(public|private|internal|external)?'
        r'\s*(pure|view|payable)?'
        r'\s*(returns\s*\([^)]*\))?\s*'
        r'({[^}]*}))',
        re.DOTALL
    )
    
    # Pattern to extract function visibility
    VISIBILITY_PATTERN = re.compile(r'(public|private|internal|external)')
    
    # Pattern to extract function modifiers
    MODIFIER_PATTERN = re.compile(r'(pure|view|payable)')
    
    def __init__(self, name: str = "FunctionExtractor"):
        """Initialize the FunctionExtractor agent"""
        self.name = name
        self.state = AgentState.IDLE
    
    @staticmethod
    def extract_functions_from_text(content: str) -> List[Dict[str, Any]]:
        """Extract Solidity functions from text"""
        functions = []
        
        # Find all functions
        matches = FunctionExtractor.FUNCTION_PATTERN.finditer(content)
        for match in matches:
            full_text = match.group(1)
            name = match.group(2)
            params = match.group(3)
            visibility = match.group(4) or "public"  # Default is public
            modifier = match.group(5) or ""
            returns = match.group(6) or ""
            body = match.group(7)
            
            # Extract line numbers
            lines_before = content[:match.start()].count('\n') + 1
            lines_body = full_text.count('\n')
            
            function = {
                "name": name,
                "params": params.strip(),
                "visibility": visibility,
                "modifier": modifier,
                "returns": returns,
                "body": body,
                "full_text": full_text,
                "start_line": lines_before,
                "end_line": lines_before + lines_body,
            }
            
            # Extract additional properties
            function["is_payable"] = "payable" in full_text
            function["is_view"] = "view" in full_text
            function["is_pure"] = "pure" in full_text
            function["is_constructor"] = name == "constructor"
            
            functions.append(function)
            
        return functions
    
    async def process(
        self,
        context: Context,
        data: Optional[str] = None,
        extract_modifiers: bool = True,
        extract_events: bool = True
    ) -> Context:
        """
        Extract functions from Solidity contracts
        
        Args:
            context: Pipeline context
            data: Optional file ID to process
            extract_modifiers: Whether to extract function modifiers
            extract_events: Whether to extract events
            
        Returns:
            Updated context with extracted functions
        """
        self.state = AgentState.INITIALIZING
        
        # Determine files to process
        if data is not None and isinstance(data, str):
            files_to_process = [data]
        else:
            files_to_process = [
                file_id for file_id, file_data in context.files.items()
                if file_data.get("is_solidity", False)
            ]
        
        self.state = AgentState.EXECUTING
        
        # Process each file
        for file_id in files_to_process:
            file_data = context.files.get(file_id)
            if not file_data:
                continue
                
            content = file_data["content"]
            
            # Extract functions
            functions = self.extract_functions_from_text(content)
            
            # Add functions to context
            for function in functions:
                function_id = f"{file_id}:{function['name']}"
                context.functions[function_id] = {
                    **function,
                    "id": function_id,
                    "file_id": file_id,
                    "file_path": file_data["path"]
                }
                
            # Update file with functions
            file_data["functions"] = [
                context.functions[f"{file_id}:{function['name']}"]
                for function in functions
            ]
            
            # Extract modifiers if requested
            if extract_modifiers:
                modifiers = self._extract_modifiers(content)
                file_data["modifiers"] = modifiers
                
            # Extract events if requested
            if extract_events:
                events = self._extract_events(content)
                file_data["events"] = events
                
            logger.info(f"Extracted {len(functions)} functions from {file_data['path']}")
        
        self.state = AgentState.COMPLETED
        return context
    
    def _extract_modifiers(self, content: str) -> List[Dict[str, Any]]:
        """Extract Solidity modifiers from text"""
        # Regex pattern to match Solidity modifiers
        modifier_pattern = re.compile(
            r'modifier\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*({[^}]*})',
            re.DOTALL
        )
        
        modifiers = []
        matches = modifier_pattern.finditer(content)
        
        for match in matches:
            name = match.group(1)
            params = match.group(2)
            body = match.group(3)
            
            # Extract line numbers
            lines_before = content[:match.start()].count('\n') + 1
            lines_body = (match.group(0)).count('\n')
            
            modifier = {
                "name": name,
                "params": params.strip(),
                "body": body,
                "start_line": lines_before,
                "end_line": lines_before + lines_body,
            }
            
            modifiers.append(modifier)
            
        return modifiers
    
    def _extract_events(self, content: str) -> List[Dict[str, Any]]:
        """Extract Solidity events from text"""
        # Regex pattern to match Solidity events
        event_pattern = re.compile(
            r'event\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*;',
            re.DOTALL
        )
        
        events = []
        matches = event_pattern.finditer(content)
        
        for match in matches:
            name = match.group(1)
            params = match.group(2)
            
            # Extract line number
            line_number = content[:match.start()].count('\n') + 1
            
            event = {
                "name": name,
                "params": params.strip(),
                "line": line_number,
                "indexed_params": self._extract_indexed_params(params)
            }
            
            events.append(event)
            
        return events
    
    def _extract_indexed_params(self, params_str: str) -> List[str]:
        """Extract indexed parameters from event params"""
        indexed_params = []
        if not params_str.strip():
            return indexed_params
            
        params = params_str.split(',')
        for param in params:
            if 'indexed' in param:
                # Extract parameter name
                parts = param.strip().split(' ')
                if len(parts) >= 2:
                    param_name = parts[-1].replace('indexed', '').strip()
                    indexed_params.append(param_name)
        
        return indexed_params


@agent_workflow
class CallGraphBuilder:
    """Build a call graph of contract functions"""
    
    def __init__(self, name: str = "CallGraphBuilder"):
        """Initialize the CallGraphBuilder agent"""
        self.name = name
        self.state = AgentState.IDLE
        
    async def process(
        self,
        context: Context,
        data: Optional[str] = None
    ) -> Context:
        """
        Build a call graph of contract functions
        
        Args:
            context: Pipeline context
            data: Optional file ID to process
            
        Returns:
            Updated context with call graph
        """
        self.state = AgentState.INITIALIZING
        
        # Determine files to process
        if data is not None and isinstance(data, str):
            files_to_process = [data]
        else:
            files_to_process = [
                file_id for file_id, file_data in context.files.items()
                if file_data.get("is_solidity", False)
            ]
            
        # Initialize call graph
        call_graph = {
            "nodes": [],  # Functions
            "edges": []   # Calls between functions
        }
        
        self.state = AgentState.EXECUTING
        
        # Process each file
        for file_id in files_to_process:
            file_data = context.files.get(file_id)
            if not file_data or not file_data.get("functions"):
                continue
                
            # Add all functions as nodes
            for function in file_data["functions"]:
                call_graph["nodes"].append({
                    "id": f"{file_id}:{function['name']}",
                    "name": function["name"],
                    "file": file_data["name"],
                    "visibility": function.get("visibility", "public"),
                    "type": "function"
                })
                
            # Add edges (calls between functions)
            for function in file_data["functions"]:
                function_id = f"{file_id}:{function['name']}"
                body = function.get("body", "")
                
                # Find function calls in the body
                for other_function in file_data["functions"]:
                    other_name = other_function["name"]
                    if other_name == function["name"]:
                        continue  # Skip self-references
                        
                    # Look for calls to this function
                    # This is a simple heuristic - more advanced parsing would be better
                    call_pattern = re.compile(
                        rf'{other_name}\s*\([^)]*\)',
                        re.DOTALL
                    )
                    
                    if call_pattern.search(body):
                        call_graph["edges"].append({
                            "source": function_id,
                            "target": f"{file_id}:{other_name}",
                            "type": "calls"
                        })
                
                # Also look for external calls and references
                self._find_external_calls(context, call_graph, file_id, function)
                
        # Add call graph to context
        context.state["call_graph"] = call_graph
        
        logger.info(f"Built call graph with {len(call_graph['nodes'])} nodes and {len(call_graph['edges'])} edges")
        
        self.state = AgentState.COMPLETED
        return context
    
    def _find_external_calls(
        self, 
        context: Context, 
        call_graph: Dict[str, List],
        file_id: str,
        function: Dict[str, Any]
    ) -> None:
        """Find external calls from a function to other contracts"""
        body = function.get("body", "")
        function_id = f"{file_id}:{function['name']}"
        
        # Look for contract instantiations: ContractName(args)
        instantiation_pattern = re.compile(
            r'new\s+([A-Z][a-zA-Z0-9_]*)\s*\(',
            re.DOTALL
        )
        
        for match in instantiation_pattern.finditer(body):
            contract_name = match.group(1)
            
            # Add contract node if it doesn't exist
            contract_id = f"contract:{contract_name}"
            if not any(node["id"] == contract_id for node in call_graph["nodes"]):
                call_graph["nodes"].append({
                    "id": contract_id,
                    "name": contract_name,
                    "type": "contract"
                })
                
            # Add instantiation edge
            call_graph["edges"].append({
                "source": function_id,
                "target": contract_id,
                "type": "instantiates"
            })
            
        # Look for contract calls: contractVar.method(args)
        # This is a simple heuristic - more advanced parsing would be better
        contract_call_pattern = re.compile(
            r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*\(',
            re.DOTALL
        )
        
        for match in contract_call_pattern.finditer(body):
            variable_name = match.group(1)
            method_name = match.group(2)
            
            # Add external call edge with best guess at target
            call_graph["edges"].append({
                "source": function_id,
                "target": f"external:{variable_name}.{method_name}",
                "type": "calls_external",
                "variable": variable_name,
                "method": method_name
            })


@agent_workflow
class ASTAnalyzer:
    """Analyze contract AST for security patterns"""
    
    def __init__(self, name: str = "ASTAnalyzer"):
        """Initialize the ASTAnalyzer agent"""
        self.name = name
        self.state = AgentState.IDLE
        
    async def process(
        self,
        context: Context,
        data: Optional[str] = None,
        analyze_patterns: bool = True,
        analyze_complexity: bool = True
    ) -> Context:
        """
        Analyze contract AST for security patterns
        
        Args:
            context: Pipeline context
            data: Optional file ID to process
            analyze_patterns: Whether to analyze for security patterns
            analyze_complexity: Whether to analyze code complexity
            
        Returns:
            Updated context with AST analysis
        """
        self.state = AgentState.INITIALIZING
        
        try:
            # Import TreeSitterAnalyzer only when needed
            try:
                from ..sitter.analyzer import TreeSitterAnalyzer
                analyzer = TreeSitterAnalyzer()
            except ImportError:
                logger.error("Tree-sitter not available, skipping AST analysis")
                context.add_error(
                    stage="ast_analyzer",
                    message="Tree-sitter not available",
                )
                self.state = AgentState.FAILED
                return context
                
            # Determine files to process
            if data is not None and isinstance(data, str):
                files_to_process = [data]
            else:
                files_to_process = [
                    file_id for file_id, file_data in context.files.items()
                    if file_data.get("is_solidity", False)
                ]
                
            self.state = AgentState.EXECUTING
                
            # Process each file with async
            semaphore = asyncio.Semaphore(3)  # Limit concurrent processing
            tasks = []
            
            for file_id in files_to_process:
                task = self._analyze_file(
                    context, analyzer, file_id, 
                    analyze_patterns, analyze_complexity, 
                    semaphore
                )
                tasks.append(task)
                
            # Wait for all files to be processed
            if tasks:
                await asyncio.gather(*tasks)
                
            self.state = AgentState.COMPLETED
            return context
            
        except Exception as e:
            logger.exception(f"Error in AST analysis: {str(e)}")
            context.add_error(
                stage="ast_analyzer",
                message="Failed to analyze AST",
                exception=e
            )
            self.state = AgentState.FAILED
            return context
            
    async def _analyze_file(
        self,
        context: Context,
        analyzer,
        file_id: str,
        analyze_patterns: bool,
        analyze_complexity: bool,
        semaphore: asyncio.Semaphore
    ) -> None:
        """Analyze a single file's AST with semaphore control"""
        async with semaphore:
            file_data = context.files.get(file_id)
            if not file_data:
                return
                
            content = file_data["content"]
            file_path = file_data["path"]
            
            # Analyze with Tree-sitter
            logger.info(f"Analyzing AST for {file_path}")
            
            # This will be made async in a future version
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, analyzer.analyze_code, content
            )
            
            # Store analysis in context
            file_data["ast_analysis"] = result
            
            # Extract security patterns if requested
            if analyze_patterns and "security_patterns" in result:
                for pattern in result["security_patterns"]:
                    finding = {
                        "title": f"Security Pattern: {pattern['name']}",
                        "description": pattern["description"],
                        "severity": pattern["severity"],
                        "location": f"{file_path}:{pattern.get('line', '?')}",
                        "source": "AST",
                        "file_id": file_id,
                        "file_path": file_path,
                        "pattern": pattern["name"],
                        "confidence": pattern.get("confidence", "Medium")
                    }
                    context.add_finding(finding)
            
            # Add complexity metrics if requested
            if analyze_complexity and "complexity_metrics" in result:
                file_data["complexity_metrics"] = result["complexity_metrics"]
                
                # Flag high-complexity functions
                if "function_complexity" in result["complexity_metrics"]:
                    for func_name, complexity in result["complexity_metrics"]["function_complexity"].items():
                        if complexity > 25:  # High complexity threshold
                            finding = {
                                "title": f"High Complexity in {func_name}",
                                "description": f"Function {func_name} has a high cyclomatic complexity of {complexity}. Consider refactoring for better maintainability and testability.",
                                "severity": "Low",
                                "location": f"{file_path}",
                                "source": "AST",
                                "file_id": file_id,
                                "file_path": file_path,
                                "complexity": complexity
                            }
                            context.add_finding(finding)
                            
            logger.info(f"Completed AST analysis for {file_path}")

class BusinessFlowExtractor:
    """
    Extract business flows from smart contract functions
    
    This transformer analyzes function calls and state transitions to identify
    business flows and logical paths within contracts.
    """
    
    def __init__(self, flow_types: Optional[List[str]] = None):
        """
        Initialize the business flow extractor
        
        Args:
            flow_types: Optional list of flow types to extract (defaults to all)
        """
        self.flow_types = flow_types or [
            "token_transfer",
            "access_control",
            "state_transition",
            "external_call",
            "fund_management"
        ]
        
    async def process(self, context: Context) -> Context:
        """
        Process the context to extract business flows
        
        Args:
            context: Pipeline context with functions
            
        Returns:
            Updated context with business flows
        """
        logger.info("Extracting business flows from functions")
        
        # Track stats for reporting
        flow_count = 0
        
        # Process each file
        for file_id, file_data in context.files.items():
            if not file_data.get("is_solidity", False) or "functions" not in file_data:
                continue
                
            # Initialize business flows list for this file
            file_data["business_flows"] = []
            
            # Get call graph if available
            call_graph = {}
            if "call_graph" in context.state:
                call_graph = context.state["call_graph"].get(file_id, {})
                
            # Extract flows from functions
            for function in file_data["functions"]:
                flows = self._extract_flows_from_function(function, call_graph)
                
                if flows:
                    # Add file info to flows
                    for flow in flows:
                        flow["file_id"] = file_id
                        flow["file_path"] = file_data["path"]
                        flow["id"] = f"{file_id}:flow:{flow['name']}"
                        file_data["business_flows"].extend(flows)
                        flow_count += len(flows)
        
        # Update context state
        context.state["business_flow_count"] = flow_count
        logger.info(f"Extracted {flow_count} business flows across all contracts")
        
        return context
        
    def _extract_flows_from_function(
        self, 
        function: Dict[str, Any], 
        call_graph: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract business flows from a single function
        
        Args:
            function: Function data
            call_graph: Call graph for related functions
            
        Returns:
            List of business flows
        """
        flows = []
        function_name = function.get("name", "")
        function_text = function.get("full_text", "")
        
        # Skip if no text to analyze
        if not function_text:
            return flows
            
        # Check for token transfer flows
        if "flow_types" in self.flow_types and self._is_token_transfer(function_text, function_name):
            flows.append({
                "name": f"{function_name}_token_flow",
                "flow_type": "token_transfer",
                "start_function": function_name,
                "description": f"Token transfer flow starting at {function_name}",
                "flow_text": self._extract_flow_text(function_text, "token_transfer")
            })
            
        # Check for access control flows
        if "access_control" in self.flow_types and self._is_access_control(function_text, function_name):
            flows.append({
                "name": f"{function_name}_access_control",
                "flow_type": "access_control",
                "start_function": function_name,
                "description": f"Access control flow in {function_name}",
                "flow_text": self._extract_flow_text(function_text, "access_control")
            })
            
        # Check for state transitions
        if "state_transition" in self.flow_types and self._is_state_transition(function_text, function_name):
            flows.append({
                "name": f"{function_name}_state_transition",
                "flow_type": "state_transition",
                "start_function": function_name,
                "description": f"State transition flow in {function_name}",
                "flow_text": self._extract_flow_text(function_text, "state_transition")
            })
            
        # Check for external calls
        if "external_call" in self.flow_types and self._is_external_call(function_text, function_name):
            flows.append({
                "name": f"{function_name}_external_call",
                "flow_type": "external_call",
                "start_function": function_name,
                "description": f"External call flow from {function_name}",
                "flow_text": self._extract_flow_text(function_text, "external_call")
            })
            
        # Check for fund management
        if "fund_management" in self.flow_types and self._is_fund_management(function_text, function_name):
            flows.append({
                "name": f"{function_name}_fund_management",
                "flow_type": "fund_management",
                "start_function": function_name,
                "description": f"Fund management flow in {function_name}",
                "flow_text": self._extract_flow_text(function_text, "fund_management")
            })
            
        return flows
        
    def _is_token_transfer(self, function_text: str, function_name: str) -> bool:
        """Check if function contains token transfer logic"""
        keywords = ["transfer", "transferFrom", "balanceOf", "approve", "allowance"]
        return any(keyword in function_text for keyword in keywords) or "transfer" in function_name.lower()
        
    def _is_access_control(self, function_text: str, function_name: str) -> bool:
        """Check if function contains access control logic"""
        keywords = ["require", "onlyOwner", "onlyAdmin", "modifier", "isOwner", "auth", "access"]
        return any(keyword in function_text for keyword in keywords)
        
    def _is_state_transition(self, function_text: str, function_name: str) -> bool:
        """Check if function contains state transition logic"""
        return "=" in function_text and any(keyword in function_text for keyword in ["state", "status", "stage"])
        
    def _is_external_call(self, function_text: str, function_name: str) -> bool:
        """Check if function makes external calls"""
        keywords = [".call{", ".call(", ".delegatecall", ".staticcall", "interface", "external"]
        return any(keyword in function_text for keyword in keywords)
        
    def _is_fund_management(self, function_text: str, function_name: str) -> bool:
        """Check if function manages funds"""
        keywords = ["wei", "ether", "value", "msg.value", "balance", "deposit", "withdraw"]
        return any(keyword in function_text for keyword in keywords)
        
    def _extract_flow_text(self, function_text: str, flow_type: str) -> str:
        """
        Extract relevant text for the business flow
        
        This is a simplified implementation. In practice, you would use
        more advanced parsing to extract only the relevant code paths.
        """
        # Simple implementation - return the full function text
        # A more advanced implementation would use AST to extract specific paths
        return function_text