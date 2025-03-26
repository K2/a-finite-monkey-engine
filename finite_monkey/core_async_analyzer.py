"""
Core Async Analyzer for Finite Monkey Engine
This module provides the foundation for the asynchronous analysis pipeline
that processes Solidity code through the following stages:
1. Parsing and chunking using Tree-Sitter
2. Initial analysis with primary LLM
3. Database-driven test generation
4. Secondary validation with confirmation LLM
5. Final report generation
The analyzer uses an async workflow to maximize throughput and supports
concurrent processing of multiple contracts and files. Configuration parameters
are controlled through nodes_config for easy tuning.
"""

import os
import re
import json
import asyncio
import traceback
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime
from pathlib import Path

# Import nodes_config for configurable parameters
from finite_monkey.nodes_config import config

# Tree-Sitter for parsing
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from .adapters import Ollama  # LLM client
from .db.manager import DatabaseManager  # Database integration

# Update imports
from llama_index.core.settings import Settings

# Configure logging
logger = logging.getLogger("core-async-analyzer")
logger.setLevel(logging.INFO)
from tree_sitter_solidity import language

"""Parse Solidity contracts using Tree-Sitter for AST-based analysis.
This parser extracts contract structures, functions, state variables,
and their relationships to enable detailed security analysis. It provides
both syntax-based parsing and semantic flow analysis to create a comprehensive
representation of contract behavior and data flows.
"""

class ContractParser:
    def __init__(self):
        """Initialize the contract parser with Tree-Sitter if available."""
        self.tree_sitter_available = True
        self.solidity_language = None
        self.parser = None

        # Try to initialize Tree-Sitter
        #if TREE_SITTER_AVAILABLE:
        try:
            # Look for the Solidity language definition
            language_path = os.path.join(os.path.dirname(__file__), "../tree_sitter_languages/solidity.so")
            self.solidity_language = Language(language())
            self.file_buffer = bytearray(1024*1024*2)  # 2MB buffer
            if os.path.exists(language_path) and not self.solidity_language:
                # Initialize a buffer for parsing
                # First, try with just the path (most common version)
                try:
                    self.solidity_language = Language(language_path)
                    logger.info("Initialized Language with simple path")
                except Exception as e:
                    logger.error(f"Failed to load Language with simple path: {e}")
                    # Only try the other methods if the simple one failed with a specific error
                    # that suggests missing language parameter
                    if "language" in str(e).lower() or "name" in str(e).lower():
                        try:
                            self.solidity_language = Language(language_path, 'solidity')
                            logger.info("Initialized Language with name parameter")
                        except Exception as e2:
                            logger.error(f"Failed to load with name param: {e2}")
                            try:
                                self.solidity_language = Language(language_path, 0)
                                logger.info("Initialized Language with index parameter")
                            except Exception as e3:
                                logger.error(f"Failed to load with index param: {e3}")
                                # We've tried all methods, resort to fallback parsing
                                self.solidity_language = None

                # Initialize parser with the language
                if self.solidity_language:
                    self.parser = Parser()
                    #self.parser.set_language(self.solidity_language)
                    self.tree_sitter_available = True
                    logger.info("Tree-Sitter initialized successfully for Solidity")
                else:
                    logger.warning("Failed to initialize Solidity language")
            #else:
            # #   logger.warning(f"Solidity language file not found at {language_path}")
        except Exception as e:
            logger.error(f"Error initializing Tree-Sitter: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")

        # Fallback regex patterns for when Tree-Sitter is not available
        self.contract_pattern = re.compile(r'contract\s+(\w+)(?:\s+is\s+([^{]+))?\s*{')
        self.function_pattern = re.compile(r'function\s+(\w+)\s*\(([^)]*)\)\s*(public|private|internal|external)?(?:\s+view|\s+pure|\s+payable|\s+virtual|\s+override)*(?:\s+returns\s*\(([^)]*)\))?\s*{')
        self.state_var_pattern = re.compile(r'^\s*([\w\[\]]+)\s+(private|public|internal)?\s*(\w+)(?:\s*=\s*([^;]+))?;', re.MULTILINE)

    async def parse_contract(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Solidity contract file asynchronously.

        Args:
            file_path: Path to the Solidity file

        Returns:
            Dictionary with contract structure
        """
        # Read file synchronously (Python 3.9+ doesn't support async file IO natively)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Use Tree-Sitter if available
        if self.tree_sitter_available and self.parser:
            return await self._parse_with_tree_sitter(content)
        else:
            return await self._parse_with_regex(content)

    async def _parse_with_tree_sitter(self, content: str) -> Dict[str, Any]:
        """Parse contract using Tree-Sitter AST."""
        try:
            # Use the pre-allocated buffer if available for better performance
            if hasattr(self, 'file_buffer'):
                # Encode content into the buffer
                content_bytes = content.encode('utf8')
                if len(content_bytes) < len(self.file_buffer):
                    self.file_buffer[:len(content_bytes)] = content_bytes
                    tree = self.parser.parse(self.file_buffer[:len(content_bytes)])
                else:
                    # If content too large for buffer, use direct bytes
                    tree = self.parser.parse(content_bytes)
            else:
                # Fallback to direct parsing
                tree = self.parser.parse(bytes(content, 'utf8'))

            root_node = tree.root_node
        except Exception as e:
            logger.error(f"Error parsing with Tree-Sitter: {e}")
            logger.error(traceback.format_exc())
            # Fallback to regex parsing on Tree-Sitter error
            return await self._parse_with_regex(content)

        # Define common query patterns for Solidity
        contract_query = """
        (contract_declaration
            name: (identifier) @contract_name
            body: (contract_body) @contract_body)
        """

        function_query = """
        (function_definition
            name: (identifier) @function_name
            parameters: (parameter_list) @parameters
            [
                (visibility_specifier) @visibility
                (state_mutability_specifier) @mutability
            ]?
            return_parameters: (return_parameter_list)? @returns
            body: (function_body) @body)
        """

        state_var_query = """
        (state_variable_declaration
            type: (type_name) @type
            (visibility_specifier)? @visibility
            name: (identifier) @name
            [
                (expression) @initial_value
            ]?)
        """

        # Compile queries
        contract_q = self.solidity_language.query(contract_query)
        function_q = self.solidity_language.query(function_query)
        state_var_q = self.solidity_language.query(state_var_query)

        # Execute queries and capture results
        contracts = {}

        # Extract contracts
        for match in contract_q.captures(root_node):
            node, name = match

            if name == "contract_name":
                contract_name = node.text.decode('utf8')
                contracts[contract_name] = {
                    "name": contract_name,
                    "functions": {},
                    "state_variables": {},
                    "modifiers": {},
                    "events": {},
                    "inheritance": []
                }

        # Extract functions for each contract
        for contract_name, contract_data in contracts.items():
            # Get contract body
            contract_body_node = None
            for match in contract_q.captures(root_node):
                node, name = match
                if name == "contract_body" and contracts.get(contract_name):
                    contract_body_node = node
                    break

            if contract_body_node:
                # Extract functions
                for match in function_q.captures(contract_body_node):
                    node, name = match
                    if name == "function_name":
                        function_name = node.text.decode('utf8')
                        function_data = {
                            "name": function_name,
                            "visibility": "internal",  # Default
                            "parameters": [],
                            "returns": [],
                            "modifiers": [],
                            "code": "",
                        }
                        contract_data["functions"][function_name] = function_data

                    # Extract function properties
                    if name == "visibility" and function_name:
                        function_data["visibility"] = node.text.decode('utf8')

                    if name == "parameters" and function_name:
                        # Parse parameters
                        params_text = node.text.decode('utf8').strip('()')
                        if params_text:
                            params = []
                            for param in params_text.split(','):
                                param = param.strip()
                                if param:
                                    param_parts = param.split()
                                    if len(param_parts) >= 2:
                                        param_type = param_parts[0]
                                        param_name = param_parts[-1].strip()
                                        params.append({
                                            "type": param_type,
                                            "name": param_name
                                        })
                            function_data["parameters"] = params

                    if name == "body" and function_name:
                        function_data["code"] = node.text.decode('utf8')

                # Extract state variables
                for match in state_var_q.captures(contract_body_node):
                    node, name = match
                    if name == "name":
                        var_name = node.text.decode('utf8')
                        var_data = {
                            "name": var_name,
                            "type": "",
                            "visibility": "internal",  # Default
                            "initial_value": None
                        }
                        contract_data["state_variables"][var_name] = var_data

                    # Extract variable properties
                    if name == "type" and var_name:
                        var_data["type"] = node.text.decode('utf8')

                    if name == "visibility" and var_name:
                        var_data["visibility"] = node.text.decode('utf8')

                    if name == "initial_value" and var_name:
                        var_data["initial_value"] = node.text.decode('utf8')

        return {
            "contracts": contracts,
            "source_code": content
        }

    async def _parse_with_regex(self, content: str) -> Dict[str, Any]:
        """Parse contract using regex patterns as fallback."""
        contracts = {}

        # Find contracts
        for match in self.contract_pattern.finditer(content):
            contract_name = match.group(1)
            inheritance = match.group(2).strip().split(',') if match.group(2) else []

            contracts[contract_name] = {
                "name": contract_name,
                "inheritance": [parent.strip() for parent in inheritance] if inheritance else [],
                "functions": {},
                "state_variables": {},
                "modifiers": {},
                "events": {}
            }

        # Find functions in each contract
        for contract_name, contract_data in contracts.items():
            # Find the contract body (basic approach)
            contract_start = content.find(f"contract {contract_name}")
            if contract_start == -1:
                continue

            contract_body_start = content.find("{", contract_start)
            if contract_body_start == -1:
                continue

            # Naive approach to find contract end - can be improved
            bracket_count = 1
            contract_end = contract_body_start + 1
            while bracket_count > 0 and contract_end < len(content):
                if content[contract_end] == '{':
                    bracket_count += 1
                elif content[contract_end] == '}':
                    bracket_count -= 1
                contract_end += 1

            contract_body = content[contract_body_start:contract_end]

            # Extract functions
            for func_match in self.function_pattern.finditer(contract_body):
                func_name = func_match.group(1)
                parameters = func_match.group(2)
                visibility = func_match.group(3) or "internal"  # Default to internal
                returns = func_match.group(4)

                # Extract function body
                func_start = func_match.end()
                bracket_count = 1
                func_end = func_start
                while bracket_count > 0 and func_end < len(contract_body):
                    if contract_body[func_end] == '{':
                        bracket_count += 1
                    elif contract_body[func_end] == '}':
                        bracket_count -= 1
                    func_end += 1

                func_body = contract_body[func_start:func_end].strip()

                # Parse parameters
                param_list = []
                if parameters:
                    for param in parameters.split(','):
                        param = param.strip()
                        if param:
                            param_parts = param.split()
                            if len(param_parts) >= 2:
                                param_type = param_parts[0]
                                param_name = param_parts[-1]
                                param_list.append({
                                    "type": param_type,
                                    "name": param_name
                                })

                # Parse return values
                return_list = []
                if returns:
                    for ret in returns.split(','):
                        ret = ret.strip()
                        if ret:
                            return_list.append({"type": ret})

                contract_data["functions"][func_name] = {
                    "name": func_name,
                    "visibility": visibility,
                    "parameters": param_list,
                    "returns": return_list,
                    "modifiers": [],
                    "code": func_body
                }

            # Extract state variables
            for var_match in self.state_var_pattern.finditer(contract_body):
                var_type = var_match.group(1)
                var_visibility = var_match.group(2) or "internal"  # Default to internal
                var_name = var_match.group(3)
                var_initial = var_match.group(4)

                contract_data["state_variables"][var_name] = {
                    "name": var_name,
                    "type": var_type,
                    "visibility": var_visibility,
                    "initial_value": var_initial
                }

        return {
            "contracts": contracts,
            "source_code": content
        }

    async def build_call_graph(self, contracts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a call graph showing function relationships.

        Args:
            contracts: Parsed contract data

        Returns:
            Call graph structure
        """
        call_graph = {}

        # For each contract
        for contract_name, contract_data in contracts.items():
            call_graph[contract_name] = {}

            # For each function
            for func_name, func_data in contract_data["functions"].items():
                calls = []
                func_code = func_data["code"]

                # Check for function calls in the code
                for other_contract, other_data in contracts.items():
                    for other_func in other_data["functions"]:
                        # Look for function calls like "otherFunc(" or "ContractName.otherFunc("
                        if re.search(r'\b' + re.escape(other_func) + r'\s*\(', func_code) or \
                            re.search(r'\b' + re.escape(other_contract) + r'\.' + re.escape(other_func) + r'\s*\(', func_code):
                            calls.append({
                                "contract": other_contract,
                                "function": other_func
                            })

                call_graph[contract_name][func_name] = calls

        return call_graph

    async def extract_control_flow(self, contracts: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        """
        Extract detailed control flow information for each function in each contract.

        This uses a combination of TreeSitter (when available) and regex pattern matching
        to identify control structures, state changes, external calls, and data flows.

        Args:
            contracts: Parsed contract data
            source_code: Original source code (for line number mapping)

        Returns:
            Control flow information
        """
        flow_data = {}
        source_lines = source_code.split('\n')

        # For each contract
        for contract_name, contract_data in contracts.items():
            flow_data[contract_name] = {}

            # For each function
            for func_name, func_data in contract_data["functions"].items():
                func_code = func_data["code"]

                # Create function flow structure
                func_flow = {
                    "name": func_name,
                    "visibility": func_data["visibility"],
                    "control_structures": [],
                    "state_changes": [],
                    "external_calls": [],
                    "value_transfers": [],
                    "condition_checks": [],
                    "line_mapping": {},
                    "parameters": func_data.get("parameters", []),
                    "returns": func_data.get("returns", []),
                }

                # Find all control structures (if/for/while)
                control_pattern = r'(if|for|while)\s*\(([^)]+)\)'
                for match in re.finditer(control_pattern, func_code):
                    control_type = match.group(1)
                    condition = match.group(2).strip()
                    position = match.start()

                    # Extract line number
                    line_num = func_code[:position].count('\n') + 1

                    func_flow["control_structures"].append({
                        "type": control_type,
                        "condition": condition,
                        "line": line_num,
                    })

                    # Add to line mapping
                    func_flow["line_mapping"][line_num] = f"{control_type} ({condition})"

                # Find state variable changes
                for var_name, var_info in contract_data["state_variables"].items():
                    # Look for assignments to state variables
                    var_pattern = r'\b' + re.escape(var_name) + r'\s*=\s*([^;]+)'
                    for match in re.finditer(var_pattern, func_code):
                        value = match.group(1).strip()
                        position = match.start()

                        # Extract line number
                        line_num = func_code[:position].count('\n') + 1

                        func_flow["state_changes"].append({
                            "variable": var_name,
                            "type": var_info["type"],
                            "value": value,
                            "line": line_num,
                        })

                        # Add to line mapping
                        func_flow["line_mapping"][line_num] = f"State Change: {var_name} = {value}"

                # Find external calls
                external_call_pattern = r'(\w+)(?:\.\w+)*\s*\.\s*(call|transfer|send|delegatecall|staticcall)(?:\.value\([^)]*\))?\s*\(([^)]*)\)'
                for match in re.finditer(external_call_pattern, func_code):
                    target = match.group(1)
                    call_type = match.group(2)
                    args = match.group(3).strip()
                    position = match.start()

                    # Extract line number
                    line_num = func_code[:position].count('\n') + 1

                    func_flow["external_calls"].append({
                        "target": target,
                        "type": call_type,
                        "arguments": args,
                        "line": line_num,
                    })

                    # Add to line mapping
                    func_flow["line_mapping"][line_num] = f"External Call: {target}.{call_type}({args})"

                    # If it's a value transfer, also add to value_transfers
                    if call_type in ["transfer", "send"] or ".value" in func_code[match.start()-20:match.start()]:
                        func_flow["value_transfers"].append({
                            "target": target,
                            "type": call_type,
                            "line": line_num,
                        })

                # Find require/assert statements
                check_pattern = r'(require|assert)\s*\(([^)]+)\)'
                for match in re.finditer(check_pattern, func_code):
                    check_type = match.group(1)
                    condition = match.group(2).strip()
                    position = match.start()

                    # Extract line number
                    line_num = func_code[:position].count('\n') + 1

                    func_flow["condition_checks"].append({
                        "type": check_type,
                        "condition": condition,
                        "line": line_num,
                    })

                    # Add to line mapping
                    func_flow["line_mapping"][line_num] = f"{check_type}({condition})"

                # Store the function flow data
                flow_data[contract_name][func_name] = func_flow

        # Use TreeSitter for more advanced analysis if available
        if self.tree_sitter_available and self.parser:
            flow_data = await self._enhance_flow_with_treesitter(flow_data, contracts, source_code)

        return flow_data

    async def _enhance_flow_with_treesitter(self, flow_data, contracts, source_code):
        """
        Enhance flow data with TreeSitter analysis for more accurate results

        Args:
            flow_data: Existing flow data from regex analysis
            contracts: Parsed contract data
            source_code: Original source code

        Returns:
            Enhanced flow data
        """
        if not self.tree_sitter_available or not self.parser:
            return flow_data

        try:
            # Parse the code with TreeSitter
            tree = self.parser.parse(bytes(source_code, "utf8"))

            # Define queries for different control flow elements
            state_var_query = """
            (assignment_expression
                left: [
                (identifier) @var_name
                (member_expression) @var_name
                ]
                right: (_) @value)
            """

            external_call_query = """
            (call_expression
                function: [
                (member_expression
                    object: (_) @target
                    property: (identifier) @method (#match? @method "^(call|transfer|send|delegatecall|staticcall)$"))
                ]
                arguments: (call_argument_list) @args)
            """

            # Compile queries
            state_var_q = self.solidity_language.query(state_var_query)
            external_call_q = self.solidity_language.query(external_call_query)

            # Extract more accurate information and enhance the flow data
            for contract_name, contract_functions in flow_data.items():
                for func_name, func_flow in contract_functions.items():
                    # More advanced analysis could be done here using TreeSitter queries
                    # This is a simple enhancement pass, but could be expanded significantly

                    # For now, we enhance the metadata with more information
                    func_flow["enhanced_with_treesitter"] = True

            return flow_data

        except Exception as e:
            logger.warning(f"Error enhancing flow with TreeSitter: {e}")
            return flow_data

    async def join_flows(self, call_graph: Dict[str, Any], flow_data: Dict[str, Any]) -> Dict[str, Any]:
        flow_paths = {}

        # For each contract
        for contract_name, contract_calls in call_graph.items():
            flow_paths[contract_name] = {}

            # For each function
            for func_name, calls in contract_calls.items():
                # Initialize the flow path for this function
                flow_paths[contract_name][func_name] = {
                    "calls": [],
                    "caller_of": [],
                    "state_dependencies": [],
                    "value_flow": [],
                    "path_conditions": [],
                }

                # Add outgoing calls information
                for call in calls:
                    other_contract = call["contract"]
                    other_func = call["function"]

                    # Get flow data for the called function
                    if other_contract in flow_data and other_func in flow_data[other_contract]:
                        called_flow = flow_data[other_contract][other_func]

                        # Add to the flow path with detailed information
                        flow_paths[contract_name][func_name]["calls"].append({
                            "contract": other_contract,
                            "function": other_func,
                            "state_changes": called_flow["state_changes"],
                            "external_calls": called_flow["external_calls"],
                            "value_transfers": called_flow["value_transfers"],
                        })

                        # Also add this function as a caller to the called function's path
                        if other_contract not in flow_paths:
                            flow_paths[other_contract] = {}
                        if other_func not in flow_paths[other_contract]:
                            flow_paths[other_contract][other_func] = {
                                "calls": [],
                                "caller_of": [],
                                "state_dependencies": [],
                                "value_flow": [],
                                "path_conditions": [],
                            }

                        flow_paths[other_contract][other_func]["caller_of"].append({
                            "contract": contract_name,
                            "function": func_name,
                        })

                # Analyze state dependencies (variables read before write)
                if contract_name in flow_data and func_name in flow_data[contract_name]:
                    current_flow = flow_data[contract_name][func_name]

                    # Extract state variables that are read but not written first
                    state_writes = set(change["variable"] for change in current_flow["state_changes"])

                    # Check for state variables in conditions
                    for control in current_flow["control_structures"] + current_flow["condition_checks"]:
                        condition = control["condition"]
                        # Simple heuristic: look for state variables in the condition
                        for var_name in flow_data[contract_name].get("state_variables", {}).keys():
                            if re.search(r'\b' + re.escape(var_name) + r'\b', condition):
                                if var_name not in state_writes:
                                    flow_paths[contract_name][func_name]["state_dependencies"].append({
                                        "variable": var_name,
                                        "context": f"Used in condition: {condition}",
                                        "line": control["line"],
                                    })

                # Analyze value flow (ETH transfers)
                if contract_name in flow_data and func_name in flow_data[contract_name]:
                    current_flow = flow_data[contract_name][func_name]

                    for transfer in current_flow["value_transfers"]:
                        flow_paths[contract_name][func_name]["value_flow"].append({
                            "from": contract_name,
                            "to": transfer["target"],
                            "via": func_name,
                            "line": transfer["line"],
                        })

                # Extract path conditions (conditions that must be satisfied for execution)
                if contract_name in flow_data and func_name in flow_data[contract_name]:
                    current_flow = flow_data[contract_name][func_name]

                    for check in current_flow["condition_checks"]:
                        if check["type"] == "require":
                            flow_paths[contract_name][func_name]["path_conditions"].append({
                                "condition": check["condition"],
                                "line": check["line"],
                                "type": "require",
                            })

        return flow_paths


class ExpressionGenerator:
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the expression generator.

        Args:
            db_manager: Database manager for storing and retrieving expressions
        """
        self.db_manager = db_manager
        self.task_engine_initialized = False

        # Initialize task engine if db_manager is provided
        if self.db_manager:
            try:
                from sqlalchemy.ext.asyncio import AsyncEngine
                from sqlalchemy.sql import text

                # Store SQLAlchemy components for task operations
                self.text = text

                # Ensure we have an async engine
                if hasattr(self.db_manager, 'engine') and isinstance(self.db_manager.engine, AsyncEngine):
                    self.engine = self.db_manager.engine
                    self.task_engine_initialized = True
                    logger.info("SQLAlchemy TaskEngine initialized successfully with AsyncEngine")
                else:
                    logger.warning("Database engine is not async, can't initialize TaskEngine")
            except Exception as e:
                logger.warning(f"Failed to initialize SQLAlchemy TaskEngine: {e}")
                self.task_engine_initialized = False

    async def generate_expressions_for_contract(self,
                                                contract_data: Dict[str, Any],
                                                call_graph: Dict[str, Any],
                                                flow_paths: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        expressions = []

        # First generate basic expressions based on code patterns
        pattern_expressions = await self._generate_pattern_expressions(contract_data, call_graph)
        expressions.extend(pattern_expressions)

        # Then enhance with flow-aware expressions if flow paths provided
        if flow_paths:
            flow_expressions = await self._generate_flow_expressions(contract_data, flow_paths)
            expressions.extend(flow_expressions)

        # Finally, enhance with database-driven expressions if task engine available
        if self.task_engine_initialized:
            db_expressions = await self._generate_database_expressions(contract_data, expressions)
            expressions.extend(db_expressions)

        return expressions

    async def _generate_pattern_expressions(self,
                                            contract_data: Dict[str, Any],
                                            call_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        expressions = []

        # Get contract name and functions
        for contract_name, contract_info in contract_data["contracts"].items():
            functions = contract_info["functions"]
            state_vars = contract_info["state_variables"]

            # Generate expressions for state variable access
            for var_name, var_info in state_vars.items():
                var_type = var_info["type"]
                visibility = var_info["visibility"]

                # Check for potentially dangerous state variable access patterns
                if visibility != "private":
                    expressions.append({
                        "type": "state_variable_access",
                        "contract": contract_name,
                        "variable": var_name,
                        "expression": f"Check if {var_name} in {contract_name} can be accessed or modified by unauthorized parties",
                        "severity": "Medium",
                        "category": "Access Control"
                    })

                # Check for potentially unchecked state variables
                if "uint" in var_type.lower() and "checked" not in var_name.lower():
                    expressions.append({
                        "type": "arithmetic",
                        "contract": contract_name,
                        "variable": var_name,
                        "expression": f"Check if {var_name} in {contract_name} can overflow or underflow",
                        "severity": "High",
                        "category": "Arithmetic"
                    })

            # Generate expressions for function call patterns
            for func_name, func_info in functions.items():
                # Check for external calls
                func_code = func_info["code"]

                # Check for reentrancy potential
                if re.search(r'\.(call|transfer|send)\s*\{?\s*value\s*:', func_code):
                    expressions.append({
                        "type": "reentrancy",
                        "contract": contract_name,
                        "function": func_name,
                        "expression": f"Check if {func_name} in {contract_name} is vulnerable to reentrancy attacks",
                        "severity": "Critical",
                        "category": "Reentrancy"
                    })

                # Check for tx.origin usage
                if "tx.origin" in func_code:
                    expressions.append({
                        "type": "tx_origin",
                        "contract": contract_name,
                        "function": func_name,
                        "expression": f"Check if {func_name} in {contract_name} uses tx.origin for authentication",
                        "severity": "High",
                        "category": "Authentication"
                    })

                # Check for unchecked return values
                if re.search(r'\.(call|transfer|send)\s*\(', func_code) and not re.search(r'require\s*\(\s*\w+\.(call|transfer|send)', func_code):
                    expressions.append({
                        "type": "unchecked_return",
                        "contract": contract_name,
                        "function": func_name,
                        "expression": f"Check if {func_name} in {contract_name} handles return values from external calls",
                        "severity": "Medium",
                        "category": "Error Handling"
                    })

                # Check for timestamp dependence
                if "block.timestamp" in func_code:
                    expressions.append({
                        "type": "timestamp_dependence",
                        "contract": contract_name,
                        "function": func_name,
                        "expression": f"Check if {func_name} in {contract_name} has dangerous timestamp dependency",
                        "severity": "Medium",
                        "category": "Randomness"
                    })

                # Analyze function parameters for unsafe input
                for param in func_info.get("parameters", []):
                    param_name = param.get("name")
                    param_type = param.get("type")

                    if param_type in ["address", "address payable"]:
                        expressions.append({
                            "type": "untrusted_address",
                            "contract": contract_name,
                            "function": func_name,
                            "parameter": param_name,
                            "expression": f"Check if {func_name} in {contract_name} properly validates address parameter {param_name}",
                            "severity": "Medium",
                            "category": "Input Validation"
                        })

        # Check call graph for dangerous patterns
        for contract_name, funcs in call_graph.items():
            for func_name, calls in funcs.items():
                # Check for direct cross-contract calls
                for call in calls:
                    other_contract = call.get("contract")
                    other_func = call.get("function")

                    if other_contract != contract_name:
                        expressions.append({
                            "type": "cross_contract_call",
                            "contract": contract_name,
                            "function": func_name,
                            "target_contract": other_contract,
                            "target_function": other_func,
                            "expression": f"Check if the call from {func_name} in {contract_name} to {other_func} in {other_contract} is secure",
                            "severity": "Medium",
                            "category": "External Interaction"
                        })

        return expressions

    async def _generate_flow_expressions(self,
                                        contract_data: Dict[str, Any],
                                        flow_paths: Dict[str, Any]) -> List[Dict[str, Any]]:
        expressions = []

        # For each contract
        for contract_name, contract_functions in flow_paths.items():
            # For each function
            for func_name, func_flow in contract_functions.items():
                # 1. Check for state changes after external calls (reentrancy)
                external_calls = []
                for call in func_flow.get("calls", []):
                    if call.get("external_calls"):
                        external_calls.append(call)

                state_changes = func_flow.get("state_changes", [])
                if external_calls and state_changes:
                    expressions.append({
                        "type": "flow_reentrancy",
                        "contract": contract_name,
                        "function": func_name,
                        "expression": f"Analyze the function flow in {func_name} for reentrancy vulnerabilities, focusing on state changes after external calls",
                        "severity": "Critical",
                        "category": "Reentrancy",
                        "flow_context": {
                            "external_calls": [call.get("function") for call in external_calls],
                            "state_changes": [change.get("variable") for change in state_changes if isinstance(change, dict) and "variable" in change]
                        }
                    })

                # 2. Check for conditional value transfers (access control)
                value_flows = func_flow.get("value_flow", [])
                path_conditions = func_flow.get("path_conditions", [])
                if value_flows and path_conditions:
                    expressions.append({
                        "type": "flow_conditional_transfer",
                        "contract": contract_name,
                        "function": func_name,
                        "expression": f"Verify that path conditions in {func_name} properly restrict value transfers",
                        "severity": "High",
                        "category": "Access Control",
                        "flow_context": {
                            "value_flows": [flow.get("to") for flow in value_flows if isinstance(flow, dict) and "to" in flow],
                            "conditions": [cond.get("condition") for cond in path_conditions if isinstance(cond, dict) and "condition" in cond]
                        }
                    })

                # 3. Check for unsafe state dependencies
                state_deps = func_flow.get("state_dependencies", [])
                if state_deps:
                    expressions.append({
                        "type": "flow_state_dependency",
                        "contract": contract_name,
                        "function": func_name,
                        "expression": f"Check if {func_name} has unsafe state variable dependencies",
                        "severity": "Medium",
                        "category": "State Management",
                        "flow_context": {
                            "dependencies": [dep.get("variable") for dep in state_deps if isinstance(dep, dict) and "variable" in dep],
                            "line_context": [dep.get("context") for dep in state_deps if isinstance(dep, dict) and "context" in dep]
                        }
                    })

                # 4. Check for functions called by multiple other functions (shared resource)
                callers = func_flow.get("caller_of", [])
                if len(callers) > 1:
                    expressions.append({
                        "type": "flow_shared_resource",
                        "contract": contract_name,
                        "function": func_name,
                        "expression": f"Analyze {func_name} for shared resource issues as it's called by multiple functions",
                        "severity": "Medium",
                        "category": "Resource Management",
                        "flow_context": {
                            "callers": [f"{caller.get('contract')}.{caller.get('function')}" for caller in callers
                                        if isinstance(caller, dict) and "contract" in caller and "function" in caller],
                        }
                    })

        return expressions

    async def _generate_database_expressions(self,
                                            contract_data: Dict[str, Any],
                                            existing_expressions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.task_engine_initialized or not self.db_manager:
            return []

        expressions = []

        try:
            # Define vulnerability patterns to query from database
            vulnerability_patterns = [
                "reentrancy", "access_control", "arithmetic", "unchecked_call",
                "timestamp_dependence", "front_running", "dos", "gas_optimization"
            ]

            # Get contract names for the query
            contract_names = list(contract_data["contracts"].keys())
            if not contract_names:
                return []

            contract_names_str = ", ".join([f"'{name}'" for name in contract_names])

            # Execute SQL query to get vulnerability tasks
            async with self.engine.connect() as conn:
                try:
                    # Check if vulnerability_tasks table exists
                    check_query = self.text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'vulnerability_tasks'
                        )
                                            """)

                    result = await conn.execute(check_query)
                    # Get the scalar result using SQLAlchemy 2.0+ API
                    table_exists = result.scalar_one_or_none()

                    if not table_exists:
                        logger.warning("vulnerability_tasks table does not exist, creating one")
                        # Create table if it doesn't exist
                        create_table_query = self.text(
                            """
                            CREATE TABLE IF NOT EXISTS vulnerability_tasks (
                                task_id SERIAL PRIMARY KEY,
                                vulnerability_type VARCHAR(50) NOT NULL,
                                contract_pattern VARCHAR(255) NOT NULL,
                                pattern TEXT,
                                description TEXT NOT NULL,
                                severity VARCHAR(20) NOT NULL,
                                category VARCHAR(50) NOT NULL,
                                created_at TIMESTAMP DEFAULT NOW()
                            )
                            """)

                        await conn.execute(create_table_query)
                        await conn.commit()

                        # Add some default vulnerability tasks
                        for vul_type in vulnerability_patterns:
                            default_task = self.text(
                                f"""
                                INSERT INTO vulnerability_tasks
                                (vulnerability_type, contract_pattern, description, severity, category)
                                VALUES
                                (:vul_type, :description, :severity, :category)
                                """)

                            severity = "Critical" if vul_type in ["reentrancy"] else "High"
                            category = vul_type.replace("_", " ").title()
                            description = f"Check for {vul_type.replace('_', ' ')} vulnerabilities in the contract"

                            await conn.execute(default_task, {
                                "vul_type": vul_type,
                                "description": description,
                                "severity": severity,
                                "category": category
                            })

                        await conn.commit()

                except Exception as e:
                    logger.error(f"Error creating vulnerability_tasks table: {e}")
                    return []

                # SQL query to get vulnerability tasks for the given contract patterns
                query = self.text(
                    f"""
                    SELECT task_id, vulnerability_type, pattern, description, severity, category
                    FROM vulnerability_tasks
                    WHERE contract_pattern IN ({contract_names_str})
                    OR contract_pattern = 'any'
                    ORDER BY severity DESC
                    """)

                try:
                    result = await conn.execute(query)
                    # Use SQLAlchemy 2.0+ API for fetching results
                    rows = result.fetchall()

                    # Convert SQL results to expressions
                    for row in rows:
                        task_id, vul_type, pattern, description, severity, category = row

                        # Check if a similar expression already exists
                        if any(e.get("type") == vul_type and e.get("category") == category for e in existing_expressions):
                            continue  # Skip duplicate patterns

                        # Create a new expression from the task
                        for contract_name in contract_names:
                            expressions.append({
                                "type": vul_type,
                                "task_id": task_id,
                                "contract": contract_name,
                                "expression": description.replace("{contract}", contract_name) if description else f"Check for {vul_type.replace('_', ' ')} in {contract_name}",
                                "severity": severity or "Medium",
                                "category": category or "Security",
                                "db_generated": True
                            })
                except Exception as e:
                    logger.error(f"Error executing SQL query: {e}")

            logger.info(f"Generated {len(expressions)} database-driven expressions")
        except Exception as e:
            logger.error(f"Error generating database expressions: {e}")

        return expressions

    async def store_expressions(self, project_id: str, file_id: str, expressions: List[Dict[str, Any]]) -> None:
        
        logger.info(f"Storing {len(expressions)} expressions for file {file_id} in project {project_id}")

        if not self.db_manager:
            # If no DB manager, just log the expressions
            for expr in expressions:
                logger.debug(f"Expression: {expr['expression']} (Severity: {expr['severity']})")
            return

        try:
            # Import TestExpression model - use absolute import path to avoid relative import issues
            from finite_monkey.db.models import TestExpression, Base

            # Create tables if they don't exist
            from sqlalchemy.ext.asyncio import AsyncEngine
            if isinstance(self.db_manager.engine, AsyncEngine):
                # Create the tables using SQLAlchemy's metadata API
                try:
                    # First try using db_manager.create_tables() if available
                    if hasattr(self.db_manager, 'create_tables') and callable(getattr(self.db_manager, 'create_tables')):
                        await self.db_manager.create_tables()
                        logger.info("Tables created successfully with db_manager.create_tables()")
                    else:
                        # Direct creation with engine
                        async with self.db_manager.engine.begin() as conn:
                            await conn.run_sync(lambda conn: Base.metadata.create_all(conn))
                            logger.info("Tables created with Base.metadata.create_all()")
                except Exception as e:
                    logger.warning(f"Error creating tables: {e}")
                    logger.warning(traceback.format_exc())

                    # For backward compatibility - create only the TestExpression table if we can't create all
                    try:
                        async with self.db_manager.engine.begin() as conn:
                            logger.info("Trying to create only TestExpression table directly...")
                            create_query = f"""
                            CREATE TABLE IF NOT EXISTS test_expressions (
                                id SERIAL PRIMARY KEY,
                                project_id VARCHAR(256),
                                file_id VARCHAR(256),
                                expression TEXT NOT NULL,
                                expression_type VARCHAR(32),
                                severity VARCHAR(32) DEFAULT 'Medium',
                                confidence FLOAT DEFAULT 0.5,
                                line_number INTEGER,
                                context TEXT,
                                expression_data JSONB,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );
                            CREATE INDEX IF NOT EXISTS idx_test_expr_project ON test_expressions(project_id);
                            CREATE INDEX IF NOT EXISTS idx_test_expr_file ON test_expressions(file_id);
                            """
                            await conn.execute(create_query)
                            logger.info("TestExpression table created directly with SQL")
                    except Exception as e2:
                        logger.error(f"Failed to create TestExpression table directly: {e2}")
                        logger.error(traceback.format_exc())

            # Try the SQL direct approach first (bypassing ORM issues)
            try:
                async with self.db_manager.engine.begin() as conn:
                    # For each expression, insert directly with SQL
                    for expr in expressions:
                        # Create insert query with parameters
                        query = f"""
                        INSERT INTO test_expressions
                        (project_id, file_id, expression, expression_type, severity, confidence,
                            line_number, context, expression_data, created_at)
                        VALUES
                        ($1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP)
                        """

                        # Convert metadata to JSON if it exists
                        import json
                        metadata_json = json.dumps(expr.get('metadata', {}))

                        # Execute insert - pass parameters as a dict for asyncpg
                        params = {
                            "project_id": project_id,
                            "file_id": file_id,
                            "expression": expr['expression'],
                            "expression_type": expr.get('type', 'vulnerability'),
                            "severity": expr.get('severity', 'Medium'),
                            "confidence": float(expr.get('confidence', 0.5)),
                            "line_number": expr.get('line_number'),
                            "context": expr.get('context'),
                            "metadata_json": metadata_json
                        }

                        # Rewrite the query to use named parameters with SQL text
                        from sqlalchemy.sql import text

                        # Check database dialect for compatibility
                        dialect_name = self.db_manager.engine.dialect.name

                        if 'postgresql' in dialect_name:
                            # PostgreSQL with SQLAlchemy's text() function
                            named_query = text("""
                            INSERT INTO test_expressions
                            (project_id, file_id, expression, expression_type, severity, confidence,
                                line_number, context, expression_data, created_at)
                            VALUES
                            (:project_id, :file_id, :expression, :expression_type, :severity, :confidence,
                                :line_number, :context, :metadata_json, CURRENT_TIMESTAMP)
                            """)
                        else:
                            # SQLite and others use JSON string directly
                            named_query = text("""
                            INSERT INTO test_expressions
                            (project_id, file_id, expression, expression_type, severity, confidence,
                                line_number, context, expression_data, created_at)
                            VALUES
                            (:project_id, :file_id, :expression, :expression_type, :severity, :confidence,
                                :line_number, :context, :metadata_json, CURRENT_TIMESTAMP)
                            """)

                        # Execute with proper SQLAlchemy 2.0+ parameter binding
                        await conn.execute(named_query, params)

                    logger.info(f"Successfully stored {len(expressions)} expressions using direct SQL")
                    return
            except Exception as e:
                logger.warning(f"Direct SQL insert failed, trying ORM approach: {e}")

            # Fallback to ORM approach if direct SQL fails
            try:
                # Create a minimal TestExpression class to avoid relationship conflicts
                from sqlalchemy import Column, Integer, String, Text, Float, JSON, DateTime
                from sqlalchemy.ext.declarative import declarative_base
                from datetime import datetime

                MinimalBase = declarative_base()

                class MinimalTestExpression(MinimalBase):
                    __tablename__ = "test_expressions"

                    id = Column(Integer, primary_key=True)
                    project_id = Column(String(256))
                    file_id = Column(String(256))
                    expression = Column(Text, nullable=False)
                    expression_type = Column(String(32))
                    severity = Column(String(32), default="Medium")
                    confidence = Column(Float, default=0.5)
                    line_number = Column(Integer)
                    context = Column(Text)
                    expression_data = Column(JSON)
                    created_at = Column(DateTime, default=datetime.utcnow)

                async with self.db_manager.async_session() as session:
                    for expr in expressions:
                        # Create TestExpression object with minimal class
                        test_expr = MinimalTestExpression(
                            project_id=project_id,
                            file_id=file_id,
                            expression=expr['expression'],
                            expression_type=expr.get('type', 'vulnerability'),
                            severity=expr.get('severity', 'Medium'),
                            confidence=expr.get('confidence', 0.5),
                            line_number=expr.get('line_number'),
                            context=expr.get('context'),
                            expression_data=expr.get('metadata', {})
                        )

                        # Add to session
                        session.add(test_expr)

                    # Commit all expressions at once
                    await session.commit()
                    logger.info(f"Successfully stored {len(expressions)} expressions using ORM")
            except Exception as e:
                logger.error(f"ORM approach also failed: {e}")
                raise

        except Exception as e:
            logger.error(f"Error storing expressions: {e}")
            # Log the full exception for debugging
            import traceback
            logger.error(traceback.format_exc())


class AsyncAnalyzer:
    """
    Main async analyzer that orchestrates the full analysis pipeline.

    This class coordinates the parsing, analysis, testing, and validation
    stages using asynchronous processing for optimal performance.    """

    def __init__(
        self,
        primary_llm_client: Optional[Ollama] = None,
        secondary_llm_client: Optional[Ollama] = None,
        db_manager: Optional[DatabaseManager] = None,
        primary_model_name: Optional[str] = None,
        secondary_model_name: Optional[str] = None,
    ):
        """
        Initialize the async analyzer.

        Args:
            primary_llm_client: Primary LLM client for initial analysis
            secondary_llm_client: Secondary LLM client for validation
            db_manager: Database manager
            primary_model_name: Name of the primary model
            secondary_model_name: Name of the secondary model        """
        # Load config

        # Set model names from arguments, config, or defaults
        self.primary_model_name = primary_model_name or config.SCAN_MODEL or "llama2:13b"  # Set a default model
        self.secondary_model_name = secondary_model_name or config.CONFIRMATION_MODEL or "llama2:70b"  # Set a default model

        # Ensure model names are not empty strings
        if not self.primary_model_name or self.primary_model_name.strip() == '':
            self.primary_model_name = "llama2:13b"  # Fallback default
            logger.warning(f"Empty primary model name provided, using default: {self.primary_model_name}")
            
        if not self.secondary_model_name or self.secondary_model_name.strip() == '':
            self.secondary_model_name = "llama2:70b"  # Fallback default
            logger.warning(f"Empty secondary model name provided, using default: {self.secondary_model_name}")

        # Set up LLM clients with fallbacks
        self.primary_llm = primary_llm_client or Ollama(
            model=self.primary_model_name,
            base_url=config.OPENAI_API_BASE or "http://localhost:11434"
        )

        self.secondary_llm = secondary_llm_client or Ollama(
            model=self.secondary_model_name,
            base_url=config.OPENAI_API_BASE
        )

        # Set up database manager
        if db_manager is None:
            try:
                # Try to initialize with PostgreSQL from config
                if config.DATABASE_URL and "postgresql" in config.DATABASE_URL:
                    # Convert the standard PostgreSQL URL to an async one
                    db_url = config.DATABASE_URL
                    if "postgresql:" in db_url and "postgresql+asyncpg:" not in db_url:
                        db_url = db_url.replace("postgresql:", "postgresql+asyncpg:")

                    from ..db.manager import DatabaseManager
                    db_manager = DatabaseManager(db_url=db_url)
                    logger.info(f"Initialized PostgreSQL database manager with {db_url}")
                    self.db_manager = db_manager
                else:
                    logger.warning("No PostgreSQL database URL found in config. Analysis results won't be persisted.")
            except Exception as e:
                logger.warning(f"Failed to initialize database manager: {e}. Analysis results won't be persisted.")
        self.db_manager = db_manager

        # Initialize parser and expression generator
        self.parser = ContractParser()

        # Only initialize expression generator if db_manager is provided
        self.expression_generator = ExpressionGenerator(db_manager) if db_manager else None

        # Track analysis state
        self.analysis_state = {}

    async def analyze_contract_file(self,
                                    file_path: str,
                                    project_id: str = "default",
                                    query: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the full analysis pipeline on a contract file.

        Args:
            file_path: Path to the Solidity file
            project_id: Project identifier
            query: Optional specific query to focus analysis

        Returns:
            Analysis results        
        """
        file_id = os.path.basename(file_path)
        logger.info(f"Starting analysis of {file_id} for project {project_id}")

        # Step 1: Parse the contract
        try:
            contract_data = await self.parser.parse_contract(file_path)
            logger.info(f"Successfully parsed {file_id} - found {len(contract_data['contracts'])} contracts")
        except Exception as e:
            logger.error(f"Error parsing {file_id}: {e}")
            return {"error": f"Parsing error: {str(e)}"}

        # Step 2: Build call graph
        try:
            call_graph = await self.parser.build_call_graph(contract_data["contracts"])
            logger.info(f"Built call graph for {file_id}")
        except Exception as e:
            logger.error(f"Error building call graph for {file_id}: {e}")
            call_graph = {}

        # Step 3: Extract control flow information
        try:
            flow_data = await self.parser.extract_control_flow(
                contract_data["contracts"], contract_data["source_code"]
            )
            logger.info(f"Extracted control flow data for {file_id}")
        except Exception as e:
            logger.error(f"Error extracting control flow for {file_id}: {e}")
            flow_data = {}

        # Step 4: Join flows to create comprehensive data flows
        try:
            flow_paths = await self.parser.join_flows(call_graph, flow_data)
            logger.info(f"Joined flow paths for {file_id}")
        except Exception as e:
            logger.error(f"Error joining flow paths for {file_id}: {e}")
            flow_paths = {}

        # Step 5: Generate test expressions using flow data (if db_manager available)
        expressions = []
        if self.expression_generator:
            try:
                expressions = await self.expression_generator.generate_expressions_for_contract(
                    contract_data, call_graph, flow_paths
                )
                
                # Store expressions in database
                await self.expression_generator.store_expressions(project_id, file_id, expressions)
                logger.info(f"Generated {len(expressions)} test expressions for {file_id}")
            except Exception as e:
                logger.error(f"Error generating expressions for {file_id}: {e}")

        # Step 6: Analyze with primary LLM including flow context
        try:
            primary_analysis = await self._run_primary_analysis(
                contract_data, call_graph, expressions, query, flow_paths
            )
            logger.info(f"Completed primary analysis for {file_id}")
        except Exception as e:
            logger.error(f"Error in primary analysis for {file_id}: {e}")
            return {"error": f"Primary analysis error: {str(e)}"}

        # Step 7: Validate with secondary LLM
        try:
            secondary_validation = await self._run_secondary_validation(
                contract_data, primary_analysis, expressions, flow_data
            )
            logger.info(f"Completed secondary validation for {file_id}")
        except Exception as e:
            logger.error(f"Error in secondary validation for {file_id}: {e}")
            secondary_validation = {"error": str(e)}

        # Step 8: Generate final report
        try:
            final_report = await self._generate_final_report(
                file_id, contract_data, primary_analysis, secondary_validation, flow_paths
            )
            logger.info(f"Generated final report for {file_id}")
        except Exception as e:
            logger.error(f"Error generating final report for {file_id}: {e}")
            final_report = {"error": str(e)}

        # Return the complete analysis results
        return {
            "file_id": file_id,
            "project_id": project_id,
            "contract_data": contract_data,
            "call_graph": call_graph,
            "flow_data": flow_data,
            "flow_paths": flow_paths,
            "primary_analysis": primary_analysis,
            "secondary_validation": secondary_validation,
            "final_report": final_report,
            "timestamp": datetime.now().isoformat()
        }

    async def analyze_file_with_chunking(
        self,
        file_path: str,
        query: str,
        project_id: str,
        max_chunk_size: int = 4000,
        include_call_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single file with enhanced chunking strategies
        
        Args:
            file_path: Path to Solidity file
            query: Analysis query
            project_id: Project identifier
            max_chunk_size: Maximum chunk size in characters
            include_call_graph: Whether to include call graph analysis
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing file with enhanced chunking: {file_path}")
        
        # Parse the contract
        contract_data = await self.parser.parse_contract(file_path)
        
        # Build call graph if requested
        call_graph = {}
        if include_call_graph:
            call_graph = await self.parser.build_call_graph(contract_data["contracts"])
            
        # Generate chunks based on contract structure
        chunks = await self._generate_hierarchical_chunks(contract_data, max_chunk_size)
        
        # Create a context summarizing the contract for each chunk
        context = await self._create_contract_context(contract_data, call_graph)
        
        # Analyze each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_result = await self._analyze_chunk(
                chunk=chunk, 
                context=context,
                query=query,
                chunk_index=i,
                total_chunks=len(chunks)
            )
            chunk_results.append(chunk_result)
        
        # Synthesize results from all chunks
        combined_result = await self._synthesize_chunk_results(chunk_results, contract_data)
        
        # Add metadata
        combined_result["file_id"] = os.path.basename(file_path)
        combined_result["project_id"] = project_id
        combined_result["enhanced_chunking"] = True
        combined_result["total_chunks"] = len(chunks)
        combined_result["contract_data"] = contract_data
        
        return combined_result
    
    async def analyze_project_with_chunking(
        self,
        project_path: str,
        project_id: str,
        query: str,
        max_chunk_size: int = 4000,
        include_call_graph: bool = True,
        file_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze a project directory with enhanced chunking strategies
        
        Args:
            project_path: Path to project directory
            project_id: Project identifier
            query: Analysis query
            max_chunk_size: Maximum chunk size in characters
            include_call_graph: Whether to include call graph analysis
            file_limit: Maximum number of files to analyze
            
        Returns:
            Project analysis results
        """
        logger.info(f"Analyzing project with enhanced chunking: {project_path}")
        
        # Find all Solidity files
        sol_files = []
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith('.sol'):
                    sol_files.append(os.path.join(root, file))
                    
        # Apply file limit if specified
        if file_limit and len(sol_files) > file_limit:
            sol_files = sol_files[:file_limit]
            
        logger.info(f"Found {len(sol_files)} Solidity files")
        
        # Create semaphore for concurrency control
        max_concurrent = config.MAX_THREADS_OF_SCAN
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Define a wrapper function that uses the semaphore
        async def analyze_with_semaphore(file_path):
            async with semaphore:
                file_result = await self.analyze_file_with_chunking(
                    file_path=file_path,
                    query=query,
                    project_id=project_id,
                    max_chunk_size=max_chunk_size,
                    include_call_graph=include_call_graph
                )
                return file_result
                
        # Analyze each file with controlled concurrency
        tasks = []
        for file_path in sol_files:
            tasks.append(analyze_with_semaphore(file_path))
            
        file_results = await asyncio.gather(*tasks)
        
        # Aggregate results
        project_result = {
            "project_id": project_id,
            "project_path": project_path,
            "file_count": len(sol_files),
            "enhanced_chunking": True,
            "files": {},
            "findings": [],
            "severity_summary": {
                "Critical": 0,
                "High": 0,
                "Medium": 0,
                "Low": 0,
                "Informational": 0
            },
            "total_chunks": 0
        }
        
        # Process each file's results
        for i, result in enumerate(file_results):
            file_name = os.path.basename(sol_files[i])
            project_result["files"][file_name] = result
            
            # Add findings to aggregate list
            if "final_report" in result and "findings" in result["final_report"]:
                for finding in result["final_report"]["findings"]:
                    # Add file info to finding
                    finding_with_file = dict(finding)
                    finding_with_file["file"] = file_name
                    project_result["findings"].append(finding_with_file)
                    
                    # Update severity summary
                    severity = finding.get("severity", "Medium")
                    if severity in project_result["severity_summary"]:
                        project_result["severity_summary"][severity] += 1
                        
            # Add chunk count
            project_result["total_chunks"] += result.get("total_chunks", 0)
            
        # Add total finding count
        project_result["finding_count"] = len(project_result["findings"])
        
        return project_result
    
    async def _generate_hierarchical_chunks(
        self, 
        contract_data: Dict[str, Any],
        max_chunk_size: int
    ) -> List[Dict[str, Any]]:
        """
        Generate chunks based on contract structure hierarchy
        
        Args:
            contract_data: Parsed contract data
            max_chunk_size: Maximum chunk size
            
        Returns:
            List of chunk dictionaries
        """
        source_code = contract_data["source_code"]
        chunks = []
        
        # If the source code is small enough, use the whole contract
        if len(source_code) <= max_chunk_size:
            chunks.append({
                "type": "full_contract",
                "content": source_code,
                "description": "Full contract source code"
            })
            return chunks
            
        # Split by contract
        for contract_name, contract_info in contract_data["contracts"].items():
            # Get contract text (approximation - ideally we'd use AST positions)
            contract_text = self._extract_contract_text(source_code, contract_name)
            
            # If contract is small enough, add whole contract
            if len(contract_text) <= max_chunk_size:
                chunks.append({
                    "type": "contract",
                    "name": contract_name,
                    "content": contract_text,
                    "description": f"Contract: {contract_name}"
                })
            else:
                # Add contract declaration and state variables
                decl_and_vars = self._extract_contract_declaration_and_vars(source_code, contract_name)
                chunks.append({
                    "type": "contract_declaration",
                    "name": contract_name,
                    "content": decl_and_vars,
                    "description": f"Contract {contract_name} declaration and state variables"
                })
                
                # Split by function
                for func_name, func_info in contract_info["functions"].items():
                    func_text = func_info.get("code", "")
                    description = f"Function {contract_name}.{func_name}"
                    
                    # Add function chunk (might need to split further for very large functions)
                    if len(func_text) <= max_chunk_size:
                        chunks.append({
                            "type": "function",
                            "contract": contract_name,
                            "name": func_name,
                            "content": func_text,
                            "description": description
                        })
                    else:
                        # Split very large functions into smaller chunks
                        func_chunks = self._split_text_into_chunks(func_text, max_chunk_size)
                        for i, chunk_text in enumerate(func_chunks):
                            chunks.append({
                                "type": "function_part",
                                "contract": contract_name,
                                "name": func_name,
                                "part": i + 1,
                                "total_parts": len(func_chunks),
                                "content": chunk_text,
                                "description": f"{description} (part {i+1} of {len(func_chunks)})"
                            })
        
        return chunks
    
    async def _create_contract_context(
        self, 
        contract_data: Dict[str, Any],
        call_graph: Dict[str, Any]
    ) -> str:
        """
        Create a context summary for the contract
        
        Args:
            contract_data: Parsed contract data
            call_graph: Call graph data
            
        Returns:
            Context summary string
        """
        context_lines = ["CONTRACT CONTEXT:"]
        
        # Add contract overview
        contracts = list(contract_data["contracts"].keys())
        context_lines.append(f"Contracts: {', '.join(contracts)}")
        
        # Add inheritance information
        for contract_name, contract_info in contract_data["contracts"].items():
            inheritance = contract_info.get("inheritance", [])
            if inheritance:
                context_lines.append(f"{contract_name} inherits from: {', '.join(inheritance)}")
                
        # Add key state variables
        for contract_name, contract_info in contract_data["contracts"].items():
            state_vars = contract_info.get("state_variables", {})
            if state_vars:
                context_lines.append(f"{contract_name} state variables:")
                for var_name, var_info in state_vars.items():
                    var_type = var_info.get("type", "unknown")
                    visibility = var_info.get("visibility", "internal")
                    context_lines.append(f"  - {var_type} {visibility} {var_name}")
                    
        # Add function signatures
        for contract_name, contract_info in contract_data["contracts"].items():
            functions = contract_info.get("functions", {})
            if functions:
                context_lines.append(f"{contract_name} functions:")
                for func_name, func_info in functions.items():
                    visibility = func_info.get("visibility", "internal")
                    params = func_info.get("parameters", [])
                    param_str = ", ".join(f"{p.get('type', '')} {p.get('name', '')}" for p in params)
                    context_lines.append(f"  - {visibility} {func_name}({param_str})")
                    
        # Add call relationships
        if call_graph:
            context_lines.append("Call relationships:")
            for contract_name, contract_calls in call_graph.items():
                for func_name, calls in contract_calls.items():
                    if calls:
                        called_funcs = [f"{call.get('contract', '')}.{call.get('function', '')}" for call in calls]
                        context_lines.append(f"  - {contract_name}.{func_name} calls: {', '.join(called_funcs)}")
                        
        return "\n".join(context_lines)
    
    async def _analyze_chunk(
        self,
        chunk: Dict[str, Any],
        context: str,
        query: str,
        chunk_index: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """
        Analyze a single chunk with the LLM
        
        Args:
            chunk: Chunk dictionary
            context: Contract context summary
            query: Analysis query
            chunk_index: Chunk index
            total_chunks: Total number of chunks
            
        Returns:
            Analysis results for this chunk
        """
        # Create prompt for this chunk
        prompt = f"""You are a Solidity smart contract security auditor analyzing code.
This is chunk {chunk_index+1} of {total_chunks} from the same contract.

AUDIT QUERY: {query}

{context}

CHUNK TYPE: {chunk["type"]}
DESCRIPTION: {chunk["description"]}

CODE:
```solidity
{chunk["content"]}
```

Analyze this chunk for security vulnerabilities and issues, focusing on:
1. Vulnerabilities present in this specific code chunk
2. Potential security issues arising from interactions with other contract components
3. Potential vulnerabilities based on common Solidity security patterns

For each finding:
1. Provide a clear title
2. Indicate severity (Critical, High, Medium, Low, Informational)
3. Describe the issue
4. Specify where in the code the issue is located
5. Explain the potential impact
6. Recommend a fix

Format your response as structured findings with clear sections.
If you don't find any issues in this particular chunk, explain why and what you checked for.
"""

        # Get analysis from LLM
        chunk_analysis = await self.primary_llm.acomplete(prompt)
        
        # Return results
        return {
            "chunk_index": chunk_index,
            "chunk_type": chunk["type"],
            "chunk_description": chunk["description"],
            "analysis": chunk_analysis
        }
    
    async def _synthesize_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]],
        contract_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize results from all chunks into a final report
        
        Args:
            chunk_results: List of chunk analysis results
            contract_data: Original contract data
            
        Returns:
            Combined analysis results
        """
        # Extract all chunk analyses
        all_analyses = []
        for result in chunk_results:
            all_analyses.append(f"CHUNK {result['chunk_index']+1}: {result['chunk_description']}\n\n{result['analysis']}")
            
        combined_analyses = "\n\n" + "="*50 + "\n\n".join(all_analyses)
        
        # Create synthesis prompt
        synthesis_prompt = f"""You are a Solidity smart contract security auditor compiling a final report.
You have received analyses from multiple chunks of the same contract.
Your job is to synthesize these analyses into a coherent final report.

CONTRACT OVERVIEW:
Number of contracts: {len(contract_data["contracts"])}
Contract names: {", ".join(contract_data["contracts"].keys())}

INDIVIDUAL CHUNK ANALYSES:
{combined_analyses}

Please create a comprehensive final report including:
1. A summary of the contract's purpose and architecture
2. Overall security assessment and risk rating (Critical, High, Medium, Low)
3. List of all findings, merging duplicate findings and eliminating false positives
4. For each finding:
   - Clear title
   - Severity (Critical, High, Medium, Low, Informational)
   - Description
   - Location in code
   - Impact
   - Recommendation for fixing

5. A summary "Severity Distribution" section showing count by severity level
6. A brief conclusion with risk assessment text

Ensure all findings are correctly categorized by severity, with appropriate justification.
"""

        # Get synthesized report from LLM
        final_report_text = await self.primary_llm.acomplete(synthesis_prompt)
        
        # Extract structured findings from text
        findings = self._extract_findings(final_report_text)
        
        # Calculate severity distribution
        severity_summary = {
            "Critical": 0,
            "High": 0,
            "Medium": 0,
            "Low": 0,
            "Informational": 0
        }
        
        for finding in findings:
            severity = finding.get("severity", "Medium")
            if severity in severity_summary:
                severity_summary[severity] += 1
                
        # Create final report structure
        final_report = {
            "summary": self._extract_summary(final_report_text),
            "findings": findings,
            "severity_summary": severity_summary,
            "risk_assessment": self._extract_risk_rating(final_report_text),
            "risk_text": self._extract_conclusion(final_report_text)
        }
        
        return {
            "final_report": final_report,
            "final_report_text": final_report_text,
            "chunk_results": chunk_results,
            "enhanced_chunking": True
        }
    
    def _extract_contract_text(self, source_code: str, contract_name: str) -> str:
        """Extract the text of a specific contract from the source code"""
        # Simple regex-based extraction, could be improved with AST positions
        pattern = rf'contract\s+{contract_name}\s*(?:is\s+[^{{]+)?\s*{{([\s\S]*?)}}(?:\s*contract|\s*$)'
        match = re.search(pattern, source_code)
        if match:
            return f"contract {contract_name} {{{match.group(1)}}}"
        return ""
    
    def _extract_contract_declaration_and_vars(self, source_code: str, contract_name: str) -> str:
        """Extract contract declaration and state variables"""
        contract_text = self._extract_contract_text(source_code, contract_name)
        
        # Extract declaration and state variables (until the first function)
        func_pattern = r'function\s+\w+\s*\('
        func_match = re.search(func_pattern, contract_text)
        
        if func_match:
            return contract_text[:func_match.start()]
        return contract_text
    
    def _split_text_into_chunks(self, text: str, max_size: int) -> List[str]:
        """Split text into chunks of maximum size"""
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # If remaining text fits in chunk, add it all
            if len(text) - current_pos <= max_size:
                chunks.append(text[current_pos:])
                break
                
            # Find a good splitting point (end of line near max_size)
            end_pos = current_pos + max_size
            while end_pos > current_pos:
                if text[end_pos] in ['\n', ';', '}']:
                    end_pos += 1  # Include the splitting character
                    break
                end_pos -= 1
                
            # If no good splitting point found, split at max_size
            if end_pos <= current_pos:
                end_pos = current_pos + max_size
                
            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos
            
        return chunks
    
    def _extract_summary(self, text: str) -> str:
        """Extract summary from report text"""
        # Look for summary section
        summary_pattern = r'(?:##?\s*Summary|\*\*Summary\*\*)(.*?)(?:##|\*\*|$)'
        match = re.search(summary_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            summary = match.group(1).strip()
            # Limit to reasonable length
            if len(summary) > 500:
                summary = summary[:497] + "..."
            return summary
            
        # Fallback to first paragraph
        paragraphs = text.split("\n\n")
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith('#') and not p.startswith('*'):
                if len(p) > 500:
                    p = p[:497] + "..."
                return p
                
        return "No summary available"
    
    def _extract_risk_rating(self, text: str) -> str:
        """Extract overall risk rating from report text"""
        risk_pattern = r'(?:risk|security)\s+(?:rating|assessment|level|score):\s*([A-Za-z]+)'
        match = re.search(risk_pattern, text, re.IGNORECASE)
        
        if match:
            rating = match.group(1).strip().capitalize()
            # Normalize to standard ratings
            if rating.lower() in ['critical', 'high', 'medium', 'low', 'informational']:
                return rating.capitalize()
        
        # Check for mentions of overall risk
        for level in ['Critical', 'High', 'Medium', 'Low']:
            if f"overall {level.lower()}" in text.lower() or f"{level.lower()} risk" in text.lower():
                return level
                
        return "Medium"  # Default to medium if no clear risk level found
    
    def _extract_conclusion(self, text: str) -> str:
        """Extract conclusion from report text"""
        conclusion_pattern = r'(?:##?\s*Conclusion|\*\*Conclusion\*\*)(.*?)(?:##|\*\*|$)'
        match = re.search(conclusion_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
            
        # Fallback to last paragraph
        paragraphs = text.split("\n\n")
        for p in reversed(paragraphs):
            p = p.strip()
            if p and not p.startswith('#') and not p.startswith('*'):
                return p
                
        return "No conclusion provided"
    
    async def validate_findings(
        self,
        findings: List[Dict[str, Any]],
        source_code: str,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Validate findings using secondary analysis
        
        Args:
            findings: List of findings to validate
            source_code: Source code for context
            model_name: Optional model name to use for validation
            
        Returns:
            List of validation results
        """
        if not findings:
            return []
            
        # Format findings for validation
        findings_text = ""
        for i, finding in enumerate(findings):
            findings_text += f"{i+1}. {finding['title']} (Severity: {finding['severity']})\n"
            findings_text += f"   Description: {finding['description']}\n"
            findings_text += f"   Location: {finding.get('location', 'Not specified')}\n"
            findings_text += f"   Impact: {finding.get('impact', 'Not specified')}\n\n"
            
        # Create validation prompt
        prompt = f"""You are a validator for smart contract security findings. Your job is to carefully review the findings from an initial analysis and provide an independent assessment.
        
        SMART CONTRACT:
        ```solidity
        {source_code}
        ```
        
        FINDINGS TO VALIDATE:
        {findings_text}
        
        For each finding above:
        1. Perform an independent verification using detailed code inspection
        2. Provide your confirmation status: Is the finding valid? (Confirmed, False Positive, or Needs More Information)
        3. Include your reasoning with specific code references and line numbers
        4. Provide adjusted severity assessment if needed (Critical, High, Medium, Low, Informational)
        
        Pay special attention to:
        - The sequence of operations (state changes vs external calls)
        - Path conditions that must be satisfied for vulnerabilities to be exploitable
        - Cross-function and cross-contract interactions
        - Value transfer flows and their security implications
        """

        # Use secondary model for validation if available
        llm_client = self.secondary_llm if self.secondary_llm else self.primary_llm

        # Get validation results
        validation_text = await llm_client.acomplete(prompt)

        # Parse validation results
        validation_results = []

        # Simple parsing based on numbered items
        finding_pattern = r'(\d+)\.\s+(.*?)(?=\d+\.\s+|$)'
        matches = re.finditer(finding_pattern, validation_text, re.DOTALL)

        for match in matches:
            finding_num = int(match.group(1))
            finding_text = match.group(2).strip()

            # Extract status
            status = "Not validated"
            status_pattern = r'(?:status|confirmation):\s*([A-Za-z\s]+)'
            status_match = re.search(status_pattern, finding_text, re.IGNORECASE)
            if status_match:
                status_text = status_match.group(1).strip().lower()
                if "confirm" in status_text or "valid" in status_text:
                    status = "Confirmed"
                elif "false" in status_text:
                    status = "False Positive"
                else:
                    status = "Needs More Information"

            validation_results.append({
                "finding_index": finding_num - 1,  # Zero-based index
                "status": status,
                "notes": finding_text
            })

        return validation_results

    async def run_example():
        sample_path = "examples/Vault.sol"

        # Initialize analyzer
        analyzer = AsyncAnalyzer()

        # Analyze the contract
        results = await analyzer.analyze_contract_file(sample_path)

        # Print summary
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Analysis complete for {results['file_id']}")
            if "final_report" in results:
                report = results["final_report"]
                print(f"Risk Assessment: {report['risk_assessment']}")
                print(f"Finding Count: {report['finding_count']}")
                print("\nSeverity Summary:")
                for severity, count in report.get("severity_summary", {}).items():
                    print(f"  {severity}: {count}")

                print("\nFindings:")
                for i, finding in enumerate(report.get("findings", []), 1):
                    print(f"{i}. {finding['title']} (Severity: {finding['severity']})")
                    print(f"   Status: {finding.get('confirmation_status', 'Not validated')}")
                    print(f"   Location: {finding.get('location', 'Not specified')}")
                    print()

        return results
