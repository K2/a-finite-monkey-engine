"""
Business flow extraction agent for the Finite Monkey framework

This agent is responsible for extracting business flows from smart contracts,
identifying logical workflows, and providing context for security analysis.
"""

import os
import re
import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from pathlib import Path

from ..models.codebase import (
    ContractDef,
    FunctionDef,
    VariableDef,
    BusinessFlow,
    CodebaseContext,
)
from ..db.manager import DatabaseManager
from ..db.models import (
    BusinessFlowTask,
    CodeContract,
    CodeFunction,
    CodeVariable,
    BusinessFlow as DBBusinessFlow,
)
from ..adapters import Ollama
from ..utils.prompting import get_business_flow_prompts


class BusinessFlowExtractor:
    """
    Business flow extraction agent
    
    This agent extracts business flows from smart contracts by analyzing the
    code structure, function relationships, and variable usage. It uses a
    multi-pass approach with LLM prompts to identify and classify different
    types of business flows.
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        validator_model: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None,
        intensity: float = 1.0,
    ):
        """
        Initialize the business flow extractor
        
        Args:
            model_name: Name of the LLM model to use
            validator_model: Name of the validator model (defaults to model_name)
            db_manager: Database manager instance
            intensity: Analysis intensity (0.0-1.0) affecting thoroughness
        """
        self.model_name = model_name
        self.validator_model = validator_model or model_name
        self.db_manager = db_manager
        self.intensity = max(0.1, min(1.0, intensity))  # Clamp between 0.1 and 1.0
        
        # Initialize LLM adapters
        self.ollama = Ollama(model=model_name)
        if validator_model and validator_model != model_name:
            self.validator = Ollama(model=validator_model)
        else:
            self.validator = self.ollama
            
        # Get prompt templates
        self.prompts = get_business_flow_prompts()
        
        # Performance tracking
        self.telemetry = {
            "contracts_analyzed": 0,
            "functions_analyzed": 0,
            "business_flows_identified": 0,
            "prompt_tokens_used": 0,
        }
    
    async def extract_from_codebase(
        self,
        codebase_context: CodebaseContext,
        project_id: str,
    ) -> Dict[str, Any]:
        """
        Extract business flows from a codebase
        
        Args:
            codebase_context: Codebase context with parsed code entities
            project_id: Project ID for database storage
            
        Returns:
            Dictionary with extraction results
        """
        # Reset telemetry
        self.telemetry = {
            "contracts_analyzed": 0,
            "functions_analyzed": 0,
            "business_flows_identified": 0,
            "prompt_tokens_used": 0,
            "start_time": asyncio.get_event_loop().time(),
        }
        
        # Check if we have contracts to analyze
        if not codebase_context.contracts:
            return {
                "status": "error",
                "message": "No contracts found in codebase",
                "business_flows": [],
                "telemetry": self.telemetry,
            }
        
        # Process each contract
        business_flows = []
        for contract_name, contract in codebase_context.contracts.items():
            # Skip interfaces and libraries if intensity is low
            if self.intensity < 0.5 and contract.contract_type != "contract":
                continue
                
            flows = await self.extract_from_contract(contract, project_id)
            business_flows.extend(flows)
            
            # Update telemetry
            self.telemetry["contracts_analyzed"] += 1
            
        # End timing
        self.telemetry["elapsed_time"] = asyncio.get_event_loop().time() - self.telemetry["start_time"]
        
        # Return results
        return {
            "status": "success",
            "message": f"Extracted {len(business_flows)} business flows from {self.telemetry['contracts_analyzed']} contracts",
            "business_flows": business_flows,
            "telemetry": self.telemetry,
        }
    
    async def extract_from_contract(
        self,
        contract: ContractDef,
        project_id: str,
    ) -> List[BusinessFlow]:
        """
        Extract business flows from a contract
        
        Args:
            contract: Contract definition
            project_id: Project ID for database storage
            
        Returns:
            List of extracted business flows
        """
        # Skip if no functions to analyze
        if not contract.functions:
            return []
            
        # Multi-pass analysis for business flow extraction
        # 1. First pass: Analyze each function individually
        function_analyses = await self._analyze_contract_functions(contract, project_id)
        
        # 2. Second pass: Identify relationships between functions
        function_relationships = await self._analyze_function_relationships(contract, function_analyses, project_id)
        
        # 3. Third pass: Extract business flows
        business_flows = await self._extract_business_flows(contract, function_analyses, function_relationships, project_id)
        
        # 4. Fourth pass: Validate and enhance business flows
        enhanced_flows = await self._validate_business_flows(contract, business_flows, project_id)
        
        # Return the enhanced flows
        return enhanced_flows
    
    async def _analyze_contract_functions(
        self,
        contract: ContractDef,
        project_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        First pass: Analyze each function in a contract
        
        Args:
            contract: Contract definition
            project_id: Project ID for database storage
            
        Returns:
            Dictionary mapping function names to their analysis results
        """
        # Prepare results
        results = {}
        
        # Skip processing if intensity is too low
        if self.intensity < 0.2:
            # Just do basic classification
            for func_name, func in contract.functions.items():
                results[func_name] = {
                    "name": func_name,
                    "type": self._classify_function_type(func),
                    "visibility": func.visibility,
                    "is_view": func.is_view,
                    "is_pure": func.is_pure,
                    "is_payable": func.is_payable,
                    "complexity": len(func.source_code.split("\n")),
                }
                
                # Update telemetry
                self.telemetry["functions_analyzed"] += 1
                
            return results
            
        # Process each function with LLM
        for func_name, func in contract.functions.items():
            # Skip trivial functions if intensity is low
            if self.intensity < 0.5 and (
                len(func.source_code.split("\n")) < 5 or 
                func.is_view or 
                func.is_pure
            ):
                # Basic classification for simple functions
                results[func_name] = {
                    "name": func_name,
                    "type": self._classify_function_type(func),
                    "visibility": func.visibility,
                    "is_view": func.is_view,
                    "is_pure": func.is_pure,
                    "is_payable": func.is_payable,
                    "business_purpose": "Unknown",
                    "complexity": len(func.source_code.split("\n")),
                }
                
                # Update telemetry
                self.telemetry["functions_analyzed"] += 1
                continue
                
            # Generate prompt for function analysis
            prompt = self.prompts["function_analysis"].format(
                contract_name=contract.name,
                function_name=func_name,
                function_code=func.source_code,
                contract_context=self._get_contract_context(contract, func),
                function_signature=self._get_function_signature(func),
            )
            
            # Get response from LLM
            response = await self.ollama.acomplete(prompt=prompt)
            
            # Parse response
            analysis = self._parse_function_analysis(response)
            
            # Enrich with function metadata
            analysis["name"] = func_name
            analysis["visibility"] = func.visibility
            analysis["is_view"] = func.is_view
            analysis["is_pure"] = func.is_pure
            analysis["is_payable"] = func.is_payable
            analysis["complexity"] = len(func.source_code.split("\n"))
            
            # Add to results
            results[func_name] = analysis
            
            # Update telemetry
            self.telemetry["functions_analyzed"] += 1
            self.telemetry["prompt_tokens_used"] += len(prompt.split())
            
            # Storage in database if db_manager is available
            if self.db_manager:
                await self._store_function_analysis(contract, func, analysis, project_id)
            
        return results
    
    def _classify_function_type(self, func: FunctionDef) -> str:
        """
        Classify function type based on name and properties
        
        Args:
            func: Function definition
            
        Returns:
            Function type classification
        """
        name = func.name.lower()
        
        # Special functions
        if func.is_constructor:
            return "constructor"
        elif func.is_fallback:
            return "fallback"
        elif func.is_receive:
            return "receive"
        elif func.is_modifier:
            return "modifier"
            
        # Access control functions
        if name.startswith("only") or "auth" in name or "admin" in name or "owner" in name:
            return "access_control"
            
        # State-changing functions
        if func.is_view or func.is_pure:
            return "view"
        elif func.is_payable:
            return "payable"
            
        # Try to classify by common patterns
        if name.startswith("get") or name.startswith("is") or name.startswith("has"):
            return "getter"
        elif name.startswith("set"):
            return "setter"
        elif "withdraw" in name:
            return "withdrawal"
        elif "deposit" in name or "receive" in name:
            return "deposit"
        elif "transfer" in name or "send" in name:
            return "transfer"
        elif "mint" in name:
            return "mint"
        elif "burn" in name:
            return "burn"
        elif "approve" in name or "allow" in name:
            return "approval"
        elif "claim" in name:
            return "claim"
        elif "stake" in name:
            return "staking"
        elif "swap" in name or "exchange" in name:
            return "swap"
        elif "initialize" in name or "init" in name:
            return "initialization"
        
        # Default
        return "unknown"
    
    def _get_contract_context(self, contract: ContractDef, function: FunctionDef) -> str:
        """
        Get contract context for a function
        
        Args:
            contract: Contract definition
            function: Function definition
            
        Returns:
            Contract context as a string
        """
        # Get contract inheritance
        inheritance = ", ".join(contract.inheritance) if contract.inheritance else "None"
        
        # Get state variables
        state_vars = []
        for var_name, var in contract.variables.items():
            if var.is_state_variable:
                state_vars.append(f"{var.variable_type} {var.name}")
                
        state_vars_str = "\n".join(state_vars) if state_vars else "None"
        
        # Get related functions (those that call or are called by this function)
        related_funcs = []
        for called_func in function.called_functions:
            if called_func.name in contract.functions:
                related_funcs.append(f"calls: {called_func.name}")
                
        for caller_func in function.called_by:
            if caller_func.name in contract.functions:
                related_funcs.append(f"called by: {caller_func.name}")
                
        related_funcs_str = "\n".join(related_funcs) if related_funcs else "None"
        
        # Return context
        return f"""
Contract Type: {contract.contract_type}
Inheritance: {inheritance}

State Variables:
{state_vars_str}

Related Functions:
{related_funcs_str}
"""
    
    def _get_function_signature(self, function: FunctionDef) -> str:
        """
        Get function signature
        
        Args:
            function: Function definition
            
        Returns:
            Function signature as a string
        """
        # Build parameter list
        params = []
        for param in function.parameters:
            param_type = param.get("type", "")
            param_name = param.get("name", "")
            params.append(f"{param_type} {param_name}")
            
        params_str = ", ".join(params)
        
        # Build modifiers list
        modifiers = []
        if function.visibility:
            modifiers.append(function.visibility)
        if function.is_view:
            modifiers.append("view")
        if function.is_pure:
            modifiers.append("pure")
        if function.is_payable:
            modifiers.append("payable")
            
        for mod in function.modifiers:
            modifiers.append(mod)
            
        modifiers_str = " ".join(modifiers)
        
        # Add return type if present
        returns = f" returns ({function.return_type})" if function.return_type else ""
        
        # Return signature
        return f"function {function.name}({params_str}) {modifiers_str}{returns}"
    
    def _parse_function_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse function analysis response from LLM
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed analysis as a dictionary
        """
        # Default values
        analysis = {
            "type": "unknown",
            "business_purpose": "Unknown",
            "description": "",
            "state_changes": [],
            "external_calls": [],
            "security_considerations": [],
            "business_flow_potential": "low",
        }
        
        # Extract type
        type_match = re.search(r"Function Type:?\s*(\w+)", response)
        if type_match:
            analysis["type"] = type_match.group(1).lower()
            
        # Extract business purpose
        purpose_match = re.search(r"Business Purpose:?\s*(.+?)(?:\n|$)", response)
        if purpose_match:
            analysis["business_purpose"] = purpose_match.group(1).strip()
            
        # Extract description
        desc_match = re.search(r"Description:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if desc_match:
            analysis["description"] = desc_match.group(1).strip()
            
        # Extract state changes
        state_changes_match = re.search(r"State Changes:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if state_changes_match:
            state_changes = state_changes_match.group(1).strip()
            analysis["state_changes"] = [s.strip() for s in state_changes.split("\n") if s.strip()]
            
        # Extract external calls
        external_calls_match = re.search(r"External Calls:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if external_calls_match:
            external_calls = external_calls_match.group(1).strip()
            analysis["external_calls"] = [s.strip() for s in external_calls.split("\n") if s.strip()]
            
        # Extract security considerations
        security_match = re.search(r"Security Considerations:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if security_match:
            security = security_match.group(1).strip()
            analysis["security_considerations"] = [s.strip() for s in security.split("\n") if s.strip()]
            
        # Extract business flow potential
        flow_match = re.search(r"Business Flow Potential:?\s*(\w+)", response)
        if flow_match:
            analysis["business_flow_potential"] = flow_match.group(1).lower()
            
        return analysis
    
    async def _analyze_function_relationships(
        self,
        contract: ContractDef,
        function_analyses: Dict[str, Dict[str, Any]],
        project_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Second pass: Analyze relationships between functions
        
        Args:
            contract: Contract definition
            function_analyses: Function analysis results from first pass
            project_id: Project ID for database storage
            
        Returns:
            Dictionary with relationship analysis results
        """
        # Skip if intensity is too low or not enough functions
        if self.intensity < 0.3 or len(contract.functions) < 2:
            return {}
            
        # Get function names with high business flow potential
        high_potential_funcs = []
        for func_name, analysis in function_analyses.items():
            if analysis.get("business_flow_potential", "low") in ["high", "medium"]:
                high_potential_funcs.append(func_name)
                
        # Skip if no high potential functions
        if not high_potential_funcs:
            return {}
            
        # Build function relationship graph
        relationship_graph = {}
        for func_name, func in contract.functions.items():
            # Skip low-potential functions if intensity is low
            if self.intensity < 0.7 and func_name not in high_potential_funcs:
                continue
                
            relationship_graph[func_name] = {
                "calls": [f.name for f in func.called_functions if f.name in contract.functions],
                "called_by": [f.name for f in func.called_by if f.name in contract.functions],
                "variables_read": [v.name for v in func.variables_read],
                "variables_written": [v.name for v in func.variables_written],
                "analysis": function_analyses.get(func_name, {}),
            }
            
        # Skip LLM analysis if intensity is low
        if self.intensity < 0.5:
            return relationship_graph
            
        # Generate prompt for relationship analysis
        contract_funcs = []
        for func_name in high_potential_funcs:
            func = contract.functions[func_name]
            analysis = function_analyses.get(func_name, {})
            
            contract_funcs.append({
                "name": func_name,
                "signature": self._get_function_signature(func),
                "purpose": analysis.get("business_purpose", "Unknown"),
                "type": analysis.get("type", "unknown"),
            })
            
        # Generate the prompt
        prompt = self.prompts["function_relationships"].format(
            contract_name=contract.name,
            contract_type=contract.contract_type,
            functions=json.dumps(contract_funcs, indent=2),
            relationship_graph=json.dumps(relationship_graph, indent=2),
        )
        
        # Get response from LLM
        response = await self.ollama.acomplete(prompt=prompt)
        
        # Parse relationship analysis
        relationships = self._parse_relationship_analysis(response)
        
        # Update telemetry
        self.telemetry["prompt_tokens_used"] += len(prompt.split())
        
        # Return relationships
        return relationships
    
    def _parse_relationship_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse relationship analysis response from LLM
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed relationship analysis
        """
        # Default structure
        relationships = {
            "workflows": [],
            "key_functions": [],
            "function_groups": [],
        }
        
        # Try to extract JSON
        json_match = re.search(r"```json(.+?)```", response, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
                if isinstance(json_data, dict):
                    # Use extracted JSON data
                    if "workflows" in json_data:
                        relationships["workflows"] = json_data["workflows"]
                    if "key_functions" in json_data:
                        relationships["key_functions"] = json_data["key_functions"]
                    if "function_groups" in json_data:
                        relationships["function_groups"] = json_data["function_groups"]
                    
                    return relationships
            except json.JSONDecodeError:
                # Fall back to regex parsing if JSON is invalid
                pass
                
        # Extract workflows
        workflow_section = re.search(r"Workflows:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if workflow_section:
            workflow_text = workflow_section.group(1).strip()
            workflow_items = re.findall(r"\d+\.\s+(.+?)(?:\n\d+\.|\n\n|\n[A-Z]|$)", workflow_text, re.DOTALL)
            for item in workflow_items:
                workflow = {
                    "name": "",
                    "functions": [],
                    "description": item.strip(),
                }
                
                # Try to extract name and functions
                name_match = re.search(r"^([^:]+):", item)
                if name_match:
                    workflow["name"] = name_match.group(1).strip()
                    
                functions_match = re.findall(r"`([^`]+)`", item)
                if functions_match:
                    workflow["functions"] = functions_match
                    
                relationships["workflows"].append(workflow)
                
        # Extract key functions
        key_section = re.search(r"Key Functions:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if key_section:
            key_text = key_section.group(1).strip()
            key_items = re.findall(r"\d+\.\s+(.+?)(?:\n\d+\.|\n\n|\n[A-Z]|$)", key_text, re.DOTALL)
            for item in key_items:
                function_match = re.search(r"`([^`]+)`", item)
                if function_match:
                    relationships["key_functions"].append(function_match.group(1))
                    
        # Extract function groups
        group_section = re.search(r"Function Groups:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if group_section:
            group_text = group_section.group(1).strip()
            group_items = re.findall(r"\d+\.\s+(.+?)(?:\n\d+\.|\n\n|\n[A-Z]|$)", group_text, re.DOTALL)
            for item in group_items:
                group = {
                    "name": "",
                    "functions": [],
                    "purpose": "",
                }
                
                # Try to extract name and functions
                name_match = re.search(r"^([^:]+):", item)
                if name_match:
                    group["name"] = name_match.group(1).strip()
                    
                functions_match = re.findall(r"`([^`]+)`", item)
                if functions_match:
                    group["functions"] = functions_match
                    
                purpose_match = re.search(r"purpose:?\s*(.+?)(?:\n|$)", item, re.IGNORECASE)
                if purpose_match:
                    group["purpose"] = purpose_match.group(1).strip()
                    
                relationships["function_groups"].append(group)
                
        return relationships
    
    async def _extract_business_flows(
        self,
        contract: ContractDef,
        function_analyses: Dict[str, Dict[str, Any]],
        function_relationships: Dict[str, Any],
        project_id: str,
    ) -> List[BusinessFlow]:
        """
        Third pass: Extract business flows
        
        Args:
            contract: Contract definition
            function_analyses: Function analysis results from first pass
            function_relationships: Function relationship results from second pass
            project_id: Project ID for database storage
            
        Returns:
            List of extracted business flows
        """
        # Skip if intensity is too low
        if self.intensity < 0.4:
            return []
            
        # Get workflows from relationship analysis
        workflows = function_relationships.get("workflows", [])
        
        # If no workflows found, and intensity is high enough, try to generate some
        if not workflows and self.intensity >= 0.6:
            # Generate basic workflows from function groups
            function_groups = function_relationships.get("function_groups", [])
            for group in function_groups:
                if group.get("functions"):
                    workflows.append({
                        "name": group.get("name", "Unnamed Flow"),
                        "functions": group.get("functions", []),
                        "description": group.get("purpose", ""),
                    })
                    
        # If still no workflows and intensity is high, try to generate from key functions
        if not workflows and self.intensity >= 0.8:
            key_functions = function_relationships.get("key_functions", [])
            if key_functions:
                workflows.append({
                    "name": f"{contract.name} Core Flow",
                    "functions": key_functions,
                    "description": "Core business flow of the contract",
                })
                
        # If still no workflows, create a single flow for the contract if it has functions
        if not workflows and contract.functions:
            # Filter to external functions only
            external_funcs = []
            for func_name, func in contract.functions.items():
                if func.visibility in ["public", "external"] and not func.is_view and not func.is_pure:
                    external_funcs.append(func_name)
                    
            if external_funcs:
                workflows.append({
                    "name": f"{contract.name} External API",
                    "functions": external_funcs,
                    "description": "External API surface of the contract",
                })
                
        # No workflows to extract
        if not workflows:
            return []
            
        # Extract business flows from workflows
        business_flows = []
        for workflow in workflows:
            # Get workflow details
            flow_name = workflow.get("name", "Unnamed Flow")
            flow_functions = workflow.get("functions", [])
            flow_description = workflow.get("description", "")
            
            # Skip if no functions
            if not flow_functions:
                continue
                
            # Get function definitions
            func_defs = []
            for func_name in flow_functions:
                if func_name in contract.functions:
                    func_defs.append(contract.functions[func_name])
                    
            # Skip if no function definitions found
            if not func_defs:
                continue
                
            # Extract code and lines for the business flow
            extracted_code = ""
            lines = []
            for func in func_defs:
                extracted_code += f"// {func.name}\n{func.source_code}\n\n"
                lines.extend(range(func.start_line, func.end_line + 1))
                
            # Get context for the business flow
            flow_context = self._generate_flow_context(contract, func_defs, function_analyses)
            
            # Create business flow
            flow = BusinessFlow(
                name=flow_name,
                flow_type=self._determine_flow_type(func_defs, function_analyses),
                description=flow_description,
                source_functions=func_defs,
                extracted_code=extracted_code,
                context=flow_context,
                lines=lines,
            )
            
            # Add to results
            business_flows.append(flow)
            
        # Update telemetry
        self.telemetry["business_flows_identified"] += len(business_flows)
        
        # If db_manager is available, store the flows
        if self.db_manager:
            await self._store_business_flows(contract, business_flows, project_id)
            
        return business_flows
    
    def _determine_flow_type(
        self,
        functions: List[FunctionDef],
        function_analyses: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Determine the type of a business flow
        
        Args:
            functions: Functions in the flow
            function_analyses: Function analysis results
            
        Returns:
            Business flow type
        """
        # Count function types
        type_counts = {}
        for func in functions:
            func_analysis = function_analyses.get(func.name, {})
            func_type = func_analysis.get("type", self._classify_function_type(func))
            type_counts[func_type] = type_counts.get(func_type, 0) + 1
            
        # Check for specific patterns
        if "deposit" in type_counts and "withdrawal" in type_counts:
            return "deposit_withdrawal"
        elif "mint" in type_counts and "burn" in type_counts:
            return "token_management"
        elif "swap" in type_counts:
            return "exchange"
        elif "stake" in type_counts or "claim" in type_counts:
            return "staking"
        elif "approval" in type_counts:
            return "approval"
            
        # Get most common type
        most_common = max(type_counts.items(), key=lambda x: x[1], default=("unknown", 0))
        
        # If most common is unknown, use a generic type
        if most_common[0] == "unknown":
            if any(f.is_payable for f in functions):
                return "value_transfer"
            elif any(f.is_view or f.is_pure for f in functions):
                return "data_access"
            else:
                return "state_change"
                
        return most_common[0]
    
    def _generate_flow_context(
        self,
        contract: ContractDef,
        functions: List[FunctionDef],
        function_analyses: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Generate context information for a business flow
        
        Args:
            contract: Contract definition
            functions: Functions in the flow
            function_analyses: Function analysis results
            
        Returns:
            Context information as a string
        """
        # Build function descriptions
        func_descs = []
        for func in functions:
            analysis = function_analyses.get(func.name, {})
            func_type = analysis.get("type", self._classify_function_type(func))
            purpose = analysis.get("business_purpose", "Unknown")
            
            func_descs.append(f"- {func.name}: {func_type.upper()} - {purpose}")
            
        # Build variable usage
        var_usage = set()
        for func in functions:
            for var in func.variables_read:
                var_usage.add(f"{var.name} (read)")
            for var in func.variables_written:
                var_usage.add(f"{var.name} (write)")
                
        # Return context
        return f"""
Contract: {contract.name}
Type: {contract.contract_type}

Functions:
{chr(10).join(func_descs)}

State Variables Used:
{chr(10).join(f"- {v}" for v in var_usage)}
"""
    
    async def _validate_business_flows(
        self,
        contract: ContractDef,
        business_flows: List[BusinessFlow],
        project_id: str,
    ) -> List[BusinessFlow]:
        """
        Fourth pass: Validate and enhance business flows
        
        Args:
            contract: Contract definition
            business_flows: Extracted business flows
            project_id: Project ID for database storage
            
        Returns:
            List of validated and enhanced business flows
        """
        # Skip if intensity is too low or no flows
        if self.intensity < 0.6 or not business_flows:
            return business_flows
            
        # Process each flow
        enhanced_flows = []
        for flow in business_flows:
            # Skip if minimal or if intensity is moderate and flow is simple
            if (self.intensity < 0.8 and len(flow.source_functions) < 2):
                enhanced_flows.append(flow)
                continue
                
            # Generate prompt for validation
            prompt = self.prompts["business_flow_validation"].format(
                contract_name=contract.name,
                flow_name=flow.name,
                flow_type=flow.flow_type,
                flow_description=flow.description,
                flow_code=flow.extracted_code,
                flow_context=flow.context,
            )
            
            # Get response from validator model
            response = await self.validator.acomplete(prompt=prompt)
            
            # Parse validation response
            validation = self._parse_flow_validation(response)
            
            # Enhance the flow
            if validation.get("enhanced_name"):
                flow.name = validation["enhanced_name"]
                
            if validation.get("enhanced_description"):
                flow.description = validation["enhanced_description"]
                
            if validation.get("enhanced_type"):
                flow.flow_type = validation["enhanced_type"]
                
            if validation.get("context_additions"):
                flow.context += "\n\nAdditional Context:\n" + validation["context_additions"]
                
            # Add to enhanced flows
            enhanced_flows.append(flow)
            
            # Update telemetry
            self.telemetry["prompt_tokens_used"] += len(prompt.split())
            
        return enhanced_flows
    
    def _parse_flow_validation(self, response: str) -> Dict[str, Any]:
        """
        Parse flow validation response from LLM
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed validation results
        """
        # Default values
        validation = {
            "is_valid": True,
            "enhanced_name": "",
            "enhanced_description": "",
            "enhanced_type": "",
            "context_additions": "",
            "security_considerations": [],
        }
        
        # Extract validity
        valid_match = re.search(r"Valid Business Flow:?\s*(Yes|No)", response, re.IGNORECASE)
        if valid_match:
            validation["is_valid"] = valid_match.group(1).lower() == "yes"
            
        # Extract enhanced name
        name_match = re.search(r"Enhanced Name:?\s*(.+?)(?:\n|$)", response)
        if name_match:
            validation["enhanced_name"] = name_match.group(1).strip()
            
        # Extract enhanced description
        desc_match = re.search(r"Enhanced Description:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if desc_match:
            validation["enhanced_description"] = desc_match.group(1).strip()
            
        # Extract enhanced type
        type_match = re.search(r"Enhanced Type:?\s*(.+?)(?:\n|$)", response)
        if type_match:
            validation["enhanced_type"] = type_match.group(1).strip().lower()
            
        # Extract context additions
        context_match = re.search(r"Additional Context:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if context_match:
            validation["context_additions"] = context_match.group(1).strip()
            
        # Extract security considerations
        security_match = re.search(r"Security Considerations:?\s*(.+?)(?:\n\n|\n[A-Z]|$)", response, re.DOTALL)
        if security_match:
            security = security_match.group(1).strip()
            validation["security_considerations"] = [s.strip() for s in security.split("\n") if s.strip()]
            
        return validation
    
    async def _store_function_analysis(
        self,
        contract: ContractDef,
        function: FunctionDef,
        analysis: Dict[str, Any],
        project_id: str,
    ) -> None:
        """
        Store function analysis in the database
        
        Args:
            contract: Contract definition
            function: Function definition
            analysis: Function analysis results
            project_id: Project ID
        """
        # Skip if no db_manager
        if not self.db_manager:
            return
            
        # Create task record
        task = BusinessFlowTask(
            key=f"{project_id}:{contract.name}:{function.name}",
            project_id=project_id,
            name=function.name,
            content=function.source_code,
            business_type=analysis.get("type", "unknown"),
            function_type=analysis.get("type", "unknown"),
            description=analysis.get("description", ""),
            relative_file_path=os.path.basename(function.file_path),
            absolute_file_path=function.file_path,
            start_line=str(function.start_line),
            end_line=str(function.end_line),
        )
        
        # Store in database
        async with self.db_manager.async_session() as session:
            session.add(task)
            await session.commit()
    
    async def _store_business_flows(
        self,
        contract: ContractDef,
        flows: List[BusinessFlow],
        project_id: str,
    ) -> None:
        """
        Store business flows in the database
        
        Args:
            contract: Contract definition
            flows: Business flows
            project_id: Project ID
        """
        # Skip if no db_manager or no flows
        if not self.db_manager or not flows:
            return
            
        # Get project and file IDs
        async with self.db_manager.async_session() as session:
            # Get or create project
            from ..db.models import Project
            result = await session.execute(
                "SELECT id FROM projects WHERE project_id = :project_id",
                {"project_id": project_id}
            )
            project_row = result.first()
            if not project_row:
                # Create project
                project = Project(
                    project_id=project_id,
                    name=project_id,
                )
                session.add(project)
                await session.commit()
                await session.refresh(project)
                project_id_db = project.id
            else:
                project_id_db = project_row[0]
                
            # Get or create file
            from ..db.models import File
            result = await session.execute(
                "SELECT id FROM files WHERE path = :path AND project_id = :project_id",
                {"path": contract.file_path, "project_id": project_id_db}
            )
            file_row = result.first()
            if not file_row:
                # Create file
                file = File(
                    project_id=project_id_db,
                    path=contract.file_path,
                    name=os.path.basename(contract.file_path),
                    extension=os.path.splitext(contract.file_path)[1].lstrip("."),
                )
                session.add(file)
                await session.commit()
                await session.refresh(file)
                file_id = file.id
            else:
                file_id = file_row[0]
                
            # Get or create contract
            from ..db.models import CodeContract
            result = await session.execute(
                "SELECT id FROM code_contracts WHERE name = :name AND project_id = :project_id AND file_id = :file_id",
                {"name": contract.name, "project_id": project_id_db, "file_id": file_id}
            )
            contract_row = result.first()
            if not contract_row:
                # Create contract
                db_contract = CodeContract(
                    project_id=project_id_db,
                    file_id=file_id,
                    name=contract.name,
                    contract_type=contract.contract_type,
                    start_line=contract.start_line,
                    end_line=contract.end_line,
                    inheritance=contract.inheritance,
                    is_abstract=contract.is_abstract,
                    docstring=contract.docstring,
                )
                session.add(db_contract)
                await session.commit()
                await session.refresh(db_contract)
                contract_id = db_contract.id
            else:
                contract_id = contract_row[0]
                
            # Store business flows
            for flow in flows:
                # Create business flow
                db_flow = DBBusinessFlow(
                    contract_id=contract_id,
                    name=flow.name,
                    flow_type=flow.flow_type,
                    description=flow.description,
                    extracted_code=flow.extracted_code,
                    context=flow.context,
                    lines=flow.lines,
                )
                session.add(db_flow)
                
                # Store function relationships
                await session.commit()
                await session.refresh(db_flow)
                
                # Update business flow tasks
                for func in flow.source_functions:
                    await session.execute(
                        """
                        UPDATE business_flow_tasks 
                        SET business_flow_code = :code,
                            business_flow_lines = :lines,
                            business_flow_context = :context,
                            if_business_flow_scan = 'yes'
                        WHERE key = :key
                        """,
                        {
                            "code": flow.extracted_code,
                            "lines": ",".join(map(str, flow.lines)),
                            "context": flow.context,
                            "key": f"{project_id}:{contract.name}:{func.name}",
                        }
                    )
                
                await session.commit()