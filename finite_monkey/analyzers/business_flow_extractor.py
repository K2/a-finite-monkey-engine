import asyncio
import os
from typing import Dict, List, Any, Optional, AsyncIterator, Literal, Union

from finite_monkey.models.business_flow import BusinessFlow
from ..llm.llama_index_adapter import LlamaIndexAdapter
import json
import logging
from loguru import logger
from llama_index.core.llms import ChatMessage
from finite_monkey.pipeline.core import Context

class BusinessFlowExtractor:
    """
    Extracts business flows from smart contracts using two approaches:
    1. Derived: Using code path traversal algorithms to detect vulnerable flows
    2. Acquired: Extracting flows from prose descriptions in issue text
    """
    
    def __init__(self, llm_adapter: Optional[LlamaIndexAdapter] = None):
        """Initialize with an optional LLM adapter"""
        # Check if we need to create a default LLM adapter
        if llm_adapter is None:
            try:
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                from finite_monkey.nodes_config import config
                
                # All configuration comes from config (Settings)
                self.llm_adapter = LlamaIndexAdapter(
                    model_name=config.BUSINESS_FLOW_MODEL,
                    provider=config.BUSINESS_FLOW_MODEL_PROVIDER,
                    base_url=config.BUSINESS_FLOW_MODEL_BASE_URL
                )
                logger.info(f"Created default LLM adapter with model: {config.BUSINESS_FLOW_MODEL}")
            except Exception as e:
                logger.warning(f"Failed to create default LLM adapter: {e}")
                self.llm_adapter = None
        else:
            self.llm_adapter = llm_adapter
        
        self.function_cache = {}  # Cache to avoid duplicate processing
    
    async def analyze_function_flow(
        self, 
        contract_name: str, 
        function_name: str, 
        contract_code: str,
        issue_text: Optional[str] = None
    ) -> BusinessFlow:
        """
        Analyze business flow of a specific function using either derived or acquired approach.
        
        Args:
            contract_name: Name of the contract
            function_name: Name of the function to analyze
            contract_code: Source code of the contract
            issue_text: Optional text from issues/discussions to derive flows from
            
        Returns:
            BusinessFlow object with the analysis results and attack surface data
        """
        # Determine analysis approach based on available data
        flow_source: Literal["derived", "acquired", "hybrid"] = "derived"
        
        if issue_text and len(issue_text.strip()) > 0:
            # If there's issue text, we can use it for acquired flow analysis
            if contract_code and len(contract_code.strip()) > 0:
                # If we have both code and issue text, use hybrid approach
                flow_source = "hybrid"
                logger.info(f"Using hybrid flow analysis (derived+acquired) for {function_name}")
            else:
                # If we only have issue text, use acquired approach
                flow_source = "acquired"
                logger.info(f"Using acquired flow analysis for {function_name} (from issue text)")
        else:
            # If no issue text, use derived approach (code path traversal)
            logger.info(f"Using derived flow analysis for {function_name} (from code)")
        
        # Create appropriate analysis prompt based on the flow source
        if flow_source == "acquired":
            business_flow = await self._analyze_acquired_flow(contract_name, function_name, issue_text)
        elif flow_source == "hybrid":
            # For hybrid, analyze both and merge results
            derived_flow = await self._analyze_derived_flow(contract_name, function_name, contract_code)
            acquired_flow = await self._analyze_acquired_flow(contract_name, function_name, issue_text)
            business_flow = self._merge_business_flows(derived_flow, acquired_flow)
        else:  # Default to derived
            business_flow = await self._analyze_derived_flow(contract_name, function_name, contract_code)
        
        # Add analysis source metadata
        if not hasattr(business_flow, "metadata"):
            business_flow.metadata = {}
        business_flow.metadata["flow_source"] = flow_source
        
        return business_flow
    
    async def _analyze_derived_flow(self, contract_name: str, function_name: str, contract_code: str) -> BusinessFlow:
        """
        Analyze business flow using code path traversal (derived approach).
        This approach derives the flow directly from code analysis.
        """
        # Create analysis prompt for code-based analysis
        prompt = f"""Analyze the business flow for the function "{function_name}" in the following Solidity contract.
        Identify all functions that are called by "{function_name}" and the sequence of these calls.
        
        Additionally, identify all attack surfaces:
        1. External functions that can be called by untrusted users
        2. Variables that can be manipulated or used in attacks
        3. Code paths or blocks that might be vulnerable to attacks
        4. Any privileged operations that could be security-critical
        
        Contract: {contract_name}
        
        ```solidity
        {contract_code}
        ```
        
        Return a structured response with:
        - FlowFunctions: An array of function calls, each with a "call" property (function name) and "flow_vars" (array of variables involved)
        - AttackSurfaces: An array of potentially vulnerable elements (functions, variables, paths)
        - Confidence: A score between 0.0 and 1.0 indicating your confidence in this analysis
        - Notes: Any additional notes or observations
        - Analysis: A brief analysis of the flow and security implications
        """
        
        system_prompt = "You are a smart contract analyzer focused on understanding business flows, function relationships, and security implications. Use static code analysis techniques to derive the business flow from the code."
        
        return await self._execute_llm_analysis(prompt, system_prompt)
    
    async def _analyze_acquired_flow(self, contract_name: str, function_name: str, issue_text: str) -> BusinessFlow:
        """
        Analyze business flow from prose descriptions (acquired approach).
        This approach acquires the flow from human descriptions rather than code analysis.
        """
        # Create analysis prompt for text-based analysis
        prompt = f"""Extract the business flow for the function "{function_name}" in the contract "{contract_name}" 
        from the following discussion or issue text. Focus on understanding how the function works and potential security issues
        based on how people are describing it.
        
        Discussion Text:
        ```
        {issue_text}
        ```
        
        Return a structured response with:
        - FlowFunctions: An array of function calls, each with a "call" property (function name) and "flow_vars" (array of variables involved)
        - AttackSurfaces: An array of potentially vulnerable elements mentioned in the discussion
        - Confidence: A score between 0.0 and 1.0 indicating your confidence in this analysis
        - Notes: Any additional notes or observations
        - Analysis: A brief analysis of the flow and security implications based on the discussion
        """
        
        system_prompt = "You are a smart contract analyzer focused on extracting business flows and security implications from natural language descriptions. Identify key elements of business logic and potential vulnerabilities from discussions."
        
        return await self._execute_llm_analysis(prompt, system_prompt)
    
    async def _execute_llm_analysis(self, prompt: str, system_prompt: str) -> BusinessFlow:
        """Execute LLM analysis with the given prompts"""
        try:
            # Check if adapter is available
            if not self.llm_adapter or not self.llm_adapter.llm:
                logger.error("No LLM adapter available for business flow analysis")
                return BusinessFlow()
            
            # Get structured LLM instance for BusinessFlow
            structured_llm = self.llm_adapter.llm.as_structured_llm(BusinessFlow)
            
            # Create chat message
            message = ChatMessage(role="user", content=prompt)
            system_message = ChatMessage(role="system", content=system_prompt)
            
            # Submit for structured analysis
            result = await structured_llm.achat([system_message, message])
            
            # The result is already a BusinessFlow object thanks to as_structured_llm
            business_flow: BusinessFlow = json.loads(result.message.content)
            return business_flow
            
        except Exception as e:
            logger.error(f"Error analyzing business flow: {e}")
            # Return an empty but valid BusinessFlow
            return BusinessFlow(
                FlowFunctions=[],
                Confidence=0.0,
                Notes=f"Error: {str(e)}"
            )
    
    def _merge_business_flows(self, derived_flow: BusinessFlow, acquired_flow: BusinessFlow) -> BusinessFlow:
        """
        Merge business flows from derived and acquired sources, prioritizing 
        derived data for technical accuracy and acquired data for human context.
        """
        # Start with the derived flow as the base
        merged_flow = derived_flow
        
        # Update confidence - use the higher confidence if available
        merged_flow.Confidence = max(
            getattr(derived_flow, 'Confidence', 0.0),
            getattr(acquired_flow, 'Confidence', 0.0)
        )
        
        # Add unique flow functions from acquired flow
        derived_calls = {f.call for f in getattr(derived_flow, 'FlowFunctions', [])}
        for acq_func in getattr(acquired_flow, 'FlowFunctions', []):
            if acq_func.call not in derived_calls:
                if not hasattr(merged_flow, 'FlowFunctions'):
                    merged_flow.FlowFunctions = []
                merged_flow.FlowFunctions.append(acq_func)
        
        # Add unique attack surfaces from acquired flow
        derived_surfaces = {
            getattr(s, 'name', '') + getattr(s, 'type', '') 
            for s in getattr(derived_flow, 'AttackSurfaces', [])
        }
        for acq_surface in getattr(acquired_flow, 'AttackSurfaces', []):
            surface_key = getattr(acq_surface, 'name', '') + getattr(acq_surface, 'type', '')
            if surface_key not in derived_surfaces:
                if not hasattr(merged_flow, 'AttackSurfaces'):
                    merged_flow.AttackSurfaces = []
                merged_flow.AttackSurfaces.append(acq_surface)
        
        # Combine notes
        derived_notes = getattr(derived_flow, 'Notes', '')
        acquired_notes = getattr(acquired_flow, 'Notes', '')
        if derived_notes and acquired_notes:
            merged_flow.Notes = f"Derived analysis: {derived_notes}\n\nAcquired analysis: {acquired_notes}"
        elif acquired_notes:
            merged_flow.Notes = acquired_notes
        
        # Combine analysis
        derived_analysis = getattr(derived_flow, 'Analysis', '')
        acquired_analysis = getattr(acquired_flow, 'Analysis', '')
        if derived_analysis and acquired_analysis:
            merged_flow.Analysis = f"Code analysis: {derived_analysis}\n\nFrom discussion: {acquired_analysis}"
        elif acquired_analysis:
            merged_flow.Analysis = acquired_analysis
        
        return merged_flow
    
    async def process(self, context: Context) -> Context:
        """Process the context to extract business flows and attack surfaces"""
        logger.info("Extracting business flows and attack surfaces from smart contracts")
        
        # Initialize attack surfaces collection
        if not hasattr(context, 'attack_surfaces'):
            context.attack_surfaces = {}
        
        # Use a semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent LLM requests
        
        # Process functions asynchronously using chunking with semaphore
        # to avoid over-consuming resources
        chunk_size = 10  # Process 10 functions at a time
        
        # Get all eligible functions
        all_functions = []
        for file_id, file_data in context.files.items():
            if not file_data.get("is_solidity", False) or "functions" not in file_data:
                continue
                
            for func in file_data["functions"]:
                # Skip already processed functions
                func_id = f"{file_id}:{func.get('name', '')}"
                if func_id not in self.function_cache:
                    self.function_cache[func_id] = True
                    all_functions.append((file_id, file_data, func))
        
        # Process in chunks to limit concurrency
        for i in range(0, len(all_functions), chunk_size):
            chunk = all_functions[i:i+chunk_size]
            
            # Create tasks for this chunk with semaphore
            tasks = []
            for file_id, file_data, func in chunk:
                task = self._process_function_with_semaphore(semaphore, context, file_id, file_data, func)
                tasks.append(task)
            
            # Wait for all tasks in this chunk to complete
            await asyncio.gather(*tasks)
            
            # Small delay to prevent overloading
            await asyncio.sleep(0.1)
        
        # After all business flows are extracted, aggregate attack surfaces
        await self._aggregate_attack_surfaces(context)
        
        logger.info(f"Business flow and attack surface extraction complete")
        return context
    
    async def _process_function_with_semaphore(self, semaphore, context, file_id, file_data, func):
        """Process a function with semaphore to limit concurrency"""
        async with semaphore:
            return await self._process_function(context, file_id, file_data, func)
    
    async def _process_function(self, context: Any, file_id: str, file_data: Dict[str, Any], func: Dict[str, Any]):
        """Process a single function asynchronously"""
        from ..models.contract import FunctionDef
        
        try:
            # Ensure func is a FunctionDef object
            if not isinstance(func, FunctionDef):
                func = FunctionDef(func)
                
            # Add contract name if not present
            if not func.contract_name and 'name' in file_data:
                func.contract_name = file_data['name']
            
            # Extract relevant issue text if available
            issue_text = self._extract_issue_text(context, func.name, func.contract_name)
                
            # Extract business flows for this function using either derived or acquired approach
            business_flow = await self.analyze_function_flow(
                contract_name=func.contract_name or "Contract", 
                function_name=func.name, 
                contract_code=func.full_text,
                issue_text=issue_text
            )
            
            # Add flow to function directly - no need for conversion
            if business_flow:
                func.add_business_flow(business_flow)
                
                # Extract attack surfaces from the business flow if available
                if hasattr(business_flow, 'AttackSurfaces'):
                    # Create an attack surface collection if it doesn't exist
                    if not hasattr(context, 'attack_surfaces'):
                        context.attack_surfaces = {}
                    
                    # Store the attack surfaces by function
                    attack_surface_id = f"{file_id}:{func.name}"
                    context.attack_surfaces[attack_surface_id] = {
                        "function": func.name,
                        "contract": func.contract_name,
                        "surfaces": business_flow.AttackSurfaces,
                        "variables": self._extract_vulnerable_variables(business_flow),
                        "paths": self._extract_vulnerable_paths(business_flow),
                        "severity": self._calculate_severity(business_flow),
                        "flow_source": getattr(business_flow, "metadata", {}).get("flow_source", "derived")
                    }
                    
                    logger.debug(f"Added attack surface data for {func.name} in {file_id} (source: {context.attack_surfaces[attack_surface_id]['flow_source']})")
                
                logger.debug(f"Added business flow to function {func.name} in {file_id}")
        
        except Exception as e:
            logger.error(f"Error extracting business flows from function {func.name if hasattr(func, 'name') else func.get('name', 'unknown')}: {str(e)}")
            context.add_error(
                stage="business_flow_extraction",
                message=f"Failed to extract business flows from function {func.name if hasattr(func, 'name') else func.get('name', 'unknown')} in {file_id}",
                exception=e
            )
    
    def _extract_issue_text(self, context: Any, function_name: str, contract_name: str) -> Optional[str]:
        """
        Extract relevant issue text from context to use for acquired flow analysis.
        
        Args:
            context: The analysis context
            function_name: Name of the function to find issue text for
            contract_name: Name of the contract to find issue text for
            
        Returns:
            Relevant issue text if found, None otherwise
        """
        # Check if we have issues in the context
        if not hasattr(context, 'issues') or not context.issues:
            return None
            
        # Extract relevant issue text
        relevant_text = []
        
        for issue in context.issues:
            # Skip issues without text
            if not hasattr(issue, 'description') or not issue.description:
                continue
                
            # Check if issue mentions the function or contract
            if (function_name.lower() in issue.description.lower() or 
                contract_name.lower() in issue.description.lower()):
                relevant_text.append(issue.description)
                
            # Also check comments if available
            if hasattr(issue, 'comments') and issue.comments:
                for comment in issue.comments:
                    if (function_name.lower() in comment.lower() or 
                        contract_name.lower() in comment.lower()):
                        relevant_text.append(comment)
        
        # Combine all relevant text
        if relevant_text:
            return "\n\n".join(relevant_text)
        
        return None
    
    def _extract_vulnerable_variables(self, business_flow: BusinessFlow) -> List[Dict[str, Any]]:
        """Extract vulnerable variables from business flow"""
        variables = []
        
        try:
            # Extract variables from flow functions
            for flow in getattr(business_flow, 'FlowFunctions', []):
                if hasattr(flow, 'flow_vars') and flow.flow_vars:
                    for var in flow.flow_vars:
                        if var not in [v.get('name') for v in variables]:
                            variables.append({
                                'name': var,
                                'usage': f"Used in {flow.call}" if hasattr(flow, 'call') else "Unknown usage",
                                'potentially_vulnerable': True
                            })
            
            # Add any explicitly identified vulnerable variables from AttackSurfaces
            if hasattr(business_flow, 'AttackSurfaces'):
                for surface in business_flow.AttackSurfaces:
                    if hasattr(surface, 'type') and surface.type == 'variable':
                        if surface.name not in [v.get('name') for v in variables]:
                            variables.append({
                                'name': surface.name,
                                'usage': getattr(surface, 'description', "Potentially vulnerable variable"),
                                'potentially_vulnerable': True
                            })
        except Exception as e:
            logger.error(f"Error extracting vulnerable variables: {str(e)}")
        
        return variables
    
    def _extract_vulnerable_paths(self, business_flow: BusinessFlow) -> List[Dict[str, Any]]:
        """Extract vulnerable paths/blocks from business flow"""
        paths = []
        
        try:
            # Extract paths from attack surfaces
            if hasattr(business_flow, 'AttackSurfaces'):
                for surface in business_flow.AttackSurfaces:
                    if hasattr(surface, 'type') and surface.type == 'code_path':
                        paths.append({
                            'description': getattr(surface, 'description', "Unknown code path"),
                            'potentially_vulnerable': True,
                            'related_to': getattr(surface, 'related_to', "Unknown")
                        })
        except Exception as e:
            logger.error(f"Error extracting vulnerable paths: {str(e)}")
        
        return paths
    
    def _calculate_severity(self, business_flow: BusinessFlow) -> str:
        """Calculate severity based on business flow analysis"""
        # Default severity
        severity = "low"
        
        try:
            # Calculate based on confidence and number of attack surfaces
            confidence = getattr(business_flow, 'Confidence', 0.5)
            num_surfaces = len(getattr(business_flow, 'AttackSurfaces', []))
            
            if num_surfaces > 5 or (num_surfaces > 2 and confidence > 0.7):
                severity = "high"
            elif num_surfaces > 2 or (num_surfaces > 0 and confidence > 0.7):
                severity = "medium"
            
        except Exception as e:
            logger.error(f"Error calculating severity: {str(e)}")
        
        return severity
    
    async def _aggregate_attack_surfaces(self, context: Context) -> None:
        """Aggregate all attack surfaces from business flows for holistic analysis"""
        try:
            if not hasattr(context, 'attack_surfaces') or not context.attack_surfaces:
                return
            
            # Create a summary of attack surfaces
            context.attack_surface_summary = {
                'vulnerable_functions': [],
                'vulnerable_variables': [],
                'vulnerable_paths': [],
                'high_severity_items': [],
                'medium_severity_items': [],
                'low_severity_items': [],
                'derived_flows': [],  # Add tracking for derived flows
                'acquired_flows': [],  # Add tracking for acquired flows
                'hybrid_flows': []    # Add tracking for hybrid flows
            }
            
            # Group attack surfaces by severity and flow source
            for surface_id, surface in context.attack_surfaces.items():
                # Add to appropriate severity list
                severity_key = f"{surface['severity']}_severity_items"
                context.attack_surface_summary[severity_key].append({
                    'id': surface_id,
                    'function': surface['function'],
                    'contract': surface['contract'],
                    'flow_source': surface.get('flow_source', 'derived')  # Track flow source
                })
                
                # Add to appropriate flow source list
                flow_source = surface.get('flow_source', 'derived')
                flow_source_key = f"{flow_source}_flows"
                if flow_source_key in context.attack_surface_summary:
                    context.attack_surface_summary[flow_source_key].append(surface_id)
                
                # Add functions to vulnerable functions list
                context.attack_surface_summary['vulnerable_functions'].append({
                    'name': surface['function'],
                    'contract': surface['contract'],
                    'severity': surface['severity'],
                    'flow_source': flow_source  # Track flow source
                })
                
                # Add variables to vulnerable variables list
                for variable in surface.get('variables', []):
                    if variable not in context.attack_surface_summary['vulnerable_variables']:
                        # Add flow source metadata
                        if isinstance(variable, dict) and 'flow_source' not in variable:
                            variable['flow_source'] = flow_source
                        context.attack_surface_summary['vulnerable_variables'].append(variable)
                
                # Add paths to vulnerable paths list
                for path in surface.get('paths', []):
                    if path not in context.attack_surface_summary['vulnerable_paths']:
                        # Add flow source metadata
                        if isinstance(path, dict) and 'flow_source' not in path:
                            path['flow_source'] = flow_source
                        context.attack_surface_summary['vulnerable_paths'].append(path)
            
            # Log flow source statistics
            derived_count = len(context.attack_surface_summary.get('derived_flows', []))
            acquired_count = len(context.attack_surface_summary.get('acquired_flows', []))
            hybrid_count = len(context.attack_surface_summary.get('hybrid_flows', []))
            
            logger.info(f"Aggregated {len(context.attack_surfaces)} attack surfaces: "
                       f"{derived_count} derived, {acquired_count} acquired, {hybrid_count} hybrid")
            
        except Exception as e:
            logger.error(f"Error aggregating attack surfaces: {str(e)}")
