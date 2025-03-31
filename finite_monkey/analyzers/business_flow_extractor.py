import asyncio
import os
from typing import Dict, List, Any, Optional, AsyncIterator

from finite_monkey.models.business_flow import BusinessFlow
from ..llm.llama_index_adapter import LlamaIndexAdapter
import json
import logging
from loguru import logger
from llama_index.core.llms import ChatMessage
from finite_monkey.pipeline.core import Context

class BusinessFlowExtractor:
    """Extracts business flows from smart contracts using LlamaIndex"""
    
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
    
    async def analyze_function_flow(self, contract_name: str, function_name: str, contract_code: str) -> BusinessFlow:
        """
        Analyze business flow of a specific function
        
        Args:
            contract_name: Name of the contract
            function_name: Name of the function to analyze
            contract_code: Source code of the contract
            
        Returns:
            BusinessFlow object with the analysis results and attack surface data
        """
        
        # Create analysis prompt
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
        
        system_prompt = "You are a smart contract analyzer focused on understanding business flows, function relationships, and security implications."
        
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
            
            # Save results if needed
            #try:
            #    savefile = f"{contract_name}/{function_name}-business-flow.json"
            #    os.makedirs(os.path.dirname(savefile), exist_ok=True)
            #    with open(savefile, "w") as f:
            #        json.dump(business_flow.model_dump(), f, indent=2)
            #except Exception as e:
            #    logger.error(f"Error saving business flow to file: {e}")
            #
            return business_flow
            
        except Exception as e:
            logger.error(f"Error analyzing business flow: {e}")
            # Return an empty but valid BusinessFlow
            return BusinessFlow(
                FlowFunctions=[],
                Confidence=0.0,
                Notes=f"Error: {str(e)}"
            )
    
    async def process(self, context: Any) -> Any:
        """Process the context to extract business flows"""
        logger.info("Extracting business flows from smart contracts")
        
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
        
        logger.info(f"Business flow extraction complete")
        return context
    
    async def _process_function_with_semaphore(self, semaphore, context, file_id, file_data, func):
        """Process a function with semaphore to limit concurrency"""
        async with semaphore:
            return await self._process_function(context, file_id, file_data, func)
    
    async def _process_function(self, context: Any, file_id: str, file_data: Dict[str, Any], func: Dict[str, Any]):
        """Process a single function asynchronously"""
        from ..models.contract import FunctionDef
        from ..models.attack_surface import AttackSurface  # New import for attack surface model
        
        try:
            # Ensure func is a FunctionDef object
            if not isinstance(func, FunctionDef):
                func = FunctionDef(func)
                
            # Add contract name if not present
            if not func.contract_name and 'name' in file_data:
                func.contract_name = file_data['name']
                
            # Extract business flows for this function
            business_flow = await self.analyze_function_flow(
                contract_name=func.contract_name or "Contract", 
                function_name=func.name, 
                contract_code=func.full_text
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
                        "severity": self._calculate_severity(business_flow)
                    }
                    
                    logger.debug(f"Added attack surface data for {func.name} in {file_id}")
                
                logger.debug(f"Added business flow to function {func.name} in {file_id}")
        
        except Exception as e:
            logger.error(f"Error extracting business flows from function {func.name if hasattr(func, 'name') else func.get('name', 'unknown')}: {str(e)}")
            context.add_error(
                stage="business_flow_extraction",
                message=f"Failed to extract business flows from function {func.name if hasattr(func, 'name') else func.get('name', 'unknown')} in {file_id}",
                exception=e
            )
    
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

    async def process(self, context: Context) -> Context:
        """Process the context to extract business flows and attack surfaces"""
        logger.info("Extracting business flows and attack surfaces from smart contracts")
        
        # Initialize attack surfaces collection
        if not hasattr(context, 'attack_surfaces'):
            context.attack_surfaces = {}
        
        # ... existing processing code ...
        
        # After all business flows are extracted, aggregate attack surfaces
        await self._aggregate_attack_surfaces(context)
        
        logger.info(f"Business flow and attack surface extraction complete")
        return context
    
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
                'low_severity_items': []
            }
            
            # Group attack surfaces by severity
            for surface_id, surface in context.attack_surfaces.items():
                # Add to appropriate severity list
                severity_key = f"{surface['severity']}_severity_items"
                context.attack_surface_summary[severity_key].append({
                    'id': surface_id,
                    'function': surface['function'],
                    'contract': surface['contract']
                })
                
                # Add functions to vulnerable functions list
                context.attack_surface_summary['vulnerable_functions'].append({
                    'name': surface['function'],
                    'contract': surface['contract'],
                    'severity': surface['severity']
                })
                
                # Add variables to vulnerable variables list
                for variable in surface.get('variables', []):
                    if variable not in context.attack_surface_summary['vulnerable_variables']:
                        context.attack_surface_summary['vulnerable_variables'].append(variable)
                
                # Add paths to vulnerable paths list
                for path in surface.get('paths', []):
                    if path not in context.attack_surface_summary['vulnerable_paths']:
                        context.attack_surface_summary['vulnerable_paths'].append(path)
            
            logger.info(f"Aggregated {len(context.attack_surfaces)} attack surfaces into summary")
            
        except Exception as e:
            logger.error(f"Error aggregating attack surfaces: {str(e)}")
