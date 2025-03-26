"""
Enhanced Async Analyzer for Finite Monkey Engine

This module extends the core_async_analyzer with enhanced chunking capabilities
using the ContractChunker with CallGraph integration.
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime
from pathlib import Path

from .core_async_analyzer import AsyncAnalyzer, ContractParser
from .utils.chunking import AsyncContractChunker, CallGraph
from .adapters import Ollama
from .db.manager import DatabaseManager
from .nodes_config import nodes_config

# Configure logging
logger = logging.getLogger("enhanced-async-analyzer")
logger.setLevel(logging.INFO)

class EnhancedAsyncAnalyzer(AsyncAnalyzer):
    """
    Enhanced version of AsyncAnalyzer with improved chunking capabilities.
    
    This class extends the AsyncAnalyzer to use the ContractChunker with CallGraph
    integration for better code structure understanding, especially for large
    codebases where hierarchical file->contract->function analysis is valuable.
    """
    
    def __init__(
        self,
        primary_llm_client: Optional[Ollama] = None,
        secondary_llm_client: Optional[Ollama] = None,
        db_manager: Optional[DatabaseManager] = None,
        primary_model_name: Optional[str] = None,
        secondary_model_name: Optional[str] = None,
    ):
        """
        Initialize the enhanced async analyzer.
        
        Args:
            primary_llm_client: Primary LLM client for initial analysis
            secondary_llm_client: Secondary LLM client for validation
            db_manager: Database manager
            primary_model_name: Name of the primary model
            secondary_model_name: Name of the secondary model
        """
        # Initialize the base class
        super().__init__(
            primary_llm_client=primary_llm_client,
            secondary_llm_client=secondary_llm_client,
            db_manager=db_manager,
            primary_model_name=primary_model_name,
            secondary_model_name=secondary_model_name,
        )
        
        # Initialize chunkers
        self.contract_chunker = None
        self.async_contract_chunker = None
    
    async def analyze_contract_file_with_chunking(self, 
                                                file_path: str, 
                                                project_id: str = "default", 
                                                query: Optional[str] = None,
                                                max_chunk_size: int = 4000,
                                                include_call_graph: bool = True) -> Dict[str, Any]:
        """
        Run the full analysis pipeline on a contract file with enhanced chunking.
        
        This method extends the base analyze_contract_file method by using the
        ContractChunker with CallGraph to better handle large contracts and
        provide more detailed code structure understanding.
        
        Args:
            file_path: Path to the Solidity file
            project_id: Project identifier
            query: Optional specific query to focus analysis
            max_chunk_size: Maximum chunk size in characters
            include_call_graph: Whether to include call graph information
            
        Returns:
            Analysis results
        """
        file_id = os.path.basename(file_path)
        logger.info(f"Starting enhanced analysis of {file_id} for project {project_id}")
        
        # Initialize chunker if needed
        if self.contract_chunker is None:
            self.contract_chunker = AsyncContractChunker(
                max_chunk_size=max_chunk_size,
                chunk_by_contract=True,
                chunk_by_function=True,
                include_call_graph=include_call_graph,
            )
            
            # Initialize call graph for the project directory
            if include_call_graph:
                project_path = os.path.dirname(file_path)
                logger.info(f"Initializing call graph for {project_path}")
                self.contract_chunker.initialize_call_graph(project_path)
        
        # Step 1: Chunk the contract with call graph
        try:
            chunks = self.contract_chunker.chunk_file(file_path)
            logger.info(f"Successfully chunked {file_id} into {len(chunks)} segments")
            
            # Extract contract data from chunks
            contract_data = self._extract_contract_data_from_chunks(chunks)
            
            # Extract call graph information
            call_graph = self._extract_call_graph_from_chunks(chunks)
            
            # Add source code to contract data
            with open(file_path, 'r', encoding='utf-8') as f:
                contract_data["source_code"] = f.read()
                
        except Exception as e:
            logger.error(f"Error chunking {file_id}: {e}")
            traceback.print_exc()
            # Fall back to standard parsing
            return await super().analyze_contract_file(file_path, project_id, query)
        
        # Step 2: Extract flow information from chunks
        try:
            flow_data = self._extract_flow_data_from_chunks(chunks)
            logger.info(f"Extracted flow data from chunks for {file_id}")
            
            # Join flows to create comprehensive data flows
            flow_paths = self._extract_flow_paths_from_chunks(chunks)
            logger.info(f"Extracted flow paths from chunks for {file_id}")
            
        except Exception as e:
            logger.error(f"Error extracting flow data from chunks for {file_id}: {e}")
            flow_data = {}
            flow_paths = {}
        
        # Step 3: Generate test expressions using flow data (if db_manager available)
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
        
        # Step 4: Analyze with primary LLM including flow context and chunk information
        try:
            primary_analysis = await self._run_primary_analysis_with_chunks(
                contract_data, call_graph, expressions, query, flow_paths, chunks
            )
            logger.info(f"Completed primary analysis for {file_id}")
        except Exception as e:
            logger.error(f"Error in primary analysis for {file_id}: {e}")
            # Fall back to standard analysis
            return await super().analyze_contract_file(file_path, project_id, query)
        
        # Step 5: Validate with secondary LLM
        try:
            secondary_validation = await self._run_secondary_validation(
                contract_data, primary_analysis, expressions, flow_data
            )
            logger.info(f"Completed secondary validation for {file_id}")
        except Exception as e:
            logger.error(f"Error in secondary validation for {file_id}: {e}")
            return {"error": f"Secondary validation error: {str(e)}"}
        
        # Step 6: Generate final report
        try:
            final_report = await self._generate_final_report(
                file_id, contract_data, primary_analysis, secondary_validation, flow_paths
            )
            logger.info(f"Generated final report for {file_id}")
        except Exception as e:
            logger.error(f"Error generating final report for {file_id}: {e}")
            return {"error": f"Report generation error: {str(e)}"}
        
        # Create result
        result = {
            "file_id": file_id,
            "project_id": project_id,
            "primary_analysis": primary_analysis,
            "secondary_validation": secondary_validation,
            "final_report": final_report,
            "chunk_count": len(chunks),
            "timestamp": datetime.now().isoformat(),
            "enhanced_chunking": True
        }
        
        return result
    
    async def analyze_project_with_chunking(self, 
                                          project_path: str, 
                                          project_id: Optional[str] = None,
                                          query: Optional[str] = None,
                                          max_chunk_size: int = 4000,
                                          include_call_graph: bool = True) -> Dict[str, Any]:
        """
        Analyze all Solidity files in a project directory with enhanced chunking.
        
        Args:
            project_path: Path to project directory
            project_id: Optional project identifier (defaults to directory name)
            query: Optional specific query to focus analysis
            max_chunk_size: Maximum chunk size in characters
            include_call_graph: Whether to include call graph information
            
        Returns:
            Project analysis results
        """
        # Load config for concurrency settings
        config = nodes_config()
        max_concurrent = config.MAX_THREADS_OF_SCAN or 4
        
        # Derive project_id from directory name if not provided
        if project_id is None:
            project_id = os.path.basename(os.path.normpath(project_path))
        
        # Find all Solidity files in the project - respecting ignore folders from config
        sol_files = []
        ignore_folders = config.IGNORE_FOLDERS.split(",") if config.IGNORE_FOLDERS else ["test", "tests", "node_modules"]
        
        for root, dirs, files in os.walk(project_path):
            # Skip ignored folders
            dirs[:] = [d for d in dirs if d.lower() not in ignore_folders]
            
            for file in files:
                if file.endswith('.sol'):
                    sol_files.append(os.path.join(root, file))
        
        if not sol_files:
            logger.warning(f"No Solidity files found in {project_path}")
            return {"error": "No Solidity files found"}
        
        logger.info(f"Found {len(sol_files)} Solidity files in project {project_id}")
        
        # Initialize AsyncContractChunker with call graph for entire project
        if self.async_contract_chunker is None and include_call_graph:
            logger.info(f"Initializing AsyncContractChunker for project {project_id}")
            self.async_contract_chunker = AsyncContractChunker(
                max_chunk_size=max_chunk_size,
                chunk_by_contract=True,
                chunk_by_function=True,
                include_call_graph=include_call_graph,
            )
            
            # Initialize call graph
            await self.async_contract_chunker.initialize_call_graph(project_path)
            logger.info(f"Call graph initialized for project {project_id}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Define a wrapper function that uses the semaphore
        async def analyze_with_semaphore(file_path):
            async with semaphore:
                return await self.analyze_contract_file_with_chunking(
                    file_path, 
                    project_id, 
                    query, 
                    max_chunk_size, 
                    include_call_graph
                )
        
        # Analyze each file with controlled concurrency
        logger.info(f"Starting analysis of {len(sol_files)} files with max concurrency: {max_concurrent}")
        tasks = [analyze_with_semaphore(file_path) for file_path in sol_files]
        results = await asyncio.gather(*tasks)
        
        # Combine results
        combined_results = {
            "project_id": project_id,
            "file_count": len(sol_files),
            "files": {os.path.basename(sol_files[i]): result for i, result in enumerate(results)},
            "timestamp": datetime.now().isoformat(),
            "enhanced_chunking": True
        }
        
        # Extract and aggregate findings
        findings = []
        for file_result in results:
            if "final_report" in file_result and "findings" in file_result["final_report"]:
                for finding in file_result["final_report"]["findings"]:
                    finding["file"] = file_result.get("file_id", "unknown")
                    findings.append(finding)
        
        combined_results["findings"] = findings
        combined_results["finding_count"] = len(findings)
        
        # Group findings by severity
        severity_counts = {}
        for finding in findings:
            severity = finding.get("severity", "Unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        combined_results["severity_summary"] = severity_counts
        
        return combined_results
    
    async def _run_primary_analysis_with_chunks(self, 
                                            contract_data: Dict[str, Any],
                                            call_graph: Dict[str, Any],
                                            expressions: List[Dict[str, Any]],
                                            query: Optional[str] = None,
                                            flow_paths: Optional[Dict[str, Any]] = None,
                                            chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run the primary analysis using LLM with chunk information.
        
        Args:
            contract_data: Parsed contract data
            call_graph: Function call graph
            expressions: Test expressions
            query: Optional specific query to focus analysis
            flow_paths: Optional flow path data for enhanced analysis
            chunks: Optional chunks from ContractChunker
            
        Returns:
            Primary analysis results
        """
        # This extends the existing _run_primary_analysis method by including
        # chunk information in the analysis prompt
        
        # Format code for analysis
        source_code = contract_data["source_code"]
        
        # Format contract structure for prompt
        contract_structure = json.dumps({
            "contracts": {
                name: {
                    "function_count": len(data["functions"]),
                    "state_var_count": len(data["state_variables"]),
                    "inheritance": data["inheritance"]
                } for name, data in contract_data["contracts"].items()
            }
        }, indent=2)
        
        # Format expressions for prompt
        expressions_text = "\n".join([
            f"- {expr['expression']} (Severity: {expr['severity']}, Category: {expr['category']})"
            for expr in expressions
        ])
        
        # Format flow data for prompt if available
        flow_analysis = ""
        if flow_paths:
            flow_snippets = []
            
            # For each contract, include critical flow data
            for contract_name, functions in flow_paths.items():
                # Add contract header
                flow_snippets.append(f"CONTRACT: {contract_name}")
                
                # For each function with interesting flow data
                for func_name, func_flow in functions.items():
                    # Only include functions with meaningful flow data
                    if (func_flow.get("calls") or func_flow.get("state_dependencies") or 
                        func_flow.get("value_flow") or func_flow.get("path_conditions")):
                        
                        flow_snippets.append(f"  FUNCTION: {func_name}")
                        
                        # Add state dependencies
                        if func_flow.get("state_dependencies"):
                            flow_snippets.append("    STATE DEPENDENCIES:")
                            for dep in func_flow.get("state_dependencies"):
                                flow_snippets.append(f"      - {dep.get('variable')}: {dep.get('context')}")
                        
                        # Add value flows
                        if func_flow.get("value_flow"):
                            flow_snippets.append("    VALUE TRANSFERS:")
                            for flow in func_flow.get("value_flow"):
                                flow_snippets.append(f"      - To: {flow.get('to')} (Line: {flow.get('line')})")
                        
                        # Add path conditions
                        if func_flow.get("path_conditions"):
                            flow_snippets.append("    PATH CONDITIONS:")
                            for cond in func_flow.get("path_conditions"):
                                flow_snippets.append(f"      - {cond.get('condition')} (Line: {cond.get('line')})")
                                
                        # Add function calls
                        if func_flow.get("calls"):
                            flow_snippets.append("    CALLS:")
                            for call in func_flow.get("calls"):
                                flow_snippets.append(f"      - {call.get('contract')}.{call.get('function')}")
            
            # Add flow data if not empty
            if flow_snippets:
                flow_analysis = "CONTROL FLOW ANALYSIS:\n" + "\n".join(flow_snippets)
        
        # Add chunk information if available
        chunk_analysis = ""
        if chunks:
            chunk_snippets = []
            
            # Add header for chunk analysis
            chunk_snippets.append(f"CONTRACT STRUCTURE (CHUNKED):")
            
            # Group chunks by contract
            contract_chunks = {}
            for chunk in chunks:
                if chunk["chunk_type"] == "contract":
                    contract_name = chunk.get("contract_name", "Unknown")
                    if contract_name not in contract_chunks:
                        contract_chunks[contract_name] = []
                    contract_chunks[contract_name].append(chunk)
            
            # Add contract chunk information
            for contract_name, chunks_list in contract_chunks.items():
                chunk_snippets.append(f"CONTRACT: {contract_name}")
                
                # Add function information if available
                function_chunks = []
                for chunk in chunks:
                    if chunk["chunk_type"] == "function" and chunk.get("contract_name") == contract_name:
                        function_chunks.append(chunk)
                
                # Show functions for this contract
                if function_chunks:
                    chunk_snippets.append(f"  FUNCTIONS:")
                    
                    for func_chunk in function_chunks:
                        func_name = func_chunk.get("function_name", "Unknown")
                        chunk_snippets.append(f"    - {func_name}")
                        
                        # Add caller/callee info if available
                        if "function_calls" in func_chunk:
                            calls = func_chunk["function_calls"]
                            if calls:
                                chunk_snippets.append(f"      CALLS:")
                                for call in calls[:3]:  # Limit to first 3
                                    chunk_snippets.append(f"        - {call['contract']}.{call['function']}")
                                if len(calls) > 3:
                                    chunk_snippets.append(f"        - ... and {len(calls) - 3} more")
                        
                        if "called_by" in func_chunk:
                            callers = func_chunk["called_by"]
                            if callers:
                                chunk_snippets.append(f"      CALLED BY:")
                                for caller in callers[:3]:  # Limit to first 3
                                    chunk_snippets.append(f"        - {caller['contract']}.{caller['function']}")
                                if len(callers) > 3:
                                    chunk_snippets.append(f"        - ... and {len(callers) - 3} more")
            
            # Add chunk data if not empty
            if chunk_snippets:
                chunk_analysis = "\n".join(chunk_snippets)
        
        # Customize query if provided, otherwise use default
        analysis_query = query or "Perform a comprehensive security audit of the smart contract"
        
        # Create analysis prompt with flow data and chunk information
        prompt = f"""
You are a smart contract security auditor analyzing Solidity code. 
Analyze the following contract for security vulnerabilities and issues.

QUERY: {analysis_query}

CONTRACT SOURCE CODE:
```solidity
{source_code}
```

CONTRACT STRUCTURE:
{contract_structure}

{flow_analysis}

{chunk_analysis}

SUGGESTED TEST EXPRESSIONS:
{expressions_text}

Analyze the contract for the following vulnerability categories:
1. Reentrancy
2. Access Control
3. Arithmetic Issues
4. Unchecked External Calls
5. Denial of Service
6. Front-Running
7. Transaction Ordering Dependence
8. Block Timestamp Manipulation
9. Unsafe Type Inference
10. Gas Optimization Issues

For each issue identified:
1. Provide a clear title
2. Describe the vulnerability in detail
3. Specify the severity (Critical, High, Medium, Low, Informational)
4. Include the exact location in the code (contract, function, line)
5. Explain the impact of the vulnerability
6. Recommend a specific fix with code examples where possible

Focus on actionable findings with clear impact, rather than theoretical or low-risk issues.
Format your response in Markdown with proper sections and code blocks.
"""
        
        # Get analysis from LLM
        analysis_result = await self.primary_llm.acomplete(prompt)
        
        # Extract findings from response
        findings = self._extract_findings(analysis_result)
        
        return {
            "raw_analysis": analysis_result,
            "findings": findings,
            "timestamp": datetime.now().isoformat(),
            "flow_enhanced": bool(flow_analysis),
            "chunk_enhanced": bool(chunk_analysis)
        }

    def _extract_contract_data_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract contract data from chunks.
        
        Args:
            chunks: List of chunks from ContractChunker
            
        Returns:
            Contract data in the format expected by AsyncAnalyzer
        """
        # Initialize contract data structure
        contracts = {}
        
        # Process contract chunks first
        for chunk in chunks:
            if chunk["chunk_type"] == "contract":
                contract_name = chunk.get("contract_name", "Unknown")
                contracts[contract_name] = {
                    "name": contract_name,
                    "functions": {},
                    "state_variables": {},
                    "modifiers": {},
                    "events": {},
                    "inheritance": []
                }
                
                # Extract contract functions from contract_functions if available
                if "contract_functions" in chunk:
                    for function_name in chunk["contract_functions"]:
                        contracts[contract_name]["functions"][function_name] = {
                            "name": function_name,
                            "visibility": "unknown",
                            "parameters": [],
                            "returns": [],
                            "modifiers": [],
                            "code": ""
                        }
        
        # Process function chunks to update function details
        for chunk in chunks:
            if chunk["chunk_type"] == "function":
                contract_name = chunk.get("contract_name")
                function_name = chunk.get("function_name")
                
                if contract_name in contracts and function_name:
                    # Update function code
                    contracts[contract_name]["functions"][function_name] = {
                        "name": function_name,
                        "visibility": "unknown",
                        "parameters": [],
                        "returns": [],
                        "modifiers": [],
                        "code": chunk["content"]
                    }
        
        return {"contracts": contracts}
    
    def _extract_call_graph_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract call graph information from chunks.
        
        Args:
            chunks: List of chunks from ContractChunker
            
        Returns:
            Call graph data
        """
        call_graph = {}
        
        # Process contract chunks first to initialize call graph structure
        for chunk in chunks:
            if chunk["chunk_type"] == "contract":
                contract_name = chunk.get("contract_name", "Unknown")
                call_graph[contract_name] = {}
        
        # Process function chunks to populate call information
        for chunk in chunks:
            if chunk["chunk_type"] == "function":
                contract_name = chunk.get("contract_name")
                function_name = chunk.get("function_name")
                
                if contract_name in call_graph and function_name:
                    calls = []
                    
                    # Extract function calls if available
                    if "function_calls" in chunk:
                        for call in chunk["function_calls"]:
                            calls.append({
                                "contract": call["contract"],
                                "function": call["function"]
                            })
                    
                    call_graph[contract_name][function_name] = calls
        
        return call_graph
    
    def _extract_flow_data_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract flow data from chunks.
        
        Args:
            chunks: List of chunks from ContractChunker
            
        Returns:
            Flow data structure
        """
        flow_data = {}
        
        # Process contract chunks first to initialize flow data structure
        for chunk in chunks:
            if chunk["chunk_type"] == "contract":
                contract_name = chunk.get("contract_name", "Unknown")
                flow_data[contract_name] = {}
        
        # Process function chunks to populate flow information
        for chunk in chunks:
            if chunk["chunk_type"] == "function":
                contract_name = chunk.get("contract_name")
                function_name = chunk.get("function_name")
                
                if contract_name in flow_data and function_name:
                    # Initialize function flow data
                    flow_data[contract_name][function_name] = {
                        "external_calls": [],
                        "state_changes": [],
                        "control_structures": [],
                        "condition_checks": []
                    }
                    
                    # Extract function calls if available
                    if "function_calls" in chunk:
                        for call in chunk["function_calls"]:
                            flow_data[contract_name][function_name]["external_calls"].append({
                                "target": f"{call['contract']}.{call['function']}",
                                "type": "call",
                                "line": 0  # Line number not available from chunks
                            })
        
        return flow_data
    
    def _extract_flow_paths_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract flow paths from chunks.
        
        Args:
            chunks: List of chunks from ContractChunker
            
        Returns:
            Flow paths structure
        """
        flow_paths = {}
        
        # Process contract chunks first to initialize flow paths structure
        for chunk in chunks:
            if chunk["chunk_type"] == "contract":
                contract_name = chunk.get("contract_name", "Unknown")
                flow_paths[contract_name] = {}
        
        # Process function chunks to populate flow path information
        for chunk in chunks:
            if chunk["chunk_type"] == "function":
                contract_name = chunk.get("contract_name")
                function_name = chunk.get("function_name")
                
                if contract_name in flow_paths and function_name:
                    # Initialize function flow paths
                    flow_paths[contract_name][function_name] = {
                        "calls": [],
                        "state_dependencies": [],
                        "value_flow": [],
                        "path_conditions": []
                    }
                    
                    # Extract function calls if available
                    if "function_calls" in chunk:
                        for call in chunk["function_calls"]:
                            flow_paths[contract_name][function_name]["calls"].append({
                                "contract": call["contract"],
                                "function": call["function"]
                            })
        
        return flow_paths