"""
Flow-based threat analyzer that combines call flow analysis with vector search to identify
potentially vulnerable execution paths in code.
"""
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path
from loguru import logger

from finite_monkey.core.pipeline import PipelineComponent, PipelineContext
from tools.vector_store_util import SimpleVectorStore

class FlowThreatAnalyzer(PipelineComponent):
    """
    A pipeline component that analyzes execution flows to identify potential security threats
    by comparing them with known vulnerable paths in the vector database.
    
    Key features:
    1. Extracts execution flows from code and ASTs
    2. Matches flows against known vulnerable patterns
    3. Identifies reachable vulnerable code paths
    4. Provides comprehensive flow-based threat analysis
    """
    
    def __init__(
        self,
        vector_store_dir: str = None,
        collection_name: str = "threats",
        similarity_threshold: float = 0.65,
        max_paths_to_analyze: int = 20,
        include_transitive_flows: bool = True,
    ):
        """
        Initialize the FlowThreatAnalyzer.
        
        Args:
            vector_store_dir: Directory containing vector stores
            collection_name: Name of the threat collection to use
            similarity_threshold: Minimum similarity score to consider a path match
            max_paths_to_analyze: Maximum number of execution paths to analyze
            include_transitive_flows: Whether to include transitive flow analysis
        """
        super().__init__()
        
        # Store configuration
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.max_paths_to_analyze = max_paths_to_analyze
        self.include_transitive_flows = include_transitive_flows
        
        # Placeholders for components
        self._vector_store = None
        self._flow_extractor = None
        self._initialization_complete = False
    
    async def initialize(self):
        """Initialize required components."""
        try:
            logger.info(f"Initializing FlowThreatAnalyzer with collection: {self.collection_name}")
            
            # Initialize vector store
            self._vector_store = SimpleVectorStore(
                storage_dir=self.vector_store_dir,
                collection_name=self.collection_name
            )
            
            # Try to import flow extractor
            try:
                from finite_monkey.analysis.flow_extractor import FlowExtractor
                self._flow_extractor = FlowExtractor()
                logger.info("Flow extractor initialized")
            except ImportError:
                logger.warning("Flow extractor not available, using basic extraction")
                self._flow_extractor = None
            
            self._initialization_complete = True
            logger.info("FlowThreatAnalyzer initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing FlowThreatAnalyzer: {e}")
            self._initialization_complete = False
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process the input context and identify flow-based threats.
        
        Args:
            context: Pipeline context containing code, AST, or existing flows
            
        Returns:
            Updated context with flow threat analysis
        """
        if not self._initialization_complete:
            await self.initialize()
            if not self._initialization_complete:
                logger.error("FlowThreatAnalyzer initialization failed, skipping processing")
                return context
        
        try:
            # First check if flows were already extracted
            execution_flows = self._get_execution_flows_from_context(context)
            
            # If no flows are found, try to extract them
            if not execution_flows:
                execution_flows = await self._extract_execution_flows(context)
                
                # If still no flows, we can't proceed
                if not execution_flows:
                    logger.warning("No execution flows found or extracted, skipping flow threat analysis")
                    return context
                
                # Add extracted flows to context for other components to use
                context.set('execution_flows', execution_flows)
            
            logger.info(f"Analyzing {len(execution_flows)} execution flows for threats")
            
            # Analyze each flow for threats
            flow_threats = []
            for flow_idx, flow in enumerate(execution_flows[:self.max_paths_to_analyze]):
                # Skip empty flows
                if not flow:
                    continue
                
                # Find matching vulnerable paths for this flow
                matches = await self._vector_store.find_matching_paths(
                    flow, 
                    similarity_threshold=self.similarity_threshold
                )
                
                if matches:
                    # Get the top matches
                    top_matches = matches[:5]  # Limit to top 5 for clarity
                    
                    # Add flow threat info
                    flow_threat = {
                        'flow_index': flow_idx,
                        'execution_flow': flow,
                        'matched_vulnerable_paths': top_matches,
                        'match_count': len(matches),
                        'highest_similarity': max(m['similarity'] for m in matches),
                        'average_similarity': sum(m['similarity'] for m in matches) / len(matches)
                    }
                    
                    flow_threats.append(flow_threat)
                    
                    logger.info(f"Found {len(matches)} vulnerable path matches for flow {flow_idx}")
            
            # Analyze transitive flows if requested
            if self.include_transitive_flows and hasattr(self, '_analyze_transitive_flows'):
                transitive_threats = await self._analyze_transitive_flows(execution_flows, context)
                if transitive_threats:
                    context.set('transitive_flow_threats', transitive_threats)
            
            # Add results to context
            if flow_threats:
                context.set('flow_threats', flow_threats)
                logger.info(f"Identified {len(flow_threats)} flows with potential threats")
            else:
                logger.info("No flow-based threats detected")
            
            return context
            
        except Exception as e:
            logger.error(f"Error in FlowThreatAnalyzer processing: {e}")
            context.set('flow_analysis_error', str(e))
            return context
    
    def _get_execution_flows_from_context(self, context: PipelineContext) -> List[List[str]]:
        """
        Extract execution flows from the context if already present.
        
        Args:
            context: Pipeline context
            
        Returns:
            List of execution flows, where each flow is a list of function calls
        """
        # Check for existing flows in different formats
        if context.has('execution_flows'):
            flows = context.get('execution_flows')
            if isinstance(flows, list):
                return flows
        
        if context.has('call_flows'):
            flows = context.get('call_flows')
            if isinstance(flows, list):
                return flows
        
        if context.has('call_graph'):
            call_graph = context.get('call_graph')
            if isinstance(call_graph, dict) and 'paths' in call_graph:
                return call_graph['paths']
        
        return []
    
    async def _extract_execution_flows(self, context: PipelineContext) -> List[List[str]]:
        """
        Extract execution flows from code or AST in the context.
        
        Args:
            context: Pipeline context with code or AST
            
        Returns:
            List of execution flows
        """
        # Use FlowExtractor if available
        if self._flow_extractor:
            try:
                if context.has('ast'):
                    return await self._flow_extractor.extract_flows_from_ast(context.get('ast'))
                
                if context.has('code'):
                    code = context.get('code')
                    return await self._flow_extractor.extract_flows_from_code(code)
                
                if context.has('file_paths'):
                    file_paths = context.get('file_paths')
                    if isinstance(file_paths, str):
                        file_paths = [file_paths]
                    
                    all_flows = []
                    for file_path in file_paths:
                        flows = await self._flow_extractor.extract_flows_from_file(file_path)
                        all_flows.extend(flows)
                    
                    return all_flows
            except Exception as e:
                logger.error(f"Error using FlowExtractor: {e}")
        
        # Fallback to basic extraction if FlowExtractor not available or fails
        return await self._basic_flow_extraction(context)
    
    async def _basic_flow_extraction(self, context: PipelineContext) -> List[List[str]]:
        """
        Basic flow extraction when FlowExtractor is not available.
        
        Args:
            context: Pipeline context
            
        Returns:
            List of extracted flows
        """
        try:
            extracted_flows = []
            
            # Handle code string
            if context.has('code'):
                code = context.get('code')
                if isinstance(code, str):
                    # Extract function calls using regex
                    import re
                    # Match function or method calls: name(args) or obj.name(args)
                    call_pattern = r'(?:\w+\.)?(\w+)\s*\('
                    calls = re.findall(call_pattern, code)
                    
                    if calls:
                        # Create a simple linear flow of all calls
                        # (This is a very simplified approach - a real implementation would need 
                        # proper parsing to build accurate control flow graphs)
                        extracted_flows.append(calls)
            
            # Handle file paths
            if context.has('file_paths'):
                file_paths = context.get('file_paths')
                if isinstance(file_paths, str):
                    file_paths = [file_paths]
                
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                file_content = f.read()
                            
                            # Extract function calls using regex
                            import re
                            call_pattern = r'(?:\w+\.)?(\w+)\s*\('
                            calls = re.findall(call_pattern, file_content)
                            
                            if calls:
                                # Create a simple linear flow
                                extracted_flows.append(calls)
                        except Exception as e:
                            logger.error(f"Error extracting flows from file {file_path}: {e}")
            
            return extracted_flows
        except Exception as e:
            logger.error(f"Error in basic flow extraction: {e}")
            return []
    
    async def _analyze_transitive_flows(self, 
                                      execution_flows: List[List[str]], 
                                      context: PipelineContext) -> List[Dict[str, Any]]:
        """
        Analyze transitive flows to identify potentially reachable vulnerable paths.
        
        Args:
            execution_flows: List of execution flows to analyze
            context: Pipeline context with additional information
            
        Returns:
            List of transitive flow threats
        """
        try:
            # Extract call graph if present
            call_graph = {}
            if context.has('call_graph'):
                call_graph = context.get('call_graph')
            elif context.has('static_analysis') and 'call_graph' in context.get('static_analysis', {}):
                call_graph = context.get('static_analysis').get('call_graph', {})
            
            # Can't analyze transitive flows without a call graph
            if not call_graph:
                return []
            
            # Get function dependencies from call graph
            function_deps = {}
            if isinstance(call_graph, dict) and 'edges' in call_graph:
                for edge in call_graph['edges']:
                    if 'from' in edge and 'to' in edge:
                        from_func = edge['from']
                        to_func = edge['to']
                        
                        if from_func not in function_deps:
                            function_deps[from_func] = set()
                        
                        function_deps[from_func].add(to_func)
            
            # If no dependencies found, can't analyze transitive flows
            if not function_deps:
                return []
            
            # Search for vulnerable functions in the vector database
            vulnerable_funcs = await self._find_vulnerable_functions()
            
            # Check which vulnerable functions are reachable from our execution flows
            transitive_threats = []
            
            for flow_idx, flow in enumerate(execution_flows):
                reachable_vulns = set()
                
                for func in flow:
                    # Find all functions transitively reachable from this one
                    reachable = self._find_transitive_closure(func, function_deps)
                    
                    # Check if any vulnerable functions are reachable
                    for vuln_func in vulnerable_funcs:
                        if vuln_func['function_name'] in reachable:
                            reachable_vulns.add(vuln_func['function_name'])
                
                if reachable_vulns:
                    # Find the detailed vulnerability information for these functions
                    vuln_details = [v for v in vulnerable_funcs if v['function_name'] in reachable_vulns]
                    
                    # Add to transitive threats
                    transitive_threat = {
                        'flow_index': flow_idx,
                        'execution_flow': flow,
                        'reachable_vulnerable_functions': list(reachable_vulns),
                        'vulnerability_details': vuln_details,
                        'count': len(reachable_vulns)
                    }
                    
                    transitive_threats.append(transitive_threat)
            
            return transitive_threats
        except Exception as e:
            logger.error(f"Error analyzing transitive flows: {e}")
            return []
    
    def _find_transitive_closure(self, start_func: str, dependencies: Dict[str, Set[str]],
                               max_depth: int = 10) -> Set[str]:
        """
        Find all functions transitively reachable from the start function.
        
        Args:
            start_func: Starting function name
            dependencies: Dictionary mapping functions to their direct dependencies
            max_depth: Maximum recursion depth to prevent infinite loops
            
        Returns:
            Set of all reachable function names
        """
        if max_depth <= 0:
            return set()
        
        # Start with direct dependencies
        reachable = set()
        direct_deps = dependencies.get(start_func, set())
        reachable.update(direct_deps)
        
        # Recursively find dependencies of dependencies
        for dep in direct_deps:
            # Avoid infinite recursion by checking if we've already visited this function
            if dep not in reachable:
                # Find transitive dependencies with decremented depth
                transitive_deps = self._find_transitive_closure(dep, dependencies, max_depth-1)
                reachable.update(transitive_deps)
        
        return reachable
    
    async def _find_vulnerable_functions(self) -> List[Dict[str, Any]]:
        """
        Query the vector database for functions known to be vulnerable.
        
        Returns:
            List of vulnerable function information
        """
        try:
            # Query for documents containing vulnerable functions
            query_results = await self._vector_store.query_with_prompts(
                "vulnerable function implementation security flaw",
                top_k=50,
                include_patterns=True
            )
            
            vulnerable_functions = []
            
            # Process results to extract function names and vulnerabilities
            for result in query_results.get('results', []):
                metadata = result.get('metadata', {})
                
                # Skip if no function name
                if 'function_name' not in metadata and 'method_name' not in metadata:
                    # Try to extract function name from text
                    function_name = self._extract_function_name_from_text(result.get('text', ''))
                    if not function_name:
                        continue
                    metadata['function_name'] = function_name
                
                # Get function name from metadata
                function_name = metadata.get('function_name', metadata.get('method_name', ''))
                
                # Skip if empty function name
                if not function_name:
                    continue
                
                # Get vulnerability information
                vulnerability_type = metadata.get('vulnerability_type', 
                                                 metadata.get('threat_type', 'Unknown'))
                
                severity = metadata.get('severity', 'medium')
                
                # Add to list of vulnerable functions
                vulnerable_functions.append({
                    'function_name': function_name,
                    'vulnerability_type': vulnerability_type,
                    'severity': severity,
                    'document_id': result.get('id'),
                    'score': result.get('score', 0),
                    'description': metadata.get('description', f"{vulnerability_type} vulnerability")
                })
            
            return vulnerable_functions
        except Exception as e:
            logger.error(f"Error finding vulnerable functions: {e}")
            return []
    
    def _extract_function_name_from_text(self, text: str) -> Optional[str]:
        """
        Extract a function name from code text.
        
        Args:
            text: Code text to analyze
            
        Returns:
            Extracted function name or None
        """
        try:
            # Try various patterns to extract function names
            import re
            
            # Python function definition
            python_match = re.search(r'def\s+(\w+)\s*\(', text)
            if python_match:
                return python_match.group(1)
            
            # JavaScript/TypeScript function
            js_match = re.search(r'function\s+(\w+)\s*\(', text)
            if js_match:
                return js_match.group(1)
            
            # Java/C# method
            java_match = re.search(r'(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(', text)
            if java_match:
                return java_match.group(1)
            
            # If no matches, return None
            return None
        except Exception as e:
            logger.error(f"Error extracting function name: {e}")
            return None
