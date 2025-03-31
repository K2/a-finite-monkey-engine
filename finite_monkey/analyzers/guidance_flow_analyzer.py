"""
Flow analyzer that combines tree-sitter with Guidance for structured analysis.
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json
from loguru import logger
from pydantic import BaseModel, Field

from ..adapters.guidance_adapter import GuidanceAdapter, GUIDANCE_AVAILABLE
from ..adapters.treesitter_adapter import TreeSitterAdapter, TREESITTER_AVAILABLE
from ..utils.guidance_program import GuidancePydanticProgram
from ..models.code_flow import CodeFlowGraph, FlowNode, FlowEdge, FlowNodeType, CodeLocation
from ..nodes_config import config


class FlowAnalysisResult(BaseModel):
    """Result of guidance-based flow analysis"""
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source nodes identified in the code"
    )
    sinks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sink nodes identified in the code"
    )
    flows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Data flows between sources and sinks"
    )
    vulnerabilities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Potential vulnerabilities identified"
    )


class GuidanceFlowAnalyzer:
    """
    Flow analyzer that combines tree-sitter with Guidance for structured analysis.
    
    This analyzer uses tree-sitter to extract code structure and guidance to
    analyze flows between sources and sinks, identifying potential vulnerabilities.
    """
    
    def __init__(
        self,
        treesitter_adapter: Optional[TreeSitterAdapter] = None,
        guidance_adapter: Optional[GuidanceAdapter] = None,
        verbose: bool = False
    ):
        """
        Initialize the flow analyzer
        
        Args:
            treesitter_adapter: Optional custom TreeSitterAdapter
            guidance_adapter: Optional custom GuidanceAdapter
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        
        # Initialize tree-sitter adapter
        self.treesitter_available = TREESITTER_AVAILABLE
        self.treesitter_adapter = treesitter_adapter or TreeSitterAdapter()
        
        # Initialize guidance adapter if available
        self.guidance_available = GUIDANCE_AVAILABLE
        if self.guidance_available:
            try:
                self.guidance_adapter = guidance_adapter or GuidanceAdapter(
                    model=getattr(config, "ANALYSIS_MODEL", config.DEFAULT_MODEL),
                    provider=getattr(config, "ANALYSIS_MODEL_PROVIDER", config.DEFAULT_PROVIDER),
                    temperature=0.1
                )
                logger.info("Initialized guidance-based flow analyzer")
            except Exception as e:
                logger.error(f"Failed to initialize guidance adapter: {e}")
                self.guidance_available = False
    
    async def analyze_function(
        self,
        contract_code: str,
        function_name: str,
        file_path: str = "",
        custom_sources: List[str] = None,
        custom_sinks: List[str] = None
    ) -> CodeFlowGraph:
        """
        Analyze the flows in a specific function
        
        Args:
            contract_code: Full contract code
            function_name: Name of function to analyze
            file_path: Path to the file
            custom_sources: Custom source patterns
            custom_sinks: Custom sink patterns
            
        Returns:
            Flow graph for the function
        """
        if not self.treesitter_available:
            logger.warning("tree-sitter not available for function analysis")
            return CodeFlowGraph(function_name=function_name)
        
        try:
            # Parse the contract
            tree = self.treesitter_adapter.parse_code(contract_code)
            if not tree:
                return CodeFlowGraph(function_name=function_name)
            
            # Extract functions
            functions = self.treesitter_adapter.extract_functions(tree)
            
            # Find the target function
            target_function = None
            for func_info in functions:
                if func_info.get('name') == function_name:
                    target_function = func_info
                    break
            
            if not target_function:
                logger.warning(f"Function {function_name} not found in contract")
                return CodeFlowGraph(function_name=function_name)
            
            # Get function body
            body = target_function.get('body')
            if not body:
                logger.warning(f"No body found for function {function_name}")
                return CodeFlowGraph(function_name=function_name)
            
            # Analyze sources and sinks
            flow_graph = self.treesitter_adapter.identify_sources_and_sinks(
                tree, custom_sources, custom_sinks
            )
            flow_graph.function_name = function_name
            
            # If guidance is available, enhance the analysis
            if self.guidance_available and self.guidance_adapter:
                enhanced_graph = await self._enhance_flow_graph_with_guidance(
                    contract_code, function_name, flow_graph
                )
                return enhanced_graph
            
            return flow_graph
            
        except Exception as e:
            logger.error(f"Error analyzing function {function_name}: {e}")
            return CodeFlowGraph(function_name=function_name)
    
    async def _enhance_flow_graph_with_guidance(
        self,
        contract_code: str,
        function_name: str,
        initial_graph: CodeFlowGraph
    ) -> CodeFlowGraph:
        """
        Enhance the flow graph with guidance-based analysis
        
        Args:
            contract_code: Full contract code
            function_name: Name of function being analyzed
            initial_graph: Initial flow graph from tree-sitter
            
        Returns:
            Enhanced flow graph
        """
        # Extract function code
        function_code = self._extract_function_code(contract_code, function_name)
        if not function_code:
            return initial_graph
        
        # Create guidance program for flow analysis
        program = GuidancePydanticProgram(
            output_cls=FlowAnalysisResult,
            prompt_template_str=self._create_flow_analysis_prompt(
                function_code, initial_graph
            ),
            guidance_adapter=self.guidance_adapter,
            verbose=self.verbose
        )
        
        # Generate the analysis
        try:
            result = await program(
                function_name=function_name,
                function_code=function_code,
                initial_sources=[node.dict() for node in initial_graph.get_sources()],
                initial_sinks=[node.dict() for node in initial_graph.get_sinks()]
            )
            
            # Enhance the graph with guidance results
            if isinstance(result, FlowAnalysisResult):
                # Add sources identified by guidance
                for source in result.sources:
                    node = FlowNode(
                        id=f"source_{len(initial_graph.nodes)}",
                        name=source.get('name', 'unknown'),
                        node_type=FlowNodeType.SOURCE,
                        metadata={'guidance_identified': True}
                    )
                    initial_graph.add_node(node)
                
                # Add sinks identified by guidance
                for sink in result.sinks:
                    node = FlowNode(
                        id=f"sink_{len(initial_graph.nodes)}",
                        name=sink.get('name', 'unknown'),
                        node_type=FlowNodeType.SINK,
                        metadata={'guidance_identified': True}
                    )
                    initial_graph.add_node(node)
                
                # Add flows between sources and sinks
                for flow in result.flows:
                    source_id = flow.get('source')
                    sink_id = flow.get('sink')
                    
                    # Find or create source node
                    source_node = None
                    for node in initial_graph.nodes:
                        if node.name == source_id and node.node_type == FlowNodeType.SOURCE:
                            source_node = node
                            break
                    
                    if not source_node:
                        source_node = FlowNode(
                            id=f"source_{len(initial_graph.nodes)}",
                            name=source_id,
                            node_type=FlowNodeType.SOURCE,
                            metadata={'guidance_identified': True}
                        )
                        initial_graph.add_node(source_node)
                    
                    # Find or create sink node
                    sink_node = None
                    for node in initial_graph.nodes:
                        if node.name == sink_id and node.node_type == FlowNodeType.SINK:
                            sink_node = node
                            break
                    
                    if not sink_node:
                        sink_node = FlowNode(
                            id=f"sink_{len(initial_graph.nodes)}",
                            name=sink_id,
                            node_type=FlowNodeType.SINK,
                            metadata={'guidance_identified': True}
                        )
                        initial_graph.add_node(sink_node)
                    
                    # Create edge between source and sink
                    edge = FlowEdge(
                        source_id=source_node.id,
                        target_id=sink_node.id,
                        edge_type="data_flow",
                        label=flow.get('description', ''),
                        metadata={'guidance_identified': True}
                    )
                    initial_graph.add_edge(edge)
                
                # Add vulnerabilities as metadata
                initial_graph.metadata['vulnerabilities'] = result.vulnerabilities
            
        except Exception as e:
            logger.error(f"Error enhancing flow graph with guidance: {e}")
        
        return initial_graph
    
    def _extract_function_code(self, contract_code: str, function_name: str) -> str:
        """
        Extract the code for a specific function
        
        Args:
            contract_code: Full contract code
            function_name: Name of function to extract
            
        Returns:
            Function code or empty string if not found
        """
        if not self.treesitter_available:
            return ""
        
        try:
            tree = self.treesitter_adapter.parse_code(contract_code)
            if not tree:
                return ""
            
            # Extract functions
            functions = self.treesitter_adapter.extract_functions(tree)
            
            # Find the target function
            for func_info in functions:
                if func_info.get('name') == function_name:
                    return func_info.get('location').code_snippet or ""
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting function code: {e}")
            return ""
    
    def _create_flow_analysis_prompt(
        self,
        function_code: str,
        initial_graph: CodeFlowGraph
    ) -> str:
        """
        Create a prompt for flow analysis with explicit JSON output instructions
        
        Args:
            function_code: Code of the function to analyze
            initial_graph: Initial flow graph from tree-sitter
            
        Returns:
            Handlebars-style prompt for guidance
        """
        return """
You are an expert smart contract security analyzer. Analyze this Solidity function 
to identify data flows between sources and sinks, and potential vulnerabilities.

Function Name: {{function_name}}

Function Code:
```solidity
{{function_code}}
```

Initial Sources Identified by Static Analysis:
{{#each initial_sources}}
- {{this.name}}
{{/each}}

Initial Sinks Identified by Static Analysis:
{{#each initial_sinks}}
- {{this.name}}
{{/each}}

Enhance this analysis by:
1. Identifying additional sources (where data/values come from)
2. Identifying additional sinks (where data/values flow to)
3. Describing data flows between sources and sinks
4. Identifying potential vulnerabilities

{{#schema}}
{
  "sources": [
    {
      "name": "source_name",
      "description": "Description of the source"
    }
  ],
  "sinks": [
    {
      "name": "sink_name",
      "description": "Description of the sink"
    }
  ],
  "flows": [
    {
      "source": "source_name",
      "sink": "sink_name",
      "description": "Description of how data flows from source to sink"
    }
  ],
  "vulnerabilities": [
    {
      "name": "vulnerability_name",
      "severity": "high/medium/low",
      "description": "Description of the vulnerability",
      "related_flow": "source_name -> sink_name"
    }
  ]
}
{{/schema}}
        """


# Create a test function to demonstrate usage
async def test_guidance_flow_analyzer():
    """Test the guidance flow analyzer"""
    logger.info("Testing guidance flow analyzer")
    
    # Create analyzer
    analyzer = GuidanceFlowAnalyzer(verbose=True)
    
    # Sample contract
    contract_code = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;
    
    contract Vulnerable {
        address owner;
        
        constructor() {
            owner = msg.sender;
        }
        
        function withdraw(uint amount) public {
            require(msg.sender == owner, "Not owner");
            payable(msg.sender).transfer(amount);
        }
        
        function deposit() public payable {
            // Just accept the ETH
        }
    }
    """
    
    # Analyze the withdraw function
    flow_graph = await analyzer.analyze_function(
        contract_code,
        "withdraw",
        custom_sources=["msg.sender", "owner"],
        custom_sinks=["transfer"]
    )
    
    # Print results
    logger.info(f"Flow graph for withdraw function:")
    logger.info(f"Sources: {[n.name for n in flow_graph.get_sources()]}")
    logger.info(f"Sinks: {[n.name for n in flow_graph.get_sinks()]}")
    logger.info(f"Edges: {len(flow_graph.edges)}")
    
    if 'vulnerabilities' in flow_graph.metadata:
        logger.info("Vulnerabilities:")
        for vuln in flow_graph.metadata['vulnerabilities']:
            logger.info(f"- {vuln.get('name')}: {vuln.get('severity')} - {vuln.get('description')}")
    
    return flow_graph
