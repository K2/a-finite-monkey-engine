"""
Comprehensive test script for the source/sink analysis components.

This script demonstrates how to use the TreeSitterAdapter and GuidanceFlowAnalyzer
to analyze smart contracts for data flows and potential vulnerabilities.
"""
import os
import sys
import asyncio
from pathlib import Path
from loguru import logger
import json

# Add root to path so we can import modules
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from finite_monkey.adapters.treesitter_adapter import TreeSitterAdapter, TREESITTER_AVAILABLE
from finite_monkey.analyzers.guidance_flow_analyzer import GuidanceFlowAnalyzer
from finite_monkey.models.code_flow import CodeFlowGraph, FlowNodeType


# Sample contracts for testing
VULNERABLE_REENTRANCY = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable to reentrancy
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
    
    function getBalance() public view returns (uint) {
        return balances[msg.sender];
    }
}
"""

ACCESS_CONTROL = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AccessControl {
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    function transferOwnership(address newOwner) public {
        // Missing access control
        owner = newOwner;
    }
    
    function withdrawFunds() public {
        require(msg.sender == owner, "Not owner");
        payable(msg.sender).transfer(address(this).balance);
    }
}
"""


async def test_treesitter_adapter():
    """Test the TreeSitterAdapter functionality"""
    if not TREESITTER_AVAILABLE:
        logger.warning("TreeSitter not available, skipping test")
        return None
        
    logger.info("Testing TreeSitterAdapter")
    
    # Create adapter
    adapter = TreeSitterAdapter()
    
    # Parse the vulnerable contract
    tree = adapter.parse_code(VULNERABLE_REENTRANCY)
    if not tree:
        logger.error("Failed to parse contract")
        return None
    
    # Extract functions
    functions = adapter.extract_functions(tree)
    logger.info(f"Found {len(functions)} functions")
    for func in functions:
        logger.info(f"- {func.get('name')}")
    
    # Test source/sink identification with default patterns
    flow_graph = adapter.identify_sources_and_sinks(tree)
    
    # Print results
    logger.info(f"Default source/sink analysis:")
    logger.info(f"- Sources: {len(flow_graph.get_sources())}")
    logger.info(f"- Sinks: {len(flow_graph.get_sinks())}")
    
    # Test with custom patterns
    custom_flow_graph = adapter.identify_sources_and_sinks(
        tree,
        source_patterns=["msg.sender", "balances"],
        sink_patterns=["call", "transfer"]
    )
    
    # Print results
    logger.info(f"Custom source/sink analysis:")
    logger.info(f"- Sources: {len(custom_flow_graph.get_sources())}")
    for source in custom_flow_graph.get_sources():
        logger.info(f"  - {source.name}")
    logger.info(f"- Sinks: {len(custom_flow_graph.get_sinks())}")
    for sink in custom_flow_graph.get_sinks():
        logger.info(f"  - {sink.name}")
    
    return adapter


async def test_guidance_flow_analyzer():
    """Test the GuidanceFlowAnalyzer functionality"""
    logger.info("Testing GuidanceFlowAnalyzer")
    
    # Create analyzer
    analyzer = GuidanceFlowAnalyzer(verbose=True)
    
    # Test on the withdraw function in the reentrancy example
    reentrancy_flow = await analyzer.analyze_function(
        VULNERABLE_REENTRANCY,
        "withdraw",
        file_path="VulnerableBank.sol",
        custom_sources=["msg.sender", "balances"],
        custom_sinks=["call", "transfer"]
    )
    
    # Print results
    logger.info(f"Reentrancy analysis results:")
    logger.info(f"- Sources: {len(reentrancy_flow.get_sources())}")
    for source in reentrancy_flow.get_sources():
        logger.info(f"  - {source.name}")
    logger.info(f"- Sinks: {len(reentrancy_flow.get_sinks())}")
    for sink in reentrancy_flow.get_sinks():
        logger.info(f"  - {sink.name}")
    logger.info(f"- Edges: {len(reentrancy_flow.edges)}")
    
    # Check for vulnerabilities
    vulnerabilities = reentrancy_flow.metadata.get('vulnerabilities', [])
    logger.info(f"- Vulnerabilities: {len(vulnerabilities)}")
    for vuln in vulnerabilities:
        logger.info(f"  - {vuln.get('name')}: {vuln.get('severity')} - {vuln.get('description')}")
    
    # Test on access control example
    access_flow = await analyzer.analyze_function(
        ACCESS_CONTROL,
        "transferOwnership",
        file_path="AccessControl.sol"
    )
    
    # Print results
    logger.info(f"Access Control analysis results:")
    logger.info(f"- Sources: {len(access_flow.get_sources())}")
    logger.info(f"- Sinks: {len(access_flow.get_sinks())}")
    logger.info(f"- Vulnerabilities: {len(access_flow.metadata.get('vulnerabilities', []))}")
    
    return {
        "reentrancy": reentrancy_flow,
        "access_control": access_flow
    }


def visualize_flow_graph(flow_graph: CodeFlowGraph, output_file: str = None) -> str:
    """
    Visualize a flow graph (simplified version)
    
    Args:
        flow_graph: The flow graph to visualize
        output_file: Optional file to save the visualization
        
    Returns:
        Text representation of the flow graph
    """
    lines = []
    lines.append(f"Flow Graph: {flow_graph.function_name or 'Unknown'}")
    lines.append(f"Contract: {flow_graph.contract_name or 'Unknown'}")
    lines.append("")
    
    # Add sources
    lines.append("Sources:")
    for source in flow_graph.get_sources():
        lines.append(f"- {source.name} ({source.id})")
    lines.append("")
    
    # Add sinks
    lines.append("Sinks:")
    for sink in flow_graph.get_sinks():
        lines.append(f"- {sink.name} ({sink.id})")
    lines.append("")
    
    # Add flows
    lines.append("Flows:")
    for edge in flow_graph.edges:
        # Find source and target nodes
        source_node = next((n for n in flow_graph.nodes if n.id == edge.source_id), None)
        target_node = next((n for n in flow_graph.nodes if n.id == edge.target_id), None)
        
        if source_node and target_node:
            lines.append(f"- {source_node.name} -> {target_node.name}: {edge.label or 'data flow'}")
    lines.append("")
    
    # Add vulnerabilities
    vulnerabilities = flow_graph.metadata.get('vulnerabilities', [])
    if vulnerabilities:
        lines.append("Vulnerabilities:")
        for vuln in vulnerabilities:
            lines.append(f"- {vuln.get('name')} ({vuln.get('severity')}): {vuln.get('description')}")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
    
    return '\n'.join(lines)


async def run_all_tests():
    """Run all tests"""
    logger.info("Starting source/sink analysis tests")
    
    # Run TreeSitter tests
    adapter = await test_treesitter_adapter()
    
    # Run GuidanceFlowAnalyzer tests
    flow_results = await test_guidance_flow_analyzer()
    
    # Visualize results
    if flow_results:
        for name, graph in flow_results.items():
            visualization = visualize_flow_graph(graph)
            logger.info(f"\nVisualization for {name}:\n{visualization}")
    
    logger.info("All tests completed")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run tests
    asyncio.run(run_all_tests())
