#!/usr/bin/env python3
"""
Test script for linguistic analysis and counterfactual generation

This script demonstrates the usage of DocumentationAnalyzer and CounterfactualGenerator
for finding inconsistencies between code comments and implementation, and generating
training scenarios for human operators.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from finite_monkey.adapters import Ollama
from finite_monkey.agents.documentation_analyzer import DocumentationAnalyzer
from finite_monkey.agents.counterfactual_generator import CounterfactualGenerator
from finite_monkey.visualization import AgentGraphRenderer


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Example vulnerable code with misleading comments
EXAMPLE_CODE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Safe Vault Contract
 * @dev A completely secure vault for storing funds safely
 * @notice Implements best practices for securing user funds
 */
contract Vault {
    mapping(address => uint256) private balances;
    address private owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @notice Deposit funds into the vault
     * @dev Safe from reentrancy attacks
     */
    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }
    
    /**
     * @notice Withdraw funds from the vault
     * @dev Protected against all attack vectors
     * @dev All security checks implemented
     */
    function withdraw() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "No funds to withdraw");
        
        // Send funds to user
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        // Update balance
        balances[msg.sender] = 0;
    }
    
    /**
     * @notice Check balance in the vault
     * @dev Only callable by the user or owner for privacy
     */
    function checkBalance(address user) external view returns (uint256) {
        // Anyone can check anyone's balance
        return balances[user];
    }
    
    /**
     * @notice Emergency function to fix potential issues
     * @dev Only owner can call this function
     */
    function emergencyWithdraw(address payable recipient) external {
        recipient.transfer(address(this).balance);
    }
}
"""

async def main():
    """Main function to run the analysis and generation"""
    # Create output directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    print("Initializing LLM client...")
    llm_model = "llama3:8b-instruct-q6_K"  # Using a smaller model for testing
    llm_client = Ollama(model=llm_model)
    
    # Initialize the DocumentationAnalyzer
    print("Analyzing code for documentation inconsistencies...")
    doc_analyzer = DocumentationAnalyzer(llm_client=llm_client, model_name=llm_model)
    
    # Analyze the code
    inconsistency_report = await doc_analyzer.analyze_code(EXAMPLE_CODE)
    
    # Add timestamp
    inconsistency_report.timestamp = datetime.now().isoformat()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Documentation Analysis Results")
    print("=" * 50)
    print(f"Analyzed {inconsistency_report.total_comments} comments")
    print(f"Found {len(inconsistency_report.inconsistencies)} inconsistencies")
    print("=" * 50)
    
    # Print some details of inconsistencies found
    for i, inc in enumerate(inconsistency_report.inconsistencies):
        print(f"\n{i+1}. {inc.get('inconsistency_type', 'Unknown')} - {inc.get('severity', 'Medium')}")
        print(f"   Comment: {inc.get('comment', {}).get('text', '')}")
        print(f"   Description: {inc.get('description', '')[:100]}...")
    
    # Generate heatmap
    print("\nGenerating linguistic heatmap...")
    inconsistencies = [
        DocumentationAnalyzer.DocumentationInconsistency(
            comment=DocumentationAnalyzer.CodeComment(
                text=inc.get("comment", {}).get("text", ""),
                line_number=inc.get("comment", {}).get("line_number", 0),
                context_before=inc.get("comment", {}).get("context_before", []),
                context_after=inc.get("comment", {}).get("context_after", []),
                comment_type=inc.get("comment", {}).get("comment_type", "inline")
            ),
            code_snippet=inc.get("code_snippet", ""),
            inconsistency_type=inc.get("inconsistency_type", ""),
            description=inc.get("description", ""),
            severity=inc.get("severity", "medium"),
            confidence=inc.get("confidence", 0.5)
        )
        for inc in inconsistency_report.inconsistencies
    ]
    
    heatmap = await doc_analyzer.generate_linguistic_heatmap(EXAMPLE_CODE, inconsistencies)
    
    # Save report and heatmap
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"reports/inconsistency_report_{timestamp}.json", "w") as f:
        json.dump(inconsistency_report.dict(), f, indent=2)
    
    with open(f"reports/heatmap_{timestamp}.json", "w") as f:
        # Convert to serializable format
        serializable_heatmap = {
            "code_lines": heatmap["code_lines"],
            "heat_levels": heatmap["heat_levels"],
            "annotations": {str(k): v for k, v in heatmap["annotations"].items()}
        }
        json.dump(serializable_heatmap, f, indent=2)
    
    print(f"Saved report to reports/inconsistency_report_{timestamp}.json")
    print(f"Saved heatmap to reports/heatmap_{timestamp}.json")
    
    # Initialize CounterfactualGenerator
    if inconsistency_report.inconsistencies:
        print("\nGenerating counterfactual scenarios for training...")
        counterfactual_gen = CounterfactualGenerator(llm_client=llm_client, model_name=llm_model)
        
        # Take the first inconsistency for demonstration
        finding = inconsistency_report.inconsistencies[0]
        
        # Generate counterfactuals
        counterfactuals = await counterfactual_gen.generate_counterfactuals(
            finding=finding,
            code_snippet=finding.get("code_snippet", EXAMPLE_CODE),
            vulnerability_type=finding.get("inconsistency_type", "security"),
            num_scenarios=2  # Limit to 2 for testing
        )
        
        # Print some details of counterfactuals
        print("\n" + "=" * 50)
        print("Counterfactual Scenarios")
        print("=" * 50)
        
        for i, scenario in enumerate(counterfactuals):
            print(f"\n{i+1}. {scenario.get('title', 'Scenario')}")
            print(f"   Learning Objective: {scenario.get('learning_objective', '')[:100]}...")
        
        # Generate exploitation path
        exploitation = await counterfactual_gen.generate_exploitation_path(
            finding=finding,
            code_snippet=finding.get("code_snippet", EXAMPLE_CODE),
            detailed=True
        )
        
        print("\n" + "=" * 50)
        print("Exploitation Path")
        print("=" * 50)
        print(f"Difficulty: {exploitation.get('difficulty_rating', 'Unknown')}/5")
        print(f"Impact: {exploitation.get('impact', 'Unknown')}")
        
        # Save counterfactuals and exploitation path
        with open(f"reports/counterfactuals_{timestamp}.json", "w") as f:
            json.dump(counterfactuals, f, indent=2)
        
        with open(f"reports/exploitation_{timestamp}.json", "w") as f:
            json.dump(exploitation, f, indent=2)
        
        print(f"Saved counterfactuals to reports/counterfactuals_{timestamp}.json")
        print(f"Saved exploitation path to reports/exploitation_{timestamp}.json")
    
    print("\nAnalysis complete. See reports/ directory for details.")

if __name__ == "__main__":
    asyncio.run(main())