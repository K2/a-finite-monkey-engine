"""
Test the cognitive bias analyzer functionality
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finite_monkey.adapters.ollama import AsyncOllamaClient
from finite_monkey.agents.cognitive_bias_analyzer import CognitiveBiasAnalyzer
from finite_monkey.models.analysis import BiasAnalysisResult, VulnerabilityReport

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample contract with various cognitive bias vulnerabilities
SAMPLE_CONTRACT = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FinancialVault {
    address public owner;
    mapping(address => uint256) public balances;
    bool public paused;
    uint256 public totalDeposits;
    address public oracleAddress;
    uint256 public exchangeRate = 100; // 1 ETH = 100 tokens
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }
    
    // Normalcy bias: No check for extreme market conditions
    function setExchangeRate(uint256 newRate) external onlyOwner {
        // No validation on rate reasonability
        // No timelock for parameter changes
        exchangeRate = newRate;
    }
    
    // Authority bias: No validation on admin-supplied parameters
    function updateOracleAddress(address newOracle) external onlyOwner {
        // No checks if this is a valid oracle
        // No timelock or multi-sig requirement
        oracleAddress = newOracle;
    }
    
    // Confirmation bias: Only checks user balance, not contract balance
    function withdraw(uint256 amount) external {
        require(!paused, "Contract is paused");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Confirms user has enough balance
        // But doesn't check if contract has enough liquidity
        // Assumes if accounting is correct, execution will succeed
        balances[msg.sender] -= amount;
        (bool success,) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
    
    // Curse of knowledge: Doesn't enforce intended usage pattern
    function deposit() external payable {
        require(!paused, "Contract is paused");
        // Developer knows this is meant for long-term staking
        // But nothing prevents immediate withdrawal
        // Protocol assumes particular usage patterns
        balances[msg.sender] += msg.value * exchangeRate;
        totalDeposits += msg.value;
    }
    
    // Hyperbolic discounting: Saves gas by not checking each iteration
    function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner {
        // No limit on batch size (gas DoS risk)
        // No validation of recipient addresses
        // No check that arrays are same length
        for (uint i = 0; i < recipients.length; i++) {
            balances[recipients[i]] += amounts[i];
        }
    }
    
    // Emergency function with no delays
    function emergencyWithdraw() external onlyOwner {
        // No timelock
        // No multi-sig requirement for critical operation
        payable(owner).transfer(address(this).balance);
    }
    
    function togglePause() external onlyOwner {
        paused = !paused;
    }
}
"""

async def test_cognitive_bias_analyzer():
    """Test the cognitive bias analyzer on the sample contract"""
    logger.info("Initializing test for cognitive bias analyzer...")
    
    # Initialize Ollama client
    try:
        llm_client = AsyncOllamaClient(model_name="llama3")
        logger.info("LLM client initialized")
    except Exception as e:
        logger.error(f"Error initializing LLM client: {e}")
        # Use a mock client for testing if the real one fails
        from unittest.mock import AsyncMock
        llm_client = AsyncMock()
        llm_client.completion = AsyncMock(return_value="Mock LLM response for testing")
        logger.info("Using mock LLM client for testing")
    
    # Initialize the cognitive bias analyzer
    bias_analyzer = CognitiveBiasAnalyzer(llm_client=llm_client)
    logger.info("Cognitive bias analyzer initialized")
    
    # Basic test of bias analysis
    contract_name = "FinancialVault"
    logger.info(f"Analyzing cognitive biases in {contract_name}...")
    
    try:
        # Analyze just a single bias type for testing
        bias_type = "authority_bias"
        bias_info = bias_analyzer.bias_categories[bias_type]
        
        result = await bias_analyzer._analyze_specific_bias(
            contract_code=SAMPLE_CONTRACT,
            bias_type=bias_type,
            bias_info=bias_info,
            previous_analysis=None
        )
        
        logger.info(f"Analysis of {bias_type} completed")
        logger.info(f"Results: {result}")
        
        # Test assumption analysis
        logger.info("Testing assumption analysis...")
        
        # Create some sample vulnerability reports
        vulnerability_reports = [
            VulnerabilityReport(
                title="Privileged Function Without Timelock",
                description="The setExchangeRate function allows the owner to change a critical parameter without a timelock.",
                severity="High",
                location="setExchangeRate function",
                vulnerability_type="Access Control"
            ),
            VulnerabilityReport(
                title="Missing Liquidity Check",
                description="The withdraw function doesn't verify if the contract has enough ETH to fulfill the request.",
                severity="Critical",
                location="withdraw function",
                vulnerability_type="Funds At Risk"
            )
        ]
        
        assumption_results = await bias_analyzer.generate_assumption_analysis(
            contract_code=SAMPLE_CONTRACT,
            vulnerability_reports=vulnerability_reports
        )
        
        logger.info("Assumption analysis completed")
        logger.info(f"Results: {assumption_results}")
        
        logger.info("Testing full bias analysis workflow...")
        
        # Now test the full analysis
        full_result = await bias_analyzer.analyze_cognitive_biases(
            contract_code=SAMPLE_CONTRACT,
            contract_name=contract_name
        )
        
        logger.info("Full cognitive bias analysis completed")
        logger.info(f"Summary:\n{full_result.get_summary()}")
        
        # Test remediation plan generation
        logger.info("Testing remediation plan generation...")
        
        remediation_plan = await bias_analyzer.generate_remediation_plan(
            contract_code=SAMPLE_CONTRACT,
            bias_analysis=full_result
        )
        
        logger.info("Remediation plan generated")
        logger.info(f"Remediation plan contains fixes for {len(remediation_plan)} bias types")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_cognitive_bias_analyzer())