#!/usr/bin/env python3
"""
Test harness for the Finite Monkey agent system

This script tests the manager/worker agent architecture with
various LLM configurations to determine optimal model pairings.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from finite_monkey.agents import ManagerAgent, Researcher, Validator, Documentor
from finite_monkey.adapters import Ollama
from finite_monkey.models import AgentMetrics, ToolUsageMetrics, WorkflowMetrics
from finite_monkey.visualization import AgentGraphRenderer
from finite_monkey.nodes_config import nodes_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Test models to evaluate
OLLAMA_MODELS = [
    "qwen2.5:14b-instruct-q6_K",
    "llama3:8b-instruct-q6_K",
    "llama3:70b-instruct-q6_K",
    "yi:34b-chat-q6_K",
    "phi3:3.8b-instruct-q6_K",
    "mistral:7b-instruct-q6_K"
]

# Test cases for each agent type
RESEARCHER_TEST_CASES = [
    {
        "id": "reentrancy-detection",
        "code": """
contract Vulnerable {
    mapping(address => uint) public balances;
    
    function withdraw() public {
        uint amount = balances[msg.sender];
        require(amount > 0);
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
        
        balances[msg.sender] = 0;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}
        """,
        "query": "Check for reentrancy vulnerabilities",
        "expected": "reentrancy",
        "validation_type": "contains"
    },
    {
        "id": "overflow-detection",
        "code": """
contract TokenSale {
    mapping(address => uint) public balances;
    uint public totalSupply = 1000000;
    uint public price = 1 ether;
    
    function buyTokens(uint amount) public payable {
        require(msg.value == amount * price);
        
        // Vulnerable to overflow if amount is large enough
        uint total = amount * price;
        
        balances[msg.sender] += amount;
        totalSupply -= amount;
    }
}
        """,
        "query": "Check for integer overflow vulnerabilities",
        "expected": "overflow",
        "validation_type": "contains"
    }
]

VALIDATOR_TEST_CASES = [
    {
        "id": "validate-reentrancy",
        "code": """
contract Vulnerable {
    mapping(address => uint) public balances;
    
    function withdraw() public {
        uint amount = balances[msg.sender];
        require(amount > 0);
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
        
        balances[msg.sender] = 0;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}
        """,
        "analysis": {
            "findings": [
                {
                    "title": "Reentrancy Vulnerability",
                    "description": "The withdraw function is vulnerable to reentrancy attacks because it sends ETH before updating the balances state.",
                    "severity": "Critical",
                    "location": "withdraw function"
                }
            ]
        },
        "expected": {
            "validated": True,
            "severity": "Critical"
        },
        "validation_type": "contains"
    },
    {
        "id": "false-positive-detection",
        "code": """
contract Secure {
    mapping(address => uint) public balances;
    
    function withdraw() public {
        uint amount = balances[msg.sender];
        require(amount > 0);
        
        balances[msg.sender] = 0;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}
        """,
        "analysis": {
            "findings": [
                {
                    "title": "Reentrancy Vulnerability",
                    "description": "The withdraw function might be vulnerable to reentrancy attacks.",
                    "severity": "Critical",
                    "location": "withdraw function"
                }
            ]
        },
        "expected": {
            "false positive",
            "state is updated before external call"
        },
        "validation_type": "contains"
    }
]

async def test_ollama_availability():
    """Test if Ollama is available"""
    try:
        ollama = Ollama(model="llama3:8b-instruct-q6_K")
        response = await ollama.acomplete("Echo the word 'available' if you can read this.")
        
        if "available" in response.lower():
            logging.info("✅ Ollama API is available and responding correctly")
            return True
        else:
            logging.error("❌ Ollama API responded but with unexpected content")
            return False
    except Exception as e:
        logging.error(f"❌ Failed to connect to Ollama API: {str(e)}")
        return False

async def test_agent_with_model(agent_type, model_name, test_cases):
    """Test an agent with a specific model"""
    logging.info(f"Testing {agent_type} with model {model_name}")
    
    # Initialize ollama client
    ollama = Ollama(model=model_name)
    
    # Initialize the appropriate agent
    if agent_type == "researcher":
        agent = Researcher(llm_client=ollama, model_name=model_name)
    elif agent_type == "validator":
        agent = Validator(llm_client=ollama, model_name=model_name)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Create a manager agent to run the tests
    manager = ManagerAgent(llm_client=ollama, model_name=model_name)
    manager.worker_agents[agent_type] = agent
    
    # Run tests
    try:
        results = await manager.test_agent(agent_type, test_cases)
        
        # Log summary
        success_rate = results["successful"] / results["total_tests"] if results["total_tests"] > 0 else 0
        logging.info(f"Model {model_name} - Success rate: {success_rate*100:.1f}% ({results['successful']}/{results['total_tests']})")
        
        # Log details
        for detail in results["details"]:
            status = "✅" if detail["success"] else "❌"
            test_id = detail.get("id", "unknown")
            logging.info(f"  {status} Test {test_id}" + (f" - {detail.get('error', '')}" if not detail["success"] else ""))
        
        return {
            "model": model_name,
            "agent": agent_type,
            "success_rate": success_rate,
            "results": results
        }
    except Exception as e:
        logging.error(f"Error testing {agent_type} with {model_name}: {str(e)}")
        return {
            "model": model_name,
            "agent": agent_type,
            "success_rate": 0,
            "error": str(e)
        }

async def run_manager_visualization_test():
    """Test the manager's workflow graph visualization"""
    logging.info("Testing manager workflow graph visualization")
    
    # Initialize a manager with some worker agents
    ollama = Ollama(model="llama3:8b-instruct-q6_K")
    manager = ManagerAgent(llm_client=ollama)
    
    # Initialize worker agents
    await manager.initialize_worker_agents()
    
    # Set some agent states for visualization
    manager.agent_states["researcher"] = "running"
    manager.agent_states["validator"] = "idle"
    manager.agent_states["documentor"] = "failed"
    
    # Generate workflow graph
    graph_data = await manager.generate_workflow_graph()
    
    # Create some sample metrics
    metrics = {
        "agents": {
            "researcher": AgentMetrics(
                name="researcher",
                success_rate=0.85,
                avg_response_time=2.3,
                calls=20,
                last_called=datetime.now().isoformat()
            ).to_dict(),
            "validator": AgentMetrics(
                name="validator",
                success_rate=0.92,
                avg_response_time=1.8,
                calls=15,
                last_called=datetime.now().isoformat()
            ).to_dict(),
            "documentor": AgentMetrics(
                name="documentor",
                success_rate=0.75,
                avg_response_time=3.2,
                calls=10,
                last_called=datetime.now().isoformat()
            ).to_dict(),
        },
        "workflow_metrics": WorkflowMetrics(
            workflow_id="test-workflow",
            start_time=datetime.now().isoformat(),
            tasks_created=45,
            tasks_completed=30,
            tasks_failed=5,
            total_api_calls=120,
            total_tokens=28500
        ).to_dict()
    }
    
    # Render the visualization
    renderer = AgentGraphRenderer(output_dir="reports")
    output_path = renderer.render_workflow_graph(graph_data, metrics)
    logging.info(f"Generated workflow graph visualization at {output_path}")
    
    # Also render metrics dashboard
    metrics_path = renderer.render_metrics_dashboard(metrics)
    logging.info(f"Generated metrics dashboard at {metrics_path}")
    
    return {
        "graph_path": output_path,
        "metrics_path": metrics_path
    }

async def run_all_tests():
    """Run all tests and compile results"""
    logging.info("Starting agent system tests")
    
    # Test Ollama availability
    if not await test_ollama_availability():
        logging.error("Cannot continue tests without Ollama")
        return False
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Test each model with researcher agent
    researcher_results = []
    for model in OLLAMA_MODELS:
        result = await test_agent_with_model("researcher", model, RESEARCHER_TEST_CASES)
        researcher_results.append(result)
    
    # Test each model with validator agent
    validator_results = []
    for model in OLLAMA_MODELS:
        result = await test_agent_with_model("validator", model, VALIDATOR_TEST_CASES)
        validator_results.append(result)
    
    # Test visualization
    visualization_results = await run_manager_visualization_test()
    
    # Compile and save results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "researcher_tests": researcher_results,
        "validator_tests": validator_results,
        "visualization_test": visualization_results
    }
    
    # Save results
    results_file = f"reports/agent_system_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    logging.info("\n==== TEST SUMMARY ====")
    
    # Researcher models summary
    researcher_results.sort(key=lambda x: x.get("success_rate", 0), reverse=True)
    logging.info("\nResearcher Agent Models (best to worst):")
    for i, result in enumerate(researcher_results):
        logging.info(f"{i+1}. {result['model']} - {result.get('success_rate', 0)*100:.1f}% success rate")
    
    # Validator models summary
    validator_results.sort(key=lambda x: x.get("success_rate", 0), reverse=True)
    logging.info("\nValidator Agent Models (best to worst):")
    for i, result in enumerate(validator_results):
        logging.info(f"{i+1}. {result['model']} - {result.get('success_rate', 0)*100:.1f}% success rate")
    
    logging.info(f"\nDetailed results saved to: {results_file}")
    return all_results

if __name__ == "__main__":
    asyncio.run(run_all_tests())