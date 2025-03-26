#!/usr/bin/env python3
"""
Simple test for the Finite Monkey agent system

This script tests the manager/worker agent architecture with
a single model to validate implementation.
"""

import os
import asyncio
import logging
from datetime import datetime

from finite_monkey.agents import ManagerAgent
from finite_monkey.adapters import Ollama
from finite_monkey.models import AgentMetrics, ToolUsageMetrics, WorkflowMetrics
from finite_monkey.visualization import AgentGraphRenderer

# Basic test tool function
async def echo_tool(text):
    """Simple echo tool for testing"""
    return f"ECHO: {text}"

async def main():
    """Main test function"""
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Initialize components with a simple model
    model_name = "llama3:8b-instruct-q6_K"
    print(f"Initializing test with model: {model_name}")
    
    # Create Ollama client
    ollama = Ollama(model=model_name)
    
    # Create manager
    manager = ManagerAgent(llm_client=ollama, model_name=model_name)
    
    # Initialize agents
    print("Initializing worker agents...")
    await manager.initialize_worker_agents()
    
    # Register a test tool
    print("Registering test tool...")
    await manager.register_tool(
        "echo", 
        echo_tool, 
        "A simple echo tool that returns the input text"
    )
    
    # Generate workflow graph
    print("Generating workflow graph...")
    graph_data = await manager.generate_workflow_graph()
    
    # Create test metrics
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
            ).to_dict()
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
    
    # Render visualization
    print("Rendering workflow visualization...")
    renderer = AgentGraphRenderer(output_dir="reports")
    try:
        output_path = renderer.render_workflow_graph(graph_data, metrics)
        print(f"Generated workflow visualization at: {output_path}")
    except Exception as e:
        print(f"Error rendering workflow graph: {str(e)}")
    
    print("Test completed successfully!")
    
if __name__ == "__main__":
    asyncio.run(main())