"""
Manager Agent for the Finite Monkey framework

This module implements the Manager agent that coordinates specialized worker agents,
handles testing, monitoring, and optimization of agent performance.
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path

from ..adapters import Ollama
from ..models import AgentMetrics, ToolUsageMetrics
from ..nodes_config import nodes_config


class ManagerAgent:
    """
    Manager agent for coordinating worker agents
    
    This agent acts as a supervisor for specialized worker agents, handling:
    - Agent initialization and configuration
    - Performance monitoring and testing
    - Error detection and recovery
    - Tool usage analysis and optimization
    """
    
    def __init__(
        self,
        config=None,
        llm_client=None,
        model_name: str = "qwen2.5:14b-instruct-q6_K",
        agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the Manager agent
        
        Args:
            config: Configuration settings
            llm_client: LLM client for the manager's own reasoning
            model_name: Model to use for the manager's reasoning
            agent_configs: Configuration for worker agents
        """
        # Load configuration
        self.config = config or nodes_config()
        
        # Set up LLM client
        self.llm_client = llm_client or Ollama(model=model_name)
        self.model_name = model_name
        
        # Initialize agent configurations
        self.agent_configs = agent_configs or {}
        
        # Initialize metrics
        self.metrics = {
            "agents": {},
            "tools": {},
            "performance": {
                "response_times": [],
                "success_rate": 1.0,
                "error_counts": {},
            }
        }
        
        # Worker agent instances
        self.worker_agents = {}
        
        # Tool registry
        self.available_tools = {}
        
        # Track agent state
        self.state = "initialized"
        self.agent_states = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize_worker_agents(self):
        """
        Initialize all worker agents based on configuration
        """
        # Import all agent types
        from .researcher import Researcher
        from .validator import Validator
        from .documentor import Documentor
        
        # Initialize standard workers
        if "researcher" not in self.worker_agents:
            self.worker_agents["researcher"] = Researcher(
                llm_client=self.llm_client,
                model_name=self.agent_configs.get("researcher", {}).get("model_name", self.model_name),
            )
            self.agent_states["researcher"] = "initialized"
            
        if "validator" not in self.worker_agents:
            self.worker_agents["validator"] = Validator(
                llm_client=self.llm_client,
                model_name=self.agent_configs.get("validator", {}).get("model_name", "claude-3-5-sonnet"),
            )
            self.agent_states["validator"] = "initialized"
            
        if "documentor" not in self.worker_agents:
            self.worker_agents["documentor"] = Documentor(
                llm_client=self.llm_client,
                model_name=self.agent_configs.get("documentor", {}).get("model_name", self.model_name),
            )
            self.agent_states["documentor"] = "initialized"
    
    async def register_tool(self, tool_name: str, tool_function: Any, description: str):
        """
        Register a tool for use by worker agents
        
        Args:
            tool_name: Name of the tool
            tool_function: Function or callable object that implements the tool
            description: Description of what the tool does
        """
        self.available_tools[tool_name] = {
            "function": tool_function,
            "description": description,
            "metrics": ToolUsageMetrics(
                name=tool_name,
                calls=0,
                success=0,
                failures=0,
                avg_latency=0,
                last_used=None,
            )
        }
    
    async def test_agent(self, agent_name: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test a worker agent against known test cases
        
        Args:
            agent_name: Name of the agent to test
            test_cases: List of test cases with inputs and expected outputs
            
        Returns:
            Test results with success rate and error details
        """
        if agent_name not in self.worker_agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.worker_agents[agent_name]
        results = {
            "agent": agent_name,
            "total_tests": len(test_cases),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        for i, test_case in enumerate(test_cases):
            test_id = test_case.get("id", f"test-{i+1}")
            
            try:
                # Execute the test
                start_time = datetime.now()
                
                if agent_name == "researcher":
                    result = await agent.analyze_code_async(
                        query=test_case["query"],
                        code_snippet=test_case["code"],
                    )
                elif agent_name == "validator":
                    result = await agent.validate_analysis(
                        code=test_case["code"],
                        analysis=test_case["analysis"],
                    )
                elif agent_name == "documentor":
                    result = await agent.generate_report_async(
                        analysis=test_case["analysis"],
                        validation=test_case["validation"],
                        project_name=test_case.get("project_name", "test-project"),
                    )
                else:
                    # Generic call for custom agents
                    method_name = test_case.get("method", "process")
                    method = getattr(agent, method_name)
                    result = await method(**test_case["inputs"])
                
                elapsed = (datetime.now() - start_time).total_seconds()
                
                # Validate result against expected output
                success = await self._validate_test_result(
                    result=result,
                    expected=test_case["expected"],
                    validation_type=test_case.get("validation_type", "contains"),
                )
                
                # Record result
                detail = {
                    "id": test_id,
                    "success": success,
                    "elapsed": elapsed,
                }
                
                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    detail["error"] = "Result did not match expected output"
                    
                results["details"].append(detail)
                
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "id": test_id,
                    "success": False,
                    "error": str(e),
                })
        
        # Update agent metrics
        self.metrics["agents"][agent_name] = AgentMetrics(
            name=agent_name,
            success_rate=results["successful"] / results["total_tests"] if results["total_tests"] > 0 else 0,
            avg_response_time=sum(d["elapsed"] for d in results["details"] if "elapsed" in d) / len(results["details"]),
            calls=len(test_cases),
            last_called=datetime.now().isoformat(),
        )
        
        return results
    
    async def _validate_test_result(self, result: Any, expected: Any, validation_type: str) -> bool:
        """
        Validate test result against expected output
        
        Args:
            result: Actual result from agent
            expected: Expected output
            validation_type: Type of validation ("contains", "exact", "semantic")
            
        Returns:
            True if validation succeeds
        """
        if validation_type == "exact":
            return result == expected
        elif validation_type == "contains":
            # For string containment
            if isinstance(result, str):
                if isinstance(expected, str):
                    return expected.lower() in result.lower()
                elif isinstance(expected, set):
                    # Check if any of the set items are contained in the result
                    return any(item.lower() in result.lower() for item in expected if isinstance(item, str))
            # For dict/list containment
            elif isinstance(result, dict) and isinstance(expected, dict):
                return all(k in result and result[k] == v for k, v in expected.items())
            elif isinstance(result, list) and isinstance(expected, list):
                return all(item in result for item in expected)
            else:
                return False
        elif validation_type == "semantic":
            # Use the LLM to check semantic equivalence
            prompt = f"""
            Compare these two outputs and determine if they are semantically equivalent:
            
            Output 1:
            ```
            {result}
            ```
            
            Output 2 (Expected):
            ```
            {expected}
            ```
            
            Are these outputs semantically equivalent? Answer with just "Yes" or "No".
            """
            
            response = await self.llm_client.acomplete(
                prompt=prompt,
                model=self.model_name,
            )
            
            return response.strip().lower().startswith("yes")
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")
    
    async def monitor_agent_performance(self, agent_name: str) -> AgentMetrics:
        """
        Monitor performance of a specific agent
        
        Args:
            agent_name: Name of the agent to monitor
            
        Returns:
            Agent performance metrics
        """
        if agent_name not in self.metrics["agents"]:
            return AgentMetrics(name=agent_name, success_rate=0, avg_response_time=0, calls=0, last_called=None)
        
        return self.metrics["agents"][agent_name]
    
    async def analyze_tool_usage(self) -> Dict[str, ToolUsageMetrics]:
        """
        Analyze tool usage patterns across all agents
        
        Returns:
            Tool usage metrics
        """
        return {name: tool["metrics"] for name, tool in self.available_tools.items()}
    
    async def create_agent_workspace(self, agent_name: str) -> str:
        """
        Create an isolated workspace for an agent using uv
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Path to the workspace
        """
        workspace_dir = os.path.join(self.config.base_dir, "workspaces", agent_name)
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Create virtual environment with uv
        venv_dir = os.path.join(workspace_dir, "venv")
        if not os.path.exists(venv_dir):
            proc = await asyncio.create_subprocess_shell(
                f"uv venv {venv_dir}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to create workspace: {stderr.decode()}")
            
            # Install dependencies
            await asyncio.create_subprocess_shell(
                f"{venv_dir}/bin/uv pip install -e .",
                cwd=self.config.base_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        
        return workspace_dir
    
    async def generate_workflow_graph(self) -> Dict[str, Any]:
        """
        Generate a graph representation of the current workflow
        
        Returns:
            Graph data structure with nodes and edges
        """
        graph = {
            "nodes": [],
            "edges": [],
        }
        
        # Add nodes for each agent
        for agent_name, agent in self.worker_agents.items():
            node = {
                "id": agent_name,
                "type": "agent",
                "state": self.agent_states.get(agent_name, "unknown"),
                "model": getattr(agent, "model_name", "unknown"),
            }
            graph["nodes"].append(node)
        
        # Add node for the manager
        graph["nodes"].append({
            "id": "manager",
            "type": "manager",
            "state": self.state,
            "model": self.model_name,
        })
        
        # Add edges based on workflow
        # Manager connects to all agents
        for agent_name in self.worker_agents:
            graph["edges"].append({
                "source": "manager",
                "target": agent_name,
                "type": "manages",
            })
        
        # Standard workflow connections
        if "researcher" in self.worker_agents and "validator" in self.worker_agents:
            graph["edges"].append({
                "source": "researcher",
                "target": "validator",
                "type": "sends_analysis",
            })
        
        if "validator" in self.worker_agents and "documentor" in self.worker_agents:
            graph["edges"].append({
                "source": "validator",
                "target": "documentor",
                "type": "sends_validation",
            })
        
        return graph