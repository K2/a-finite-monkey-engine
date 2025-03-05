# Finite Monkey Engine - Agent Architecture

This document outlines the agent architecture used in the Finite Monkey Engine, explaining how the system is organized, how agents interact, and how to extend the system with new agent types.

## Overview

Finite Monkey uses a hierarchical multi-agent architecture with the following components:

1. **Manager Agent**: Coordinates worker agents, handles testing, monitoring, and optimization
2. **Worker Agents**: Specialized agents for specific tasks (researcher, validator, documentor)
3. **Tool Registry**: Shared utilities that agents can access to complete tasks
4. **Visualization System**: Represents agent interactions and workflow state graphically

The system supports both synchronous and asynchronous execution models, with built-in observability and telemetry.

## Manager-Worker Architecture

```
┌───────────────────────┐
│                       │
│    Manager Agent      │
│                       │
└─────────┬─────────────┘
          │
          │ manages
          ▼
┌─────────┬─────────────┬────────────┐
│         │             │            │
│Researcher│  Validator  │ Documentor │
│ Agent   │   Agent     │  Agent     │
│         │             │            │
└─────────┴─────────────┴────────────┘
     │           │            │
     │           │            │
     └───────────┼────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  Tool Registry│
         └───────────────┘
```

### Manager Agent

The `ManagerAgent` is responsible for:

1. **Initialization**: Creating and configuring worker agents
2. **Coordination**: Managing workflow between agents
3. **Monitoring**: Tracking agent performance and efficiency
4. **Testing**: Self-validation through test cases
5. **Resource Management**: Allocating computation resources
6. **Visualization**: Generating workflow representations

The manager extends beyond simple orchestration by adding adaptive capabilities:

```python
# Example: Dynamic model selection based on task complexity
async def select_model_for_task(self, task_type, complexity):
    if complexity > 0.8:
        return "llama3:70b-instruct"  # Complex task
    elif complexity > 0.5:
        return "qwen2.5:14b-instruct"  # Moderate task
    else:
        return "llama3:8b-instruct"  # Simple task
```

### Worker Agents

Worker agents are specialized for specific tasks:

1. **Researcher Agent**: Analyzes code for vulnerabilities
2. **Validator Agent**: Verifies analysis results and identifies false positives
3. **Documentor Agent**: Creates comprehensive reports from findings

Each worker agent follows a common interface but has specialized behaviors:

```python
class BaseWorkerAgent:
    async def process(self, task):
        """Process a task and return results"""
        raise NotImplementedError
        
    async def handle_error(self, error):
        """Handle and recover from errors"""
        raise NotImplementedError
```

### Agent Communication

Agents communicate through structured messages that include:

1. **Message ID**: Unique identifier for the message
2. **Sender/Receiver**: Source and destination agents
3. **Message Type**: Type of message (task, result, error, control)
4. **Content**: The actual message content
5. **Metadata**: Additional information (timestamps, priorities, etc.)

Example message flow:

```
Manager → Researcher: "Analyze this contract for vulnerabilities"
Researcher → Manager: "Analysis complete, found 3 issues"
Manager → Validator: "Verify these 3 issues"
Validator → Manager: "2 issues confirmed, 1 is a false positive"
Manager → Documentor: "Create report with 2 confirmed issues"
Documentor → Manager: "Report completed successfully"
```

## Self-Testing Framework

The architecture includes a self-testing framework that allows agents to verify their capabilities:

1. **Test Cases**: Predefined tasks with expected outputs
2. **Test Runner**: Executes tests and compares results
3. **Validation Types**:
   - Exact Match: Result must exactly match expected output
   - Contains: Result must contain expected output
   - Semantic: Result must be semantically equivalent

Example test case:

```python
{
    "id": "reentrancy-detection",
    "code": "...",  # Vulnerable contract
    "query": "Check for reentrancy vulnerabilities",
    "expected": "reentrancy",
    "validation_type": "contains"
}
```

## Isolated Workspaces

To prevent conflicts and ensure clean execution environments, the system supports isolated workspaces using `uv`:

1. **Workspace Creation**: Each agent can have its own isolated environment
2. **Dependency Management**: Dependencies are installed per workspace
3. **Resource Isolation**: Prevents resource conflicts between agents

```python
# Example: Creating an isolated workspace
workspace_dir = await manager.create_agent_workspace("researcher")
```

## Tool Registry

The tool registry provides a collection of utilities that agents can use:

1. **Registration**: Tools are registered with the manager
2. **Discovery**: Agents can discover available tools
3. **Usage Tracking**: Tool usage is monitored for optimization
4. **Permission Control**: Access can be restricted per agent

```python
# Example: Registering a tool
await manager.register_tool(
    name="code_analyzer",
    function=analyze_code_patterns,
    description="Analyzes code patterns for security issues"
)
```

## Visualization System

The visualization system provides graphical representations of:

1. **Agent Workflow**: How agents interact and information flows
2. **Agent States**: Current status of each agent
3. **Metrics Dashboard**: Performance and telemetry data
4. **Tool Usage**: How agents utilize available tools

The system uses D3.js for interactive visualizations and supports real-time updates.

## LLM Optimization

The architecture is optimized for multi-LLM usage:

1. **Model Selection**: Dynamic model selection based on task requirements
2. **Batching**: Similar tasks are batched for efficiency
3. **Caching**: Common queries are cached to reduce API calls
4. **Fallbacks**: System can fall back to alternative models if primary is unavailable

## Adding New Agents

To add a new worker agent to the system:

1. Create a new class that inherits from `BaseWorkerAgent`
2. Implement the required interface methods
3. Register with the manager agent
4. Add test cases for the new agent
5. Update visualization to include the new agent

Example:

```python
class PatternMatcherAgent(BaseWorkerAgent):
    """Agent that specializes in pattern matching for vulnerabilities"""
    
    async def process(self, task):
        """Process a pattern matching task"""
        # Implementation
        
    async def handle_error(self, error):
        """Handle errors in pattern matching"""
        # Implementation

# Register with manager
await manager.register_worker_agent("pattern_matcher", PatternMatcherAgent())
```

## Human-in-the-Loop Integration

The architecture supports human interaction through:

1. **Intervention Points**: Defined points where human input is requested
2. **Feedback Loops**: Mechanisms for incorporating human feedback
3. **Override Capabilities**: Ways to override agent decisions
4. **Explanation Interfaces**: Methods for agents to explain reasoning

```python
# Example: Human validation request
if confidence < 0.8:
    human_input = await manager.request_human_input(
        question="Is this a valid vulnerability?",
        context={"finding": finding, "code": code},
        timeout=300  # Wait up to 5 minutes
    )
```

## Future Directions

The agent architecture is designed to evolve in several directions:

1. **Agent Specialization**: More specialized agents for specific vulnerability classes
2. **Learning Capabilities**: Agents that improve through experience
3. **Multi-Node Deployment**: Distributed execution across multiple machines
4. **Plugin System**: User-defined extensions to the framework
5. **Dynamic Agent Creation**: Creating agents on-demand based on needs

## Example Workflow

A typical workflow in the system would look like:

1. **Input**: User provides code and query through CLI or web interface
2. **Planning**: Manager agent determines which worker agents to use
3. **Research**: Researcher agent analyzes code for vulnerabilities
4. **Validation**: Validator agent verifies research findings
5. **Documentation**: Documentor agent creates comprehensive report
6. **Visualization**: System generates visual representation of workflow
7. **Output**: Final report and visualization delivered to user

Throughout this process, the manager monitors performance, handles errors, and optimizes resource usage.