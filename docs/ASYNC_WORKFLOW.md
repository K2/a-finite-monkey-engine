# Asynchronous Workflow Architecture

This document describes the architecture and implementation of the asynchronous workflow system in the Finite Monkey Engine.

## Overview

The asynchronous workflow system is designed to enable:

1. Parallel execution of tasks across multiple files
2. Background processing of long-running tasks
3. Task dependency management
4. Automatic retry mechanisms
5. Persistence of task state

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                     WorkflowOrchestrator                        │
│                                                                 │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐               │
│   │           │    │           │    │           │               │
│   │ Researcher│    │ Validator │    │ Documentor│               │
│   │           │    │           │    │           │               │
│   └───────────┘    └───────────┘    └───────────┘               │
│          │               │               │                      │
└──────────┼───────────────┼───────────────┼──────────────────────┘
           │               │               │
           ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                        TaskManager                              │
│                                                                 │
│   ┌───────────┐    ┌────────────┐     ┌───────────┐             │
│   │           │    │            │     │           │             │
│   │ Task Queue│───▶│ Semaphore │───▶│ Worker    │             │
│   │           │    │            │     │ Loop      │             │
│   └───────────┘    └────────────┘     └───────────┘             │
│                                          │                      │
└──────────────────────────────────────────┼──────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                      DatabaseManager                            │
│                                                                 │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐               │
│   │           │    │           │    │           │               │
│   │ Project   │    │ File      │    │ Audit     │               │
│   │           │    │           │    │           │               │
│   └───────────┘    └───────────┘    └───────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### DatabaseManager

The `DatabaseManager` provides an asynchronous interface to the database using SQLAlchemy with asyncpg as the database driver. It handles:

- Project, file, and audit tracking
- Persistence of analysis results
- Query functionality for retrieving results

### TaskManager

The `TaskManager` extends the `DatabaseManager` with task management capabilities:

- Task queuing with controlled concurrency
- Task state management
- Automatic retries for failed tasks
- Task status querying
- Dependency chaining between tasks

### WorkflowOrchestrator

The `WorkflowOrchestrator` coordinates the atomic agents (Researcher, Validator, Documentor) using the task management infrastructure:

- Creates and initializes agent instances
- Manages the workflow execution
- Handles both synchronous and asynchronous execution models
- Aggregates results across multiple files

## Task Execution Flow

1. **Task Creation**: `orchestrator.run_audit_workflow()` creates tasks for each file
2. **Task Queuing**: Tasks are added to the queue via `task_manager.add_task()`
3. **Task Processing**: Worker loop picks up tasks and executes them within concurrency limits
4. **Task Chaining**: Each task can spawn dependent tasks upon completion
5. **Result Aggregation**: Results from all tasks are combined into a final report

## Dependency Resolution

Tasks are naturally ordered through explicit dependencies:

1. Analysis task → Validation task → Report task

When a task completes, it automatically schedules the next task in the sequence.

## Error Handling

The system includes several error handling mechanisms:

1. **Automatic retries**: Failed tasks are retried up to a configured limit
2. **State persistence**: Task state is persisted to survive application restarts
3. **Timeout handling**: Tasks can be configured with timeouts
4. **Error reporting**: Detailed error information is captured and stored

## Extension Points

The architecture includes several extension points:

1. **Custom agents**: Additional specialized agents can be added for specific analysis types
2. **Priority queuing**: Task prioritization can be implemented
3. **Distributed execution**: The system can be extended for multi-node execution
4. **Real-time monitoring**: Status reporting can be enhanced with real-time updates
5. **Tool integration**: External security tools can be integrated into the workflow

## Configuration

The system is configured through the `nodes_config` system, which provides a unified configuration interface from various sources (environment variables, config files, command line arguments).

Key configuration values:

- `MAX_THREADS_OF_SCAN`: Controls maximum concurrent tasks
- `ASYNC_DB_URL`: Database connection string
- `WORKFLOW_MODEL`: Default model to use for analysis

## Usage Examples

### Basic Async Workflow

```python
# Initialize orchestrator
orchestrator = WorkflowOrchestrator()

# Run async workflow
task_ids = await orchestrator.run_audit_workflow(
    solidity_paths=["contract.sol"],
    query="Perform a security audit",
    wait_for_completion=False
)

# Get task status
for file_path, tasks in task_ids.items():
    status = await orchestrator.task_manager.get_task_status(tasks["analysis"])
    print(f"Status: {status['status']}")
```

### Waiting for Results

```python
# Run async workflow and wait for completion
report = await orchestrator.run_audit_workflow(
    solidity_paths=["contract.sol"],
    query="Perform a security audit",
    wait_for_completion=True
)

# Access results
print(f"Findings: {len(report.findings)}")
```

## Future Improvements

1. **Task Cancellation**: Ability to cancel running tasks
2. **Progress Reporting**: Real-time progress updates
3. **Workflow Visualization**: Visual representation of task dependencies
4. **Task Prioritization**: Priority-based scheduling
5. **Resource Throttling**: Dynamic adjustment of concurrency based on load

## Conclusion

The asynchronous workflow system provides a robust foundation for parallel execution of security analysis tasks. It enables efficient processing of large codebases while maintaining state and handling errors gracefully.