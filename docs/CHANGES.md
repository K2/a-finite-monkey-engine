# Finite Monkey Engine - Recent Changes

## Multi-LLM Support

The framework now supports configuring different LLMs for different agent types:

- **Researcher Agent**: Configurable using `SCAN_MODEL` setting
- **Validator Agent**: Configurable using `CONFIRMATION_MODEL` setting (typically Claude)
- **Documentor Agent**: Configurable using `RELATION_MODEL` setting

For development purposes, all agents currently use the same Ollama instance with different models, but the infrastructure for using different API providers is in place.

## Async TaskManager

Added a new asynchronous task management system:

- Background processing of long-running tasks
- Concurrent execution with controlled parallelism
- Automatic retry for failed tasks
- Task status monitoring and dependency tracking
- Persistence with SQLAlchemy for reliability

## Web Interface

Added a comprehensive web interface for monitoring and interacting with the framework:

- **Telemetry Dashboard**: Real-time monitoring of tasks and workflow state
- **Configuration Management**: Web-based configuration editing
- **IPython Console**: Interactive debugging and exploration of agents

## Improvements and Bug Fixes

- Added more detailed telemetry tracking in WorkflowOrchestrator
- Enhanced configuration management with pydantic-settings
- Improved error handling in the async workflow
- Added test examples and developer tools

## Test Assets

Added a comprehensive DeFi project for testing:

- Token.sol: ERC20-like token with various vulnerabilities
- Vault.sol: Token vault with security issues
- StakingPool.sol: Staking pool with reward calculation flaws

## Documentation

- Added detailed documentation for the async workflow
- Added web interface documentation with usage examples
- Created ASCII diagrams explaining component relationships

## TODO

- Add Claude-specific adapter for validation agent
- Implement more sophisticated task prioritization
- Add support for distributed execution across nodes
- Enhance telemetry with visualization
- Add user authentication for production deployments