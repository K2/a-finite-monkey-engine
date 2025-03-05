# Web Interface for Finite Monkey Engine

This document provides an overview of the web interface for the Finite Monkey Engine.

## Overview

The Finite Monkey Engine web interface provides a user-friendly way to:

1. Monitor ongoing audits with real-time telemetry
2. Configure the engine through a web UI
3. Debug and explore component state using an embedded IPython console
4. Access the API for programmatic control

## Architecture

The web interface is built with:

- FastAPI for the API and web server
- WebSockets for real-time communication
- IPython for interactive debugging
- HTML/CSS/JavaScript for the frontend

```
┌───────────────────────────────────────────────────────┐
│                                                       │
│                      Web Browser                      │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │             │  │             │  │             │   │
│  │  Config UI  │  │  Telemetry  │  │  IPython    │   │
│  │             │  │             │  │  Console    │   │
│  └─────────────┘  └─────────────┘  └─────────────┘   │
│          │               │                │          │
└──────────┼───────────────┼────────────────┼──────────┘
           │               │                │
┌──────────┼───────────────┼────────────────┼──────────┐
│          ▼               ▼                ▼          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │             │  │             │  │             │   │
│  │  Config API │  │  Telemetry  │  │  WebSocket  │   │
│  │             │  │  API        │  │  Server     │   │
│  └─────────────┘  └─────────────┘  └─────────────┘   │
│          │               │                │          │
│          │               │                │          │
│  ┌───────┴───────────────┴────────────────┴────────┐ │
│  │                                                  │ │
│  │             FastAPI Application                  │ │
│  │                                                  │ │
│  └───────────────────────┬──────────────────────────┘ │
│                          │                            │
└──────────────────────────┼────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────┐
│                          ▼                            │
│  ┌────────────────────────────────────────────────┐   │
│  │                                                │   │
│  │             WorkflowOrchestrator               │   │
│  │                                                │   │
│  │   ┌───────────┐   ┌───────────┐   ┌──────────┐ │   │
│  │   │           │   │           │   │          │ │   │
│  │   │ Researcher│   │ Validator │   │Documentor│ │   │
│  │   │           │   │           │   │          │ │   │
│  │   └───────────┘   └───────────┘   └──────────┘ │   │
│  │                                                │   │
│  └────────────────────────┬───────────────────────┘   │
│                           │                           │
│                           ▼                           │
│  ┌────────────────────────────────────────────────┐   │
│  │                                                │   │
│  │               TaskManager                      │   │
│  │                                                │   │
│  └────────────────────────┬───────────────────────┘   │
│                           │                           │
│                           ▼                           │
│  ┌────────────────────────────────────────────────┐   │
│  │                                                │   │
│  │               Database                         │   │
│  │                                                │   │
│  └────────────────────────────────────────────────┘   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

## Components

### Web Server

The web server is implemented using FastAPI and provides:

1. HTML pages for the user interface
2. REST API endpoints for CRUD operations
3. WebSocket endpoints for real-time communication with the IPython console

### Telemetry System

The telemetry system tracks:

1. Task creation, completion, and failure metrics
2. Workflow state and execution time
3. Agent-specific metrics and resource usage

### Configuration Interface

The configuration interface allows:

1. Viewing all available configuration options
2. Modifying configuration values through a web form
3. Configuration value validation and error reporting

### IPython Console

The embedded IPython console provides:

1. Direct access to framework components
2. Ability to run ad-hoc analysis and debugging
3. Live state inspection and modification

## API Routes

The web interface exposes the following API routes:

### Configuration

- `GET /api/config` - Get current configuration
- `POST /api/config` - Update configuration

### Audit Management

- `POST /api/audit` - Start a new audit
- `GET /api/tasks` - Get all task statuses
- `GET /api/tasks/{task_id}` - Get status of a specific task

### Telemetry

- `GET /api/telemetry` - Get telemetry data

## WebSocket Endpoints

The web interface exposes the following WebSocket endpoints:

- `/ws/terminal/{terminal_id}` - IPython terminal WebSocket

## Usage

To start the web interface:

```bash
./run_web_interface.py --host 0.0.0.0 --port 8000
```

Then open a browser and navigate to:

```
http://localhost:8000/
```

## IPython Terminal Usage

The IPython terminal provides access to the following objects:

- `orchestrator` - The WorkflowOrchestrator instance
- `task_manager` - The TaskManager instance
- `researcher` - The Researcher agent
- `validator` - The Validator agent
- `documentor` - The Documentor agent
- `config` - The configuration object

Example commands:

```python
# Check task status
tasks = await task_manager.get_task_status('some-task-id')

# Run a simple analysis
analysis = await researcher.analyze_code_async(
    query="Check for reentrancy",
    code_snippet="// Some code here"
)

# Get configuration
print(config.WORKFLOW_MODEL)
```

## Extension Opportunities

The web interface can be extended in several ways:

1. **Dashboard Improvements**: Add charts and visualizations for telemetry
2. **User Authentication**: Add user authentication and authorization
3. **Real-time Notifications**: Add websocket-based notifications for task events
4. **Project Management**: Add project management features
5. **Result Viewing**: Add a dedicated view for audit results and findings
6. **Integrated File Browser**: Add a file browser for selecting files to audit

## Security Considerations

When deploying the web interface:

1. **Authentication**: Consider adding authentication for production environments
2. **HTTPS**: Use HTTPS for production deployments
3. **Network Isolation**: Consider network isolation for production deployments
4. **Input Validation**: All inputs are validated, but additional validation may be needed
5. **Rate Limiting**: Consider adding rate limiting for production deployments