# Web Interface for Finite Monkey Engine

This document provides an overview of the web interfaces for the Finite Monkey Engine.

## Overview

The Finite Monkey Engine provides two web interfaces:

1. **FastAPI Interface** (original): A feature-rich interface using FastAPI, WebSockets, and IPython
2. **FastHTML Interface** (new): A modernized, responsive interface with improved UI/UX

Both interfaces provide user-friendly ways to:

1. Monitor ongoing audits with real-time telemetry
2. Configure the engine through a web UI
3. Debug and explore component state using an embedded IPython console
4. Access the API for programmatic control
5. View and analyze security reports
6. Visualize results with interactive charts

## Architecture

### FastAPI Interface

The original web interface is built with:

- FastAPI for the API and web server
- WebSockets for real-time communication
- IPython for interactive debugging
- HTML/CSS/JavaScript for the frontend

### FastHTML Interface

The new web interface is built with:

- FastHTML framework (based on Starlette/Uvicorn)
- HTMX for dynamic UI updates without complex JavaScript
- WebSockets for real-time communication
- SQLAlchemy for data persistence
- Markdown rendering for reports
- Matplotlib/Seaborn for visualizations

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

### FastAPI Interface

To start the original FastAPI web interface:

```bash
./run_web.sh
# Or with options
./run_web_interface.py --host 0.0.0.0 --port 8000
```

### FastHTML Interface

To start the new FastHTML web interface:

```bash
./run_fasthtml_web.sh
```

This script will:
- Activate or create a virtual environment
- Install required dependencies
- Start the server with hot reload enabled

For either interface, then open a browser and navigate to:

```
http://localhost:8000/
```

### FastHTML Interface Features

The FastHTML interface provides these key features:

1. **Dashboard**: Overview of projects, metrics, and recent activities
2. **Interactive Terminal**: Real-time IPython terminal with syntax highlighting
3. **Code Editor**: Integrated code editor with syntax highlighting and terminal integration
4. **Reports Viewer**: Browse, search, and view security audit reports with markdown rendering
5. **Visualizations**: Interactive charts for security analysis insights

#### Code Editor Section
The code editor provides:
- ACE editor with syntax highlighting and autocompletion
- Multiple language support including Python, JavaScript, and Solidity
- Direct integration with the IPython terminal
- Ability to run code with a single click or keyboard shortcut (Ctrl+Enter)
- Real-time output display in the terminal
- Support for saving code snippets

#### Reports Section
The reports section allows you to:
- Browse all security reports
- Search and filter by project name or report type
- View detailed reports with proper markdown rendering
- See code snippets with syntax highlighting
- View vulnerability details with severity indicators
- Access associated graph visualizations

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

## FastHTML Improvements

The FastHTML interface implements several improvements compared to the original FastAPI interface:

1. **Modern UI/UX**: A cleaner, more responsive interface with dark theme
2. **Simplified Code**: More maintainable code structure using FastHTML components
3. **Enhanced WebSockets**: Improved WebSocket handling for terminal communication
4. **Reports Rendering**: Proper markdown rendering for security reports
5. **Interactive Visualizations**: Charts and graphs powered by Matplotlib/Seaborn
6. **Database Integration**: SQLAlchemy models for terminal sessions and commands
7. **Scope Inspector**: Real-time object inspection for debugging
8. **Mobile Support**: Responsive design that works on all device sizes

## Extension Opportunities

The web interfaces can be extended in several ways:

1. **Dashboard Improvements**: Add more charts and visualizations for telemetry
2. **User Authentication**: Add user authentication and authorization
3. **Real-time Notifications**: Add websocket-based notifications for task events
4. **Project Management**: Add project management features
5. **Code Editor**: Integrate a code editor for direct contract editing
6. **Integrated File Browser**: Add a file browser for selecting files to audit
7. **Configuration Editor**: Add a dedicated configuration editor
8. **Agent Telemetry**: Enhanced monitoring of agent activity and performance
9. **PDF Export**: Export reports to PDF format

## Security Considerations

When deploying the web interfaces:

1. **Authentication**: Consider adding authentication for production environments
2. **HTTPS**: Use HTTPS for production deployments
3. **Network Isolation**: Consider network isolation for production deployments
4. **Input Validation**: All inputs are validated, but additional validation may be needed
5. **Rate Limiting**: Consider adding rate limiting for production deployments
6. **WebSocket Security**: Ensure WebSocket connections are properly validated
7. **Code Execution Safety**: Restrict terminal access in production environments