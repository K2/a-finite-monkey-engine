"""
Main web application for the Finite Monkey framework.

This module provides a FastAPI-based web interface for monitoring, configuration,
and debugging of the Finite Monkey framework.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from ..agents import WorkflowOrchestrator
from ..db.manager import TaskManager
from ..nodes_config import nodes_config
from .api.routes import router as api_router, get_orchestrator
from .ui.ipython_terminal import AsyncIPythonBridge


# Create FastAPI application
app = FastAPI(
    title="Finite Monkey Web Interface",
    description="Web interface for the Finite Monkey framework",
    version="0.1.0",
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templating
from . import TEMPLATE_DIR, STATIC_DIR

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include API routes
app.include_router(api_router)

# Terminal connections
terminals: Dict[str, AsyncIPythonBridge] = {}
terminal_connections: Dict[str, List[WebSocket]] = {}

async def get_or_create_terminal(terminal_id: str, orchestrator: WorkflowOrchestrator = None) -> AsyncIPythonBridge:
    """Get or create a terminal with the given ID."""
    global terminals
    
    if terminal_id not in terminals:
        # Create namespace with important components
        namespace = {}
        
        if orchestrator is None:
            orchestrator = await get_orchestrator()
        
        # Add components to namespace
        namespace.update({
            "orchestrator": orchestrator,
            "task_manager": orchestrator.task_manager,
            "researcher": orchestrator.researcher,
            "validator": orchestrator.validator,
            "documentor": orchestrator.documentor,
            "config": nodes_config(),
        })
        
        # Create terminal
        terminals[terminal_id] = AsyncIPythonBridge(namespace)
        
        # Start output polling
        await terminals[terminal_id].start_polling()
        
        # Set output callback to broadcast to all connections
        async def broadcast_output(output: str):
            if terminal_id in terminal_connections:
                for connection in terminal_connections[terminal_id]:
                    try:
                        await connection.send_text(output)
                    except Exception:
                        pass
        
        terminals[terminal_id].set_output_callback(broadcast_output)
    
    return terminals[terminal_id]

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint that returns the HTML for the web interface."""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        # Fallback to inline HTML if template not found
        print(f"Error loading template: {e}")
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Finite Monkey Engine</title>
                <link rel="stylesheet" href="/static/css/main.css">
            </head>
            <body>
                <h1>Finite Monkey Engine</h1>
                <p>Web interface for the Finite Monkey framework</p>
                <ul>
                    <li><a href="/config">Configuration</a></li>
                    <li><a href="/telemetry">Telemetry</a></li>
                    <li><a href="/terminal/main">IPython Terminal</a></li>
                </ul>
            </body>
        </html>
        """)

@app.get("/config", response_class=HTMLResponse)
async def config_page():
    """Config page that renders a form for updating configuration."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Configuration - Finite Monkey Engine</title>
            <link rel="stylesheet" href="/static/css/main.css">
            <script>
                async function loadConfig() {
                    const response = await fetch('/api/config');
                    const config = await response.json();
                    
                    const configContainer = document.getElementById('config-container');
                    configContainer.innerHTML = '';
                    
                    for (const [key, value] of Object.entries(config)) {
                        const group = document.createElement('div');
                        group.className = 'config-group';
                        
                        const label = document.createElement('label');
                        label.textContent = key;
                        
                        const input = document.createElement('input');
                        input.type = 'text';
                        input.name = key;
                        input.value = value;
                        
                        group.appendChild(label);
                        group.appendChild(input);
                        configContainer.appendChild(group);
                    }
                }
                
                async function saveConfig() {
                    const form = document.getElementById('config-form');
                    const formData = new FormData(form);
                    
                    const settings = {};
                    for (const [key, value] of formData.entries()) {
                        settings[key] = value;
                    }
                    
                    const response = await fetch('/api/config', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ settings })
                    });
                    
                    const result = await response.json();
                    alert('Configuration saved');
                }
                
                window.addEventListener('load', loadConfig);
            </script>
        </head>
        <body>
            <h1>Configuration</h1>
            <form id="config-form">
                <div id="config-container"></div>
                <button type="button" onclick="saveConfig()">Save</button>
            </form>
        </body>
    </html>
    """)

@app.get("/telemetry", response_class=HTMLResponse)
async def telemetry_page():
    """Telemetry page that displays metrics."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Telemetry - Finite Monkey Engine</title>
            <link rel="stylesheet" href="/static/css/main.css">
            <script>
                async function loadTelemetry() {
                    const response = await fetch('/api/telemetry');
                    const telemetry = await response.json();
                    
                    document.getElementById('tasks-created').textContent = telemetry.tasks_created;
                    document.getElementById('tasks-completed').textContent = telemetry.tasks_completed;
                    document.getElementById('tasks-failed').textContent = telemetry.tasks_failed;
                    
                    if (telemetry.audit_start_time) {
                        document.getElementById('audit-start-time').textContent = new Date(telemetry.audit_start_time).toLocaleString();
                    }
                    
                    if (telemetry.audit_end_time) {
                        document.getElementById('audit-end-time').textContent = new Date(telemetry.audit_end_time).toLocaleString();
                    }
                    
                    const tasksContainer = document.getElementById('tasks-container');
                    tasksContainer.innerHTML = '';
                    
                    for (const [taskId, taskData] of Object.entries(telemetry.active_tasks)) {
                        const taskElem = document.createElement('div');
                        taskElem.className = `task task-${taskData.status}`;
                        
                        taskElem.innerHTML = `
                            <div class="task-header">
                                <span class="task-id">${taskId}</span>
                                <span class="task-status">${taskData.status}</span>
                            </div>
                            <div class="task-details">
                                <div class="task-file">${taskData.file || ''}</div>
                                <div class="task-type">${taskData.type || ''}</div>
                                <div class="task-created">${taskData.created_at ? new Date(taskData.created_at).toLocaleString() : ''}</div>
                                ${taskData.completed_at ? `<div class="task-completed">Completed: ${new Date(taskData.completed_at).toLocaleString()}</div>` : ''}
                                ${taskData.failed_at ? `<div class="task-failed">Failed: ${new Date(taskData.failed_at).toLocaleString()}</div>` : ''}
                                ${taskData.error ? `<div class="task-error">Error: ${taskData.error}</div>` : ''}
                            </div>
                        `;
                        
                        tasksContainer.appendChild(taskElem);
                    }
                    
                    // Schedule next update
                    setTimeout(loadTelemetry, 2000);
                }
                
                window.addEventListener('load', loadTelemetry);
            </script>
        </head>
        <body>
            <h1>Telemetry</h1>
            <div class="metrics">
                <div class="metric">
                    <label>Tasks Created:</label>
                    <span id="tasks-created">0</span>
                </div>
                <div class="metric">
                    <label>Tasks Completed:</label>
                    <span id="tasks-completed">0</span>
                </div>
                <div class="metric">
                    <label>Tasks Failed:</label>
                    <span id="tasks-failed">0</span>
                </div>
                <div class="metric">
                    <label>Audit Start Time:</label>
                    <span id="audit-start-time">-</span>
                </div>
                <div class="metric">
                    <label>Audit End Time:</label>
                    <span id="audit-end-time">-</span>
                </div>
            </div>
            <h2>Tasks</h2>
            <div id="tasks-container"></div>
        </body>
    </html>
    """)

@app.get("/terminal/{terminal_id}", response_class=HTMLResponse)
async def terminal_page(terminal_id: str):
    """Terminal page that provides an IPython terminal."""
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Terminal - Finite Monkey Engine</title>
            <link rel="stylesheet" href="/static/css/main.css">
            <style>
                .terminal {
                    background-color: #000;
                    color: #fff;
                    font-family: monospace;
                    padding: 10px;
                    border-radius: 5px;
                    width: 100%;
                    height: 400px;
                    overflow-y: auto;
                    white-space: pre-wrap;
                }
                
                .terminal-input {
                    display: flex;
                    margin-top: 10px;
                }
                
                .terminal-input input {
                    flex: 1;
                    font-family: monospace;
                    padding: 5px;
                }
                
                .terminal-input button {
                    padding: 5px 10px;
                    margin-left: 5px;
                }
            </style>
            <script>
                let ws;
                
                function initWebSocket() {{
                    ws = new WebSocket(`${{window.location.protocol === 'https:' ? 'wss' : 'ws'}}://${{window.location.host}}/ws/terminal/{terminal_id}`);
                    
                    ws.onopen = function(e) {{
                        console.log('Connected to terminal');
                        document.getElementById('terminal-status').textContent = 'Connected';
                    }};
                    
                    ws.onmessage = function(e) {{
                        const terminal = document.getElementById('terminal');
                        terminal.textContent += e.data;
                        terminal.scrollTop = terminal.scrollHeight;
                    }};
                    
                    ws.onclose = function(e) {{
                        console.log('Disconnected from terminal');
                        document.getElementById('terminal-status').textContent = 'Disconnected';
                        
                        // Try to reconnect
                        setTimeout(initWebSocket, 1000);
                    }};
                    
                    ws.onerror = function(e) {{
                        console.error('WebSocket error:', e);
                    }};
                }}
                
                function sendCommand() {{
                    const input = document.getElementById('terminal-input');
                    const command = input.value;
                    
                    if (command) {{
                        ws.send(command);
                        input.value = '';
                    }}
                }}
                
                window.addEventListener('load', initWebSocket);
            </script>
        </head>
        <body>
            <h1>IPython Terminal</h1>
            <p>Status: <span id="terminal-status">Connecting...</span></p>
            <p>
                This terminal provides access to the following objects:
                <ul>
                    <li><code>orchestrator</code> - The WorkflowOrchestrator instance</li>
                    <li><code>task_manager</code> - The TaskManager instance</li>
                    <li><code>researcher</code> - The Researcher agent</li>
                    <li><code>validator</code> - The Validator agent</li>
                    <li><code>documentor</code> - The Documentor agent</li>
                    <li><code>config</code> - The configuration object</li>
                </ul>
            </p>
            <div class="terminal" id="terminal"></div>
            <div class="terminal-input">
                <input type="text" id="terminal-input" placeholder="Enter command...">
                <button onclick="sendCommand()">Send</button>
            </div>
        </body>
    </html>
    """)

@app.websocket("/ws/terminal/{terminal_id}")
async def terminal_websocket(websocket: WebSocket, terminal_id: str):
    """WebSocket handler for IPython terminal."""
    await websocket.accept()
    
    # Get terminal
    terminal = await get_or_create_terminal(terminal_id)
    
    # Add to connections
    if terminal_id not in terminal_connections:
        terminal_connections[terminal_id] = []
    
    terminal_connections[terminal_id].append(websocket)
    
    try:
        # Send welcome message
        await websocket.send_text("Welcome to the Finite Monkey IPython Terminal!\n")
        await websocket.send_text("Type Python commands to interact with the framework.\n\n")
        
        # Process incoming messages
        while True:
            data = await websocket.receive_text()
            result = await terminal.run_code(data)
            
            # If no output, send a newline
            if not result:
                await websocket.send_text("\n")
    except WebSocketDisconnect:
        # Remove from connections
        if terminal_id in terminal_connections:
            if websocket in terminal_connections[terminal_id]:
                terminal_connections[terminal_id].remove(websocket)
    except Exception as e:
        # Log error
        print(f"Error in terminal websocket: {e}")
        
        # Try to send error message
        try:
            await websocket.send_text(f"Error: {str(e)}\n")
        except:
            pass

# Application lifecycle
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    # Create static and template directories if they don't exist
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    
    # Create CSS directory
    css_dir = STATIC_DIR / "css"
    os.makedirs(css_dir, exist_ok=True)
    
    # Create basic CSS file if it doesn't exist
    css_file = css_dir / "main.css"
    if not css_file.exists():
        with open(css_file, "w") as f:
            f.write("""
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            
            h1, h2, h3 {
                color: #333;
            }
            
            .config-group {
                margin-bottom: 10px;
            }
            
            .config-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            
            .config-group input {
                width: 100%;
                padding: 5px;
                box-sizing: border-box;
            }
            
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .metric {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
            }
            
            .metric label {
                display: block;
                font-weight: bold;
                margin-bottom: 5px;
            }
            
            .task {
                margin-bottom: 10px;
                padding: 10px;
                border-radius: 5px;
            }
            
            .task-pending {
                background-color: #f5f5f5;
            }
            
            .task-running {
                background-color: #e6f7ff;
            }
            
            .task-completed {
                background-color: #f6ffed;
            }
            
            .task-failed {
                background-color: #fff2f0;
            }
            
            .task-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            }
            
            .task-id {
                font-family: monospace;
                font-weight: bold;
            }
            
            .task-status {
                font-weight: bold;
            }
            
            .task-details {
                font-size: 0.9em;
            }
            
            .task-error {
                color: #f5222d;
                margin-top: 5px;
            }
            """)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    # Stop all terminals
    for terminal_id, terminal in terminals.items():
        await terminal.stop_polling()


def run_server():
    """
    Function to run the web server as an entry point.
    This is used when the package is installed and the command-line script is invoked.
    """
    import os
    import sys
    import argparse
    import uvicorn
    from pathlib import Path
    
    from finite_monkey.nodes_config import nodes_config
    
    # Load configuration
    config = nodes_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Finite Monkey - Web Interface",
    )
    
    parser.add_argument(
        "--host",
        default=config.WEB_HOST or "0.0.0.0",
        help=f"Host to bind to (default: {config.WEB_HOST or '0.0.0.0'})",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=config.WEB_PORT or 8000,
        help=f"Port to bind to (default: {config.WEB_PORT or 8000})",
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    parser.add_argument(
        "--enable-ipython",
        action="store_true",
        help="Enable IPython terminal functionality",
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print("Finite Monkey Engine - Web Interface")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    print(f"IPython: {'Enabled' if args.enable_ipython else 'Disabled'}")
    print("=" * 60)
    
    # Set up database directory
    db_dir = Path(config.DB_DIR)
    db_dir.mkdir(exist_ok=True)
    
    # Set up outputs directory
    output_dir = Path(config.output)
    output_dir.mkdir(exist_ok=True)
    
    # Run with uvicorn
    uvicorn.run(
        "finite_monkey.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload or args.debug,
        log_level="debug" if args.debug else "info",
    )
    
    return 0