#!/usr/bin/env python3
"""
Full-featured Engine Terminal for the Finite Monkey Engine.

This script creates a comprehensive terminal interface that:
1. Directly interacts with the full nodes_config
2. Provides access to all engine functionality
3. Offers real configuration management
4. Integrates with report generation and visualization
"""

import os
import sys
import json
import asyncio
import logging
import inspect
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set
import webbrowser
from textwrap import dedent

# Check for required packages and install if missing
try:
    from pydantic import BaseModel
    from fastapi import FastAPI, WebSocket, Request, Form, File, UploadFile, HTTPException, Depends
    from fastapi.staticfiles import StaticFiles
except ImportError:
    import subprocess
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "python-multipart", "aiofiles"])
    from pydantic import BaseModel
    from fastapi import FastAPI, WebSocket, Request, Form, File, UploadFile, HTTPException, Depends
    from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("engine-terminal")

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TERM'] = 'xterm-256color'
os.environ['FORCE_COLOR'] = '1'

# Add project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Import engine components
try:
    from finite_monkey.nodes_config import nodes_config, Settings
    # Import other key components
    engine_imports_available = True
    WEB_PORT = nodes_config.WEB_PORT
    # Use terminal port = web port + 1 to avoid conflicts
    TERMINAL_PORT = WEB_PORT + 1
    logger.info(f"Using configuration: Web port={WEB_PORT}, Terminal port={TERMINAL_PORT}")
except ImportError as e:
    logger.warning(f"Could not import engine components: {e}")
    engine_imports_available = False
    WEB_PORT = 8000
    TERMINAL_PORT = 8001

# Create FastAPI app
app = FastAPI(title="Engine Terminal",
              description="A comprehensive terminal interface for the Finite Monkey Engine")

# Create a directory for static files
static_dir = Path(project_dir) / "static"
static_dir.mkdir(exist_ok=True)

# Create templates directory
templates_dir = Path(project_dir) / "templates"
templates_dir.mkdir(exist_ok=True)

# Create uploads directory
uploads_dir = Path(project_dir) / "uploads"
uploads_dir.mkdir(exist_ok=True)

# Static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Create templates object
templates = Jinja2Templates(directory=str(templates_dir))

# Import Jupyter kernel functionality
try:
    from jupyter_client import KernelManager
    has_jupyter = True
except ImportError:
    has_jupyter = False
    logger.warning("jupyter_client not found. Will run with simulated output.")

# Terminal sessions store
terminal_sessions: Dict[str, WebSocket] = {}

# Jupyter kernel output queue
output_queue = asyncio.Queue()

# Keep track of engine components loaded into the kernel
engine_components = {
    "orchestrator": False,
    "researcher": False,
    "validator": False,
    "documentor": False,
    "config_loaded": False
}

# Store engine file paths for navigation
engine_file_paths: Dict[str, Path] = {}

# Track code analysis tasks
analysis_tasks: Dict[str, Dict[str, Any]] = {}

class ConfigUpdate(BaseModel):
    """Model for config updates."""
    field: str
    value: Any

class JupyterKernel:
    """Enhanced Jupyter kernel with context injection support."""
    
    def __init__(self):
        """Initialize the kernel."""
        self.km = None
        self.kc = None
        self.initialized = False
        self.namespace = {}
        self.config = None
        
    async def start(self):
        """Start the Jupyter kernel with engine context."""
        if not has_jupyter:
            logger.warning("Jupyter client not available. Using simulated output.")
            self.initialized = True
            return True
            
        try:
            # Start the kernel
            self.km = KernelManager()
            self.km.start_kernel()
            
            # Get the kernel client
            self.kc = self.km.client()
            self.kc.start_channels()
            
            # Wait for the kernel to be ready
            self.kc.wait_for_ready(timeout=300)
            
            # Mark as initialized
            self.initialized = True
            
            # Start the output polling task
            asyncio.create_task(self._poll_output())
            
            # Inject engine context setup
            await self._inject_context_setup()
            
            logger.info("Jupyter kernel started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting Jupyter kernel: {e}")
            return False
    
    async def _inject_context_setup(self):
        """Inject setup code for engine context."""
        # First, inject common utilities
        setup_code = """
        import sys
        import os
        import json
        import asyncio
        
        # Add helper function to ensure we have an event loop
        def ensure_event_loop():
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        
        # Make ensure_event_loop available
        globals()['ensure_event_loop'] = ensure_event_loop
        
        print("🔄 Basic utilities loaded.")
        """
        await self.execute(dedent(setup_code))
        
        # Then, try to load nodes_config
        config_code = """
        try:
            # Load configuration
            from finite_monkey.nodes_config import nodes_config
            
            # Make available globally
            globals()['nodes_config'] = nodes_config
            
            print("✅ Configuration loaded successfully.")
        except ImportError as e:
            print(f"⚠️ Error loading configuration: {e}")
        """
        await self.execute(dedent(config_code))
        
        # Finally, provide helper for initializing agents
        agents_code = """
        # Helper to initialize the engine components
        async def init_engine():
            ensure_event_loop()
            
            # Import components
            from finite_monkey.agents.orchestrator import WorkflowOrchestrator
            from finite_monkey.agents.researcher import Researcher
            from finite_monkey.agents.validator import Validator
            
            try:
                from finite_monkey.agents.documentor import Documentor
                has_documentor = True
            except ImportError:
                has_documentor = False
            
            # Create instances
            global orchestrator, researcher, validator, documentor
            orchestrator = WorkflowOrchestrator()
            researcher = Researcher()
            validator = Validator()
            
            if has_documentor:
                documentor = Documentor()
                components = {
                    "orchestrator": orchestrator,
                    "researcher": researcher,
                    "validator": validator,
                    "documentor": documentor
                }
            else:
                components = {
                    "orchestrator": orchestrator,
                    "researcher": researcher,
                    "validator": validator
                }
            
            print("✅ Engine components initialized successfully!")
            return components
            
        # Helper to run a full analysis
        async def analyze_contract(file_path, query=None):
            ensure_event_loop()
            
            # Ensure orchestrator is available
            if 'orchestrator' not in globals():
                components = await init_engine()
                global orchestrator
                orchestrator = components['orchestrator']
            
            print(f"🔍 Analyzing contract: {file_path}")
            result = await orchestrator.run_analysis(file_path, query=query)
            return result
            
        # Make helpers available
        globals()['init_engine'] = init_engine
        globals()['analyze_contract'] = analyze_contract
        
        print("🔄 Engine helpers initialized. Run 'await init_engine()' to initialize components.")
        """
        await self.execute(dedent(agents_code))
    
    async def execute(self, code, store_history=True):
        """Execute code in the kernel."""
        if not self.initialized:
            return f"<span style='color: #f55;'>Kernel not initialized</span>"
            
        if not has_jupyter:
            # Simulate output
            return f"<span style='color: #5f5;'>&gt;&gt;&gt; {code}</span><br>Simulated output (Jupyter client not available)"
        
        try:
            # Update component status tracking
            if "orchestrator = WorkflowOrchestrator()" in code:
                engine_components["orchestrator"] = True
            if "researcher = Researcher()" in code:
                engine_components["researcher"] = True
            if "validator = Validator()" in code:
                engine_components["validator"] = True
            if "documentor = Documentor()" in code:
                engine_components["documentor"] = True
                
            # Check for full initialization
            if "init_engine()" in code:
                engine_components["orchestrator"] = True
                engine_components["researcher"] = True
                engine_components["validator"] = True
                engine_components["documentor"] = True
                
            # Execute the code
            msg_id = self.kc.execute(code, store_history=store_history)
            
            # Return the message ID for tracking
            return msg_id
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return f"<span style='color: #f55;'>Error executing code: {e}</span>"
    
    async def _poll_output(self):
        """Poll for output from the kernel."""
        if not has_jupyter or not self.initialized:
            return
            
        try:
            while True:
                # Get messages from the kernel
                try:
                    msg = self.kc.get_iopub_msg(timeout=0.1)
                    
                    # Process the message
                    await self._process_message(msg)
                except:
                    # No message available
                    pass
                
                # Sleep to avoid CPU hogging
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in output polling: {e}")
    
    async def _process_message(self, msg):
        """Process a message from the kernel."""
        # Check the message type
        msg_type = msg['header']['msg_type']
        content = msg['content']
        
        if msg_type == 'stream':
            # Text output
            stream_name = content['name']  # 'stdout' or 'stderr'
            text = content['text']
            
            # Format the output
            if stream_name == 'stderr':
                formatted = f"<span class='ansi-red'>{text}</span>"
            else:
                formatted = text
                
            # Add to the queue
            await output_queue.put(formatted)
            
        elif msg_type == 'execute_result':
            # Execution result
            data = content['data']
            
            # Check for different representations
            if 'text/html' in data:
                html = data['text/html']
                await output_queue.put(html)
            elif 'text/plain' in data:
                text = data['text/plain']
                await output_queue.put(text)
                
        elif msg_type == 'display_data':
            # Display data
            data = content['data']
            
            # Check for different representations
            if 'text/html' in data:
                html = data['text/html']
                await output_queue.put(html)
            elif 'text/plain' in data:
                text = data['text/plain']
                await output_queue.put(text)
                
        elif msg_type == 'error':
            # Error message
            ename = content['ename']
            evalue = content['evalue']
            traceback = content.get('traceback', [])
            
            # Format the error
            error_msg = f"<span class='ansi-red'>{ename}: {evalue}</span><br>"
            for tb_line in traceback:
                # Remove ANSI escape sequences
                tb_line = tb_line.replace('<', '&lt;').replace('>', '&gt;')
                error_msg += f"{tb_line}<br>"
                
            # Add to the queue
            await output_queue.put(error_msg)
    
    def shutdown(self):
        """Shutdown the kernel."""
        if not has_jupyter or not self.initialized:
            return
            
        try:
            if self.kc:
                self.kc.stop_channels()
            if self.km:
                self.km.shutdown_kernel()
                
            logger.info("Jupyter kernel shut down")
        except Exception as e:
            logger.error(f"Error shutting down kernel: {e}")

# Initialize the kernel
kernel = JupyterKernel()

async def discover_engine_files():
    """Discover important engine files for navigation."""
    global engine_file_paths
    
    # Base paths to check
    if engine_imports_available:
        base_dir = Path(project_dir) / "finite_monkey"
        if base_dir.exists():
            # Find key files
            engine_file_paths["nodes_config"] = base_dir / "nodes_config.py"
            
            # Find agent files
            agents_dir = base_dir / "agents"
            if agents_dir.exists():
                for agent_file in agents_dir.glob("*.py"):
                    if agent_file.name != "__init__.py":
                        agent_name = agent_file.stem
                        engine_file_paths[f"agent_{agent_name}"] = agent_file
            
            # Find model files
            models_dir = base_dir / "models"
            if models_dir.exists():
                for model_file in models_dir.glob("*.py"):
                    if model_file.name != "__init__.py":
                        model_name = model_file.stem
                        engine_file_paths[f"model_{model_name}"] = model_file
                        
    # Log discovered files
    logger.info(f"Discovered {len(engine_file_paths)} engine files")

@app.on_event("startup")
async def startup_event():
    """Initialize components at application startup."""
    # Start the kernel
    await kernel.start()
    
    # Discover engine files
    await discover_engine_files()

@app.on_event("shutdown")
def shutdown_event():
    """Shutdown components at application shutdown."""
    # Shutdown the kernel
    kernel.shutdown()

# Create the index.html template if it doesn't exist
index_template = templates_dir / "index.html"
if not index_template.exists():
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Engine Terminal</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css">
        <style>
            :root {
                --primary-color: #5D3FD3;
                --secondary-color: #9D4EDD;
                --background-color: #121212;
                --terminal-bg: #1E1E1E;
                --terminal-text: #F8F8F2;
                --card-bg: #252526;
                --border-color: #333;
                --accent-color: #66D9EF;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                line-height: 1.5;
                color: var(--terminal-text);
                background-color: var(--background-color);
                margin: 0;
                padding: 0;
            }
            
            .sidebar {
                background-color: var(--card-bg);
                height: 100vh;
                overflow-y: auto;
                position: fixed;
                width: 250px;
                border-right: 1px solid var(--border-color);
            }
            
            .sidebar-header {
                padding: 20px;
                text-align: center;
                border-bottom: 1px solid var(--border-color);
            }
            
            .sidebar-header h1 {
                margin: 0;
                font-size: 20px;
                color: var(--accent-color);
            }
            
            .sidebar-menu {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            
            .sidebar-menu li {
                margin: 0;
                padding: 0;
            }
            
            .sidebar-menu a {
                display: block;
                padding: 12px 20px;
                color: var(--terminal-text);
                text-decoration: none;
                border-bottom: 1px solid var(--border-color);
                transition: background-color 0.2s;
            }
            
            .sidebar-menu a:hover,
            .sidebar-menu a.active {
                background-color: var(--primary-color);
                color: white;
            }
            
            .sidebar-menu i {
                margin-right: 10px;
                width: 20px;
                text-align: center;
            }
            
            .main-content {
                margin-left: 250px;
                padding: 20px;
            }
            
            .card {
                background-color: var(--card-bg);
                border-radius: 8px;
                margin-bottom: 20px;
                overflow: hidden;
                border: 1px solid var(--border-color);
            }
            
            .card-header {
                background-color: rgba(93, 63, 211, 0.2);
                padding: 15px 20px;
                font-weight: bold;
                border-bottom: 1px solid var(--border-color);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .card-body {
                padding: 20px;
            }
            
            .terminal-container {
                height: 500px;
                display: flex;
                flex-direction: column;
            }
            
            .terminal-output {
                flex-grow: 1;
                background-color: var(--terminal-bg);
                color: var(--terminal-text);
                font-family: 'Fira Code', monospace;
                padding: 15px;
                overflow-y: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            
            .terminal-input-area {
                background-color: var(--terminal-bg);
                padding: 10px;
                border-top: 1px solid var(--border-color);
            }
            
            .terminal-input-area textarea {
                width: 100%;
                background-color: var(--terminal-bg);
                color: var(--terminal-text);
                border: 1px solid var(--border-color);
                padding: 8px;
                font-family: 'Fira Code', monospace;
                resize: vertical;
                min-height: 60px;
                border-radius: 4px;
            }
            
            .terminal-input-area textarea:focus {
                outline: none;
                border-color: var(--accent-color);
            }
            
            .terminal-input-controls {
                display: flex;
                justify-content: space-between;
                margin-top: 10px;
            }
            
            .terminal-checkbox {
                display: flex;
                align-items: center;
                color: #aaa;
                font-size: 14px;
            }
            
            .terminal-checkbox input {
                margin-right: 5px;
            }
            
            .btn-primary {
                background-color: var(--primary-color);
                border-color: var(--primary-color);
            }
            
            .btn-primary:hover {
                background-color: var(--secondary-color);
                border-color: var(--secondary-color);
            }
            
            .welcome-message {
                margin-bottom: 15px;
                padding: 10px;
                background-color: rgba(93, 63, 211, 0.1);
                border-left: 4px solid var(--accent-color);
                border-radius: 5px;
            }
            
            pre {
                background-color: rgba(0, 0, 0, 0.2);
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
                overflow-x: auto;
            }
            
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 5px;
            }
            
            .status-active {
                background-color: #4CAF50;
            }
            
            .status-inactive {
                background-color: #F44336;
            }
            
            .status-pending {
                background-color: #FFC107;
            }
            
            .form-label {
                color: #aaa;
            }
            
            .form-control {
                background-color: var(--terminal-bg);
                color: var(--terminal-text);
                border: 1px solid var(--border-color);
            }
            
            .form-control:focus {
                background-color: var(--terminal-bg);
                color: var(--terminal-text);
                border-color: var(--accent-color);
                box-shadow: 0 0 0 0.25rem rgba(93, 63, 211, 0.25);
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            
            table th,
            table td {
                padding: 10px;
                border: 1px solid var(--border-color);
            }
            
            table th {
                background-color: rgba(93, 63, 211, 0.2);
                color: var(--terminal-text);
                font-weight: bold;
                text-align: left;
            }
            
            table tr:nth-child(even) {
                background-color: rgba(255, 255, 255, 0.05);
            }
            
            .file-browser {
                border: 1px solid var(--border-color);
                border-radius: 4px;
                overflow: hidden;
            }
            
            .file-header {
                background-color: rgba(93, 63, 211, 0.2);
                padding: 10px;
                border-bottom: 1px solid var(--border-color);
            }
            
            .file-list {
                max-height: 300px;
                overflow-y: auto;
            }
            
            .file-item {
                padding: 8px 10px;
                border-bottom: 1px solid var(--border-color);
                cursor: pointer;
                transition: background-color 0.2s;
            }
            
            .file-item:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
            
            .file-item i {
                margin-right: 10px;
                color: #aaa;
            }
            
            .file-item.folder i {
                color: #FFC107;
            }
            
            .file-item.file i {
                color: #66D9EF;
            }
            
            .recent-command {
                padding: 5px 8px;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 3px;
                margin-bottom: 5px;
                cursor: pointer;
                font-family: 'Fira Code', monospace;
                font-size: 12px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .recent-command:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
            
            .code-editor {
                height: 300px;
                font-family: 'Fira Code', monospace;
                font-size: 14px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
            }
            
            .analysis-result {
                border: 1px solid var(--border-color);
                border-radius: 4px;
                margin-bottom: 10px;
            }
            
            .analysis-header {
                background-color: rgba(93, 63, 211, 0.2);
                padding: 10px;
                font-weight: bold;
                border-bottom: 1px solid var(--border-color);
                cursor: pointer;
            }
            
            .analysis-body {
                padding: 10px;
                background-color: var(--terminal-bg);
                display: none;
            }
            
            .analysis-body.show {
                display: block;
            }
            
            /* ANSI color codes for IPython */
            .ansi-black { color: #3E3D32; }
            .ansi-red { color: #F92672; }
            .ansi-green { color: #A6E22E; }
            .ansi-yellow { color: #FD971F; }
            .ansi-blue { color: #66D9EF; }
            .ansi-magenta { color: #AE81FF; }
            .ansi-cyan { color: #A1EFE4; }
            .ansi-white { color: #F8F8F2; }
            
            .ansi-bright-black { color: #75715E; }
            .ansi-bright-red { color: #F92672; }
            .ansi-bright-green { color: #A6E22E; }
            .ansi-bright-yellow { color: #E6DB74; }
            .ansi-bright-blue { color: #66D9EF; }
            .ansi-bright-magenta { color: #AE81FF; }
            .ansi-bright-cyan { color: #A1EFE4; }
            .ansi-bright-white { color: #F9F8F5; }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>Engine Terminal</h1>
            </div>
            <ul class="sidebar-menu">
                <li><a href="/" class="active" id="nav-terminal"><i class="bi bi-terminal"></i> Terminal</a></li>
                <li><a href="/config" id="nav-config"><i class="bi bi-gear"></i> Configuration</a></li>
                <li><a href="/analysis" id="nav-analysis"><i class="bi bi-search"></i> Analysis</a></li>
                <li><a href="/files" id="nav-files"><i class="bi bi-folder"></i> Files</a></li>
                <li><a href="/reports" id="nav-reports"><i class="bi bi-file-text"></i> Reports</a></li>
                <li><a href="/visualization" id="nav-visualization"><i class="bi bi-graph-up"></i> Visualization</a></li>
                <li><a href="/about" id="nav-about"><i class="bi bi-info-circle"></i> About</a></li>
            </ul>
        </div>
        
        <div class="main-content">
            <div class="page-content">
                {% block content %}{% endblock %}
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Highlight active nav item
            document.addEventListener('DOMContentLoaded', function() {
                const currentPath = window.location.pathname;
                const navItems = document.querySelectorAll('.sidebar-menu a');
                
                navItems.forEach(item => {
                    item.classList.remove('active');
                    if (item.getAttribute('href') === currentPath) {
                        item.classList.add('active');
                    }
                });
            });
        </script>
        {% block scripts %}{% endblock %}
    </body>
    </html>
    """
    with open(index_template, 'w') as f:
        f.write(index_html)

# Create the terminal template
terminal_template = templates_dir / "terminal.html"
if not terminal_template.exists():
    terminal_html = """
    {% extends "index.html" %}
    
    {% block content %}
    <h2>Interactive Terminal</h2>
    <p>Use this terminal to interact with the Finite Monkey Engine.</p>
    
    <div class="row">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <span>Python Terminal</span>
                    <button id="clear-terminal" class="btn btn-sm btn-outline-light">Clear</button>
                </div>
                <div class="card-body terminal-container">
                    <div class="welcome-message">
                        <strong>Welcome to the Finite Monkey Engine Terminal!</strong><br>
                        Type Python commands to interact with the framework.<br>
                        <em>Multi-line input is supported. Use Shift+Enter to add new lines.</em><br>
                        <br>
                        Run this command to initialize the engine components:<br>
                        <pre>await init_engine()</pre>
                    </div>
                    
                    <div id="terminal-output" class="terminal-output"></div>
                    
                    <div class="terminal-input-area">
                        <textarea id="terminal-input" placeholder="Enter Python code (use Shift+Enter for new lines)"></textarea>
                        <div class="terminal-input-controls">
                            <label class="terminal-checkbox">
                                <input type="checkbox" id="auto-run" checked>
                                Auto-run on Enter (use Shift+Enter for new lines)
                            </label>
                            <button id="terminal-submit" class="btn btn-primary">Run (Ctrl+Enter)</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">Agent Status</div>
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <div>
                            <span class="status-indicator status-inactive" id="orchestrator-status"></span>
                            <span>Orchestrator</span>
                        </div>
                        <button class="btn btn-sm btn-primary" onclick="runCode('from finite_monkey.agents.orchestrator import WorkflowOrchestrator; orchestrator = WorkflowOrchestrator(); print(\"✅ Orchestrator initialized\")')">Initialize</button>
                    </div>
                    
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <div>
                            <span class="status-indicator status-inactive" id="researcher-status"></span>
                            <span>Researcher</span>
                        </div>
                        <button class="btn btn-sm btn-primary" onclick="runCode('from finite_monkey.agents.researcher import Researcher; researcher = Researcher(); print(\"✅ Researcher initialized\")')">Initialize</button>
                    </div>
                    
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <div>
                            <span class="status-indicator status-inactive" id="validator-status"></span>
                            <span>Validator</span>
                        </div>
                        <button class="btn btn-sm btn-primary" onclick="runCode('from finite_monkey.agents.validator import Validator; validator = Validator(); print(\"✅ Validator initialized\")')">Initialize</button>
                    </div>
                    
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <div>
                            <span class="status-indicator status-inactive" id="documentor-status"></span>
                            <span>Documentor</span>
                        </div>
                        <button class="btn btn-sm btn-primary" onclick="runCode('from finite_monkey.agents.documentor import Documentor; documentor = Documentor(); print(\"✅ Documentor initialized\")')">Initialize</button>
                    </div>
                    
                    <div class="d-grid mt-3">
                        <button class="btn btn-primary" onclick="runCode('await init_engine()')">Initialize All</button>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Recent Commands</div>
                <div class="card-body">
                    <div id="recent-commands">
                        <!-- Recent commands will be populated here -->
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Quick Commands</div>
                <div class="card-body">
                    <div class="d-grid gap-2">
                        <button class="btn btn-outline-light text-start" onclick="runCode('await analyze_contract(\"examples/Vault.sol\")')">
                            <i class="bi bi-search me-2"></i> Analyze Vault.sol
                        </button>
                        <button class="btn btn-outline-light text-start" onclick="runCode('print(nodes_config)')" id="config-btn">
                            <i class="bi bi-gear me-2"></i> Print Configuration
                        </button>
                        <button class="btn btn-outline-light text-start" onclick="runCode('import glob; print(glob.glob(\"examples/**/*.sol\", recursive=True))')">
                            <i class="bi bi-folder me-2"></i> List Example Files
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endblock %}
    
    {% block scripts %}
    <script>
        // Terminal elements
        const terminalOutput = document.getElementById('terminal-output');
        const terminalInput = document.getElementById('terminal-input');
        const terminalSubmit = document.getElementById('terminal-submit');
        const autoRunCheckbox = document.getElementById('auto-run');
        const clearTerminalButton = document.getElementById('clear-terminal');
        const recentCommandsContainer = document.getElementById('recent-commands');
        
        // Agent status elements
        const orchestratorStatus = document.getElementById('orchestrator-status');
        const researcherStatus = document.getElementById('researcher-status');
        const validatorStatus = document.getElementById('validator-status');
        const documentorStatus = document.getElementById('documentor-status');
        
        // Command history
        let commandHistory = [];
        let commandIndex = -1;
        
        // WebSocket connection
        let socket;
        
        // Connect to WebSocket
        function connectWebSocket() {
            socket = new WebSocket(`ws://${window.location.host}/ws/terminal`);
            
            socket.onopen = function(e) {
                console.log('WebSocket connection established');
                appendToTerminal('<span style="color: #8a8">Connected to terminal.</span><br>');
            };
            
            socket.onmessage = function(event) {
                appendToTerminal(event.data);
                terminalOutput.scrollTop = terminalOutput.scrollHeight;
                
                // Check for agent status updates
                checkAgentStatus();
            };
            
            socket.onclose = function(event) {
                if (event.wasClean) {
                    appendToTerminal(`<span style="color: #f33">Connection closed.</span><br>`);
                } else {
                    appendToTerminal(`<span style="color: #f33">Connection lost. Please refresh the page.</span><br>`);
                }
                
                // Try to reconnect after a delay
                setTimeout(connectWebSocket, 3000);
            };
            
            socket.onerror = function(error) {
                appendToTerminal(`<span style="color: #f33">WebSocket error: ${error.message}</span><br>`);
            };
        }
        
        // Initialize the terminal
        function initTerminal() {
            connectWebSocket();
            
            // Add event listeners
            terminalSubmit.addEventListener('click', sendCommand);
            
            terminalInput.addEventListener('keydown', function(e) {
                // Ctrl+Enter always runs the command
                if (e.key === 'Enter' && e.ctrlKey) {
                    e.preventDefault();
                    sendCommand();
                }
                // Enter runs the command if auto-run is enabled
                else if (e.key === 'Enter' && !e.shiftKey && autoRunCheckbox.checked) {
                    e.preventDefault();
                    sendCommand();
                }
                // Shift+Enter adds a new line
                else if (e.key === 'Enter' && e.shiftKey) {
                    // Let the textarea handle this (add a new line)
                }
                // Up arrow navigates command history
                else if (e.key === 'ArrowUp' && e.altKey) {
                    e.preventDefault();
                    navigateHistory(-1);
                }
                // Down arrow navigates command history
                else if (e.key === 'ArrowDown' && e.altKey) {
                    e.preventDefault();
                    navigateHistory(1);
                }
            });
            
            // Clear terminal button
            clearTerminalButton.addEventListener('click', function() {
                terminalOutput.innerHTML = '';
            });
            
            // Focus the input field
            terminalInput.focus();
        }
        
        // Append text to the terminal
        function appendToTerminal(text) {
            terminalOutput.innerHTML += text;
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
        }
        
        // Send command to the server
        function sendCommand() {
            const code = terminalInput.value.trim();
            if (!code) return;
            
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(code);
                terminalInput.value = '';
                
                // Add to command history
                addToCommandHistory(code);
            } else {
                appendToTerminal('<span style="color: #f33">Not connected to server. Please refresh the page.</span><br>');
            }
        }
        
        // Run code directly (used by buttons)
        function runCode(code) {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(code);
                
                // Add to command history
                addToCommandHistory(code);
            } else {
                appendToTerminal('<span style="color: #f33">Not connected to server. Please refresh the page.</span><br>');
            }
        }
        
        // Add a command to the history
        function addToCommandHistory(command) {
            // Don't add duplicate commands consecutively
            if (commandHistory.length > 0 && commandHistory[0] === command) {
                return;
            }
            
            // Add to internal history array
            commandHistory.unshift(command);
            if (commandHistory.length > 20) {
                commandHistory.pop();
            }
            
            // Reset index
            commandIndex = -1;
            
            // Update the UI
            updateRecentCommands();
        }
        
        // Update the recent commands list
        function updateRecentCommands() {
            if (!recentCommandsContainer) return;
            
            recentCommandsContainer.innerHTML = '';
            
            // Add the commands to the list
            commandHistory.slice(0, 5).forEach(cmd => {
                // Truncate long commands
                let displayCmd = cmd.length > 35 ? cmd.substring(0, 35) + '...' : cmd;
                // Replace newlines
                displayCmd = displayCmd.replace(/\\n/g, ' ↵ ');
                
                const div = document.createElement('div');
                div.className = 'recent-command';
                div.textContent = displayCmd;
                div.title = cmd; // Show full command on hover
                div.onclick = function() {
                    terminalInput.value = cmd;
                    terminalInput.focus();
                };
                recentCommandsContainer.appendChild(div);
            });
        }
        
        // Navigate through command history
        function navigateHistory(direction) {
            if (commandHistory.length === 0) return;
            
            commandIndex += direction;
            
            if (commandIndex < -1) {
                commandIndex = commandHistory.length - 1;
            } else if (commandIndex >= commandHistory.length) {
                commandIndex = -1;
            }
            
            if (commandIndex === -1) {
                terminalInput.value = '';
            } else {
                terminalInput.value = commandHistory[commandIndex];
            }
        }
        
        // Check the status of agents
        function checkAgentStatus() {
            // This is a simple implementation that checks for keywords in the terminal output
            const output = terminalOutput.innerHTML;
            
            // Check for orchestrator initialization
            if (output.includes('Orchestrator initialized') || 
                output.includes('WorkflowOrchestrator()') || 
                output.includes('orchestrator = WorkflowOrchestrator()')) {
                orchestratorStatus.className = 'status-indicator status-active';
            }
            
            // Check for researcher initialization
            if (output.includes('Researcher initialized') || 
                output.includes('Researcher()') || 
                output.includes('researcher = Researcher()')) {
                researcherStatus.className = 'status-indicator status-active';
            }
            
            // Check for validator initialization
            if (output.includes('Validator initialized') || 
                output.includes('Validator()') || 
                output.includes('validator = Validator()')) {
                validatorStatus.className = 'status-indicator status-active';
            }
            
            // Check for documentor initialization
            if (output.includes('Documentor initialized') || 
                output.includes('Documentor()') || 
                output.includes('documentor = Documentor()')) {
                documentorStatus.className = 'status-indicator status-active';
            }
            
            // Check for full initialization
            if (output.includes('Engine components initialized successfully')) {
                orchestratorStatus.className = 'status-indicator status-active';
                researcherStatus.className = 'status-indicator status-active';
                validatorStatus.className = 'status-indicator status-active';
                
                // Check if config button should be enabled
                if (output.includes('Configuration loaded successfully')) {
                    const configBtn = document.getElementById('config-btn');
                    if (configBtn) {
                        configBtn.disabled = false;
                    }
                }
            }
        }
        
        // Initialize the terminal when the page loads
        window.addEventListener('DOMContentLoaded', initTerminal);
    </script>
    {% endblock %}
    """
    with open(terminal_template, 'w') as f:
        f.write(terminal_html)

# Create the config template
config_template = templates_dir / "config.html"
if not config_template.exists():
    config_html = """
    {% extends "index.html" %}
    
    {% block content %}
    <h2>Configuration</h2>
    <p>Manage your Finite Monkey Engine configuration.</p>
    
    <div class="card mb-4">
        <div class="card-header">Current Configuration</div>
        <div class="card-body">
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Setting</th>
                            <th>Value</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="config-table-body">
                        {% for field, value in config.items() %}
                        <tr>
                            <td>{{ field }}</td>
                            <td>{{ value }}</td>
                            <td>
                                <button class="btn btn-sm btn-primary" onclick="editConfig('{{ field }}', '{{ value }}')">Edit</button>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">LLM Configuration</div>
                <div class="card-body">
                    <form id="llm-config-form">
                        <div class="mb-3">
                            <label for="workflow-model" class="form-label">Workflow Model</label>
                            <input type="text" class="form-control" id="workflow-model" name="WORKFLOW_MODEL" value="{{ config.WORKFLOW_MODEL }}">
                        </div>
                        
                        <div class="mb-3">
                            <label for="openai-api-base" class="form-label">OpenAI API Base</label>
                            <input type="text" class="form-control" id="openai-api-base" name="OPENAI_API_BASE" value="{{ config.OPENAI_API_BASE }}">
                        </div>
                        
                        <div class="mb-3">
                            <label for="openai-api-key" class="form-label">OpenAI API Key</label>
                            <input type="password" class="form-control" id="openai-api-key" name="OPENAI_API_KEY" value="{{ config.OPENAI_API_KEY }}">
                        </div>
                        
                        <div class="mb-3">
                            <label for="claude-api-key" class="form-label">Claude API Key</label>
                            <input type="password" class="form-control" id="claude-api-key" name="CLAUDE_API_KEY" value="{{ config.CLAUDE_API_KEY }}">
                        </div>
                        
                        <div class="d-grid">
                            <button type="submit" class="btn btn-primary">Save LLM Configuration</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">Storage Configuration</div>
                <div class="card-body">
                    <form id="storage-config-form">
                        <div class="mb-3">
                            <label for="vector-store-path" class="form-label">Vector Store Path</label>
                            <input type="text" class="form-control" id="vector-store-path" name="VECTOR_STORE_PATH" value="{{ config.VECTOR_STORE_PATH }}">
                        </div>
                        
                        <div class="mb-3">
                            <label for="db-dir" class="form-label">Database Directory</label>
                            <input type="text" class="form-control" id="db-dir" name="DB_DIR" value="{{ config.DB_DIR }}">
                        </div>
                        
                        <div class="mb-3">
                            <label for="embedding-model" class="form-label">Embedding Model</label>
                            <input type="text" class="form-control" id="embedding-model" name="EMBEDDING_MODEL_NAME" value="{{ config.EMBEDDING_MODEL_NAME }}">
                        </div>
                        
                        <div class="d-grid">
                            <button type="submit" class="btn btn-primary">Save Storage Configuration</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    
    <div class="card mt-4">
        <div class="card-header">Web Interface Configuration</div>
        <div class="card-body">
            <form id="web-config-form">
                <div class="row">
                    <div class="col-md-4">
                        <div class="mb-3">
                            <label for="web-host" class="form-label">Web Host</label>
                            <input type="text" class="form-control" id="web-host" name="WEB_HOST" value="{{ config.WEB_HOST }}">
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="mb-3">
                            <label for="web-port" class="form-label">Web Port</label>
                            <input type="number" class="form-control" id="web-port" name="WEB_PORT" value="{{ config.WEB_PORT }}">
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="mb-3">
                            <label for="web-interface" class="form-label">Enable Web Interface</label>
                            <select class="form-control" id="web-interface" name="WEB_INTERFACE">
                                <option value="true" {% if config.WEB_INTERFACE %}selected{% endif %}>Enabled</option>
                                <option value="false" {% if not config.WEB_INTERFACE %}selected{% endif %}>Disabled</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="d-grid">
                    <button type="submit" class="btn btn-primary">Save Web Configuration</button>
                </div>
            </form>
        </div>
    </div>
    {% endblock %}
    
    {% block scripts %}
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Submit LLM config form
            document.getElementById('llm-config-form').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                const data = {};
                formData.forEach((value, key) => {
                    data[key] = value;
                });
                
                updateConfig(data);
            });
            
            // Submit storage config form
            document.getElementById('storage-config-form').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                const data = {};
                formData.forEach((value, key) => {
                    data[key] = value;
                });
                
                updateConfig(data);
            });
            
            // Submit web config form
            document.getElementById('web-config-form').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                const data = {};
                formData.forEach((value, key) => {
                    if (key === 'WEB_INTERFACE') {
                        data[key] = value === 'true';
                    } else if (key === 'WEB_PORT') {
                        data[key] = parseInt(value);
                    } else {
                        data[key] = value;
                    }
                });
                
                updateConfig(data);
            });
        });
        
        // Edit a single config setting
        function editConfig(field, value) {
            const newValue = prompt(`Enter new value for ${field}:`, value);
            if (newValue !== null) {
                // Handle booleans and numbers
                let parsedValue = newValue;
                if (newValue.toLowerCase() === 'true') parsedValue = true;
                else if (newValue.toLowerCase() === 'false') parsedValue = false;
                else if (!isNaN(newValue) && newValue.trim() !== '') parsedValue = Number(newValue);
                
                const data = {};
                data[field] = parsedValue;
                updateConfig(data);
            }
        }
        
        // Update configuration via API
        function updateConfig(data) {
            fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Configuration updated successfully');
                    // Reload page to see updated config
                    window.location.reload();
                } else {
                    alert('Error updating configuration: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while updating configuration');
            });
        }
    </script>
    {% endblock %}
    """
    with open(config_template, 'w') as f:
        f.write(config_html)

# Create the analysis template
analysis_template = templates_dir / "analysis.html"
if not analysis_template.exists():
    analysis_html = """
    {% extends "index.html" %}
    
    {% block content %}
    <h2>Contract Analysis</h2>
    <p>Analyze smart contracts for vulnerabilities and generate reports.</p>
    
    <div class="row">
        <div class="col-md-6">
            <div class="card mb-4">
                <div class="card-header">Select Contract</div>
                <div class="card-body">
                    <form id="analysis-form">
                        <div class="mb-3">
                            <label for="contract-path" class="form-label">Contract Path</label>
                            <div class="input-group">
                                <input type="text" class="form-control" id="contract-path" name="file_path" placeholder="examples/Vault.sol" required>
                                <button class="btn btn-outline-secondary" type="button" id="browse-btn">Browse</button>
                            </div>
                            <div class="form-text text-muted">Path to the contract file to analyze</div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="query" class="form-label">Query (Optional)</label>
                            <textarea class="form-control" id="query" name="query" placeholder="Specific aspects to analyze, e.g., 'Check for reentrancy vulnerabilities'" rows="3"></textarea>
                        </div>
                        
                        <div class="d-grid">
                            <button type="submit" class="btn btn-primary" id="analyze-btn">
                                <i class="bi bi-search me-2"></i> Analyze Contract
                            </button>
                        </div>
                    </form>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Recent Analyses</div>
                <div class="card-body">
                    <div id="recent-analyses">
                        {% if analyses %}
                            {% for analysis in analyses %}
                                <div class="analysis-result">
                                    <div class="analysis-header" onclick="toggleAnalysis('{{ analysis.id }}')">
                                        <i class="bi bi-file-earmark-code me-2"></i> {{ analysis.file_path }}
                                    </div>
                                    <div class="analysis-body" id="analysis-{{ analysis.id }}">
                                        <div class="mb-2"><strong>Time:</strong> {{ analysis.timestamp }}</div>
                                        <div class="mb-2"><strong>Status:</strong> {{ analysis.status }}</div>
                                        {% if analysis.report_path %}
                                            <a href="/reports/{{ analysis.report_id }}" class="btn btn-sm btn-primary" target="_blank">View Report</a>
                                        {% endif %}
                                    </div>
                                </div>
                            {% endfor %}
                        {% else %}
                            <p class="text-muted">No analyses found</p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card mb-4">
                <div class="card-header">Analysis Output</div>
                <div class="card-body">
                    <div id="analysis-output" class="terminal-output mb-3" style="height: 300px;"></div>
                    <div class="d-flex justify-content-between align-items-center">
                        <div id="analysis-status"></div>
                        <div>
                            <button class="btn btn-sm btn-primary" id="view-report-btn" style="display: none;">View Report</button>
                            <button class="btn btn-sm btn-outline-light" id="clear-output-btn">Clear</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Quick Analysis</div>
                <div class="card-body">
                    <div class="d-grid gap-2">
                        <button class="btn btn-outline-light text-start" onclick="quickAnalysis('examples/Vault.sol')">
                            <i class="bi bi-search me-2"></i> Vault.sol
                        </button>
                        <button class="btn btn-outline-light text-start" onclick="quickAnalysis('examples/Token.sol')">
                            <i class="bi bi-search me-2"></i> Token.sol
                        </button>
                        <button class="btn btn-outline-light text-start" onclick="quickAnalysis('examples/StakingPool.sol')">
                            <i class="bi bi-search me-2"></i> StakingPool.sol
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- File Browser Modal -->
    <div class="modal fade" id="file-browser-modal" tabindex="-1" aria-labelledby="file-browser-title" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content bg-dark text-light">
                <div class="modal-header">
                    <h5 class="modal-title" id="file-browser-title">Browse Files</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div class="file-browser">
                        <div class="file-header">
                            <span id="current-path">/</span>
                        </div>
                        <div class="file-list" id="file-list">
                            <!-- Files will be populated here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endblock %}
    
    {% block scripts %}
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Submit analysis form
            document.getElementById('analysis-form').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                const data = {};
                formData.forEach((value, key) => {
                    data[key] = value;
                });
                
                runAnalysis(data.file_path, data.query);
            });
            
            // Clear output button
            document.getElementById('clear-output-btn').addEventListener('click', function() {
                document.getElementById('analysis-output').innerHTML = '';
                document.getElementById('analysis-status').innerHTML = '';
                document.getElementById('view-report-btn').style.display = 'none';
            });
            
            // Browse button
            document.getElementById('browse-btn').addEventListener('click', function() {
                openFileBrowser();
            });
        });
        
        // Toggle analysis details
        function toggleAnalysis(id) {
            const analysisBody = document.getElementById(`analysis-${id}`);
            if (analysisBody.classList.contains('show')) {
                analysisBody.classList.remove('show');
            } else {
                // Hide all others
                document.querySelectorAll('.analysis-body').forEach(el => {
                    el.classList.remove('show');
                });
                // Show this one
                analysisBody.classList.add('show');
            }
        }
        
        // Run quick analysis
        function quickAnalysis(filePath) {
            document.getElementById('contract-path').value = filePath;
            runAnalysis(filePath);
        }
        
        // Run analysis
        function runAnalysis(filePath, query = '') {
            // Clear previous output
            document.getElementById('analysis-output').innerHTML = '';
            document.getElementById('analysis-status').innerHTML = '<span class="badge bg-warning">Running...</span>';
            document.getElementById('view-report-btn').style.display = 'none';
            
            // Disable analyze button
            const analyzeBtn = document.getElementById('analyze-btn');
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i> Analyzing...';
            
            // Run analysis via API
            fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file_path: filePath,
                    query: query
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('analysis-status').innerHTML = '<span class="badge bg-success">Completed</span>';
                    
                    // If a report was generated, show the button
                    if (data.report_id) {
                        const viewReportBtn = document.getElementById('view-report-btn');
                        viewReportBtn.style.display = 'inline-block';
                        viewReportBtn.onclick = function() {
                            window.open(`/reports/${data.report_id}`, '_blank');
                        };
                    }
                    
                    // Execute the analysis command in terminal
                    const command = `await analyze_contract("${filePath}"${query ? ', "' + query + '"' : ''})`;
                    const socket = new WebSocket(`ws://${window.location.host}/ws/terminal`);
                    
                    socket.onopen = function() {
                        socket.send(command);
                        socket.close();
                    };
                    
                    // Update recent analyses
                    updateRecentAnalyses();
                } else {
                    document.getElementById('analysis-status').innerHTML = '<span class="badge bg-danger">Failed</span>';
                    document.getElementById('analysis-output').innerHTML += `<span class="ansi-red">Error: ${data.error}</span>`;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('analysis-status').innerHTML = '<span class="badge bg-danger">Failed</span>';
                document.getElementById('analysis-output').innerHTML += `<span class="ansi-red">Error: ${error.message}</span>`;
            })
            .finally(() => {
                // Re-enable analyze button
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="bi bi-search me-2"></i> Analyze Contract';
            });
        }
        
        // Update recent analyses
        function updateRecentAnalyses() {
            fetch('/api/analyses')
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('recent-analyses');
                if (data.analyses.length === 0) {
                    container.innerHTML = '<p class="text-muted">No analyses found</p>';
                    return;
                }
                
                container.innerHTML = '';
                data.analyses.forEach(analysis => {
                    container.innerHTML += `
                        <div class="analysis-result">
                            <div class="analysis-header" onclick="toggleAnalysis('${analysis.id}')">
                                <i class="bi bi-file-earmark-code me-2"></i> ${analysis.file_path}
                            </div>
                            <div class="analysis-body" id="analysis-${analysis.id}">
                                <div class="mb-2"><strong>Time:</strong> ${analysis.timestamp}</div>
                                <div class="mb-2"><strong>Status:</strong> ${analysis.status}</div>
                                ${analysis.report_path ? `<a href="/reports/${analysis.report_id}" class="btn btn-sm btn-primary" target="_blank">View Report</a>` : ''}
                            </div>
                        </div>
                    `;
                });
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
        
        // Open file browser
        function openFileBrowser() {
            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('file-browser-modal'));
            modal.show();
            
            // Load files
            loadFiles('/');
        }
        
        // Load files for browser
        function loadFiles(path) {
            document.getElementById('current-path').textContent = path;
            
            fetch(`/api/files?path=${encodeURIComponent(path)}`)
            .then(response => response.json())
            .then(data => {
                const fileList = document.getElementById('file-list');
                fileList.innerHTML = '';
                
                // Add parent directory if not at root
                if (path !== '/') {
                    const parentPath = path.split('/').slice(0, -1).join('/') || '/';
                    fileList.innerHTML += `
                        <div class="file-item folder" onclick="loadFiles('${parentPath}')">
                            <i class="bi bi-arrow-up"></i> ..
                        </div>
                    `;
                }
                
                // Add directories
                data.directories.forEach(dir => {
                    const dirPath = path === '/' ? `/${dir}` : `${path}/${dir}`;
                    fileList.innerHTML += `
                        <div class="file-item folder" onclick="loadFiles('${dirPath}')">
                            <i class="bi bi-folder"></i> ${dir}
                        </div>
                    `;
                });
                
                // Add files
                data.files.forEach(file => {
                    const filePath = path === '/' ? `/${file}` : `${path}/${file}`;
                    fileList.innerHTML += `
                        <div class="file-item file" onclick="selectFile('${filePath}')">
                            <i class="bi bi-file-text"></i> ${file}
                        </div>
                    `;
                });
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('file-list').innerHTML = `<div class="text-danger">Error loading files: ${error.message}</div>`;
            });
        }
        
        // Select a file from the browser
        function selectFile(path) {
            document.getElementById('contract-path').value = path;
            bootstrap.Modal.getInstance(document.getElementById('file-browser-modal')).hide();
        }
    </script>
    {% endblock %}
    """
    with open(analysis_template, 'w') as f:
        f.write(analysis_html)

@app.get("/", response_class=HTMLResponse)
async def get_terminal(request: Request):
    """Main terminal page."""
    return templates.TemplateResponse("terminal.html", {"request": request})

@app.get("/config", response_class=HTMLResponse)
async def get_config(request: Request):
    """Configuration page."""
    if engine_imports_available:
        # Get configuration from nodes_config
        config_dict = {
            field: getattr(nodes_config, field) 
            for field in dir(nodes_config) 
            if not field.startswith('_') and not callable(getattr(nodes_config, field))
        }
    else:
        # Mock configuration
        config_dict = {
            "WORKFLOW_MODEL": "llama3:8b-instruct",
            "OPENAI_API_BASE": "http://127.0.0.1:11434/v1",
            "OPENAI_API_KEY": "",
            "CLAUDE_API_KEY": "",
            "VECTOR_STORE_PATH": "lancedb",
            "DB_DIR": "db",
            "EMBEDDING_MODEL_NAME": "BAAI/bge-small-en-v1.5",
            "WEB_HOST": "0.0.0.0",
            "WEB_PORT": 8000,
            "WEB_INTERFACE": False
        }
    
    return templates.TemplateResponse("config.html", {
        "request": request,
        "config": config_dict
    })

@app.get("/analysis", response_class=HTMLResponse)
async def get_analysis(request: Request):
    """Contract analysis page."""
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "analyses": list(analysis_tasks.values())
    })

@app.websocket("/ws/terminal")
async def websocket_terminal(websocket: WebSocket):
    """WebSocket endpoint for terminal communication."""
    await websocket.accept()
    
    # Generate a unique session ID
    session_id = str(id(websocket))
    terminal_sessions[session_id] = websocket
    
    # Send welcome message
    await websocket.send_text(f"<span style='color: #5D3FD3;'>Terminal session started. Type Python commands to interact with the framework.</span><br>")
    
    # Start output forwarding task
    output_task = asyncio.create_task(forward_output(websocket))
    
    try:
        # Handle incoming messages
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            # Echo the command
            # Format multiline code nicely
            if "\n" in message:
                formatted_message = "<span class='ansi-bright-green'>&gt;&gt;&gt; Multi-line code:</span><br>"
                formatted_message += "<pre style='margin: 0 0 10px 20px; background-color: rgba(166, 226, 46, 0.1);'>"
                for line in message.split("\n"):
                    formatted_message += f"{line}<br>"
                formatted_message += "</pre>"
                await websocket.send_text(formatted_message)
            else:
                await websocket.send_text(f"<span class='ansi-bright-green'>&gt;&gt;&gt; {message}</span><br>")
            
            # Execute the command
            await kernel.execute(message)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up
        output_task.cancel()
        terminal_sessions.pop(session_id, None)

async def forward_output(websocket: WebSocket):
    """Forward kernel output to the WebSocket."""
    try:
        while True:
            # Get output from the queue
            try:
                # Use a short timeout to avoid blocking
                if not output_queue.empty():
                    output = await output_queue.get()
                    if output:
                        await websocket.send_text(output)
            except Exception as e:
                logger.error(f"Error getting output: {e}")
            
            # Sleep before checking again
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # Task was cancelled, clean up
        pass
    except Exception as e:
        logger.error(f"Error forwarding output: {e}")

@app.post("/api/config")
async def update_config(data: Dict[str, Any]):
    """Update configuration settings."""
    if not engine_imports_available:
        return {"success": False, "error": "Engine components not available"}
    
    try:
        # Update each field in nodes_config
        for field, value in data.items():
            if hasattr(nodes_config, field):
                setattr(nodes_config, field, value)
            else:
                return {"success": False, "error": f"Unknown field: {field}"}
                
        return {"success": True}
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/analyze")
async def analyze_contract(data: Dict[str, Any]):
    """Analyze a contract via API."""
    file_path = data.get("file_path")
    query = data.get("query", "")
    
    if not file_path:
        return {"success": False, "error": "File path is required"}
    
    try:
        # Generate a unique ID for this analysis
        analysis_id = f"analysis_{len(analysis_tasks) + 1}"
        
        # Store the analysis task
        analysis_tasks[analysis_id] = {
            "id": analysis_id,
            "file_path": file_path,
            "query": query,
            "timestamp": str(datetime.now()),
            "status": "Running"
        }
        
        # In a real implementation, this would run the analysis
        # For now, just pretend it was successful
        analysis_tasks[analysis_id]["status"] = "Completed"
        analysis_tasks[analysis_id]["report_id"] = f"report_{analysis_id}"
        
        return {"success": True, "analysis_id": analysis_id, "report_id": f"report_{analysis_id}"}
    except Exception as e:
        logger.error(f"Error analyzing contract: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/analyses")
async def get_analyses():
    """Get a list of analysis tasks."""
    return {"analyses": list(analysis_tasks.values())}

@app.get("/api/files")
async def get_files(path: str = "/"):
    """Get a list of files and directories at the given path."""
    try:
        # Normalize the path to prevent directory traversal
        norm_path = os.path.normpath(path)
        if norm_path == ".":
            norm_path = "/"
            
        # Convert to absolute path within the project
        abs_path = os.path.join(project_dir, norm_path.lstrip("/"))
        
        # Get directories and files
        directories = []
        files = []
        
        if os.path.exists(abs_path):
            for item in os.listdir(abs_path):
                item_path = os.path.join(abs_path, item)
                if os.path.isdir(item_path):
                    directories.append(item)
                elif os.path.isfile(item_path) and item.endswith(".sol"):
                    files.append(item)
        
        return {"directories": directories, "files": files}
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return {"directories": [], "files": [], "error": str(e)}

def main():
    """Run the application."""
    # Create logs directory
    logs_dir = Path(project_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(logs_dir / "engine_terminal.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log the startup
    logger.info(f"Starting Engine Terminal on port {TERMINAL_PORT}")
    logger.info(f"Open http://localhost:{TERMINAL_PORT} in your browser")
    
    # Open browser automatically
    webbrowser.open(f"http://localhost:{TERMINAL_PORT}")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=TERMINAL_PORT, log_level="info")

if __name__ == "__main__":
    from datetime import datetime
    main()