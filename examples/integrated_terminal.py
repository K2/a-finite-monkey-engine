#!/usr/bin/env python3
"""
Integrated Terminal for the Finite Monkey Engine.

This script creates an integrated Jupyter terminal that:
1. Extracts port configuration from nodes_config
2. Supports multi-line input for code blocks
3. Injects the Finite Monkey Engine context into the IPython session
4. Provides a simple but effective dashboard UI
"""

import os
import sys
import json
import asyncio
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import webbrowser
from textwrap import dedent
from fastapi import FastAPI, WebSocket, Request, Form, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("integrated-terminal")

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TERM'] = 'xterm-256color'
os.environ['FORCE_COLOR'] = '1'

# Add project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Import nodes_config to get port configuration
try:
    from finite_monkey.nodes_config import nodes_config
    WEB_PORT = nodes_config.WEB_PORT
    # Use terminal port = web port + 1 to avoid conflicts
    TERMINAL_PORT = WEB_PORT + 1
    logger.info(f"Using configuration: Web port={WEB_PORT}, Terminal port={TERMINAL_PORT}")
except ImportError:
    logger.warning("Could not import nodes_config, using default ports")
    WEB_PORT = 8000
    TERMINAL_PORT = 8001

# Create FastAPI app
app = FastAPI(title="Integrated Finite Monkey Terminal",
              description="An integrated terminal for the Finite Monkey Engine")

# Create a directory for static files
static_dir = Path(project_dir) / "static"
static_dir.mkdir(exist_ok=True)

# Static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Import Jupyter kernel functionality
try:
    from jupyter_client import KernelManager
    has_jupyter = True
except ImportError:
    has_jupyter = False
    logger.warning("jupyter_client not found. Will run with simulated output.")

# Terminal sessions store
terminal_sessions = {}

# Jupyter kernel output queue
output_queue = asyncio.Queue()

# Keep track of engine components loaded into the kernel
engine_components = {
    "orchestrator": False,
    "researcher": False,
    "validator": False,
    "documentor": False
}

class JupyterKernel:
    """Enhanced Jupyter kernel with context injection support."""
    
    def __init__(self):
        """Initialize the kernel."""
        self.km = None
        self.kc = None
        self.initialized = False
        self.namespace = {}
        
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
        setup_code = """
        import sys
        import os
        import asyncio
        
        # Add helper function to ensure we have an event loop
        def ensure_event_loop():
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        
        # Helper to initialize the engine components
        async def init_engine():
            # Import components
            from finite_monkey.agents.orchestrator import WorkflowOrchestrator
            from finite_monkey.agents.researcher import Researcher
            from finite_monkey.agents.validator import Validator
            
            # Create instances
            global orchestrator, researcher, validator
            orchestrator = WorkflowOrchestrator()
            researcher = Researcher()
            validator = Validator()
            
            print("✅ Engine components initialized successfully!")
            return {
                "orchestrator": orchestrator,
                "researcher": researcher,
                "validator": validator
            }
            
        # Make ensure_event_loop available
        globals()['ensure_event_loop'] = ensure_event_loop
        globals()['init_engine'] = init_engine
        
        print("🔄 Engine context setup complete. Run 'await init_engine()' to initialize components.")
        """
        
        # Execute the setup code
        await self.execute(dedent(setup_code))
    
    async def execute(self, code, store_history=True):
        """Execute code in the kernel."""
        if not self.initialized:
            return f"<span style='color: #f55;'>Kernel not initialized</span>"
            
        if not has_jupyter:
            # Simulate output
            return f"<span style='color: #5f5;'>&gt;&gt;&gt; {code}</span><br>Simulated output (Jupyter client not available)"
        
        try:
            # Check if the code initializes engine components
            if "orchestrator = WorkflowOrchestrator()" in code:
                engine_components["orchestrator"] = True
            if "researcher = Researcher()" in code:
                engine_components["researcher"] = True
            if "validator = Validator()" in code:
                engine_components["validator"] = True
            if "documentor = Documentor()" in code:
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
        
        # Debug: print message type
        # logger.debug(f"Message type: {msg_type}")
        
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

# HTML template for the terminal
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Finite Monkey Engine - Integrated Terminal</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }
        
        header h1 {
            margin: 0;
            color: var(--accent-color);
            font-size: 24px;
        }
        
        .header-controls {
            display: flex;
            gap: 10px;
        }
        
        .header-controls a {
            color: var(--accent-color);
            text-decoration: none;
            border: 1px solid var(--accent-color);
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .header-controls a:hover {
            background-color: var(--accent-color);
            color: var(--background-color);
        }
        
        .row {
            display: flex;
            flex-wrap: wrap;
            margin: 0 -10px;
        }
        
        .col-md-8 {
            flex: 0 0 66.666667%;
            max-width: 66.666667%;
            padding: 0 10px;
        }
        
        .col-md-4 {
            flex: 0 0 33.333333%;
            max-width: 33.333333%;
            padding: 0 10px;
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .card-header {
            background-color: var(--primary-color);
            color: white;
            padding: 10px 15px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .card-body {
            padding: 15px;
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
        
        .terminal-submit {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .terminal-submit:hover {
            background-color: var(--secondary-color);
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
        
        .agent-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px;
            border-radius: 4px;
            background-color: rgba(255, 255, 255, 0.05);
            margin-bottom: 10px;
        }
        
        .agent-status {
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
        
        .agent-info {
            display: flex;
            align-items: center;
        }
        
        .agent-actions button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            cursor: pointer;
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
        
        .quick-commands button {
            display: block;
            width: 100%;
            text-align: left;
            background-color: rgba(255, 255, 255, 0.05);
            color: var(--terminal-text);
            border: none;
            padding: 8px 10px;
            border-radius: 4px;
            margin-bottom: 8px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .quick-commands button:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .shortcuts hr {
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 10px 0;
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
        
        @media (max-width: 768px) {
            .row {
                flex-direction: column;
            }
            
            .col-md-8, .col-md-4 {
                flex: 0 0 100%;
                max-width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Finite Monkey Engine Terminal</h1>
            <div class="header-controls">
                <a href="/status" target="_blank">Status</a>
                <a href="/reports" target="_blank">Reports</a>
                <a href="http://localhost:${WEB_PORT}" target="_blank">FastHTML UI</a>
            </div>
        </header>
        
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <span>Python Terminal</span>
                        <button id="clear-terminal" class="terminal-submit">Clear</button>
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
                                <button id="terminal-submit" class="terminal-submit">Run (Ctrl+Enter)</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Agent Status</div>
                    <div class="card-body">
                        <div class="agent-card">
                            <div class="agent-info">
                                <span class="agent-status status-inactive" id="orchestrator-status"></span>
                                <span>Orchestrator</span>
                            </div>
                            <div class="agent-actions">
                                <button onclick="runCode('from finite_monkey.agents.orchestrator import WorkflowOrchestrator; orchestrator = WorkflowOrchestrator(); print(\"✅ Orchestrator initialized\")')">Initialize</button>
                            </div>
                        </div>
                        
                        <div class="agent-card">
                            <div class="agent-info">
                                <span class="agent-status status-inactive" id="researcher-status"></span>
                                <span>Researcher</span>
                            </div>
                            <div class="agent-actions">
                                <button onclick="runCode('from finite_monkey.agents.researcher import Researcher; researcher = Researcher(); print(\"✅ Researcher initialized\")')">Initialize</button>
                            </div>
                        </div>
                        
                        <div class="agent-card">
                            <div class="agent-info">
                                <span class="agent-status status-inactive" id="validator-status"></span>
                                <span>Validator</span>
                            </div>
                            <div class="agent-actions">
                                <button onclick="runCode('from finite_monkey.agents.validator import Validator; validator = Validator(); print(\"✅ Validator initialized\")')">Initialize</button>
                            </div>
                        </div>
                        
                        <div class="agent-card">
                            <div class="agent-info">
                                <span class="agent-status status-inactive" id="documentor-status"></span>
                                <span>Documentor</span>
                            </div>
                            <div class="agent-actions">
                                <button onclick="runCode('from finite_monkey.agents.documentor import Documentor; documentor = Documentor(); print(\"✅ Documentor initialized\")')">Initialize</button>
                            </div>
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
                        <div class="quick-commands">
                            <button onclick="runCode('await init_engine()')">Initialize All Agents</button>
                            <button onclick="runCode('await orchestrator.run_analysis(\"examples/Vault.sol\")')">Analyze Vault.sol</button>
                            <button onclick="runCode('await researcher.analyze_code_async(\"contract Test { uint value; }\")')">Test Researcher</button>
                            <button onclick="runCode('print(validator)')">Print Validator Info</button>
                        </div>
                        
                        <div class="shortcuts">
                            <hr>
                            <h4 style="margin-top: 0; color: #aaa; font-size: 14px;">Keyboard Shortcuts</h4>
                            <ul style="color: #aaa; font-size: 14px; padding-left: 20px;">
                                <li><strong>Enter</strong>: Run code (if auto-run enabled)</li>
                                <li><strong>Shift+Enter</strong>: New line</li>
                                <li><strong>Ctrl+Enter</strong>: Run code (always)</li>
                                <li><strong>Up/Down</strong>: Navigate history</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
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
                let displayCmd = cmd.length > 40 ? cmd.substring(0, 40) + '...' : cmd;
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
                orchestratorStatus.className = 'agent-status status-active';
            }
            
            // Check for researcher initialization
            if (output.includes('Researcher initialized') || 
                output.includes('Researcher()') || 
                output.includes('researcher = Researcher()')) {
                researcherStatus.className = 'agent-status status-active';
            }
            
            // Check for validator initialization
            if (output.includes('Validator initialized') || 
                output.includes('Validator()') || 
                output.includes('validator = Validator()')) {
                validatorStatus.className = 'agent-status status-active';
            }
            
            // Check for documentor initialization
            if (output.includes('Documentor initialized') || 
                output.includes('Documentor()') || 
                output.includes('documentor = Documentor()')) {
                documentorStatus.className = 'agent-status status-active';
            }
            
            // Check for full initialization
            if (output.includes('Engine components initialized successfully')) {
                orchestratorStatus.className = 'agent-status status-active';
                researcherStatus.className = 'agent-status status-active';
                validatorStatus.className = 'agent-status status-active';
            }
        }
        
        // Initialize the terminal when the page loads
        window.addEventListener('DOMContentLoaded', initTerminal);
    </script>
</body>
</html>
""".replace("${WEB_PORT}", str(WEB_PORT))

@app.on_event("startup")
async def startup_event():
    """Start the kernel on application startup."""
    await kernel.start()

@app.on_event("shutdown")
def shutdown_event():
    """Shutdown the kernel on application shutdown."""
    kernel.shutdown()

@app.get("/", response_class=HTMLResponse)
async def get_terminal():
    """Return the terminal HTML page."""
    return HTMLResponse(content=HTML_TEMPLATE)

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

@app.get("/status")
async def status():
    """Get the status of the engine components."""
    return {
        "kernel_initialized": kernel.initialized,
        "components": engine_components,
        "terminal_sessions": len(terminal_sessions)
    }

@app.get("/reports")
async def get_reports():
    """Get a list of reports."""
    reports_dir = Path(project_dir) / "reports"
    if not reports_dir.exists():
        return HTMLResponse(content="<h1>No reports directory found</h1>")
        
    # List files in the reports directory
    files = list(reports_dir.glob("*.md"))
    
    # Create a simple HTML page
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Finite Monkey Reports</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #5D3FD3; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            a {{ color: #5D3FD3; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>Finite Monkey Reports</h1>
        <ul>
    """
    
    for file in files:
        html += f"""
        <li>
            <a href="/reports/{file.stem}">{file.name}</a>
            <div>{file.stat().st_mtime}</div>
        </li>
        """
    
    html += """
        </ul>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Get a report by ID."""
    report_file = Path(project_dir) / "reports" / f"{report_id}.md"
    
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report not found")
        
    try:
        content = report_file.read_text()
        
        # Create a simple HTML page
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Report: {report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #5D3FD3; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                code {{ font-family: monospace; }}
            </style>
        </head>
        <body>
            <a href="/reports">&larr; Back to reports</a>
            <h1>Report: {report_id}</h1>
            <div>
                {content}
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {str(e)}")

def main():
    """Run the application."""
    # Create logs directory
    logs_dir = Path(project_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(logs_dir / "integrated_terminal.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log the startup
    logger.info(f"Starting Integrated Terminal on port {TERMINAL_PORT}")
    logger.info(f"Open http://localhost:{TERMINAL_PORT} in your browser")
    
    # Open browser automatically
    webbrowser.open(f"http://localhost:{TERMINAL_PORT}")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=TERMINAL_PORT, log_level="info")

if __name__ == "__main__":
    main()