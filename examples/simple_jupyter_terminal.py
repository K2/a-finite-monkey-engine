#!/usr/bin/env python3
"""
Simple Jupyter Terminal for the Finite Monkey Engine.

This script creates a simple Jupyter kernel and provides
a basic web interface to interact with it.
"""

import os
import sys
import asyncio
import logging
import webbrowser
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("simple-jupyter-terminal")

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TERM'] = 'xterm-256color'
os.environ['FORCE_COLOR'] = '1'

# Create FastAPI app
app = FastAPI(title="Simple Jupyter Terminal", 
              description="A simple Jupyter terminal for the Finite Monkey Engine")

# HTML template for the terminal
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Jupyter Terminal</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #252526;
            color: #f0f0f0;
            padding: 20px;
            margin: 0;
        }
        
        h1 {
            color: #5D3FD3;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .terminal {
            background-color: #1e1e1e;
            color: #f0f0f0;
            font-family: 'Courier New', monospace;
            padding: 15px;
            border-radius: 5px;
            height: 400px;
            overflow-y: auto;
            margin-bottom: 15px;
            white-space: pre-wrap;
        }
        
        .terminal-input {
            display: flex;
            margin-bottom: 20px;
        }
        
        .terminal-input input {
            flex: 1;
            padding: 8px 12px;
            background-color: #333;
            color: #fff;
            border: 1px solid #555;
            font-family: 'Courier New', monospace;
            border-radius: 3px 0 0 3px;
        }
        
        .terminal-input button {
            padding: 8px 16px;
            background-color: #5D3FD3;
            color: white;
            border: none;
            border-radius: 0 3px 3px 0;
            cursor: pointer;
        }
        
        .terminal-input button:hover {
            background-color: #4a3ca8;
        }
        
        .welcome-message {
            margin-bottom: 15px;
            padding: 10px;
            background-color: rgba(93, 63, 211, 0.1);
            border-left: 4px solid #5D3FD3;
            border-radius: 5px;
        }
        
        pre {
            background-color: #333;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            overflow-x: auto;
        }
        
        /* ANSI colors */
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
    <div class="container">
        <h1>Finite Monkey IPython Terminal</h1>
        <p>Interactive IPython terminal for the Finite Monkey Engine</p>
        
        <div class="welcome-message">
            <strong>Welcome to the Finite Monkey IPython Terminal!</strong><br>
            Type Python commands to interact with the framework.<br>
            <br>
            Run these commands to initialize the framework objects:<br>
            <pre>
import asyncio

# Ensure we have a proper event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from finite_monkey.agents.orchestrator import WorkflowOrchestrator
from finite_monkey.agents.researcher import Researcher
from finite_monkey.agents.validator import Validator

# Initialize the agents
orchestrator = WorkflowOrchestrator()
researcher = Researcher()
validator = Validator()

print("✅ Framework objects initialized successfully!")
            </pre>
        </div>
        
        <div id="terminal" class="terminal"></div>
        
        <div class="terminal-input">
            <input type="text" id="command" placeholder="Enter Python command...">
            <button onclick="sendCommand()">Run</button>
        </div>
        
        <script>
            // WebSocket connection
            const terminalElement = document.getElementById('terminal');
            const commandInput = document.getElementById('command');
            
            let socket = new WebSocket('ws://' + window.location.host + '/ws/terminal');
            
            socket.onopen = function(e) {
                console.log('WebSocket connection established');
                appendToTerminal('<span style="color: #8a8">Connected to terminal.</span>');
            };
            
            socket.onmessage = function(event) {
                appendToTerminal(event.data);
                terminalElement.scrollTop = terminalElement.scrollHeight;
            };
            
            socket.onclose = function(event) {
                if (event.wasClean) {
                    appendToTerminal(`<span style="color: #f33">Connection closed.</span>`);
                } else {
                    appendToTerminal(`<span style="color: #f33">Connection lost. Please refresh the page.</span>`);
                }
            };
            
            socket.onerror = function(error) {
                appendToTerminal(`<span style="color: #f33">WebSocket error: ${error.message}</span>`);
            };
            
            function appendToTerminal(message) {
                terminalElement.innerHTML += message;
                terminalElement.scrollTop = terminalElement.scrollHeight;
            }
            
            function sendCommand() {
                const command = commandInput.value;
                if (command) {
                    socket.send(command);
                    commandInput.value = '';
                }
            }
            
            // Submit on Enter key
            commandInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendCommand();
                }
            });
            
            // Focus the input field
            commandInput.focus();
        </script>
    </div>
</body>
</html>
"""

# Terminal sessions store
terminal_sessions = {}

# Jupyter kernel output queue
output_queue = asyncio.Queue()

# Import the kernel manager
try:
    from jupyter_client import KernelManager
    has_jupyter = True
except ImportError:
    has_jupyter = False
    logger.warning("jupyter_client not found. Will run with simulated output.")

class JupyterKernel:
    """Simple wrapper for Jupyter kernel interactions."""
    
    def __init__(self):
        """Initialize the kernel."""
        self.km = None
        self.kc = None
        self.initialized = False
        
    async def start(self):
        """Start the Jupyter kernel."""
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
            
            logger.info("Jupyter kernel started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting Jupyter kernel: {e}")
            return False
    
    async def execute(self, code):
        """Execute code in the kernel."""
        if not self.initialized:
            return f"<span style='color: #f55;'>Kernel not initialized</span>"
            
        if not has_jupyter:
            # Simulate output
            return f"<span style='color: #5f5;'>&gt;&gt;&gt; {code}</span><br>Simulated output (Jupyter client not available)"
        
        try:
            # Execute the code
            msg_id = self.kc.execute(code)
            
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
            while not output_queue.empty():
                output = await output_queue.get()
                if output:
                    await websocket.send_text(output)
            
            # Sleep before checking again
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # Task was cancelled, clean up
        pass
    except Exception as e:
        logger.error(f"Error forwarding output: {e}")

def main():
    """Run the application."""
    port = 8899  # Use a high port number to avoid conflicts
    
    # Create logs directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = Path(project_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(logs_dir / "simple_jupyter_terminal.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log the startup
    logger.info(f"Starting Simple Jupyter Terminal on port {port}")
    logger.info(f"Open http://localhost:{port} in your browser")
    
    # Open browser
    webbrowser.open(f"http://localhost:{port}")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()