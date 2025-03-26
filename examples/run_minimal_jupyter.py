#!/usr/bin/env python3
"""
Minimal FastHTML app with embedded IPython terminal.
"""

import os
import sys
import asyncio
import threading
import queue
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config import Config

# FastHTML imports
from fasthtml.common import *
from fasthtml.fastapp import FastHTML
import uvicorn

class MinimalIPythonTerminal:
    """Minimal IPython terminal embedding for FastHTML."""
    
    def __init__(self, namespace=None):
        """Initialize the terminal with optional namespace."""
        self.namespace = namespace or {}
        
        # Input/output queues for IPython communication
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.ipython_thread = None
        
        # Configure IPython
        c = Config()
        c.InteractiveShell.colors = 'Linux'
        c.InteractiveShell.confirm_exit = False
        c.TerminalInteractiveShell.term_title = False
        c.TerminalInteractiveShell.simple_prompt = True
        c.InteractiveShell.banner1 = 'Minimal IPython Terminal'
        
        self.ipython_config = c
    
    def _io_redirect(self):
        """Set up custom I/O redirection for IPython."""
        # Create custom stdin/stdout
        class TerminalStdin:
            def __init__(self, queue):
                self.queue = queue
                self.encoding = 'utf-8'
                
            def readline(self):
                return self.queue.get() + '\n'
                
            def isatty(self):
                return True
        
        class TerminalStdout:
            def __init__(self, queue):
                self.queue = queue
                self.encoding = 'utf-8'
                
            def write(self, data):
                if data:
                    self.queue.put(data)
                return len(data) if data else 0
                
            def flush(self):
                pass
                
        return TerminalStdin(self.input_queue), TerminalStdout(self.output_queue)
    
    def _ipython_thread_func(self):
        """Function to run IPython in a separate thread."""
        try:
            # Set terminal colors
            os.environ['TERM'] = 'xterm-256color'
            os.environ['FORCE_COLOR'] = '1'
            
            # Create redirected I/O
            stdin, stdout = self._io_redirect()
            
            # Save original stdin/stdout/stderr
            orig_stdin = sys.stdin
            orig_stdout = sys.stdout
            orig_stderr = sys.stderr
            
            try:
                # Redirect stdin/stdout/stderr
                sys.stdin = stdin
                sys.stdout = stdout
                sys.stderr = stdout
                
                # Create and run IPython shell
                shell = InteractiveShellEmbed(
                    config=self.ipython_config,
                    user_ns=self.namespace
                )
                shell()
            finally:
                # Restore original stdin/stdout/stderr
                sys.stdin = orig_stdin
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                
            print("IPython thread exited")
        except Exception as e:
            print(f"Error in IPython thread: {e}")
    
    def start(self):
        """Start the IPython terminal."""
        if self.ipython_thread is None or not self.ipython_thread.is_alive():
            self.ipython_thread = threading.Thread(
                target=self._ipython_thread_func,
                daemon=True
            )
            self.ipython_thread.start()
    
    def stop(self):
        """Stop the IPython terminal."""
        if self.ipython_thread and self.ipython_thread.is_alive():
            self.input_queue.put("exit")
            self.ipython_thread.join(timeout=1.0)
    
    def execute_command(self, command):
        """Execute a command in the IPython shell."""
        self.input_queue.put(command)
    
    def get_output(self, timeout=0.1):
        """Get output from the IPython shell."""
        output = []
        try:
            # Try to get all output
            while True:
                item = self.output_queue.get(block=False)
                if item:
                    output.append(item)
        except queue.Empty:
            pass
            
        return ''.join(output)

# Create a FastHTML app
app = FastHTML(
    title="Minimal IPython Terminal"
)

# Create a terminal instance
terminal = MinimalIPythonTerminal({
    'test_var': 'Hello from IPython!',
    'os': os,
    'sys': sys
})

# Start the terminal
terminal.start()

# WebSocket handler for terminal communication
@app.ws('/ws/terminal')
async def terminal_ws(msg: str = None, send = None):
    """WebSocket handler for terminal communication."""
    if msg is None:
        # Send welcome message on connection
        welcome_msg = """
        <div class="welcome-message">
            <strong>Welcome to the Minimal IPython Terminal!</strong><br>
            Type Python commands to interact with IPython.
        </div>
        """
        await send(welcome_msg)
        
        # Start background task to poll for output
        asyncio.create_task(poll_output(send))
        return
    
    # Process command
    try:
        # Echo the command
        await send(f'<span class="command-echo">&gt;&gt;&gt; {msg}</span><br>')
        
        # Execute the command
        terminal.execute_command(msg)
        
        # Note: The output will be handled by the polling task
        return ""
    except Exception as e:
        return f'<span class="error">Error: {str(e)}</span>'

async def poll_output(send_func):
    """Poll for output from the terminal and send it to the client."""
    while True:
        # Get new output
        output = terminal.get_output()
        if output:
            # Process output for display
            output = output.replace('\n', '<br>')
            await send_func(output)
        
        # Wait before checking again
        await asyncio.sleep(0.1)

@app.route('/')
def index():
    """Main terminal page."""
    return Div(
        H1("Minimal IPython Terminal"),
        
        # Terminal output
        Div(
            id="terminal-output",
            cls="terminal-output",
            hx_ext="ws",
            ws_connect="/ws/terminal",
            style="height: 400px; overflow-y: auto; background-color: #1e1e1e; color: #f0f0f0; font-family: monospace; padding: 10px; border-radius: 5px; margin-bottom: 10px;"
        ),
        
        # Input form
        Form(
            Input(
                id="terminal-input",
                name="command",
                placeholder="Enter a command...",
                style="width: 80%; padding: 8px; border-radius: 5px; border: 1px solid #888;"
            ),
            Button(
                "Run",
                type="submit",
                style="margin-left: 10px; padding: 8px 16px; background-color: #5D3FD3; color: white; border: none; border-radius: 5px; cursor: pointer;"
            ),
            hx_ws_send="true",
            style="display: flex; margin-top: 10px;"
        ),
        
        # Styles
        Style("""
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        h1 {
            color: #333;
        }
        
        .terminal-output {
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
        
        .welcome-message {
            margin-bottom: 10px;
            padding: 10px;
            background-color: rgba(93, 63, 211, 0.1);
            border-left: 4px solid #5D3FD3;
        }
        
        .command-echo {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .error {
            color: #F44336;
        }
        """),
        
        # Script to handle input form submission with Enter key
        Script("""
        document.addEventListener('DOMContentLoaded', function() {
            const terminalInput = document.getElementById('terminal-input');
            if (terminalInput) {
                terminalInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.form.requestSubmit();
                    }
                });
            }
            
            // Auto-scroll terminal to bottom on content changes
            const terminalOutput = document.getElementById('terminal-output');
            if (terminalOutput) {
                const observer = new MutationObserver(function() {
                    terminalOutput.scrollTop = terminalOutput.scrollHeight;
                });
                
                observer.observe(terminalOutput, { childList: true, subtree: true });
            }
        });
        """)
    )

# Run the app when this script is executed directly
if __name__ == "__main__":
    # Start the uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)