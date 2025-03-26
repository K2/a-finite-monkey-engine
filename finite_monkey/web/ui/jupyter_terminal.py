"""
FastHTML Jupyter integration for the Finite Monkey IPython Terminal.

This module provides integration between FastHTML's JupyUviAsync and HTMX
components for proper iframe isolation of the Jupyter terminal.
"""

import os
import asyncio
import threading
import queue
import sys
from fasthtml.common import *
from fasthtml.jupyter import JupyUviAsync, HTMX, ws_client
import json

class JupyterTerminal:
    """
    FastHTML Jupyter integration for IPython terminal.
    
    This class provides a high-level interface for embedding a Jupyter terminal
    in a FastHTML web application using FastHTML's JupyUviAsync and HTMX.
    """
    
    def __init__(self, namespace=None, app=None, port=8000):
        """
        Initialize the JupyterTerminal.
        
        Args:
            namespace: Dictionary of objects to include in the IPython namespace
            app: FastHTML application instance
            port: Port number for the application
        """
        self.namespace = namespace or {}
        self.app = app
        self.port = port
        self._initialized = False
        
        # Set up the app routes if provided
        if self.app:
            self._init_app()
        
    async def start(self):
        """Start the Jupyter terminal."""
        if self._initialized:
            return self
        
        # Mark as initialized
        self._initialized = True
        return self
        
    async def stop(self):
        """Stop the Jupyter terminal."""
        self._initialized = False
        
    def _init_app(self):
        """Initialize the FastHTML app routes."""
        # Define the WebSocket handler for the terminal
        @self.app.ws('/ws/jupyter_terminal')
        async def terminal_ws(msg: str = None, send = None):
            """WebSocket handler for Jupyter terminal."""
            if msg is None:
                # Send welcome message on connection
                welcome_html = """
                <div class="welcome-message">
                    <strong>Welcome to the Finite Monkey IPython Terminal!</strong><br>
                    Type Python commands to interact with the framework.<br>
                    <br>
                    Run these commands to initialize the framework objects:<br>
                    <pre style="background-color: #333; padding: 10px; margin: 5px 0; border-radius: 5px;">
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
                """
                await send(welcome_html)
                return
                
            # Provide basic echo functionality for testing - in the real app, 
            # the JupyUvi integration will handle execution
            await send(f'<span class="ansi-bright-green">&gt;&gt;&gt; {msg}</span><br>')
            await send(f'Command received: {msg}<br>Use the embedded Jupyter terminal for execution.')
                
    def get_terminal_component(self, height="500px", width="100%"):
        """
        Get the FastHTML component for embedding the terminal.
        
        Args:
            height: Height of the terminal iframe
            width: Width of the terminal iframe
            
        Returns:
            FastHTML component for the terminal
        """
        # Define the terminal route if not already defined
        @self.app.get('/jupyter_terminal')
        def terminal_view():
            """Create a minimal HTML page for the JupyUvi iframe to load."""
            return Div(
                H3("IPython Terminal"),
                P("Type Python commands to interact with the framework."),
                
                # A minimal client-side terminal for testing - the actual terminal
                # will be rendered by JupyUvi in the iframe
                Div(
                    id="terminal-output",
                    cls="terminal-output",
                    hx_ext="ws",
                    ws_connect="/ws/jupyter_terminal",
                    style=f"height: {height}; width: {width}; overflow: auto; padding: 1em; background-color: #1e1e1e; color: #f0f0f0; font-family: monospace; white-space: pre-wrap;"
                ),
                Form(
                    Input(
                        id="terminal-input",
                        name="command",
                        placeholder="Enter command...",
                        style="width: 80%; padding: 0.5em; margin-right: 1em; background-color: #333; color: #fff; border: 1px solid #555;"
                    ),
                    Button(
                        "Run",
                        id="run-command",
                        style="padding: 0.5em 1em; background-color: #5D3FD3; color: white; border: none; border-radius: 3px; cursor: pointer;"
                    ),
                    hx_ws_send="true",
                    style="margin-top: 1em; display: flex;"
                ),
                
                # Note about using the iframe
                P(
                    "For full IPython functionality, use the terminal in the iframe below.",
                    style="margin-top: 1.5em; font-style: italic; color: #ccc;"
                ),
                
                style="background-color: #252526; padding: 1em; border-radius: 5px;"
            )
        
        # Use the ws_client component to create a terminal client
        terminal_client = ws_client(
            self.app, 
            'jupyter_terminal',
            port=self.port,
            ws_connect='/ws/jupyter_terminal',
            frame=False  # Don't automatically create an iframe - we'll use get_iframe_component
        )
        
        return terminal_client
        
    def get_iframe_component(self):
        """
        Get an iframe component for embedding the terminal.
        
        Returns:
            FastHTML HTMX component for the terminal iframe
        """
        return HTMX(
            '/jupyter_terminal',
            app=self.app,
            port=self.port,
            height="500px",
            link=True  # Show a link to open in a new tab
        )