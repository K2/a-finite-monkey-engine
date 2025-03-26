#!/usr/bin/env python3
"""
Standalone Jupyter terminal test server.

This script creates a simple FastHTML app with a built-in 
jupyter terminal page for testing.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from fasthtml.common import *
from fasthtml.fastapp import FastHTML
from fasthtml.jupyter import JupyUviAsync, HTMX
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("jupyter-standalone")

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TERM'] = 'xterm-256color'
os.environ['FORCE_COLOR'] = '1'

# Create a simple FastHTML app
app = FastHTML(title="Jupyter Standalone Test")

@app.get("/")
def index():
    return Div(
        H1("Jupyter Terminal Test"),
        P("This is a standalone test for the Jupyter terminal."),
        
        # Add a link to the jupyter terminal
        Div(
            A("Open Jupyter Terminal", href="/jupyter_terminal", cls="button"),
            style="margin: 20px 0;"
        ),
        
        # Add some styling
        Style("""
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        h1 {
            color: #4a2dbf;
        }
        
        .button {
            display: inline-block;
            background-color: #4a2dbf;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
        }
        
        .button:hover {
            background-color: #6549d5;
        }
        """)
    )

@app.get("/jupyter_terminal")
def jupyter_terminal():
    return Div(
        H1("Jupyter Terminal"),
        P("This is a standalone Jupyter terminal for testing."),
        
        # Link back to home
        Div(
            A("Back to Home", href="/", cls="button"),
            style="margin: 20px 0;"
        ),
        
        # Jupyter terminal container
        Div(
            id="jupyter-term",
            style="height: 400px; background-color: #1e1e1e; color: white; padding: 20px; border-radius: 5px; font-family: monospace; overflow: auto;"
        ),
        
        # Add some styling
        Style("""
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        h1 {
            color: #4a2dbf;
        }
        
        .button {
            display: inline-block;
            background-color: #4a2dbf;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .button:hover {
            background-color: #6549d5;
        }
        
        #jupyter-term {
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
        """)
    )

# Port configuration
PORT = 8889  # Use a different port to avoid conflicts

async def main():
    """Start the standalone Jupyter server."""
    logger.info(f"Starting standalone Jupyter server on port {PORT}...")
    logger.info(f"Open http://localhost:{PORT} in your browser")
    
    # Create the server
    config = uvicorn.Config(app=app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    
    # Create the JupyUvi integration for the existing server
    jupyter_server = JupyUviAsync(app, port=PORT, log_level="info")
    await jupyter_server.start()
    
    # Run the server
    await server.serve()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    project_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = Path(project_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler(logs_dir / "jupyter_standalone.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)