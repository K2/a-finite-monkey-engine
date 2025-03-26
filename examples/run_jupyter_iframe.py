#!/usr/bin/env python3
"""
Run the Finite Monkey Engine FastHTML web interface with Jupyter integration.

This script creates a FastHTML app with the Jupyter terminal embedded in an iframe
using FastHTML's JupyUviAsync to properly handle the Jupyter kernel.
"""

import os
import sys
import asyncio
from fasthtml.common import *
from fasthtml.fastapp import FastHTML
from fasthtml.jupyter import JupyUviAsync, HTMX
import uvicorn

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TERM'] = 'xterm-256color'
os.environ['FORCE_COLOR'] = '1'

# Import FastHTML app from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finite_monkey.web.fasthtml_app import asgi_app

# Port configuration
PORT = 8888

async def main():
    """Start the FastHTML web app with Jupyter integration."""
    print("Starting FastHTML web interface with Jupyter integration...")
    print(f"Open http://localhost:{PORT} in your browser")
    
    # Create the Jupyter server for iframe integration
    jupyter_server = JupyUviAsync(asgi_app, port=PORT, log_level="info")
    await jupyter_server.start()
    
    try:
        # Keep the script running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        jupyter_server.stop()
        print("Server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
