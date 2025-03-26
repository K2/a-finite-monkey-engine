"""
Jupyter kernel integration for the Finite Monkey Engine.

This module provides an IPython kernel that can be embedded in the FastHTML
web interface using FastHTML's JupyUviAsync.
"""

import os
import sys
import asyncio
from IPython import embed_kernel
from ipykernel.kernelapp import IPKernelApp
from traitlets.config import Config

def create_kernel(namespace=None):
    """
    Create an IPython kernel with the specified namespace.
    
    Args:
        namespace: Dictionary of objects to include in the kernel namespace
    
    Returns:
        The kernel application instance
    """
    # Configure the kernel
    c = Config()
    c.IPKernelApp.matplotlib = 'inline'
    c.IPKernelApp.pylab = 'inline'
    c.ZMQInteractiveShell.colors = 'Linux'
    c.ZMQInteractiveShell.confirm_exit = False
    c.ZMQInteractiveShell.pdb = False
    
    # Set environment variables for better color support
    os.environ['TERM'] = 'xterm-256color'
    os.environ['FORCE_COLOR'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Initialize the kernel with the namespace
    if namespace is None:
        namespace = {}
    
    # Add helper functions and modules to the namespace
    namespace.update({
        'asyncio': asyncio,
        'os': os,
        'sys': sys,
    })
    
    # Start the kernel
    kernel = IPKernelApp.instance(user_ns=namespace)
    kernel.initialize([])
    
    return kernel

async def start_kernel(namespace=None):
    """
    Start an IPython kernel in a background thread.
    
    Args:
        namespace: Dictionary of objects to include in the kernel namespace
    """
    kernel = create_kernel(namespace)
    
    # Start the kernel in a background thread
    import threading
    kernel_thread = threading.Thread(target=kernel.start, daemon=True)
    kernel_thread.start()
    
    # Wait for the kernel to be ready
    while not kernel.shell:
        await asyncio.sleep(0.1)
    
    # Return connection info
    return kernel.connection_file

if __name__ == "__main__":
    # When run directly, this creates a standalone kernel
    asyncio.run(start_kernel())