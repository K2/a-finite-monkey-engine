#!/usr/bin/env python3
"""
Run script for the Finite Monkey Web Interface
"""

import os
import sys
import asyncio
import argparse
import uvicorn
from typing import Optional

from finite_monkey.nodes_config import nodes_config


def main():
    """
    Main entry point for the web interface server
    """
    # Load config
    config = nodes_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Finite Monkey - Web Interface Server",
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
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print("Finite Monkey Engine - Web Interface")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    print("=" * 60)
    
    # Run with uvicorn
    uvicorn.run(
        "finite_monkey.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    

if __name__ == "__main__":
    sys.exit(main())