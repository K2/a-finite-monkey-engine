#!/usr/bin/env python3
"""
Simple script to test the Jupyter terminal implementation directly.
"""

import asyncio
import sys
import uvicorn
from finite_monkey.web.fasthtml_app import asgi_app

async def main():
    """Run the FastHTML web server."""
    config = uvicorn.Config(
        app=asgi_app, 
        host="0.0.0.0", 
        port=8888, 
        reload=True,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped")
        sys.exit(0)
