"""
Web interface for the Finite Monkey framework

This module provides a web interface for monitoring, configuration, and debugging
of the Finite Monkey framework.
"""

from pathlib import Path

# Set the template and static directories
TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"