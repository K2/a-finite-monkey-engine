"""
Main CLI entry point for Finite Monkey Engine
"""

import sys
import argparse
import logging

from ..utils.logger import setup_logger, logger
from ..nodes_config import nodes_config

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Finite Monkey Engine - Smart Contract Analysis Framework"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version information"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to run"
    )
    
    # Add enhanced audit subcommand
    audit_parser = subparsers.add_parser(
        "audit",
        help="Run the enhanced audit pipeline"
    )
    audit_parser.add_argument(
        "--project-dir", "-p",
        type=str,
        help="Path to the project directory to analyze"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(level=log_level)
    
    # Show version information
    if args.version:
        from .. import __version__
        print(f"Finite Monkey Engine v{__version__}")
        return 0
    
    # Handle command
    if args.command == "audit":
        # Import and run the enhanced audit command
        from .enhanced_audit import run as run_audit
        return run_audit([
            "--project-dir", args.project_dir
        ] if args.project_dir else [])
    else:
        # No command specified, show help
        parser.print_help()
    
    return 0

def run():
    """Entry point for the CLI script"""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if logger.getEffectiveLevel() <= logging.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run()
