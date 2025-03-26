"""
Command-line interface for enhanced audit pipeline
"""

import os
import sys
import asyncio
import argparse
from typing import Optional, List
from pathlib import Path

from ..nodes_config import nodes_config
from ..pipeline.enhanced_audit import create_enhanced_audit_pipeline
from ..utils.logger import setup_logger, logger

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Smart Contract Audit Pipeline"
    )
    
    parser.add_argument(
        "--project-dir", "-p",
        type=str,
        help="Path to the project directory to analyze (default: current directory)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for reports (default: ./reports)"
    )
    
    parser.add_argument(
        "--scan-model",
        type=str,
        help="Model to use for primary scanning"
    )
    
    parser.add_argument(
        "--confirmation-model",
        type=str,
        help="Model to use for validation and confirmation"
    )
    
    parser.add_argument(
        "--threads", "-t",
        type=int,
        help="Maximum number of concurrent analysis threads"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Custom audit query to focus the analysis"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def update_config_from_args(args: argparse.Namespace) -> None:
    """Update configuration from command line arguments"""
    config = nodes_config()
    
    # Update config values from args if provided
    if args.project_dir:
        config.base_dir = args.project_dir
    
    if args.output_dir:
        config.output = args.output_dir
    
    if args.scan_model:
        config.SCAN_MODEL = args.scan_model
    
    if args.confirmation_model:
        config.CONFIRMATION_MODEL = args.confirmation_model
    
    if args.threads:
        config.MAX_THREADS_OF_SCAN = args.threads
    
    if args.query:
        config.USER_QUERY = args.query

async def main() -> int:
    """Main entry point for enhanced audit CLI"""
    args = parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(level=log_level)
    
    # Update configuration from args
    update_config_from_args(args)
    config = nodes_config()
    
    project_dir = Path(config.base_dir).resolve()
    
    # Validate project directory
    if not project_dir.exists():
        logger.error(f"Project directory does not exist: {project_dir}")
        return 1
    
    logger.info(f"Starting enhanced audit for project: {project_dir}")
    logger.info(f"Configuration: scan_model={config.SCAN_MODEL}, "
                f"confirmation_model={config.CONFIRMATION_MODEL}, "
                f"threads={config.MAX_THREADS_OF_SCAN}")
    
    try:
        # Run the enhanced audit pipeline
        results = await create_enhanced_audit_pipeline(str(project_dir))
        
        # Output report paths
        if "report_results" in results:
            html_report = results["report_results"].get("html_report")
            md_report = results["report_results"].get("markdown_report")
            
            logger.info(f"Analysis complete. Reports generated at:")
            if html_report:
                logger.info(f"HTML Report: {html_report}")
            if md_report:
                logger.info(f"Markdown Report: {md_report}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Enhanced audit pipeline failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

def run() -> None:
    """Entry point for the CLI script"""
    sys.exit(asyncio.run(main()))

if __name__ == "__main__":
    run()
