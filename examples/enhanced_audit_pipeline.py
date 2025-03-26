#!/usr/bin/env python3
"""
Enhanced Audit Pipeline for Finite Monkey Engine

This script implements a pipeline-based approach to smart contract auditing,
integrating enhanced chunking with the existing pipeline infrastructure.
It provides a command-line interface for running security audits on Solidity files.
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import glob

# Import core components
from finite_monkey.pipeline.base import Pipeline, PipelineStep
from finite_monkey.pipeline.executor import PipelineExecutor
from finite_monkey.core_async_analyzer import AsyncAnalyzer
from finite_monkey.db.manager import DatabaseManager
from finite_monkey.nodes_config import nodes_config
from finite_monkey.utils.logger import setup_logger, logger
from finite_monkey.utils.package_utils import ensure_packages
from finite_monkey.visualizer.graph_factory import GraphFactory

# Define pipeline steps for enhanced chunking audit
class ProjectPreparationStep(PipelineStep):
    """Initialize project and analyze structure for optimal chunking"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Preparing project: {context.get('project_id', 'Unknown')}")
        
        # Determine if we're analyzing a file or directory
        if context.get("file_path"):
            context["is_single_file"] = True
            if not os.path.exists(context["file_path"]):
                raise FileNotFoundError(f"File not found: {context['file_path']}")
            
            # Extract solidity files (just the one file in this case)
            context["solidity_files"] = [context["file_path"]]
        else:
            context["is_single_file"] = False
            dir_path = context.get("project_path")
            if not dir_path or not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            
            # Get all solidity files
            context["solidity_files"] = glob.glob(
                os.path.join(dir_path, "**", "*.sol"), 
                recursive=True
            )
            
            # Apply file limit if specified
            if context.get("file_limit") and len(context["solidity_files"]) > context["file_limit"]:
                context["solidity_files"] = context["solidity_files"][:context["file_limit"]]
                
        logger.info(f"Found {len(context['solidity_files'])} Solidity files to analyze")
        
        # Create output directory if needed
        os.makedirs(context["output_dir"], exist_ok=True)
        
        return context


class EnhancedChunkingAnalysisStep(PipelineStep):
    """Analyze contracts using enhanced chunking strategies"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        analyzer = context["analyzer"]
        
        if context["is_single_file"]:
            # Single file analysis with chunking
            logger.info(f"Analyzing file with enhanced chunking: {context['file_path']}")
            results = await analyzer.analyze_file_with_chunking(
                file_path=context["file_path"],
                query=context["query"],
                project_id=context["project_id"],
                max_chunk_size=context["chunk_size"],
                include_call_graph=context["include_call_graph"]
            )
            context["results"] = results
        else:
            # Project analysis with chunking
            logger.info(f"Analyzing project with enhanced chunking: {context['project_path']}")
            results = await analyzer.analyze_project_with_chunking(
                project_path=context["project_path"],
                query=context["query"],
                project_id=context["project_id"],
                max_chunk_size=context["chunk_size"],
                include_call_graph=context["include_call_graph"],
                file_limit=context.get("file_limit")
            )
            context["results"] = results
            
        return context


class ValidationStep(PipelineStep):
    """Validate findings using secondary analysis"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results = context["results"]
        
        # Check if we need to perform validation
        if context.get("skip_validation"):
            logger.info("Validation step skipped")
            return context
        
        logger.info("Validating findings with secondary analysis")
        
        # Here we'd interface with the existing validation framework
        # This could be through the analyzer's validation methods
        # or through a dedicated validation service
        
        if context["is_single_file"] and "final_report" in results:
            # Example of validating a single file's findings
            findings = results["final_report"].get("findings", [])
            
            if findings:
                logger.info(f"Validating {len(findings)} findings")
                
                # Here you would connect to your existing validation infrastructure
                # For now, we'll add a placeholder indicating validation status
                for finding in findings:
                    if "confirmation_status" not in finding:
                        finding["confirmation_status"] = "Pending validation"
        
        return context


class VisualizationStep(PipelineStep):
    """Generate visualizations for contract analysis"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Skip if visualization is disabled
        if context.get("skip_visualization"):
            return context
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if context["is_single_file"]:
            # Generate visualization for single file
            try:
                logger.info("Generating contract visualization...")
                graph_file = os.path.join(
                    context["output_dir"], 
                    f"{context['project_id']}_graph_{timestamp}.html"
                )
                graph = GraphFactory.analyze_solidity_file(context["file_path"])
                graph.export_html(graph_file)
                logger.success(f"Visualization saved to: {graph_file}")
                context["visualization_file"] = graph_file
            except Exception as e:
                logger.error(f"Failed to generate visualization: {e}")
        else:
            # Generate project-level visualization 
            try:
                logger.info("Generating project visualization...")
                graph_file = os.path.join(
                    context["output_dir"], 
                    f"{context['project_id']}_project_graph_{timestamp}.html"
                )
                
                # Create a combined graph for all solidity files
                # Assuming GraphFactory has or could have a method for multiple files
                # If not, we could iterate through files and combine results
                combined_graph = GraphFactory.analyze_solidity_directory(context["project_path"])
                combined_graph.export_html(graph_file)
                
                logger.success(f"Project visualization saved to: {graph_file}")
                context["visualization_file"] = graph_file
            except Exception as e:
                logger.error(f"Failed to generate project visualization: {e}")
                
        return context


class ReportGenerationStep(PipelineStep):
    """Generate comprehensive audit reports"""
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        results = context["results"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw JSON results
        output_file = os.path.join(
            context["output_dir"], 
            f"{context['project_id']}_results_{timestamp}.json"
        )
        
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            # Handle non-serializable objects
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=2)
        
        logger.success(f"Raw results saved to: {output_file}")
        context["result_file"] = output_file
        
        # Generate markdown report
        report_file = os.path.join(
            context["output_dir"], 
            f"{context['project_id']}_report_{timestamp}.md"
        )
        
        await self._generate_markdown_report(results, report_file)
        logger.success(f"Report saved to: {report_file}")
        context["report_file"] = report_file
        
        return context
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization by handling non-serializable types"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_json(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._prepare_for_json(obj.__dict__)
        else:
            # Convert other types to string
            return str(obj)
            
    async def _generate_markdown_report(self, results, output_path):
        """Generate a detailed Markdown report from analysis results."""
        if "file_id" in results and "final_report" in results:
            # Single file report
            report = results["final_report"]
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Security Audit Report: {results['file_id']}\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
                f.write(f"**Risk Assessment:** {report.get('risk_assessment', 'Not provided')}\n\n")
                
                # Note enhanced analysis if used
                if results.get("enhanced_chunking"):
                    f.write(f"**Analysis Method:** Enhanced hierarchical analysis with CallGraph\n\n")
                
                # Add validation info if present
                if "validation" in results:
                    f.write(f"## Validation Summary\n\n")
                    validation_summary = results["validation"][:500] + "..." if len(results["validation"]) > 500 else results["validation"]
                    f.write(f"{validation_summary}\n\n")
                
                f.write(f"## Summary\n\n")
                f.write(f"{report.get('summary', 'No summary available.')}\n\n")
                
                f.write(f"## Severity Distribution\n\n")
                f.write("| Severity | Count |\n")
                f.write("|----------|-------|\n")
                for severity in ["Critical", "High", "Medium", "Low", "Informational"]:
                    count = report.get("severity_summary", {}).get(severity, 0)
                    f.write(f"| {severity} | {count} |\n")
                
                f.write(f"\n## Findings\n\n")
                for i, finding in enumerate(report.get("findings", []), 1):
                    f.write(f"### {i}. {finding['title']}\n\n")
                    f.write(f"**Severity:** {finding['severity']}\n\n")
                    f.write(f"**Status:** {finding.get('confirmation_status', 'Not validated')}\n\n")
                    f.write(f"**Description:**\n\n{finding.get('description', 'No description available.')}\n\n")
                    
                    if finding.get("location"):
                        f.write(f"**Location:**\n\n{finding['location']}\n\n")
                    
                    if finding.get("impact"):
                        f.write(f"**Impact:**\n\n{finding['impact']}\n\n")
                    
                    if finding.get("recommendation"):
                        f.write(f"**Recommendation:**\n\n{finding['recommendation']}\n\n")
                    
                    if finding.get("validation_notes"):
                        f.write(f"**Validation Notes:**\n\n{finding['validation_notes']}\n\n")
                    
                    f.write("---\n\n")
                
                f.write(f"## Conclusion\n\n")
                f.write(f"{report.get('risk_text', 'No conclusion available.')}\n\n")
                
                # Add full final report if present
                if "final_report_text" in results:
                    f.write(f"## Full Analysis Report\n\n")
                    f.write(results["final_report_text"])
                    
        else:
            # Project report
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Project Security Audit Report: {results.get('project_id', 'Unknown')}\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
                
                # Note enhanced analysis if used
                if results.get("enhanced_chunking"):
                    f.write(f"**Analysis Method:** Enhanced hierarchical analysis with CallGraph\n\n")
                
                f.write(f"## Project Overview\n\n")
                f.write(f"- **Files Analyzed:** {results.get('file_count', 0)}\n")
                f.write(f"- **Total Findings:** {results.get('finding_count', 0)}\n\n")
                
                f.write(f"## Severity Distribution\n\n")
                f.write("| Severity | Count |\n")
                f.write("|----------|-------|\n")
                for severity in ["Critical", "High", "Medium", "Low", "Informational"]:
                    count = results.get("severity_summary", {}).get(severity, 0)
                    f.write(f"| {severity} | {count} |\n")
                
                f.write(f"\n## Findings\n\n")
                for i, finding in enumerate(results.get("findings", []), 1):
                    f.write(f"### {i}. {finding['title']} ({finding.get('file', 'Unknown')})\n\n")
                    f.write(f"**Severity:** {finding['severity']}\n\n")
                    f.write(f"**Status:** {finding.get('confirmation_status', 'Not validated')}\n\n")
                    f.write(f"**Description:**\n\n{finding.get('description', 'No description available.')}\n\n")
                    
                    if finding.get("location"):
                        f.write(f"**Location:**\n\n{finding['location']}\n\n")
                    
                    if finding.get("impact"):
                        f.write(f"**Impact:**\n\n{finding['impact']}\n\n")
                    
                    if finding.get("recommendation"):
                        f.write(f"**Recommendation:**\n\n{finding['recommendation']}\n\n")
                    
                    if finding.get("validation_notes"):
                        f.write(f"**Validation Notes:**\n\n{finding['validation_notes']}\n\n")
                    
                    f.write("---\n\n")
                
                # Add file details
                f.write(f"## Files Analyzed\n\n")
                for file_name in results.get("files", {}).keys():
                    f.write(f"- {file_name}\n")


# Define the enhanced audit pipeline
class EnhancedAuditPipeline(Pipeline):
    """Pipeline for enhanced smart contract auditing with chunking"""
    
    def __init__(self, context: Dict[str, Any]):
        super().__init__(
            name="Enhanced Audit Pipeline",
            steps=[
                ProjectPreparationStep(name="Project Preparation"),
                EnhancedChunkingAnalysisStep(name="Enhanced Chunking Analysis"),
                ValidationStep(name="Findings Validation"),
                VisualizationStep(name="Contract Visualization"),
                ReportGenerationStep(name="Report Generation")
            ],
            initial_context=context
        )


# Main application function
async def main():
    """Main entry point for the enhanced analyzer pipeline CLI."""
    # Set up logging
    setup_logger()
    
    # Ensure required packages are installed
    ensure_packages()
    
    # Load config for defaults
    config = nodes_config()
    
    parser = argparse.ArgumentParser(
        description="Finite Monkey Engine - Enhanced Smart Contract Analyzer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single contract
  python enhanced_audit_pipeline.py -f examples/SimpleVault.sol
  
  # Analyze a directory of contracts
  python enhanced_audit_pipeline.py -d examples/src
  
  # Specify output directory for reports
  python enhanced_audit_pipeline.py -f examples/SimpleVault.sol -o reports
  
  # Use specific models
  python enhanced_audit_pipeline.py -f examples/SimpleVault.sol -m llama3:70b -v claude-3-sonnet-20240229
  
  # Disable call graph integration (faster for simple contracts)
  python enhanced_audit_pipeline.py -f examples/SimpleVault.sol --no-call-graph
        """
    )
    
    # File or directory input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-f", "--file",
        help="Path to a Solidity file to analyze"
    )
    input_group.add_argument(
        "-d", "--directory",
        help="Path to a directory containing Solidity files to analyze"
    )
    
    # Analysis options
    parser.add_argument(
        "--no-call-graph",
        action="store_true",
        help="Disable call graph integration (faster but less comprehensive)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4000,
        help="Maximum chunk size in characters (default: 4000)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of files to analyze (for large projects)"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip the validation step"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip visualization generation"
    )
    
    # Additional options
    parser.add_argument(
        "-o", "--output-dir",
        default=config.output or "reports",
        help=f"Directory to store analysis reports (default: {config.output or 'reports'})"
    )
    parser.add_argument(
        "-m", "--model",
        default=config.SCAN_MODEL,
        help=f"LLM model to use for analysis (default: {config.SCAN_MODEL})"
    )
    parser.add_argument(
        "-v", "--validator-model",
        default=config.CONFIRMATION_MODEL,
        help=f"LLM model to use for validation (default: {config.CONFIRMATION_MODEL})"
    )
    parser.add_argument(
        "-q", "--query",
        default="Perform a comprehensive security audit",
        help="Analysis query (e.g., 'Check for reentrancy vulnerabilities')"
    )
    parser.add_argument(
        "-n", "--project-name",
        help="Project name (defaults to directory or file name)"
    )
    parser.add_argument(
        "--db-url",
        help="Database URL (defaults to config or SQLite)"
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable database persistence (use in-memory only)"
    )
    
    args = parser.parse_args()
    
    # Set up database manager if enabled
    db_manager = None
    db_url = None
    if not args.no_db:
        try:
            # Try to connect to database
            db_url = args.db_url
            if not db_url:
                # Try to get from config, prioritizing PostgreSQL
                if config.DATABASE_URL and "postgresql" in config.DATABASE_URL:
                    # Convert standard PostgreSQL URL to async version
                    db_url = config.DATABASE_URL
                    if "postgresql:" in db_url and "postgresql+asyncpg:" not in db_url:
                        db_url = db_url.replace("postgresql:", "postgresql+asyncpg:")
                        
                elif config.ASYNC_DB_URL and "postgresql" in config.ASYNC_DB_URL:
                    db_url = config.ASYNC_DB_URL
                elif config.DATABASE_URL:  # Fallback to any DATABASE_URL with asyncpg
                    db_url = config.DATABASE_URL
                    # Ensure async driver
                    if "postgresql:" in db_url and "postgresql+asyncpg:" not in db_url:
                        db_url = db_url.replace("postgresql:", "postgresql+asyncpg:")
                else:
                    # Default PostgreSQL connection with async driver
                    db_url = "postgresql+asyncpg://postgres:1234@127.0.0.1:5432/postgres"
            
            logger.info(f"Connecting to database: {db_url}")
            # Initialize database manager
            db_manager = DatabaseManager(db_url=db_url)
            
            # Create tables
            await db_manager.create_tables()
            logger.success(f"Connected to database: {db_url}")
        except Exception as e:
            logger.warning(f"Failed to initialize database. Running in memory-only mode: {e}")
            db_manager = None
            db_url = None
    
    # Set up API base from config
    api_base = config.OPENAI_API_BASE or "http://localhost:11434"
    
    # Determine project name
    project_name = args.project_name
    if not project_name:
        if args.file:
            project_name = Path(args.file).stem
        elif args.directory:
            project_name = Path(args.directory).name
    
    # Create analyzer
    primary_model = args.model
    secondary_model = args.validator_model or primary_model
    
    # Initialize analyzer
    analyzer = AsyncAnalyzer(
        primary_model_name=primary_model,
        secondary_model_name=secondary_model,
        db_manager=db_manager
    )
    
    # Print banner
    logger.info("=" * 70)
    logger.info(f"Finite Monkey Engine - Enhanced Smart Contract Analyzer Pipeline")
    logger.info("=" * 70)
    logger.info(f"Project: {project_name}")
    if args.file:
        logger.info(f"File: {args.file}")
    else:
        logger.info(f"Directory: {args.directory}")
    logger.info(f"Analysis model: {primary_model}")
    logger.info(f"Validation model: {secondary_model}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Database enabled: {db_manager is not None}")
    logger.info(f"Call graph integration: {not args.no_call_graph}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info("=" * 70)
    
    # Setup pipeline context
    context = {
        "project_id": project_name,
        "file_path": args.file if args.file else None,
        "project_path": args.directory if args.directory else None,
        "output_dir": args.output_dir,
        "query": args.query,
        "chunk_size": args.chunk_size,
        "include_call_graph": not args.no_call_graph,
        "skip_validation": args.no_validation,
        "skip_visualization": args.no_visualization,
        "file_limit": args.limit,
        "analyzer": analyzer,
        "db_manager": db_manager,
        "primary_model": primary_model,
        "secondary_model": secondary_model,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Create and run the pipeline
    pipeline = EnhancedAuditPipeline(context)
    executor = PipelineExecutor()
    
    try:
        # Execute the pipeline
        logger.info("Starting enhanced audit pipeline")
        result_context = await executor.execute(pipeline)
        
        # Display results
        if "report_file" in result_context:
            logger.success(f"Pipeline completed successfully.")
            logger.success(f"Report saved to: {result_context['report_file']}")
            
            # Open report if on a system with a supported browser
            if os.name == 'posix' and sys.platform != 'darwin':
                try:
                    os.system(f"xdg-open {result_context['report_file']}")
                except:
                    pass
            elif sys.platform == 'darwin':  # macOS
                os.system(f"open {result_context['report_file']}")
            elif os.name == 'nt':  # Windows
                os.system(f"start {result_context['report_file']}")
        else:
            logger.error("Pipeline did not generate a report file.")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    import traceback
    
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
