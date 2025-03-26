#!/usr/bin/env python3
"""
Run the enhanced analyzer with CallGraph integration on Solidity files or projects.

This script provides an improved analysis pipeline that leverages the ContractChunker
with CallGraph integration for a more comprehensive hierarchical analysis
of file->contract->function relationships.
"""

from loguru import logger
import os
import sys
import json
import asyncio
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from finite_monkey.enhanced_async_analyzer import EnhancedAsyncAnalyzer
from finite_monkey.adapters import Ollama
from finite_monkey.db.manager import DatabaseManager, TaskManager
from finite_monkey.nodes_config import nodes_config
from finite_monkey.agents import WorkflowOrchestrator
from finite_monkey.workflow import AgentController
from finite_monkey.visualization import GraphFactory


# Check for required packages using uv
def ensure_packages():
    """Ensure required packages are installed using uv."""
    try:
        # Check if uv is available
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        
        # Install required packages
        required_packages = [
            "tree-sitter", 
            "fastapi", 
            "uvicorn", 
            "sqlalchemy[asyncio]", 
            "asyncpg",  # PostgreSQL async driver
            "psycopg2-binary",  # PostgreSQL synchronous driver for admin ops
            "loguru",   # Better logging
            "rich"      # Rich terminal output
        ]
        
        print(f"Checking for required packages using uv...")
        subprocess.run(["uv", "pip", "install", "--upgrade"] + required_packages, check=True)
        return True
    except Exception as e:
        print(f"Warning: Could not ensure packages with uv: {e}")
        print("Please install required packages manually if needed.")
        return False


class AgentOrchestrator:
    """
    Orchestrates the workflow between high-level atomic agents and low-level LlamaIndex agents.
    
    This class provides a unified interface to the dual-layer agent architecture:
    - Inner core: LlamaIndex-powered agents for efficient code analysis
    - Outer layer: Atomic agents for orchestration, validation, and reporting
    """
    
    def __init__(
        self,
        primary_model: str,
        secondary_model: str,
        api_base: str,
        db_manager: Optional[DatabaseManager] = None,
        db_url: Optional[str] = None,
        base_dir: Optional[str] = None
    ):
        """Initialize the agent orchestrator."""
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.api_base = api_base
        self.db_manager = db_manager
        self.db_url = db_url
        self.base_dir = base_dir or os.getcwd()
        
        # Telemetry and metrics
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "files_processed": 0,
            "chunks_processed": 0,
            "findings_count": 0,
            "task_status": {}
        }
        
        # Initialize clients and components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize the agent components."""
        logger.debug("Initializing agent components...")
        
        # Initialize LLM clients
        self.primary_llm = Ollama(model=self.primary_model, base_url=self.api_base)
        self.secondary_llm = Ollama(model=self.secondary_model, base_url=self.api_base)
        
        # Initialize enhanced analyzer for low-level analysis
        self.analyzer = EnhancedAsyncAnalyzer(
            primary_llm_client=self.primary_llm,
            secondary_llm_client=self.secondary_llm,
            db_manager=self.db_manager,
            primary_model_name=self.primary_model,
            secondary_model_name=self.secondary_model,
        )
        
        # Initialize task manager if we have a database URL
        if self.db_url:
            logger.debug(f"Initializing TaskManager with DB: {self.db_url}")
            self.task_manager = TaskManager(db_url=self.db_url)
        else:
            logger.debug("No DB URL provided, running without TaskManager")
            self.task_manager = None
        
        # Initialize workflow orchestrator for high-level orchestration
        self.orchestrator = WorkflowOrchestrator(
            model_name=self.primary_model,
            validator_model=self.secondary_model,
            task_manager=self.task_manager,
            base_dir=self.base_dir,
            db_url=self.db_url,
        )
        
        # Initialize agent controller for task coordination
        self.agent_controller = AgentController(
            llm_client=self.primary_llm,
            model_name=self.primary_model
        )
        
    async def start_services(self):
        """Start required services like TaskManager."""
        if self.task_manager:
            logger.info("Starting TaskManager...")
            await self.task_manager.start()
    
    async def stop_services(self):
        """Stop all services."""
        if self.task_manager:
            logger.info("Stopping TaskManager...")
            await self.task_manager.stop()
    
    async def analyze_file_with_chunking(
        self,
        file_path: str,
        query: str,
        project_id: str,
        max_chunk_size: int = 4000,
        include_call_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single contract file with smart chunking.
        
        Args:
            file_path: Path to the file to analyze
            query: Analysis query
            project_id: Project identifier
            max_chunk_size: Maximum chunk size
            include_call_graph: Whether to include call graph analysis
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing file: {file_path}")
        self.metrics["start_time"] = datetime.now().isoformat()
        
        # Step 1: Enhanced analysis with chunking
        results = await self.analyzer.analyze_contract_file_with_chunking(
            file_path=file_path,
            project_id=project_id,
            query=query,
            max_chunk_size=max_chunk_size,
            include_call_graph=include_call_graph
        )
        
        self.metrics["files_processed"] += 1
        if "chunk_count" in results:
            self.metrics["chunks_processed"] += results["chunk_count"]
        
        # Step 2: High-level validation with atomic agents
        logger.info("Running high-level validation with atomic agents")
        
        # Extract the content and findings for validation
        with open(file_path, "r", encoding="utf-8") as f:
            code_content = f.read()
        
        # Use orchestrator to validate findings
        validation_prompt = await self.agent_controller.generate_agent_prompt(
            agent_type="validator",
            task=f"Validate the security analysis for {os.path.basename(file_path)}",
            context=f"Code:\n```solidity\n{code_content}\n```\n\nAnalysis Results:\n{json.dumps(results.get('final_report', {}), indent=2)}"
        )
        
        validation_response = await self.secondary_llm.acomplete(
            prompt=validation_prompt
        )
        
        # Update the results with validation feedback
        results["validation"] = validation_response
        
        # Generate final report with documentor agent
        documentor_prompt = await self.agent_controller.generate_agent_prompt(
            agent_type="documentor",
            task=f"Create a comprehensive security report for {os.path.basename(file_path)}",
            context=(
                f"Code:\n```solidity\n{code_content}\n```\n\n"
                f"Analysis Results:\n{json.dumps(results.get('final_report', {}), indent=2)}\n\n"
                f"Validation:\n{validation_response}"
            )
        )
        
        report_text = await self.primary_llm.acomplete(
            prompt=documentor_prompt
        )
        
        # Add report to results
        results["final_report_text"] = report_text
        
        # Update metrics
        self.metrics["end_time"] = datetime.now().isoformat()
        if "final_report" in results and "findings" in results["final_report"]:
            self.metrics["findings_count"] += len(results["final_report"]["findings"])
        
        return results
        
    async def analyze_project_with_chunking(
        self,
        project_path: str,
        query: str,
        project_id: str,
        max_chunk_size: int = 4000,
        include_call_graph: bool = True,
        file_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze an entire project with smart chunking.
        
        Args:
            project_path: Path to the project directory
            query: Analysis query
            project_id: Project identifier
            max_chunk_size: Maximum chunk size
            include_call_graph: Whether to include call graph analysis
            file_limit: Maximum number of files to analyze
            
        Returns:
            Project analysis results dictionary
        """
        logger.info(f"Analyzing project: {project_path}")
        self.metrics["start_time"] = datetime.now().isoformat()
        
        # Step 1: Enhanced project analysis with chunking
        results = await self.analyzer.analyze_project_with_chunking(
            project_path=project_path,
            project_id=project_id,
            query=query,
            max_chunk_size=max_chunk_size,
            include_call_graph=include_call_graph,
            #file_limit=file_limit
            
        )
        
        self.metrics["files_processed"] += results.get("file_count", 0)
        self.metrics["chunks_processed"] += results.get("total_chunks", 0)
        
        # Use the WorkflowOrchestrator to provide a high-level summary and analysis
        # This connects the enhanced analysis with the atomic agent framework
        if self.task_manager:
            logger.info("Running high-level workflow with atomic agents")
            
            # Get all solidity files in the project
            import glob
            solidity_files = glob.glob(os.path.join(project_path, "**", "*.sol"), recursive=True)
            
            if solidity_files:
                # Run async workflow with task manager for tracking
                workflow_results = await self.orchestrator.run_audit_workflow(
                    solidity_paths=solidity_files[:file_limit] if file_limit else solidity_files,
                    query=query,
                    project_name=project_id,
                    wait_for_completion=True
                )
                
                # Add workflow results to our results
                results["workflow_results"] = workflow_results.metadata
        
        # Update metrics
        self.metrics["end_time"] = datetime.now().isoformat()
        if "finding_count" in results:
            self.metrics["findings_count"] += results["finding_count"]
        
        return results


async def generate_markdown_report(results, output_path):
    """
    Generate a detailed Markdown report from analysis results.
    
    Args:
        results: Analysis results
        output_path: Path to save the report
    """
    if "file_id" in results and "final_report" in results:
        # Single file report
        report = results["final_report"]
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Security Audit Report: {results['file_id']}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
            f.write(f"**Risk Assessment:** {report['risk_assessment']}\n\n")
            
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


async def main():
    """Main entry point for the enhanced analyzer CLI."""
    # Set up logging
    setup_logger()
    
    # Ensure required packages are installed
    ensure_packages()
    
    # Load config for defaults
    config = nodes_config()
    
    parser = argparse.ArgumentParser(
        description="Finite Monkey Engine - Enhanced Smart Contract Analyzer with CallGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single contract
  python run_enhanced_chunking_audit.py -f examples/SimpleVault.sol
  
  # Analyze a directory of contracts
  python run_enhanced_chunking_audit.py -d examples/src
  
  # Specify output directory for reports
  python run_enhanced_chunking_audit.py -f examples/SimpleVault.sol -o reports
  
  # Use specific models
  python run_enhanced_chunking_audit.py -f examples/SimpleVault.sol -m llama3:70b -v claude-3-sonnet-20240229
  
  # Disable call graph integration (faster for simple contracts)
  python run_enhanced_chunking_audit.py -f examples/SimpleVault.sol --no-call-graph
        """
    )
    
    # File or directory input
    #input_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "-f", "--file",
        help="Path to a Solidity file to analyze"
    )
    parser.add_argument(
        "-d", "--directory",
        default="examples/src",
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
    
    # Additional options
    parser.add_argument(
        "-o", "--output-dir",
        default=config.output or "reports",
        help=f"Directory to store analysis reports (default: {config.output or 'reports'})"
    )
    parser.add_argument(
        "-m", "--model",
        default=config.SCAN_MODEL or "llama3:8b-instruct-q6_K",
        help=f"LLM model to use for analysis (default: {config.SCAN_MODEL or 'llama3:8b-instruct-q6_K'})"
    )
    parser.add_argument(
        "-v", "--validator-model",
        default=config.CONFIRMATION_MODEL or None,
        help=f"LLM model to use for validation (default: {config.CONFIRMATION_MODEL or 'same as analysis model'})"
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
    # parser.add_argument(
    #     "--verbose", "-v",
    #     action="store_true",
    #     help="Enable verbose logging"
    # )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Create the agent orchestrator
    primary_model = args.model
    secondary_model = args.validator_model or primary_model
    
    try:
        # Initialize orchestrator
        logger.info("Initializing agent orchestrator...")
        orchestrator = AgentOrchestrator(
            primary_model=primary_model,
            secondary_model=secondary_model,
            api_base=api_base,
            db_manager=db_manager,
            db_url=db_url,
            base_dir=os.getcwd()
        )
        
        # Start services
        await orchestrator.start_services()
        
        # Determine project name
        project_name = args.project_name
        if not project_name:
            if args.file:
                project_name = Path(args.file).stem
            elif args.directory:
                project_name = Path(args.directory).name
        
        # Print banner
        logger.info("=" * 70)
        logger.info(f"Finite Monkey Engine - Enhanced Smart Contract Analyzer")
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
        
        # Run analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = None
        
        if args.file:
            # Analyze single file
            logger.info(f"Analyzing file: {args.file}")
            results = await orchestrator.analyze_file_with_chunking(
                file_path=args.file,
                query=args.query,
                project_id=project_name,
                max_chunk_size=args.chunk_size,
                include_call_graph=not args.no_call_graph
            )
            
            # Format output filename
            output_file = f"{project_name}_results_{timestamp}.json"
            report_file = f"{project_name}_report_{timestamp}.md"
            
            # Generate visualization
            try:
                logger.info("Generating contract visualization...")
                graph_file = os.path.join(args.output_dir, f"{project_name}_graph_{timestamp}.html")
                graph = GraphFactory.analyze_solidity_file(args.file)
                graph.export_html(graph_file)
                logger.success(f"Visualization saved to: {graph_file}")
            except Exception as e:
                logger.error(f"Failed to generate visualization: {e}")
            
        else:
            # Analyze entire project
            logger.info(f"Analyzing project directory: {args.directory}")
            results = await orchestrator.analyze_project_with_chunking(
                project_path=args.directory,
                query=args.query,
                project_id=project_name,
                max_chunk_size=args.chunk_size,
                include_call_graph=not args.no_call_graph,
                file_limit=args.limit
            )
            
            # Format output filename
            output_file = f"{project_name}_project_results_{timestamp}.json"
            report_file = f"{project_name}_project_report_{timestamp}.md"
        
        # Save results to file
        output_path = os.path.join(args.output_dir, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        logger.success(f"Analysis results saved to: {output_path}")
        
        # Generate and save markdown report
        report_path = os.path.join(args.output_dir, report_file)
        await generate_markdown_report(results, report_path)
        
        logger.success(f"Report saved to: {report_path}")
        
        # Print summary
        if "error" in results:
            logger.error(f"Error: {results['error']}")
            return 1
        
        if args.file and "final_report" in results:
            report = results["final_report"]
            logger.info(f"\nAnalysis complete for {results.get('file_id', 'unknown')}")
            logger.info(f"Risk Assessment: {report.get('risk_assessment', 'Unknown')}")
            logger.info(f"Finding Count: {report.get('finding_count', 0)}")
            
            logger.info("\nSeverity Summary:")
            for severity, count in report.get("severity_summary", {}).items():
                logger.info(f"  {severity}: {count}")
            
            # Show chunk information if available
            if "chunk_count" in results:
                logger.info(f"Contract chunked into {results['chunk_count']} segments")
        
        elif "finding_count" in results:
            logger.info(f"\nProject analysis complete")
            logger.info(f"Total files analyzed: {results.get('file_count', 0)}")
            logger.info(f"Total findings: {results.get('finding_count', 0)}")
            
            logger.info("\nSeverity Summary:")
            for severity, count in results.get("severity_summary", {}).items():
                logger.info(f"  {severity}: {count}")
        
        # Stop services
        await orchestrator.stop_services()
        return 0
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Set up logging
    logger.remove(0)
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        sys.exit(1)