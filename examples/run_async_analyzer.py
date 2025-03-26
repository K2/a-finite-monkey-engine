#!/usr/bin/env python3
"""
Run the async analyzer on Solidity files or projects.

This script provides a command-line interface for the async analyzer,
allowing users to analyze individual Solidity files or entire projects.

The analyzer leverages the nodes_config system for configurable parameters
and uses uv for package management.
"""

import os
import sys
import json
import asyncio
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from finite_monkey.core_async_analyzer import AsyncAnalyzer, ContractParser
from finite_monkey.adapters import Ollama
from finite_monkey.db.manager import DatabaseManager
from finite_monkey.nodes_config import nodes_config

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
            "psycopg2-binary"  # PostgreSQL synchronous driver for admin ops
        ]
        
        print(f"Checking for required packages using uv...")
        subprocess.run(["uv", "pip", "install", "--upgrade"] + required_packages, check=True)
        return True
    except Exception as e:
        print(f"Warning: Could not ensure packages with uv: {e}")
        print("Please install required packages manually if needed.")
        return False


async def main():
    """Main entry point for the async analyzer CLI."""
    # Ensure required packages are installed
    ensure_packages()
    
    # Load config for defaults
    config = nodes_config()
    
    parser = argparse.ArgumentParser(
        description="Finite Monkey Engine - Async Smart Contract Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single contract
  python run_async_analyzer.py -f examples/Vault.sol
  
  # Analyze a directory of contracts
  python run_async_analyzer.py -d examples/defi_project
  
  # Specify output directory for reports
  python run_async_analyzer.py -f examples/Vault.sol -o reports
  
  # Use specific models
  python run_async_analyzer.py -f examples/Vault.sol -m llama3:70b -v claude-3-sonnet-20240229
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
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up database manager if enabled
    db_manager = None
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
            
            print(f"Connecting to database: {db_url}")
            # Initialize database manager
            db_manager = DatabaseManager(db_url=db_url)
            
            # Create tables
            await db_manager.create_tables()
            print(f"Connected to database: {db_url}")
        except Exception as e:
            print(f"Warning: Failed to initialize database. Running in memory-only mode: {e}")
            db_manager = None
    
    # Set up LLM clients
    primary_model = args.model
    secondary_model = args.validator_model or primary_model
    
    try:
        # Use the base URL from config
        api_base = config.OPENAI_API_BASE or "http://localhost:11434"
        
        primary_llm = Ollama(model=primary_model, base_url=api_base)
        secondary_llm = Ollama(model=secondary_model, base_url=api_base)
        print(f"Using models: {primary_model} (primary) and {secondary_model} (validator)")
        print(f"Using API base URL: {api_base}")
    except Exception as e:
        print(f"Error setting up LLM clients: {e}")
        return 1
    
    # Initialize analyzer
    analyzer = AsyncAnalyzer(
        primary_llm_client=primary_llm,
        secondary_llm_client=secondary_llm,
        db_manager=db_manager,
        primary_model_name=primary_model,
        secondary_model_name=secondary_model,
    )
    
    # Determine project name
    project_name = args.project_name
    if not project_name:
        if args.file:
            project_name = Path(args.file).stem
        elif args.directory:
            project_name = Path(args.directory).name
    
    # Print banner
    print("=" * 70)
    print(f"Finite Monkey Engine - Async Smart Contract Analyzer")
    print("=" * 70)
    print(f"Project: {project_name}")
    if args.file:
        print(f"File: {args.file}")
    else:
        print(f"Directory: {args.directory}")
    print(f"Analysis model: {primary_model}")
    print(f"Validation model: {secondary_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Database enabled: {not args.no_db}")
    print("=" * 70)
    
    # Run analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = None
    
    try:
        if args.file:
            # Analyze single file
            print(f"Analyzing file: {args.file}")
            results = await analyzer.analyze_contract_file(
                file_path=args.file,
                project_id=project_name,
                query=args.query
            )
            
            # Format output filename
            output_file = f"{project_name}_results_{timestamp}.json"
            report_file = f"{project_name}_report_{timestamp}.md"
        else:
            # Analyze entire project
            print(f"Analyzing project directory: {args.directory}")
            results = await analyzer.analyze_project(
                project_path=args.directory,
                project_id=project_name,
                query=args.query
            )
            
            # Format output filename
            output_file = f"{project_name}_project_results_{timestamp}.json"
            report_file = f"{project_name}_project_report_{timestamp}.md"
        
        # Save results to file
        output_path = os.path.join(args.output_dir, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis results saved to: {output_path}")
        
        # Generate and save markdown report
        report_path = os.path.join(args.output_dir, report_file)
        await generate_markdown_report(results, report_path)
        
        print(f"Report saved to: {report_path}")
        
        # Print summary
        if "error" in results:
            print(f"Error: {results['error']}")
            return 1
        
        if args.file and "final_report" in results:
            report = results["final_report"]
            print(f"\nAnalysis complete for {results['file_id']}")
            print(f"Risk Assessment: {report['risk_assessment']}")
            print(f"Finding Count: {report['finding_count']}")
            
            print("\nSeverity Summary:")
            for severity, count in report.get("severity_summary", {}).items():
                print(f"  {severity}: {count}")
        
        elif "finding_count" in results:
            print(f"\nProject analysis complete")
            print(f"Total files analyzed: {results.get('file_count', 0)}")
            print(f"Total findings: {results.get('finding_count', 0)}")
            
            print("\nSeverity Summary:")
            for severity, count in results.get("severity_summary", {}).items():
                print(f"  {severity}: {count}")
        
        return 0
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


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
            
    else:
        # Project report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Project Security Audit Report: {results.get('project_id', 'Unknown')}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
            
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


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)