#!/usr/bin/env python3
"""
Command-line interface for the Finite Monkey framework
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from nodes_config import nodes_config

async def main():
    """
    Main entry point for the command-line interface
    """
    # Import here to avoid circular imports
    from agents import WorkflowOrchestrator
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Finite Monkey - Smart Contract Audit & Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single contract
  python -m finite_monkey analyze -f examples/Vault.sol
  
  # Start the web interface
  python -m finite_monkey web
  
  # Show help for a specific command
  python -m finite_monkey web --help
        """
    )
    
    # Set up subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze smart contracts for security vulnerabilities")
    
    # Required arguments - file or directory
    file_group = analyze_parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument(
        "-f", "--file",
        help="Path to a single Solidity file to audit",
    )
    file_group.add_argument(
        "-d", "--directory",
        help="Path to a directory containing Solidity files to audit",
    )
    file_group.add_argument(
        "--files",
        nargs="+",
        help="Multiple Solidity files to audit (space-separated)",
    )
    
    # Optional arguments
    analyze_parser.add_argument(
        "-q", "--query",
        default="Perform a comprehensive security audit",
        help="Audit query (e.g., 'Check for reentrancy vulnerabilities')",
    )
    
    analyze_parser.add_argument(
        "--pattern",
        default="*.sol",
        help="Glob pattern for Solidity files when using --directory (default: *.sol)",
    )
    
    analyze_parser.add_argument(
        "-n", "--name",
        default="default",
        help="Project name (defaults to 'default')",
    )
    
    analyze_parser.add_argument(
        "-m", "--model",
        default="llama3",
        help="LLM model to use (default: llama3)",
    )
    
    analyze_parser.add_argument(
        "-o", "--output",
        default=str(Path("reports") / "<project>_report_<timestamp>.md"),
        help="Output file for the report (default: reports/<project>_report_<timestamp>.md)",
    )
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Start the web interface")
    web_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    web_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    web_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    # GitHub loader command
    github_parser = subparsers.add_parser("github", help="Load code from GitHub repository")
    github_parser.add_argument(
        "repo_url",
        help="GitHub repository URL (e.g., https://github.com/owner/repo)",
    )
    github_parser.add_argument(
        "-q", "--query",
        default="Perform a comprehensive security audit",
        help="Audit query (e.g., 'Check for reentrancy vulnerabilities')",
    )
    github_parser.add_argument(
        "-n", "--name",
        default=None,
        help=f"Project name (defaults to repository name)",
    )
    github_parser.add_argument(
        "-m", "--model",
        default="llama3",
        help="LLM model to use (default: llama3)",
    )
    github_parser.add_argument(
        "-o", "--output",
        default=str(Path("reports") / "<project>_report_<timestamp>.md"),
        help="Output file for the report (default: reports/<project>_report_<timestamp>.md)",
    )
    github_parser.add_argument(
        "--issues",
        action="store_true",
        help="Load and analyze GitHub issues in addition to code",
    )
    github_parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub API token for private repositories (default: GITHUB_TOKEN env var)",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 0
        
    # Handle the analyze command
    if args.command == "analyze":
        # Get the solidity files to analyze
        solidity_files = get_solidity_files(args)
        if not solidity_files:
            return 1
    
        # Set default project name if not provided
        project_name = args.name if hasattr(args, 'name') else Path(solidity_files[0]).stem
        
        # Set default output file if not provided
        output_file = args.output if hasattr(args, 'output') else None
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"report_{timestamp}.md"
        
        # Format output file path
        if "<project>" in output_file:
            output_file = output_file.replace("<project>", project_name)
        if "<timestamp>" in output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_file.replace("<timestamp>", timestamp)
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Initialize the workflow orchestrator
        orchestrator = WorkflowOrchestrator(
            model_name=args.model if hasattr(args, 'model') else nodes_config.WORKFLOW_MODEL,
        )
        
        try:
            # Print banner
            print_banner("Finite Monkey - Smart Contract Analyzer", [
                f"Project: {project_name}",
                f"Files: {len(solidity_files)} Solidity file(s)",
            ] + [f"  - {file}" for file in solidity_files[:5]] + 
            ([f"  - ... and {len(solidity_files) - 5} more"] if len(solidity_files) > 5 else []) + [
                f"Model: {args.model if hasattr(args, 'model') else nodes_config.WORKFLOW_MODEL}",
                f"Output: {output_file}",
            ])
            
            # Run the audit workflow
            report = await orchestrator.run_atomic_agent_workflow(
                solidity_path=solidity_files,
                query=args.query if hasattr(args, 'query') else "Perform a comprehensive security audit",
                project_name=project_name,
            )
            
            # Save the report
            print(f"Saving report to {output_file}...")
            await report.save(output_file)
            
            # Print summary
            print("\nAudit completed successfully!")
            print(f"Project: {project_name}")
            print(f"Report saved to: {output_file}")
            
            # Print findings summary using the common function
            print_findings_summary(report)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
            
    # Handle the web command
    elif args.command == "web":
        try:
            # Print banner
            print_banner("Finite Monkey - Web Interface", [
                f"Host: {args.host}",
                f"Port: {args.port}",
                f"Debug: {args.debug}",
                f"Auto-reload: {'Enabled' if args.reload else 'Disabled'}",
            ])
            
            # Import the web server
            import uvicorn
            from .web.app import app
            
            # Set up database directory
            db_dir = Path("db")
            db_dir.mkdir(exist_ok=True)
            
            # Set up outputs directory
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            
            # Start Uvicorn server - need to handle it differently when we're already in an async context
            config = uvicorn.Config(
                "finite_monkey.web.app:app",
                host=args.host,
                port=args.port,
                reload=args.reload or args.debug,
                log_level="debug" if args.debug else "info",
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            print(f"Error starting web server: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Handle the GitHub command
    elif args.command == "github":
        try:
            # Extract repository owner and name from URL
            import re
            repo_url = args.repo_url
            match = re.search(r"github\.com[:/]([^/]+)/([^/\.]+)", repo_url)
            if not match:
                print(f"Error: Invalid GitHub repository URL: {repo_url}")
                return 1
            
            owner, repo = match.groups()
            repo = repo.rstrip(".git")
            
            # Set project name if not provided
            project_name = args.name or repo
            
            # Set output file path
            output_file = args.output
            if "<project>" in output_file:
                output_file = output_file.replace("<project>", project_name)
            if "<timestamp>" in output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_file.replace("<timestamp>", timestamp)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Print banner
            print_banner("Finite Monkey - GitHub Repository Analyzer", [
                f"Repository: {owner}/{repo}",
                f"URL: {repo_url}",
                f"Project Name: {project_name}",
                f"Model: {args.model}",
                f"Output: {output_file}",
                f"Include Issues: {'Yes' if args.issues else 'No'}",
            ])
            
            # Clone the repository to a temporary directory
            import tempfile
            import subprocess
            
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Cloning repository to {temp_dir}...")
                
                # Set git command
                git_cmd = ["git", "clone"]
                if args.token:
                    # Use token for authentication
                    auth_url = f"https://{args.token}@github.com/{owner}/{repo}.git"
                    git_cmd.append(auth_url)
                else:
                    git_cmd.append(repo_url)
                
                git_cmd.append(temp_dir)
                
                # Clone repository
                process = subprocess.run(git_cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    print(f"Error cloning repository: {process.stderr}")
                    return 1
                
                print(f"Repository cloned successfully.")
                
                # Find Solidity files
                import glob
                solidity_files = glob.glob(os.path.join(temp_dir, "**", "*.sol"), recursive=True)
                
                if not solidity_files:
                    print(f"No Solidity files found in repository {owner}/{repo}")
                    return 1
                
                print(f"Found {len(solidity_files)} Solidity files:")
                for i, file in enumerate(solidity_files[:5]):
                    rel_path = os.path.relpath(file, temp_dir)
                    print(f"  - {rel_path}")
                
                if len(solidity_files) > 5:
                    print(f"  - ... and {len(solidity_files) - 5} more")
                
                # Initialize the workflow orchestrator
                from .agents import WorkflowOrchestrator
                orchestrator = WorkflowOrchestrator(
                    model_name=args.model,
                )
                
                # Load GitHub issues if requested
                if args.issues:
                    print(f"Loading GitHub issues for {owner}/{repo}...")
                    
                    from .llama_index.loaders import AsyncGithubIssueLoader
                    from .llama_index.vector_store import VectorStoreManager
                    
                    # Initialize loader and vector store
                    issue_loader = AsyncGithubIssueLoader(
                        repo_url=repo_url,
                        token=args.token,
                    )
                    
                    vector_store = VectorStoreManager()
                    
                    # Load issues
                    print("Fetching issues...")
                    issues = await issue_loader.load()
                    
                    if not issues:
                        print("No issues found in the repository.")
                    else:
                        print(f"Found {len(issues)} issues.")
                        
                        # Store issues in vector store
                        print("Storing issues in vector database...")
                        collection_name = f"github_issues_{owner}_{repo}"
                        collection = await vector_store.get_or_create_collection(collection_name)
                        await vector_store.add_documents(collection, issues)
                        
                        print(f"Issues stored in collection: {collection_name}")
                
                # Run the audit workflow
                print(f"Running audit workflow...")
                report = await orchestrator.run_atomic_agent_workflow(
                    solidity_path=solidity_files,
                    query=args.query,
                    project_name=project_name,
                )
                
                # Save the report
                print(f"Saving report to {output_file}...")
                await report.save(output_file)
                
                # Count findings by severity
                severity_counts = {}
                for finding in report.findings:
                    severity = finding.get("severity", "Unknown")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # Print findings summary
                print("\nFindings Summary:")
                if not severity_counts:
                    print("No findings identified.")
                else:
                    for severity in ["Critical", "High", "Medium", "Low", "Informational"]:
                        if severity in severity_counts:
                            print(f"- {severity}: {severity_counts[severity]}")
                
                # Also save JSON results
                json_file = output_file.replace(".md", ".json")
                import json
                results_data = {
                    "project": project_name,
                    "repository": f"{owner}/{repo}",
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "model": args.model,
                    "query": args.query,
                    "files_analyzed": len(solidity_files),
                    "issues_loaded": len(issues) if args.issues and 'issues' in locals() else 0,
                    "findings": report.findings,
                }
                
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(results_data, f, indent=2)
                
                print(f"\nResults saved to: {json_file}")
                print(f"Report saved to: {output_file}")
                
                return 0
            
        except Exception as e:
            print(f"Error analyzing GitHub repository: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


def get_solidity_files(args):
    """Get Solidity files from arguments"""
    solidity_files = []
    
    if hasattr(args, 'file') and args.file:
        # Single file mode
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}")
            return []
        solidity_files = [args.file]
    elif hasattr(args, 'directory') and args.directory:
        # Directory mode
        if not os.path.isdir(args.directory):
            print(f"Error: Directory not found: {args.directory}")
            return []
        
        import glob
        # Find all Solidity files matching the pattern in the directory
        pattern = os.path.join(args.directory, args.pattern)
        solidity_files = glob.glob(pattern)
        
        if not solidity_files:
            print(f"Error: No files matching pattern '{args.pattern}' found in {args.directory}")
            return []
    elif hasattr(args, 'files') and args.files:
        # Multiple files mode
        for file_path in args.files:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found: {file_path}")
            else:
                solidity_files.append(file_path)
        
        if not solidity_files:
            print("Error: None of the specified files were found")
            return []
    
    return solidity_files


def print_banner(title, lines=None):
    """Print a nice banner with a title and optional lines"""
    print("=" * 60)
    print(title)
    print("=" * 60)
    
    if lines:
        for line in lines:
            print(line)
        print("=" * 60)


def print_findings_summary(report):
    """Print summary of findings from a report"""
    # Count findings by severity
    severity_counts = {}
    for finding in report.findings:
        severity = finding.get("severity", "Unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    # Print findings summary
    print("\nFindings Summary:")
    if not severity_counts:
        print("No findings identified.")
    else:
        for severity in ["Critical", "High", "Medium", "Low", "Informational"]:
            if severity in severity_counts:
                print(f"- {severity}: {severity_counts[severity]}")


def run_main():
    """Wrapper for main function that handles asyncio properly"""
    try:
        # Run the main function with asyncio
        result = asyncio.run(main())
        return result
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run the main function with asyncio
    sys.exit(run_main())