#!/usr/bin/env python3
"""
Run script for the Finite Monkey atomic agent workflow
"""

import os
import asyncio
import argparse
from datetime import datetime

from finite_monkey.agents import WorkflowOrchestrator


async def main():
    """
    Main entry point for the atomic agent workflow
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Finite Monkey - Atomic Agent Workflow"
    )
    
    # Required arguments - file or directory
    file_group = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument(
        "-q", "--query",
        default="Perform a comprehensive security audit",
        help="Audit query (e.g., 'Check for reentrancy vulnerabilities')",
    )
    
    parser.add_argument(
        "--pattern",
        default="*.sol",
        help="Glob pattern for Solidity files when using --directory (default: *.sol)",
    )
    
    parser.add_argument(
        "-n", "--name",
        help="Project name (defaults to filename)",
    )
    
    parser.add_argument(
        "-m", "--model",
        default="llama3",
        help="LLM model to use (default: llama3)",
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file for the report (default: atomic_report_YYYYMMDD_HHMMSS.md)",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine the input files
    solidity_files = []
    
    if args.file:
        # Single file mode
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}")
            return 1
        solidity_files = [args.file]
    elif args.directory:
        # Directory mode
        if not os.path.isdir(args.directory):
            print(f"Error: Directory not found: {args.directory}")
            return 1
        
        import glob
        # Find all Solidity files matching the pattern in the directory
        pattern = os.path.join(args.directory, args.pattern)
        solidity_files = glob.glob(pattern)
        
        if not solidity_files:
            print(f"Error: No files matching pattern '{args.pattern}' found in {args.directory}")
            return 1
    elif args.files:
        # Multiple files mode
        for file_path in args.files:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found: {file_path}")
            else:
                solidity_files.append(file_path)
        
        if not solidity_files:
            print("Error: None of the specified files were found")
            return 1
    
    # Set default project name if not provided
    project_name = args.name
    if project_name is None:
        if len(solidity_files) == 1:
            # Single file - use filename
            project_name = os.path.basename(solidity_files[0]).split(".")[0]
        else:
            # Multiple files - try to use common directory name
            common_dir = os.path.commonpath([os.path.abspath(p) for p in solidity_files])
            project_name = os.path.basename(common_dir)
            if not project_name or project_name == ".":
                # Fall back to timestamp-based name
                project_name = f"solidity_project_{datetime.now().strftime('%Y%m%d')}"
    
    # Set default output file if not provided
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{project_name}_report_{timestamp}.md"
    
    # Print banner
    print("=" * 60)
    print(f"Finite Monkey - Atomic Agent Workflow")
    print(f"Smart Contract Audit & Analysis Framework")
    print("=" * 60)
    print(f"Project: {project_name}")
    print(f"Files: {len(solidity_files)} Solidity file(s)")
    for file in solidity_files[:5]:  # Show up to 5 files
        print(f"  - {file}")
    if len(solidity_files) > 5:
        print(f"  - ... and {len(solidity_files) - 5} more")
    print(f"Model: {args.model}")
    print(f"Output: {output_file}")
    print("=" * 60)
    
    try:
        # Initialize the workflow orchestrator
        orchestrator = WorkflowOrchestrator(
            model_name=args.model,
        )
        
        # Run the atomic agent workflow
        print(f"Starting atomic agent workflow...")
        report = await orchestrator.run_atomic_agent_workflow(
            solidity_path=solidity_files,
            query=args.query,
            project_name=project_name,
        )
        
        # Save the report
        print(f"Saving report to {output_file}...")
        await report.save(output_file)
        
        # Print summary
        print("\nAudit completed successfully!")
        print(f"Project: {project_name}")
        print(f"Report saved to: {output_file}")
        
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
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run the main function with asyncio
    exitcode = asyncio.run(main())
    exit(exitcode)