#!/usr/bin/env python3
"""
Example script for running the async workflow with the TaskManager
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from finite_monkey.agents import WorkflowOrchestrator
from finite_monkey.nodes_config import nodes_config
from finite_monkey.db.manager import TaskManager


async def main():
    """
    Main entry point for the async workflow example
    """
    # Load config
    config = nodes_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Finite Monkey - Async Workflow Example",
    )
    
    parser.add_argument(
        "-f", "--file",
        help="Path to a Solidity file to audit",
    )
    
    parser.add_argument(
        "-d", "--directory",
        help="Path to a directory containing Solidity files to audit",
    )
    
    parser.add_argument(
        "-p", "--pattern",
        default="*.sol",
        help="File pattern for directory scanning (default: *.sol)",
    )
    
    parser.add_argument(
        "-q", "--query",
        default="Perform a comprehensive security audit",
        help="Audit query",
    )
    
    parser.add_argument(
        "-m", "--model",
        default=config.WORKFLOW_MODEL,
        help="LLM model to use",
    )
    
    parser.add_argument(
        "-w", "--wait",
        action="store_true",
        help="Wait for workflow completion (default: start tasks and exit)",
    )
    
    parser.add_argument(
        "-o", "--output",
        default=config.output,
        help="Output directory",
    )

    args = parser.parse_args()
    
    # Validate arguments
    if not args.file and not args.directory:
        print("Error: Either --file or --directory must be specified")
        parser.print_help()
        return 1
    
    # Determine files to process
    files_to_process = []
    
    if args.file:
        # Process single file
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}")
            return 1
        files_to_process.append(args.file)
    
    elif args.directory:
        # Process directory
        if not os.path.isdir(args.directory):
            print(f"Error: Directory not found: {args.directory}")
            return 1
        
        # Find matching files
        import glob
        pattern = os.path.join(args.directory, args.pattern)
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print(f"Error: No files matching '{args.pattern}' found in '{args.directory}'")
            return 1
        
        files_to_process.extend(matching_files)
    
    # Print summary
    print(f"Starting async workflow for {len(files_to_process)} file(s):")
    for i, file_path in enumerate(files_to_process, 1):
        print(f"  {i}. {file_path}")
    
    # Initialize workflow orchestrator with task manager
    task_manager = TaskManager(db_url=config.ASYNC_DB_URL)
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        task_manager=task_manager,
        model_name=args.model,
        base_dir=os.getcwd(),
        db_url=config.ASYNC_DB_URL,
    )
    
    # Determine project name from files
    if len(files_to_process) == 1:
        # Single file - use filename
        project_name = os.path.basename(files_to_process[0]).split(".")[0]
    else:
        # Multiple files - use common directory
        try:
            common_dir = os.path.commonpath([os.path.abspath(p) for p in files_to_process])
            project_name = os.path.basename(common_dir)
            if not project_name or project_name == ".":
                # Fall back to timestamp
                project_name = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        except ValueError:
            # Fall back to timestamp
            project_name = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Project name: {project_name}")
    print(f"Model: {args.model}")
    print(f"Query: {args.query}")
    print(f"Output directory: {args.output}")
    print(f"Waiting for completion: {args.wait}")
    
    # Start the workflow
    print(f"Starting workflow...")
    result = await orchestrator.run_audit_workflow(
        solidity_paths=files_to_process,
        query=args.query,
        project_name=project_name,
        wait_for_completion=args.wait,
    )
    
    if args.wait:
        # Print report summary
        print("\nWorkflow completed!")
        print(f"Findings: {len(result.findings)}")
        
        # Print findings
        if result.findings:
            print("\nFindings:")
            for i, finding in enumerate(result.findings, 1):
                print(f"  {i}. {finding.get('title', 'Untitled')} (Severity: {finding.get('severity', 'Unknown')})")
        
        # Print report paths
        if "report_paths" in result.metadata:
            print("\nReport paths:")
            for path in result.metadata["report_paths"]:
                print(f"  - {path}")
    else:
        # Print task IDs
        print("\nWorkflow started in the background.")
        print("Task IDs:")
        for file_path, tasks in result.items():
            print(f"  File: {file_path}")
            print(f"    Audit ID: {tasks['audit_id']}")
            print(f"    Analysis task: {tasks['analysis']}")
        
        print("\nYou can check the status of these tasks in the database.")
        print("Reports will be saved to the output directory when tasks complete.")
    
    # Clean up
    if args.wait:
        await orchestrator.task_manager.stop()
        
    return 0


if __name__ == "__main__":
    # Run with asyncio
    exitcode = asyncio.run(main())
    sys.exit(exitcode)