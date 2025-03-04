#!/usr/bin/env python3
"""
Command-line interface for the Finite Monkey framework
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from tkinter.filedialog import Directory

from .agents import WorkflowOrchestrator
from nodes_config import nodes_config

async def main():
    # """
    # Main entry point for the command-line interface
    # """
    # # Parse command-line arguments
    # parser = argparse.ArgumentParser(
    #     description="Finite Monkey - Smart Contract Audit & Analysis Framework"
    # )
    
    # # Required arguments
    # parser.add_argument(
    #     "file_path",
    #     help="Path to the Solidity file to audit",
    # )
    
    # # Optional arguments
    # parser.add_argument(
    #     "-q", "--query",
    #     default="Perform a comprehensive security audit",
    #     help="Audit query (e.g., 'Check for reentrancy vulnerabilities')",
    # )
    
    # parser.add_argument(
    #     "-n", "--name",
    #     help="Project name (defaults to filename)",
    # )
    
    # parser.add_argument(
    #     "-m", "--model",
    #     default="llama3",
    #     help="LLM model to use (default: llama3)",
    # )
    
    # parser.add_argument(
    #     "-o", "--output",
    #     help="Output file for the report (default: report_YYYYMMDD_HHMMSS.md)",
    # )
    
    # # Parse arguments
    # args = parser.parse_args()
    
    # Validate file path
    if not Directory(nodes_config.path):
        print(f"Error: File not found: {nodes_config.path}")
        sys.exit(1)
    
    # Set default project name if not provided
    project_name = nodes_config.id
    
    # Set default output file if not provided
    output_file = nodes_config.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"report_{timestamp}.md"
    
    # Initialize the workflow orchestrator
    orchestrator = WorkflowOrchestrator(
        model_name=nodes_config.WORKFLOW_MODEL,
    )
    
    try:
        QUERY_MODEL = args.query
        
        # Run the audit
        print(f"Running audit on {nodes_config.base_dir}...")
        print(f"Query: {nodes_config.UESR_QUERY}")
        print(f"Model: {nodes_config.QUERY_MODEL}")
        
        # Run the audit workflow
        report = await orchestrator.run_audit(
            solidity_path=nodes_config.base_dir,
            query=nodes_config.UESR_QUERY,
            project_name=nodes_config.id,
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
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function with asyncio
    asyncio.run(main())