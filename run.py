#!/usr/bin/env python3
"""
Run script for the Finite Monkey framework

This script provides a unified command-line interface for the various
components of the Finite Monkey smart contract security analysis framework.
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime

from finite_monkey.agents import WorkflowOrchestrator
from finite_monkey.visualization import GraphFactory


async def main():
    """
    Main entry point for the Finite Monkey framework
    """
    parser = argparse.ArgumentParser(
        description="Finite Monkey - Smart Contract Audit & Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single contract
  ./run.py analyze -f examples/Vault.sol
  
  # Analyze all contracts in a directory
  ./run.py analyze -d examples/
  
  # Generate visualization for a contract
  ./run.py visualize examples/Vault.sol
  
  # Analyze contracts and generate visualization in one step
  ./run.py full-audit -f examples/Vault.sol
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
        help="Project name (defaults to filename or directory name)",
    )
    
    analyze_parser.add_argument(
        "-m", "--model",
        default="llama3",
        help="LLM model to use (default: llama3)",
    )
    
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output file for the report (default: <project>_report_<timestamp>.md)",
    )
    
    analyze_parser.add_argument(
        "--simple",
        action="store_true",
        help="Use the simple workflow instead of the atomic agent workflow",
    )
    
    # Visualization command
    visualize_parser = subparsers.add_parser("visualize", help="Generate visualization for smart contracts")
    visualize_parser.add_argument(
        "file_path",
        help="Path to the Solidity file to visualize",
    )
    
    visualize_parser.add_argument(
        "-o", "--output",
        help="Output HTML file for the visualization (default: <filename>_graph.html)",
    )
    
    # Full audit command
    full_parser = subparsers.add_parser("full-audit", help="Analyze and visualize smart contracts")
    
    # Required arguments - file or directory
    file_group = full_parser.add_mutually_exclusive_group(required=True)
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
    full_parser.add_argument(
        "-q", "--query",
        default="Perform a comprehensive security audit",
        help="Audit query (e.g., 'Check for reentrancy vulnerabilities')",
    )
    
    full_parser.add_argument(
        "--pattern",
        default="*.sol",
        help="Glob pattern for Solidity files when using --directory (default: *.sol)",
    )
    
    full_parser.add_argument(
        "-n", "--name",
        help="Project name (defaults to filename or directory name)",
    )
    
    full_parser.add_argument(
        "-m", "--model",
        default="llama3",
        help="LLM model to use (default: llama3)",
    )
    
    full_parser.add_argument(
        "-o", "--output-dir",
        default="reports",
        help="Output directory for reports and visualizations (default: reports/)",
    )
    args = parser.parse_args()
    
    # Parse arguments
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 0
    
    # Handle the analyze command
    if args.command == "analyze":
        # Determine the input files
        solidity_files = get_solidity_files(args)
        if not solidity_files:
            return 1
        
        # Set default project name if not provided
        project_name = get_project_name(args, solidity_files)
        
        # Set default output file if not provided
        output_file = args.output
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{project_name}_report_{timestamp}.md"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Print banner
        print_banner("Finite Monkey - Smart Contract Analyzer", [
            f"Project: {project_name}",
            f"Files: {len(solidity_files)} Solidity file(s)",
        ] + [f"  - {file}" for file in solidity_files[:5]] + 
        ([f"  - ... and {len(solidity_files) - 5} more"] if len(solidity_files) > 5 else []) + [
            f"Model: {args.model}",
            f"Output: {output_file}",
            f"Workflow: {'Simple' if args.simple else 'Atomic Agent'}"
        ])
        
        # Initialize the workflow orchestrator
        orchestrator = WorkflowOrchestrator(
            model_name=args.model,
        )
        
        # Run the appropriate workflow
        print(f"Starting {'simple' if args.simple else 'atomic agent'} workflow...")
        
        if args.simple:
            # Import the controller here to avoid circular imports
            from finite_monkey.workflow import AgentController
            from finite_monkey.adapters import Ollama
            from finite_monkey.visualization import GraphFactory
            
            # Use the simple workflow for single files
            if len(solidity_files) != 1:
                print("Error: Simple workflow only supports a single file. Use the atomic agent workflow for multiple files.")
                return 1
            
            # Initialize components
            ollama = Ollama(model=args.model)
            controller = AgentController(ollama, args.model)
            
            # Read the file content
            with open(solidity_files[0], "r", encoding="utf-8") as f:
                code_content = f.read()
            
            # Run the simple workflow
            from run_simple_workflow import run_simple_workflow
            await run_simple_workflow(
                controller=controller,
                ollama=ollama,
                file_path=solidity_files[0],
                code_content=code_content,
                project_name=project_name,
                output_file=output_file,
                query=args.query,
            )
        else:
            # Run the atomic agent workflow
            report = orchestrator.run_atomic_agent_workflow(
                solidity_path=solidity_files,
                query=args.query,
                project_name=project_name,
            )
            
            # Save the report
            print(f"Saving report to {output_file}...")
            await report.save(output_file)
            
            # Print findings summary
            print_findings_summary(report)
    
    # Handle the visualize command
    elif args.command == "visualize":
        # Validate file path
        if not os.path.isfile(args.file_path):
            print(f"Error: File not found: {args.file_path}")
            return 1
        
        # Set default output file if not provided
        output_file = args.output
        if output_file is None:
            filename = os.path.basename(args.file_path).split(".")[0]
            output_file = f"{filename}_graph.html"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Print banner
        print_banner("Finite Monkey - Contract Visualizer", [
            f"File: {args.file_path}",
            f"Output: {output_file}"
        ])
        
        # Generate the visualization
        print(f"Generating contract visualization...")
        graph = GraphFactory.analyze_solidity_file(args.file_path)
        
        # Export as HTML
        print(f"Exporting visualization to {output_file}...")
        graph.export_html(output_file)
        
        print("\nVisualization generated successfully!")
        print(f"Visualization saved to: {output_file}")
    
    # Handle the full-audit command
    elif args.command == "full-audit":
        # Determine the input files
        solidity_files = get_solidity_files(args)
        if not solidity_files:
            return 1
        
        # Set default project name if not provided
        project_name = get_project_name(args, solidity_files)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate output file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(args.output_dir, f"{project_name}_report_{timestamp}.md")
        graph_files = [os.path.join(args.output_dir, f"{os.path.basename(file).split('.')[0]}_graph_{timestamp}.html") 
                     for file in solidity_files]
        
        # Print banner
        print_banner("Finite Monkey - Full Audit", [
            f"Project: {project_name}",
            f"Files: {len(solidity_files)} Solidity file(s)",
        ] + [f"  - {file}" for file in solidity_files[:5]] + 
        ([f"  - ... and {len(solidity_files) - 5} more"] if len(solidity_files) > 5 else []) + [
            f"Model: {args.model}",
            f"Output Directory: {args.output_dir}",
            f"Report: {os.path.basename(report_file)}",
            f"Visualizations: {len(graph_files)} files"
        ])
        
        # Initialize the workflow orchestrator
        orchestrator = WorkflowOrchestrator(
            model_name=args.model,
        )
        
        # Run the atomic agent workflow
        print(f"Starting audit...")
        report = await orchestrator.run_atomic_agent_workflow(
            solidity_path=solidity_files,
            query=args.query,
            project_name=project_name,
        )
        
        # Save the report
        print(f"Saving report to {report_file}...")
        await report.save(report_file)
        
        # Generate visualizations for each file
        print(f"Generating contract visualizations...")
        for i, file_path in enumerate(solidity_files):
            graph_file = graph_files[i]
            print(f"  Processing {os.path.basename(file_path)}...")
            graph = GraphFactory.analyze_solidity_file(file_path)
            graph.export_html(graph_file)
        
        # Print summary
        print("\nAudit completed successfully!")
        print(f"Project: {project_name}")
        print(f"Report saved to: {report_file}")
        print(f"Visualizations saved to: {args.output_dir}/")
        
        # Print findings summary
        print_findings_summary(report)
    
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
    elif hasattr(args, 'file_path') and args.file_path:
        # Direct file path
        if not os.path.isfile(args.file_path):
            print(f"Error: File not found: {args.file_path}")
            return []
        solidity_files = [args.file_path]
    
    return solidity_files


def get_project_name(args, solidity_files):
    """Get project name from arguments or files"""
    if hasattr(args, 'name') and args.name:
        return args.name
    
    if len(solidity_files) == 1:
        # Single file - use filename
        return os.path.basename(solidity_files[0]).split(".")[0]
    else:
        # Multiple files - try to use common directory name
        common_dir = os.path.commonpath([os.path.abspath(p) for p in solidity_files])
        project_name = os.path.basename(common_dir)
        if not project_name or project_name == ".":
            # Fall back to timestamp-based name
            project_name = f"solidity_project_{datetime.now().strftime('%Y%m%d')}"
        return project_name


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


async def run_simple_workflow(controller, ollama, file_path, code_content, project_name, output_file, query):
    """Run the simple workflow for a single file"""
    # STEP 1: Generate Researcher prompt
    print(f"Generating researcher prompt...")
    researcher_prompt = await controller.generate_agent_prompt(
        agent_type="researcher",
        task=f"Analyze the {project_name} contract for security vulnerabilities",
        context=code_content,
    )
    
    # Use the prompt to generate a research analysis
    print(f"Researcher agent analyzing code...")
    research_response = await ollama.acomplete(
        prompt=researcher_prompt,
    )
    
    # Monitor and provide feedback
    print(f"Monitoring researcher results...")
    researcher_feedback = await controller.monitor_agent(
        agent_type="researcher",
        state="completed",
        results=research_response,
    )
    
    # STEP 2: Generate Validator prompt with feedback
    print(f"Generating validator prompt...")
    validator_prompt = await controller.generate_agent_prompt(
        agent_type="validator",
        task=f"Validate the security analysis for the {project_name} contract",
        context=f"Code:\n```solidity\n{code_content}\n```\n\nResearch Results:\n{research_response}\n\nFeedback:\n{researcher_feedback}",
    )
    
    # Use the prompt to generate validation
    print(f"Validator agent validating analysis...")
    validation_response = await ollama.acomplete(
        prompt=validator_prompt,
    )
    
    # Monitor and provide feedback
    print(f"Monitoring validator results...")
    validator_feedback = await controller.monitor_agent(
        agent_type="validator",
        state="completed",
        results=validation_response,
    )
    
    # STEP 3: Get coordination instructions
    print(f"Coordinating workflow...")
    coordination_instructions = await controller.coordinate_workflow(
        research_results=research_response,
        validation_results=validation_response,
    )
    
    # STEP 4: Generate Documentor prompt with coordination
    print(f"Generating documentor prompt...")
    documentor_prompt = await controller.generate_agent_prompt(
        agent_type="documentor",
        task=f"Create a comprehensive security report for the {project_name} contract",
        context=(
            f"Code:\n```solidity\n{code_content}\n```\n\n"
            f"Research Results:\n{research_response}\n\n"
            f"Validation Results:\n{validation_response}\n\n"
            f"Coordination Instructions:\n{coordination_instructions}"
        ),
    )
    
    # Use the prompt to generate report
    print(f"Documentor agent generating report...")
    report_text = await ollama.acomplete(
        prompt=documentor_prompt,
    )
    
    # STEP 5: Generate visualization
    print(f"Generating contract visualization...")
    graph = GraphFactory.analyze_solidity_file(file_path)
    graph_file = output_file.replace(".md", ".html")
    graph.export_html(graph_file)
    
    # STEP 6: Save outputs
    print(f"Saving report...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    # Save intermediate results
    results_file = output_file.replace(".md", ".json")
    results = {
        "project": project_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "file_path": file_path,
        "researcher": {
            "prompt": researcher_prompt,
            "response": research_response,
            "feedback": researcher_feedback,
        },
        "validator": {
            "prompt": validator_prompt,
            "response": validation_response,
            "feedback": validator_feedback,
        },
        "coordinator": {
            "instructions": coordination_instructions,
        },
        "documentor": {
            "prompt": documentor_prompt,
            "report": report_text,
        },
    }
    
    import json
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nWorkflow completed successfully!")
    print(f"Report: {output_file}")
    print(f"Graph: {graph_file}")
    print(f"Results: {results_file}")


if __name__ == "__main__":
    # Run the main function with asyncio
    exitcode = asyncio.run(main())
    sys.exit(exitcode)