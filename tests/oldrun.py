#!/usr/bin/env python3
"""
Run script for the Finite Monkey framework

This script provides a unified command-line interface for the various
components of the Finite Monkey smart contract security analysis framework.

It supports a "zero-configuration" mode when run without arguments for
quick testing and debugging.
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from finite_monkey.agents import WorkflowOrchestrator
from finite_monkey.visualization import GraphFactory
from finite_monkey.nodes_config import nodes_config


# Zero-configuration default mode when no arguments are provided
async def run_default_analysis():
    """
    Run a default analysis with zero-configuration
    
    This provides a simple "just works" approach for quick usage and debugging.
    It will:
    1. Use all contracts under examples/src
    2. Run a comprehensive security audit with business flow analysis
    3. Save the report to the reports directory
    """
    print("=" * 60)
    print("Finite Monkey Engine - Default Analysis Mode")
    print("=" * 60)
    print("Running with default settings (zero-configuration mode)")
    
    # Load configuration from nodes_config
    config = nodes_config()
    
    # Create the orchestrator with default settings
    # Use configured model from nodes_config or fall back to qwen2.5
    model_name = config.WORKFLOW_MODEL or "qwen2.5-coder:14b-instruct-q8_0"
    orchestrator = WorkflowOrchestrator(model_name=model_name)
    
    try:
        # Default to analyzing all contracts in examples/src
        example_path = "examples/src"
        if not os.path.exists(example_path):
            # Fall back to examples/defi_project/contracts if src doesn't exist
            example_path = "examples/defi_project/contracts"
            if not os.path.exists(example_path):
                # Last resort fallback
                example_path = "examples/Vault.sol"
                project_name = "Vault"
            else:
                project_name = "DeFiProject"
        else:
            project_name = "SmartContracts"
        
        print(f"Analyzing contracts in: {example_path}")
        print(f"Using model: {model_name}")
        
        # Prepare output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("reports")
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, f"{project_name}_report_{timestamp}.md")
        results_path = os.path.join(output_dir, f"{project_name}_results_{timestamp}.json")
        
        print(f"Report will be saved to: {report_path}")
        print(f"Analysis starting...")
        
        query = "Perform a comprehensive security audit with business flow analysis"
        
        # Use the new pipeline for business flow analysis
        print(f"Running business flow analysis for {project_name}...")
        
        # Step 1: Set up database and prompt templates
        from finite_monkey.db.prompts.prompt_service import prompt_service
        
        # Initialize prompt service
        await prompt_service.initialize()
        
        # Step 2: Run the analysis using proper agents
        from finite_monkey.agents.researcher import Researcher
        from finite_monkey.agents.validator import Validator
        from finite_monkey.agents.documentor import Documentor
        from finite_monkey.utils.chunking import chunk_solidity_file, ContractChunker
        from finite_monkey.adapters import Ollama
        
        # Set up pipeline components
        llm_client = Ollama(model=model_name)
        researcher = Researcher(query_engine=None, llm_client=llm_client, model_name=model_name)
        validator = Validator(llm_client=llm_client, model_name=model_name)
        documentor = Documentor(llm_client=llm_client, model_name=model_name)
        
        print("Reading contract files...")
        # Read contract files and prepare for chunking
        contract_code = {}
        contract_files = []
        
        if os.path.isdir(example_path):
            # Collect all .sol files
            for root, _, files in os.walk(example_path):
                for file in files:
                    if file.endswith('.sol'):
                        file_path = os.path.join(root, file)
                        contract_files.append(file_path)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            contract_code[file_path] = f.read()
        else:
            # Single file
            contract_files.append(example_path)
            with open(example_path, 'r', encoding='utf-8') as f:
                contract_code[example_path] = f.read()
        
        print("Analyzing code with dual-layer agent architecture...")
        # Analyze with agents
        analysis_results = {}
        
        for file_path in contract_files:
            code = contract_code[file_path]
            file_name = os.path.basename(file_path)
            
            # Check if file is too large and needs chunking
            if len(code) > 8000:
                print(f"File {file_name} is large ({len(code)} chars), using chunking...")
                chunks = chunk_solidity_file(file_path, max_chunk_size=8000)
                print(f"Divided into {len(chunks)} chunks for analysis")
                
                # Analyze each chunk
                chunk_analyses = []
                chunk_validations = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = chunk["chunk_id"]
                    chunk_content = chunk["content"]
                    chunk_type = chunk["chunk_type"]
                    
                    # Research phase
                    print(f"Analyzing chunk {i+1}/{len(chunks)} ({chunk_type})...")
                    chunk_analysis = await researcher.analyze_code_async(
                        query=query,
                        code_snippet=chunk_content,
                    )
                    chunk_analyses.append(chunk_analysis)
                    
                    # Validation phase
                    print(f"Validating chunk {i+1}/{len(chunks)}...")
                    chunk_validation = await validator.validate_analysis(
                        code=chunk_content,
                        analysis=chunk_analysis,
                    )
                    chunk_validations.append(chunk_validation)
                
                # Combine results from all chunks
                print(f"Combining results from {len(chunks)} chunks...")
                
                # Create an aggregate analysis
                aggregate_findings = []
                seen_findings = set()
                
                for analysis in chunk_analyses:
                    for finding in analysis.findings:
                        finding_title = finding.get("title", "")
                        if finding_title and finding_title not in seen_findings:
                            seen_findings.add(finding_title)
                            aggregate_findings.append(finding)
                
                aggregate_validation = {
                    "structured_validation": {},
                    "issues": []
                }
                
                for validation in chunk_validations:
                    if hasattr(validation, "issues"):
                        for issue in validation.issues:
                            issue_dict = {
                                "title": issue.title,
                                "description": issue.description,
                                "severity": issue.severity,
                                "confidence": issue.confidence
                            }
                            if issue_dict not in aggregate_validation["issues"]:
                                aggregate_validation["issues"].append(issue_dict)
                
                # Store the combined results
                analysis_results[file_path] = {
                    "analysis": {"structured_findings": aggregate_findings},
                    "validation": aggregate_validation
                }
            else:
                # Regular analysis for small files
                print(f"Analyzing {file_name}...")
                
                # Research phase
                file_analysis = await researcher.analyze_code_async(
                    query=query,
                    code_snippet=code,
                )
                
                # Validation phase
                findings = file_analysis.findings
                validation = await validator.validate_analysis(
                    code=code,
                    analysis=file_analysis,
                )
                
                # Store results
                analysis_results[file_path] = {
                    "analysis": {"structured_findings": findings},
                    "validation": {"structured_validation": {}, "issues": validation.issues}
                }
        
        # Generate final report
        print("Generating final report...")
        # Combine all analysis results
        all_findings = []
        all_validations = {}
        
        for file_path, results in analysis_results.items():
            findings = results["analysis"].get("structured_findings", [])
            all_findings.extend(findings)
            
            validation = results["validation"]
            all_validations[file_path] = validation
        
        # Generate report with documentor
        report_data = await documentor.generate_report_async(
            analysis={"findings": all_findings},
            validation={"issues": [issue for file_val in all_validations.values() for issue in file_val.get("issues", [])]},
            project_name=project_name,
            target_files=contract_files,
            query=query,
        )
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_data["markdown"])
            
        # Also save results as JSON for future reference
        import json
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                "project": project_name,
                "timestamp": timestamp,
                "findings": all_findings,
                "validation": all_validations
            }, f, indent=2)
        
        print("\nAnalysis complete!")
        print(f"Report saved to: {report_path}")
        
        # Print findings summary
        print("\nFindings Summary:")
        severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Informational": 0}
        
        for finding in all_findings:
            severity = finding.get("severity", "Medium")
            # Normalize severity (handle different formats)
            severity = severity.capitalize()
            if severity in severity_counts:
                severity_counts[severity] += 1
            else:
                severity_counts["Medium"] += 1
        
        for severity, count in severity_counts.items():
            if count > 0:
                print(f"- {severity}: {count}")
                
        return 0
    except Exception as e:
        print(f"Error in default analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


async def main():
    """
    Main entry point for the Finite Monkey framework
    """
    # Initialize settings from all sources (pydantic-settings automatically handles precedence)
    # Order of precedence: class-default → pyproject.toml → .env → env vars → cmdline args
    config = nodes_config()
    
    # Check if command was provided
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print_help(config)
        return 0
    
    command = sys.argv[1]
    
    # Handle commands
    if command == "analyze":
        # For compatibility with argparse version, extract arguments
        file_path = get_arg_value(["-f", "--file"], None)
        directory = get_arg_value(["-d", "--directory"], None)
        files = get_arg_list("--files")
        query = get_arg_value(["-q", "--query"], config.USER_QUERY or "Perform a comprehensive security audit")
        pattern = get_arg_value(["--pattern"], "*.sol")
        project_name = get_arg_value(["-n", "--name"], config.id)
        model = get_arg_value(["-m", "--model"], config.WORKFLOW_MODEL or "llama2")
        output = get_arg_value(["-o", "--output"], str(Path(config.output) / "<project>_report_<timestamp>.md"))
        simple = "--simple" in sys.argv
        
        # Determine the input files
        solidity_files = get_solidity_files_from_args(file_path, directory, files, pattern)
        if not solidity_files:
            return 1
        
        # Run analysis
        return await run_analyze(
            solidity_files=solidity_files,
            query=query,
            project_name=project_name,
            model=model,
            output=output,
            simple=simple,
            config=config
        )
    
    elif command == "visualize":
        # Get required file_path (positional argument after command)
        if len(sys.argv) < 3:
            print("Error: Missing file path for visualization")
            return 1
            
        file_path = sys.argv[2]
        output = get_arg_value(["-o", "--output"], str(Path(config.output) / "<filename>_graph.html"))
        
        # Run visualization
        return await run_visualize(
            file_path=file_path,
            output=output,
            config=config
        )
    
    elif command == "full-audit":
        # Extract arguments
        file_path = get_arg_value(["-f", "--file"], None)
        directory = get_arg_value(["-d", "--directory"], None)
        files = get_arg_list("--files")
        query = get_arg_value(["-q", "--query"], config.USER_QUERY or "Perform a comprehensive security audit")
        pattern = get_arg_value(["--pattern"], "*.sol")
        project_name = get_arg_value(["-n", "--name"], config.id)
        model = get_arg_value(["-m", "--model"], config.WORKFLOW_MODEL or "llama2")
        output_dir = get_arg_value(["-o", "--output-dir"], config.output)
        
        # Determine the input files
        solidity_files = get_solidity_files_from_args(file_path, directory, files, pattern)
        if not solidity_files:
            return 1
            
        # Run full audit
        return await run_full_audit(
            solidity_files=solidity_files,
            query=query,
            project_name=project_name,
            model=model,
            output_dir=output_dir,
            config=config
        )
    
    else:
        print(f"Unknown command: {command}")
        print_help(config)
        return 1


def print_help(config):
    """Print help message"""
    print("Finite Monkey - Smart Contract Audit & Analysis Framework\n")
    print("Commands:")
    print("  analyze     Analyze smart contracts for security vulnerabilities")
    print("  visualize   Generate visualization for smart contracts")
    print("  full-audit  Analyze and visualize smart contracts\n")
    
    print("Examples:")
    print("  # Analyze a single contract")
    print("  ./run.py analyze -f examples/Vault.sol\n")
    
    print("  # Analyze all contracts in a directory")
    print("  ./run.py analyze -d examples/\n")
    
    print("  # Generate visualization for a contract")
    print("  ./run.py visualize examples/Vault.sol\n")
    
    print("  # Analyze contracts and generate visualization")
    print("  ./run.py full-audit -f examples/Vault.sol\n")
    
    print("For help on specific commands, use: ./run.py <command> --help")


def get_arg_value(options, default=None):
    """Get argument value from command line"""
    for i, arg in enumerate(sys.argv):
        for option in options:
            if arg == option and i + 1 < len(sys.argv):
                return sys.argv[i + 1]
    return default


def get_arg_list(option):
    """Get list argument value from command line"""
    for i, arg in enumerate(sys.argv):
        if arg == option and i + 1 < len(sys.argv):
            # Collect all arguments that don't start with '-'
            result = []
            j = i + 1
            while j < len(sys.argv) and not sys.argv[j].startswith('-'):
                result.append(sys.argv[j])
                j += 1
            return result
    return None


def get_solidity_files_from_args(file_path, directory, files, pattern):
    """Get Solidity files from arguments"""
    solidity_files = []
    
    if file_path:
        # Single file mode
        if not os.path.isfile(file_path):
            print(f"Error: File not found: {file_path}")
            return []
        solidity_files = [file_path]
    elif directory:
        # Directory mode
        if not os.path.isdir(directory):
            print(f"Error: Directory not found: {directory}")
            return []
        
        import glob
        # Find all Solidity files matching the pattern in the directory
        glob_pattern = os.path.join(directory, pattern)
        solidity_files = glob.glob(glob_pattern)
        
        if not solidity_files:
            print(f"Error: No files matching pattern '{pattern}' found in {directory}")
            return []
    elif files:
        # Multiple files mode
        for file_path in files:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found: {file_path}")
            else:
                solidity_files.append(file_path)
        
        if not solidity_files:
            print("Error: None of the specified files were found")
            return []
    
    return solidity_files


async def run_analyze(solidity_files, query, project_name, model, output, simple, config):
    """Run the analyze command"""
    # Set default project name if not provided
    project_name = get_project_name(project_name, solidity_files, config)
    
    # Set default output file if not provided
    output_file = output
    if "<project>" in output_file:
        # Replace placeholder with actual project name
        output_file = output_file.replace("<project>", project_name)
    if "<timestamp>" in output_file:
        # Replace placeholder with actual timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_file.replace("<timestamp>", timestamp)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Print banner
    print_banner("Finite Monkey - Smart Contract Analyzer", [
        f"Project: {project_name}",
        f"Files: {len(solidity_files)} Solidity file(s)",
    ] + [f"  - {file}" for file in solidity_files[:5]] + 
    ([f"  - ... and {len(solidity_files) - 5} more"] if len(solidity_files) > 5 else []) + [
        f"Model: {model}",
        f"Output: {output_file}",
        f"Workflow: {'Simple' if simple else 'Atomic Agent'}"
    ])
    
    # Initialize the workflow orchestrator
    orchestrator = WorkflowOrchestrator(
        model_name=model,
    )
    
    # Run the appropriate workflow
    print(f"Starting {'simple' if simple else 'atomic agent'} workflow...")
    
    if simple:
        # Import the controller here to avoid circular imports
        from finite_monkey.workflow import AgentController
        from finite_monkey.adapters import Ollama
        from finite_monkey.visualization import GraphFactory
        
        # Use the simple workflow for single files
        if len(solidity_files) != 1:
            print("Error: Simple workflow only supports a single file. Use the atomic agent workflow for multiple files.")
            return 1
        
        # Initialize components - use model from args or config
        model_name = model or config.WORKFLOW_MODEL
        #ollama_api_base = config.OPENAI_API_BASE  # Using OPENAI base as it's likely pointing to Ollama
        
        # Initialize components
        ollama = Ollama(model=model_name, api_base=ollama_api_base)
        controller = AgentController(ollama, model_name)
        
        # Read the file content
        with open(solidity_files[0], "r", encoding="utf-8") as f:
            code_content = f.read()
        
        # Run the simple workflow
        await run_simple_workflow(
            controller=controller,
            ollama=ollama,
            file_path=solidity_files[0],
            code_content=code_content,
            
            project_name=project_name,
            output_file=output_file,
            query=query,
        )
    else:
        # Run the atomic agent workflow
        report = await orchestrator.run_atomic_agent_workflow(
            solidity_path=solidity_files,
            query=query,
            project_name=project_name,
        )
        
        # Save the report
        print(f"Saving report to {output_file}...")
        await report.save(output_file)
        
        # Print findings summary
        print_findings_summary(report)
    
    return 0


async def run_visualize(file_path, output, config):
    """Run the visualize command"""
    # Validate file path
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        return 1
    
    # Set default output file if not provided
    output_file = output
    filename = os.path.basename(file_path).split(".")[0]
    
    if "<filename>" in output_file:
        # Replace placeholder with actual filename
        output_file = output_file.replace("<filename>", filename)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Print banner
    print_banner("Finite Monkey - Contract Visualizer", [
        f"File: {file_path}",
        f"Output: {output_file}"
    ])
    
    # Generate the visualization
    print(f"Generating contract visualization...")
    graph = GraphFactory.analyze_solidity_file(file_path)
    
    # Export as HTML
    print(f"Exporting visualization to {output_file}...")
    graph.export_html(output_file)
    
    print("\nVisualization generated successfully!")
    print(f"Visualization saved to: {output_file}")
    
    return 0


async def run_full_audit(solidity_files, query, project_name, model, output_dir, config):
    """Run the full-audit command"""
    # Set default project name if not provided
    project_name = get_project_name(project_name, solidity_files, config)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"{project_name}_report_{timestamp}.md")
    graph_files = [os.path.join(output_dir, f"{os.path.basename(file).split('.')[0]}_graph_{timestamp}.html") 
                 for file in solidity_files]
    
    # Also create a JSON results file
    results_file = os.path.join(output_dir, f"{project_name}_results_{timestamp}.json")
    
    # Print banner
    print_banner("Finite Monkey - Full Audit", [
        f"Project: {project_name}",
        f"Files: {len(solidity_files)} Solidity file(s)",
    ] + [f"  - {file}" for file in solidity_files[:5]] + 
    ([f"  - ... and {len(solidity_files) - 5} more"] if len(solidity_files) > 5 else []) + [
        f"Model: {model}",
        f"Output Directory: {output_dir}",
        f"Report: {os.path.basename(report_file)}",
        f"Results: {os.path.basename(results_file)}",
        f"Visualizations: {len(graph_files)} files"
    ])
    
    # Initialize the workflow orchestrator
    orchestrator = WorkflowOrchestrator(
        model_name=model,
    )
    
    # Run the atomic agent workflow
    print(f"Starting audit...")
    report = await orchestrator.run_atomic_agent_workflow(
        solidity_path=solidity_files,
        query=query,
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
    
    # Save JSON results
    import json
    results_data = {
        "project": project_name,
        "timestamp": timestamp,
        "files": solidity_files,
        "model": model,
        "query": query,
        "findings": report.findings if hasattr(report, "findings") else [],
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print("\nAudit completed successfully!")
    print(f"Project: {project_name}")
    print(f"Report saved to: {report_file}")
    print(f"Results saved to: {results_file}")
    print(f"Visualizations saved to: {output_dir}/")
    
    # Print findings summary
    print_findings_summary(report)
    
    return 0
    
    # Handle the analyze command
    
    if args.command == "analyze":
        # Determine the input files
        solidity_files = get_solidity_files(args)
        if not solidity_files:
            return 1
        
        # Set default project name if not provided
        project_name = get_project_name(args, solidity_files, config)
        
        # Set default output file if not provided
        output_file = args.output
        if "<project>" in output_file:
            # Replace placeholder with actual project name
            output_file = output_file.replace("<project>", project_name)
        if "<timestamp>" in output_file:
            # Replace placeholder with actual timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_file.replace("<timestamp>", timestamp)
        
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
            
            # Initialize components - use model from args or config
            model = args.model or config.WORKFLOW_MODEL
            #ollama_api_base = config.OPENAI_API_BASE  # Using OPENAI base as it's likely pointing to Ollama
            
            # Initialize components
            ollama = Ollama(model=model)
            controller = AgentController(ollama, model)
            
            # Read the file content
            with open(solidity_files[0], "r", encoding="utf-8") as f:
                code_content = f.read()
            
            # Run the simple workflow
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
            report = await orchestrator.run_atomic_agent_workflow(
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
        filename = os.path.basename(args.file_path).split(".")[0]
        
        if "<filename>" in output_file:
            # Replace placeholder with actual filename
            output_file = output_file.replace("<filename>", filename)
        
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
        project_name = get_project_name(args, solidity_files, config)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate output file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(args.output_dir, f"{project_name}_report_{timestamp}.md")
        graph_files = [os.path.join(args.output_dir, f"{os.path.basename(file).split('.')[0]}_graph_{timestamp}.html") 
                     for file in solidity_files]
        
        # Also create a JSON results file
        results_file = os.path.join(args.output_dir, f"{project_name}_results_{timestamp}.json")
        
        # Print banner
        print_banner("Finite Monkey - Full Audit", [
            f"Project: {project_name}",
            f"Files: {len(solidity_files)} Solidity file(s)",
        ] + [f"  - {file}" for file in solidity_files[:5]] + 
        ([f"  - ... and {len(solidity_files) - 5} more"] if len(solidity_files) > 5 else []) + [
            f"Model: {args.model}",
            f"Output Directory: {args.output_dir}",
            f"Report: {os.path.basename(report_file)}",
            f"Results: {os.path.basename(results_file)}",
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
        
        # Save JSON results
        import json
        results_data = {
            "project": project_name,
            "timestamp": timestamp,
            "files": solidity_files,
            "model": args.model,
            "query": args.query,
            "findings": report.findings if hasattr(report, "findings") else [],
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)
        
        # Print summary
        print("\nAudit completed successfully!")
        print(f"Project: {project_name}")
        print(f"Report saved to: {report_file}")
        print(f"Results saved to: {results_file}")
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


def get_project_name(args, solidity_files, config=None):
    """Get project name from arguments or files"""
    # Check arguments first (highest precedence)
    if hasattr(args, 'name') and args.name:
        return args.name
    
    # Next, try to infer from files
    if len(solidity_files) == 1:
        # Single file - use filename
        return os.path.basename(solidity_files[0]).split(".")[0]
    elif len(solidity_files) > 1:
        # Multiple files - try to use common directory name
        common_dir = os.path.commonpath([os.path.abspath(p) for p in solidity_files])
        project_name = os.path.basename(common_dir)
        if not project_name or project_name == ".":
            # Fall back to config or timestamp-based name
            if config and config.id and config.id != "default":
                return config.id
            else:
                return f"solidity_project_{datetime.now().strftime('%Y%m%d')}"
        return project_name
    
    # Finally, fall back to config (lowest precedence)
    if config and config.id:
        return config.id
    
    # Ultimate fallback
    return f"solidity_project_{datetime.now().strftime('%Y%m%d')}"


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
    # Check if any arguments were provided
    if len(sys.argv) == 1:
        # No arguments, run in zero-configuration mode
        exitcode = asyncio.run(run_default_analysis())
    else:
        # Arguments provided, run normal flow
        exitcode = asyncio.run(main())
    
    sys.exit(exitcode)