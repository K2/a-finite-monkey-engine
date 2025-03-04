#!/usr/bin/env python3
"""
Simplified workflow runner for the Finite Monkey framework
"""

import os
import asyncio
import argparse
import json
from datetime import datetime

from finite_monkey.workflow import AgentController
from finite_monkey.adapters import Ollama
from finite_monkey.visualization import GraphFactory


async def main():
    """
    Main entry point for the simplified workflow
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Finite Monkey - Simplified Workflow"
    )
    
    # Required arguments
    parser.add_argument(
        "file_path",
        help="Path to the Solidity file to audit",
    )
    
    # Optional arguments
    parser.add_argument(
        "-m", "--model",
        default="llama3",
        help="LLM model to use (default: llama3)",
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output directory for reports (default: reports/)",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate file path
    if not os.path.isfile(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return 1
    
    # Set default project name and extract code
    project_name = os.path.basename(args.file_path).split(".")[0]
    
    with open(args.file_path, "r", encoding="utf-8") as f:
        code_content = f.read()
    
    # Set up output directory
    output_dir = args.output or "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"{project_name}_report_{timestamp}.md")
    graph_file = os.path.join(output_dir, f"{project_name}_graph_{timestamp}.html")
    
    # Print banner
    print("=" * 60)
    print(f"Finite Monkey - Simplified Workflow")
    print(f"Smart Contract Audit & Analysis Framework")
    print("=" * 60)
    print(f"Project: {project_name}")
    print(f"Model: {args.model}")
    print(f"Report: {report_file}")
    print(f"Graph: {graph_file}")
    print("=" * 60)
    
    try:
        # Initialize components
        ollama = Ollama(model=args.model)
        controller = AgentController(ollama, args.model)
        
        # STEP 1: Research phase
        print(f"Generating researcher prompt...")
        researcher_prompt = await controller.generate_agent_prompt(
            agent_type="researcher",
            task=f"Analyze the {project_name} contract for security vulnerabilities",
            context=code_content,
        )
        
        print(f"Researcher agent analyzing code...")
        research_response = await ollama.acomplete(
            prompt=researcher_prompt,
        )
        
        print(f"Monitoring researcher results...")
        researcher_feedback = await controller.monitor_agent(
            agent_type="researcher",
            state="completed",
            results=research_response,
        )
        
        # STEP 2: Validation phase
        print(f"Generating validator prompt...")
        validator_prompt = await controller.generate_agent_prompt(
            agent_type="validator",
            task=f"Validate the security analysis for the {project_name} contract",
            context=f"Code:\n```solidity\n{code_content}\n```\n\nResearch Results:\n{research_response}\n\nFeedback:\n{researcher_feedback}",
        )
        
        print(f"Validator agent validating analysis...")
        validation_response = await ollama.acomplete(
            prompt=validator_prompt,
        )
        
        print(f"Monitoring validator results...")
        validator_feedback = await controller.monitor_agent(
            agent_type="validator",
            state="completed",
            results=validation_response,
        )
        
        # STEP 3: Coordination
        print(f"Coordinating workflow...")
        coordination_instructions = await controller.coordinate_workflow(
            research_results=research_response,
            validation_results=validation_response,
        )
        
        # STEP 4: Documentation phase
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
        
        print(f"Documentor agent generating report...")
        report_text = await ollama.acomplete(
            prompt=documentor_prompt,
        )
        
        # STEP 5: Generate visualization
        print(f"Generating contract visualization...")
        graph = GraphFactory.analyze_solidity_file(args.file_path)
        graph.export_html(graph_file)
        
        # STEP 6: Save outputs
        print(f"Saving report and results...")
        
        # Save report
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        # Save intermediate results
        results_file = os.path.join(output_dir, f"{project_name}_results_{timestamp}.json")
        results = {
            "project": project_name,
            "timestamp": timestamp,
            "file_path": args.file_path,
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
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\nWorkflow completed successfully!")
        print(f"Report: {report_file}")
        print(f"Graph: {graph_file}")
        print(f"Results: {results_file}")
        
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