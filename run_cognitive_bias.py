#!/usr/bin/env python3
"""
Run cognitive bias analysis on a smart contract
"""

import asyncio
import argparse
import json
import os
import logging
from datetime import datetime
from pathlib import Path

from finite_monkey.agents.orchestrator import WorkflowOrchestrator
from finite_monkey.models.analysis import BiasAnalysisResult, AssumptionAnalysis
from finite_monkey.visualization.bias_visualizer import BiasVisualizer


async def analyze_cognitive_biases(
    contract_path: str,
    output_dir: str = "reports",
    include_remediation: bool = True,
    include_assumption_analysis: bool = True
):
    """
    Analyze cognitive biases in a smart contract
    
    Args:
        contract_path: Path to the smart contract file
        output_dir: Directory to save reports
        include_remediation: Whether to generate remediation plans
        include_assumption_analysis: Whether to analyze developer assumptions
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize orchestrator
    logger.info("Initializing orchestrator...")
    orchestrator = WorkflowOrchestrator()
    
    # Extract project name from file path
    project_name = os.path.basename(contract_path).split(".")[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Run cognitive bias analysis
    logger.info(f"Analyzing {project_name} for cognitive biases...")
    bias_analysis, remediation_plan, assumption_analysis = await orchestrator.analyze_cognitive_biases(
        solidity_path=contract_path,
        project_name=project_name,
        include_remediation=include_remediation,
        include_assumption_analysis=include_assumption_analysis
    )
    
    # Create output file paths
    bias_report_path = os.path.join(output_dir, f"{project_name}_cognitive_bias_{timestamp}.md")
    json_results_path = os.path.join(output_dir, f"{project_name}_cognitive_bias_{timestamp}.json")
    
    # Generate and save markdown report
    logger.info("Generating markdown report...")
    with open(bias_report_path, "w") as f:
        f.write(f"# Cognitive Bias Analysis for {project_name}\n\n")
        f.write(bias_analysis.get_summary())
        
        if assumption_analysis:
            f.write("\n\n## Developer Assumption Analysis\n\n")
            f.write(assumption_analysis.get_summary())
        
        if remediation_plan:
            f.write("\n\n## Remediation Plans\n\n")
            for bias_type, steps in remediation_plan.items():
                bias_name = bias_type.replace("_", " ").title()
                f.write(f"### {bias_name}\n\n")
                
                for i, step in enumerate(steps, 1):
                    f.write(f"#### {i}. {step['title']}\n\n")
                    f.write(f"{step['description']}\n\n")
                    
                    if step.get('code_example'):
                        f.write("```solidity\n")
                        f.write(step['code_example'])
                        f.write("\n```\n\n")
                    
                    if step.get('validation'):
                        f.write(f"**Validation:** {step['validation']}\n\n")
    
    # Save JSON results
    logger.info("Saving JSON results...")
    json_results = {
        "project_name": project_name,
        "timestamp": timestamp,
        "bias_analysis": bias_analysis.dict(),
    }
    
    if assumption_analysis:
        json_results["assumption_analysis"] = assumption_analysis.dict()
    
    if remediation_plan:
        json_results["remediation_plan"] = remediation_plan
    
    with open(json_results_path, "w") as f:
        # Convert to JSON-serializable format
        json.dump(json_results, f, indent=2, default=str)
    
    # Generate interactive HTML visualization
    logger.info("Generating interactive HTML visualization...")
    
    # Read the contract code
    with open(contract_path, "r") as f:
        contract_code = f.read()
    
    # Create the visualizer
    visualizer = BiasVisualizer()
    
    # Generate visualization
    html_path = visualizer.create_bias_analysis_visualization(
        bias_analysis=bias_analysis,
        assumption_analysis=assumption_analysis,
        remediation_plan=remediation_plan,
        contract_code=contract_code,
        output_path=os.path.join(output_dir, f"{project_name}_bias_viz_{timestamp}.html")
    )
    
    logger.info(f"Analysis complete! Reports saved to:")
    logger.info(f" - {bias_report_path}")
    logger.info(f" - {json_results_path}")
    logger.info(f" - {html_path}")

    return bias_report_path, json_results_path, html_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze cognitive biases in smart contracts")
    parser.add_argument("contract_path", help="Path to the smart contract file")
    parser.add_argument("--output-dir", default="reports", help="Directory to save reports")
    parser.add_argument("--skip-remediation", action="store_true", help="Skip remediation plan generation")
    parser.add_argument("--skip-assumption-analysis", action="store_true", help="Skip developer assumption analysis")
    
    args = parser.parse_args()
    
    asyncio.run(analyze_cognitive_biases(
        contract_path=args.contract_path,
        output_dir=args.output_dir,
        include_remediation=not args.skip_remediation,
        include_assumption_analysis=not args.skip_assumption_analysis
    ))