#!/usr/bin/env python3
"""
Run an enhanced audit with cognitive bias analysis, documentation consistency checking,
and counterfactual generation for smart contract security analysis.
"""

import asyncio
import argparse
import json
import os
import logging
from datetime import datetime
from pathlib import Path

from finite_monkey.agents.orchestrator import WorkflowOrchestrator
from finite_monkey.models.analysis import (
    BiasAnalysisResult, AssumptionAnalysis, VulnerabilityReport
)
from finite_monkey.visualization.bias_visualizer import BiasVisualizer


async def run_enhanced_audit(
    contract_path: str,
    output_dir: str = "reports",
    query: str = "Conduct a comprehensive security audit of this smart contract",
):
    """
    Run an enhanced audit using all specialized agents
    
    Args:
        contract_path: Path to the smart contract file
        output_dir: Directory to save reports
        query: Query for the basic security audit
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
    
    # Create output file paths
    main_report_path = os.path.join(output_dir, f"{project_name}_enhanced_audit_{timestamp}.md")
    json_results_path = os.path.join(output_dir, f"{project_name}_enhanced_audit_{timestamp}.json")
    
    # STEP 1: Run traditional audit
    logger.info(f"Starting traditional security audit for {project_name}...")
    audit_report = await orchestrator.run_audit(
        solidity_path=contract_path,
        query=query,
        project_name=project_name
    )
    
    # Extract vulnerability reports from audit
    vulnerability_reports = []
    for finding in audit_report.findings:
        vulnerability_reports.append(
            VulnerabilityReport(
                title=finding.get("title", "Unnamed Vulnerability"),
                description=finding.get("description", ""),
                severity=finding.get("severity", "Medium"),
                location=finding.get("location", ""),
                vulnerability_type=finding.get("type", "Unknown")
            )
        )
    
    # STEP 2: Run cognitive bias analysis
    logger.info(f"Analyzing cognitive biases in {project_name}...")
    bias_analysis, remediation_plan, assumption_analysis = await orchestrator.analyze_cognitive_biases(
        solidity_path=contract_path,
        project_name=project_name
    )
    
    # STEP 3: Analyze documentation consistency
    logger.info(f"Analyzing documentation consistency in {project_name}...")
    doc_consistency_results = await orchestrator.analyze_documentation_consistency(
        solidity_path=contract_path,
        project_name=project_name
    )
    
    # STEP 4: Generate counterfactuals (if vulnerabilities were found)
    counterfactual_results = None
    if vulnerability_reports:
        logger.info(f"Generating counterfactual scenarios for {project_name}...")
        counterfactual_results = await orchestrator.generate_counterfactuals(
            solidity_path=contract_path,
            vulnerability_reports=vulnerability_reports,
            project_name=project_name
        )
    
    # Generate and save markdown report
    logger.info("Generating enhanced markdown report...")
    with open(main_report_path, "w") as f:
        f.write(f"# Enhanced Security Audit for {project_name}\n\n")
        f.write("## Traditional Security Audit\n\n")
        f.write(audit_report.summary + "\n\n")
        
        # Add findings
        f.write("### Findings\n\n")
        for i, finding in enumerate(audit_report.findings, 1):
            f.write(f"#### {i}. {finding.get('title', 'Unnamed Issue')}\n\n")
            f.write(f"**Severity**: {finding.get('severity', 'Medium')}\n\n")
            f.write(f"**Location**: {finding.get('location', 'Unknown')}\n\n")
            f.write(f"{finding.get('description', '')}\n\n")
        
        # Add recommendations
        f.write("### Recommendations\n\n")
        for i, rec in enumerate(audit_report.recommendations, 1):
            f.write(f"{i}. {rec}\n")
        
        # Add cognitive bias analysis
        f.write("\n\n## Cognitive Bias Analysis\n\n")
        f.write(bias_analysis.get_summary())
        
        # Add assumption analysis
        if assumption_analysis:
            f.write("\n\n## Developer Assumption Analysis\n\n")
            f.write(assumption_analysis.get_summary())
        
        # Add remediation plan
        if remediation_plan:
            f.write("\n\n## Cognitive Bias Remediation Plans\n\n")
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
        
        # Add documentation consistency results
        f.write("\n\n## Documentation-Code Consistency Analysis\n\n")
        consistency_analysis = doc_consistency_results.get("consistency_analysis", {})
        
        total_comments = consistency_analysis.get("total_comments", 0)
        inconsistencies = consistency_analysis.get("inconsistencies", [])
        
        f.write(f"Analyzed {total_comments} comments and found {len(inconsistencies)} inconsistencies.\n\n")
        
        if inconsistencies:
            f.write("### Documentation Inconsistencies\n\n")
            for i, inconsistency in enumerate(inconsistencies, 1):
                f.write(f"#### {i}. {inconsistency.get('inconsistency_type', 'Inconsistency')}\n\n")
                f.write(f"**Severity**: {inconsistency.get('severity', 'Medium')}\n\n")
                f.write(f"**Comment**: `{inconsistency.get('comment', {}).get('text', '')}`\n\n")
                f.write(f"**Code**:\n```solidity\n{inconsistency.get('code_snippet', '')}\n```\n\n")
                f.write(f"**Description**: {inconsistency.get('description', '')}\n\n")
        
        # Add counterfactual results if available
        if counterfactual_results:
            f.write("\n\n## Counterfactual Analysis\n\n")
            
            # Add scenarios
            scenarios = counterfactual_results.get("counterfactual_scenarios", {})
            if scenarios:
                f.write("### Alternative Scenarios\n\n")
                for vuln_title, scenario_list in scenarios.items():
                    f.write(f"#### For vulnerability: {vuln_title}\n\n")
                    for i, scenario in enumerate(scenario_list, 1):
                        f.write(f"**Scenario {i}**: {scenario.get('description', '')}\n\n")
                        if scenario.get('code'):
                            f.write("```solidity\n")
                            f.write(scenario.get('code', ''))
                            f.write("\n```\n\n")
            
            # Add exploitation paths
            exploitation_paths = counterfactual_results.get("exploitation_paths", {})
            if exploitation_paths:
                f.write("### Exploitation Paths\n\n")
                for vuln_title, path in exploitation_paths.items():
                    f.write(f"#### For vulnerability: {vuln_title}\n\n")
                    f.write(f"{path.get('description', '')}\n\n")
                    f.write("**Steps**:\n\n")
                    for i, step in enumerate(path.get('steps', []), 1):
                        f.write(f"{i}. {step}\n")
                    f.write("\n")
            
            # Add training scenarios
            training_scenarios = counterfactual_results.get("training_scenarios", {})
            if training_scenarios:
                f.write("### Training Scenarios\n\n")
                for i, scenario in enumerate(training_scenarios, 1):
                    f.write(f"#### Training Scenario {i}: {scenario.get('title', '')}\n\n")
                    f.write(f"**Description**: {scenario.get('description', '')}\n\n")
                    f.write(f"**Learning Objective**: {scenario.get('learning_objective', '')}\n\n")
                    if scenario.get('code'):
                        f.write("**Code Sample**:\n```solidity\n")
                        f.write(scenario.get('code', ''))
                        f.write("\n```\n\n")
    
    # Save JSON results
    logger.info("Saving JSON results...")
    json_results = {
        "project_name": project_name,
        "timestamp": timestamp,
        "traditional_audit": {
            "summary": audit_report.summary,
            "findings": audit_report.findings,
            "recommendations": audit_report.recommendations
        },
        "cognitive_bias_analysis": bias_analysis.dict(),
    }
    
    if assumption_analysis:
        json_results["assumption_analysis"] = assumption_analysis.dict()
    
    if remediation_plan:
        json_results["remediation_plan"] = remediation_plan
        
    json_results["documentation_consistency"] = doc_consistency_results
    
    if counterfactual_results:
        json_results["counterfactual_analysis"] = counterfactual_results
    
    with open(json_results_path, "w") as f:
        # Convert to JSON-serializable format
        json.dump(json_results, f, indent=2, default=str)
    
    # Generate interactive HTML visualization for cognitive bias analysis
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
        output_path=os.path.join(output_dir, f"{project_name}_viz_{timestamp}.html")
    )
    
    logger.info(f"Enhanced audit complete! Reports saved to:")
    logger.info(f" - {main_report_path}")
    logger.info(f" - {json_results_path}")
    logger.info(f" - {html_path}")

    return main_report_path, json_results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run enhanced smart contract audit with cognitive bias analysis")
    parser.add_argument("contract_path", help="Path to the smart contract file")
    parser.add_argument("--output-dir", default="reports", help="Directory to save reports")
    parser.add_argument("--query", default="Conduct a comprehensive security audit of this smart contract",
                        help="Query for the basic security audit")
    
    args = parser.parse_args()
    
    asyncio.run(run_enhanced_audit(
        contract_path=args.contract_path,
        output_dir=args.output_dir,
        query=args.query
    ))