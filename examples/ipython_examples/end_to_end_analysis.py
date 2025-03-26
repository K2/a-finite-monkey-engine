"""
Example script for running a comprehensive end-to-end analysis
via the IPython terminal in the web interface.

Usage:
1. Copy and paste this code into the IPython terminal
2. The script will perform a multi-stage analysis on a DeFi project
3. Results will be displayed in the terminal and saved to reports

This demonstrates a complete workflow using multiple agents and features.
"""

import os
import asyncio
from pathlib import Path
from pprint import pprint
import json

print("End-to-End DeFi Security Analysis")
print("=" * 50)

# Get the example DeFi project path
examples_dir = Path(os.getcwd()) / "examples" / "defi_project"
contracts_dir = examples_dir / "contracts"

# List all Solidity files in the project
print("Project Structure:")
print("-" * 40)

solidity_files = list(contracts_dir.glob("*.sol"))
for i, file_path in enumerate(solidity_files, 1):
    print(f"{i}. {file_path.relative_to(os.getcwd())}")

print("\nStarting comprehensive analysis...")

# 1. Research Phase - Learn about common DeFi vulnerabilities
print("\n--- PHASE 1: RESEARCH ---")
research_query = "What are the common vulnerabilities in DeFi lending and staking protocols?"
research_result = await researcher.research(research_query)

print("\nResearch Findings:")
print("-" * 40)
print(research_result.findings[:1000] + "...\n" if len(research_result.findings) > 1000 else research_result.findings)

# 2. Analysis Phase - Analyze each contract
print("\n--- PHASE 2: CONTRACT ANALYSIS ---")

# A dictionary to store issues for each contract
all_issues = {}

# Analyze each contract individually
for contract_path in solidity_files:
    rel_path = contract_path.relative_to(os.getcwd())
    print(f"\nAnalyzing {rel_path}...")
    
    with open(contract_path, "r") as f:
        code = f.read()
    
    # Validate the contract
    validation_result = await validator.validate_contract(
        file_path=str(contract_path),
        code=code,
        specific_concerns=["reentrancy", "access control", "flash loans"]
    )
    
    all_issues[str(rel_path)] = validation_result.issues
    
    print(f"Issues found: {len(validation_result.issues)}")
    for issue in validation_result.issues:
        print(f"- {issue.title} (Severity: {issue.severity})")

# 3. Documentation Analysis Phase
print("\n--- PHASE 3: DOCUMENTATION ANALYSIS ---")

from finite_monkey.agents.documentation_analyzer import DocumentationAnalyzer

# Create a documentation analyzer
doc_analyzer = DocumentationAnalyzer(
    model_name=config.WORKFLOW_MODEL,
    base_dir=os.getcwd()
)

# Analyze each contract for documentation issues
for contract_path in solidity_files:
    rel_path = contract_path.relative_to(os.getcwd())
    print(f"\nAnalyzing documentation for {rel_path}...")
    
    with open(contract_path, "r") as f:
        code = f.read()
    
    # Analyze documentation
    doc_result = await doc_analyzer.analyze_documentation(
        file_path=str(contract_path),
        code=code
    )
    
    print(f"Documentation completeness: {doc_result.completeness_score}/10")
    print(f"Documentation quality: {doc_result.quality_score}/10")
    
    if doc_result.missing_documentation:
        print("\nMissing documentation:")
        for item in doc_result.missing_documentation[:3]:  # Show first 3 items
            print(f"- {item}")
        
        if len(doc_result.missing_documentation) > 3:
            print(f"  (and {len(doc_result.missing_documentation) - 3} more...)")

# 4. Business Logic Analysis
print("\n--- PHASE 4: BUSINESS LOGIC ANALYSIS ---")

from finite_monkey.agents.business_flow_extractor import BusinessFlowExtractor

# Create a business flow extractor
flow_extractor = BusinessFlowExtractor(
    model_name=config.WORKFLOW_MODEL,
    base_dir=os.getcwd()
)

# Process all contracts together to understand project-wide flows
project_files = [str(path) for path in solidity_files]
flows_result = await flow_extractor.extract_business_flows(
    file_paths=project_files
)

print(f"\nIdentified {len(flows_result.flows)} business flows:")
for i, flow in enumerate(flows_result.flows[:3], 1):  # Show first 3 flows
    print(f"\n{i}. {flow.name}")
    print(f"   Description: {flow.description}")
    print(f"   Entry Points: {', '.join(flow.entry_points[:2])}...")
    print(f"   Risk Level: {flow.risk_level}")

if len(flows_result.flows) > 3:
    print(f"\n(and {len(flows_result.flows) - 3} more flows...)")

# 5. Generate comprehensive report
print("\n--- PHASE 5: GENERATING COMPREHENSIVE REPORT ---")

# Combine all analyses into a comprehensive report
report = {
    "project_name": "Example DeFi Project Analysis",
    "timestamp": "2025-01-01T00:00:00",
    "files_analyzed": [str(path.relative_to(os.getcwd())) for path in solidity_files],
    "issues": [],
    "business_flows": [flow.model_dump() for flow in flows_result.flows],
    "documentation_quality": {
        "average_completeness": sum([doc_result.completeness_score for _ in solidity_files]) / len(solidity_files),
        "average_quality": sum([doc_result.quality_score for _ in solidity_files]) / len(solidity_files),
    }
}

# Collect all issues
for contract, issues in all_issues.items():
    for issue in issues:
        issue_dict = issue.model_dump()
        issue_dict["file"] = contract
        report["issues"].append(issue_dict)

# Save report to file
reports_dir = Path(os.getcwd()) / "reports"
reports_dir.mkdir(exist_ok=True)

report_path = reports_dir / "example_comprehensive_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nComprehensive report saved to: {report_path}")
print("\nAnalysis Summary:")
print(f"- Files analyzed: {len(solidity_files)}")
print(f"- Issues found: {len(report['issues'])}")
print(f"- Business flows identified: {len(report['business_flows'])}")
print(f"- Documentation quality: {report['documentation_quality']['average_quality']}/10")

print("\nEnd-to-End Analysis completed!")