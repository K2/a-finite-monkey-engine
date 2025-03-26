"""
Example script for running a simple smart contract security analysis 
via the IPython terminal in the web interface.

Usage:
1. Copy and paste this code into the IPython terminal
2. The script will analyze the example Vault.sol contract
3. Results will be displayed in the terminal

This demonstrates a basic workflow using the orchestrator.
"""

# Display the example contract we'll analyze
import os
from pathlib import Path

# Get the example contract path
examples_dir = Path(os.getcwd()) / "examples"
vault_path = str(examples_dir / "Vault.sol")

print(f"Analyzing contract at: {vault_path}")
print("\nContract Source:")
print("-" * 40)

with open(vault_path, "r") as f:
    print(f.read())

print("-" * 40)
print("\nRunning security analysis...\n")

# Run the analysis using the orchestrator
# This will extract business logic, identify vulnerabilities, and generate a report
result = await orchestrator.run_audit_workflow(
    solidity_paths=[vault_path],
    query="Perform a security audit of this smart contract focusing on reentrancy and access control vulnerabilities",
    project_name="Example Vault Analysis",
    wait_for_completion=True
)

# Display the analysis results
print("\nAnalysis Results:")
print("=" * 50)
print(f"Project: {result.project_name}")
print(f"Files analyzed: {result.files_analyzed}")
print(f"Issues found: {len(result.issues)}")
print("\nVulnerabilities:")

for i, issue in enumerate(result.issues, 1):
    print(f"\n{i}. {issue.title} (Severity: {issue.severity})")
    print(f"   Location: {issue.location}")
    print(f"   Description: {issue.description}")
    print(f"   Recommendation: {issue.recommendation}")

print("\nAnalysis completed!")
print(f"Full report saved to: {result.report_path}")