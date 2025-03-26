#\!/usr/bin/env python3
"""
Simplified run script for demonstration and testing
Uses a mock LLM implementation to avoid external dependencies
"""

import os
import asyncio
from datetime import datetime
from pathlib import Path

from finite_monkey.agents import WorkflowOrchestrator
from finite_monkey.models.reports import AuditReport


async def run_mock_analysis():
    """
    Run a mock analysis with zero-configuration
    """
    print("=" * 60)
    print("Finite Monkey Engine - Mock Analysis Mode")
    print("=" * 60)
    print("Running with simulated LLM responses")
    
    # Create the orchestrator with default settings
    orchestrator = WorkflowOrchestrator()
    
    # Use Vault.sol for the analysis
    example_path = "examples/Vault.sol"
    project_name = "Vault"
    query = "Perform a comprehensive security audit"
    
    # Build a mock report with findings
    findings = [
        {
            "title": "Reentrancy Vulnerability",
            "description": "The withdraw() function is vulnerable to reentrancy attacks because it performs an external call before updating the user's balance.",
            "severity": "High",
            "location": "withdraw() function",
            "validated": True
        },
        {
            "title": "Missing Access Control",
            "description": "The destroyContract() function allows anyone to destroy the contract and send all funds to themselves.",
            "severity": "Critical",
            "location": "destroyContract() function",
            "validated": True
        },
        {
            "title": "Dangerous Use of tx.origin",
            "description": "The isUser() function uses tx.origin for authentication, which is unsafe as it is vulnerable to phishing attacks.",
            "severity": "Medium",
            "location": "isUser() function",
            "validated": True
        }
    ]
    
    # Build a mock report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = AuditReport(
        project_id=f"audit-{project_name}",
        project_name=project_name,
        target_files=[example_path],
        query=query,
        summary="This is a mock analysis for demonstration purposes.",
        findings=findings,
        recommendations=["Implement proper access controls", "Use nonReentrant modifier on withdraw()", "Replace tx.origin with msg.sender"],
        analysis_details={
            "summary": "Mock analysis to demonstrate the framework's capability",
            "full_analysis": "This is a simulated analysis response."
        },
        validation_results={
            "summary": "Mock validation to demonstrate the framework's capability",
            "full_validation": "This is a simulated validation response."
        },
        metadata={
            "is_mock": True
        }
    )
    
    # Create a reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Save to a default location
    report_path = reports_dir / f"{report.project_name}_report_{timestamp}.md"
    await report.save(str(report_path))
    
    print("\nAnalysis complete\!")
    print(f"Report saved to: {report_path}")
    
    # Print findings summary
    print("\nFindings Summary:")
    severity_counts = {}
    for finding in report.findings:
        severity = finding.get("severity", "Unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    for severity in ["Critical", "High", "Medium", "Low", "Informational"]:
        if severity in severity_counts:
            print(f"- {severity}: {severity_counts[severity]}")
    
    return 0


if __name__ == "__main__":
    asyncio.run(run_mock_analysis())
