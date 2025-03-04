"""
Prompt templates and utilities for LLM interactions
"""

from typing import Dict, List, Any


def get_analysis_prompt(
    query: str,
    code: str,
    context_nodes: List[Dict[str, Any]],
    related_functions: List[Dict[str, Any]],
) -> str:
    """
    Generate a prompt for code analysis
    
    Args:
        query: The query or task
        code: The source code to analyze
        context_nodes: Context nodes from the vector store
        related_functions: Related functions
        
    Returns:
        Formatted prompt
    """
    # Build the prompt
    prompt = f"""
You are an expert smart contract security auditor. Analyze the following code:

```
{code}
```

Query: {query}

Your task is to carefully analyze the code and provide a security assessment.
Focus specifically on identifying potential vulnerabilities, logical flaws, and security risks.

"""
    
    # Add context if available
    if context_nodes:
        prompt += "\nAdditional context from codebase:\n\n"
        for i, node in enumerate(context_nodes[:5]):
            content = node.get("text", "")
            metadata = node.get("metadata", {})
            file_path = metadata.get("file_path", "Unknown")
            
            prompt += f"File: {file_path}\n```\n{content}\n```\n\n"
    
    # Add related functions if available
    if related_functions:
        prompt += "\nRelated functions:\n\n"
        for i, func in enumerate(related_functions[:3]):
            content = func.get("text", "")
            prompt += f"```\n{content}\n```\n\n"
    
    # Add instructions for output format
    prompt += """
Provide your analysis in the following format:

1. Start with a brief summary of the code and its purpose.
2. FINDINGS: List any vulnerabilities or security issues you detect.
   For each finding, specify the severity (Critical, High, Medium, Low, Informational).
3. RECOMMENDATIONS: Provide specific recommendations to fix the issues.

Be thorough, but focus on significant security concerns rather than style issues.
"""
    
    return prompt


def get_validation_prompt(
    code: str,
    analysis: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> str:
    """
    Generate a prompt for validation
    
    Args:
        code: The source code to validate
        analysis: Previous analysis results
        issues: Potential issues to validate
        
    Returns:
        Formatted prompt
    """
    # Extract summary and findings
    summary = analysis.get("summary", "No summary available.")
    findings = analysis.get("findings", [])
    
    # Build the prompt
    prompt = f"""
You are a thorough smart contract security validator. Your job is to independently validate the following analysis:

```
{code}
```

Previous analysis summary:
{summary}

Potential issues identified:
"""
    
    # Add findings
    for i, finding in enumerate(findings):
        title = finding.get("title", f"Finding {i+1}")
        severity = finding.get("severity", "Medium")
        description = finding.get("description", "")
        
        prompt += f"\n{i+1}. {title} (Severity: {severity})\n   {description}\n"
    
    # Add validation instructions
    prompt += """
Your task is to independently assess each issue:

1. Carefully validate each issue by examining the code
2. Determine if the issue is a true positive or false positive
3. Provide additional context or insights not covered in the initial analysis
4. Identify any issues that might have been missed

For each potential issue, provide:
- Confirmation status (Confirmed/False Positive/Needs More Context)
- Your reasoning with specific code references
- Severity assessment (Critical, High, Medium, Low, Informational)
- Suggestions for remediation if confirmed

Be specific and provide code line references where possible.
"""
    
    return prompt


def get_report_prompt(
    analysis: Dict[str, Any],
    validation: Dict[str, Any],
    project_name: str,
) -> str:
    """
    Generate a prompt for report generation
    
    Args:
        analysis: Analysis results
        validation: Validation results
        project_name: Name of the project
        
    Returns:
        Formatted prompt
    """
    # Extract relevant information
    summary = analysis.get("summary", "No summary available.")
    findings = analysis.get("findings", [])
    recommendations = analysis.get("recommendations", [])
    
    validation_summary = validation.get("summary", "No validation summary available.")
    validation_issues = validation.get("issues", [])
    
    # Build the prompt
    prompt = f"""
You are a professional smart contract audit report writer. Create a comprehensive audit report for the project '{project_name}'.

Analysis Summary:
{summary}

Validation Summary:
{validation_summary}

Findings:
"""
    
    # Add findings
    for i, finding in enumerate(findings):
        title = finding.get("title", f"Finding {i+1}")
        severity = finding.get("severity", "Medium")
        description = finding.get("description", "")
        
        prompt += f"\n{i+1}. {title} (Severity: {severity})\n   {description}\n"
    
    # Add recommendations
    prompt += "\nRecommendations:\n"
    for i, rec in enumerate(recommendations):
        prompt += f"\n{i+1}. {rec}"
    
    # Add report instructions
    prompt += """
Create a professional and comprehensive audit report with the following sections:

1. Executive Summary
   - Brief overview of the project
   - Summary of findings
   - Overall security assessment

2. Findings
   - Detailed description of each issue
   - Severity classification
   - Impact analysis
   - Recommendations for remediation

3. Recommendations
   - Specific, actionable recommendations
   - Best practices for improving security

4. Conclusion
   - Final assessment
   - Next steps

Format the report using clean, professional Markdown formatting.
Make it accessible to both technical and non-technical stakeholders.
"""
    
    return prompt