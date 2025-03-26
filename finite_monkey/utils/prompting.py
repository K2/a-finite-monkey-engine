"""
Prompt templates and utilities for LLM interactions

This module provides templates and utilities for generating prompts for LLM
interactions. It integrates with the database-driven prompt system to allow
dynamic prompt updates.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union

# Global variable to hold prompt service instance (initialized lazily)
_prompt_service = None


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


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with the given arguments
    
    Args:
        template: Name of the template to format
        **kwargs: Arguments to format the template with
        
    Returns:
        Formatted prompt
    """
    templates = {
        **get_business_flow_prompts(),
        **get_cognitive_bias_prompts(),
    }
    
    if template not in templates:
        raise ValueError(f"Template {template} not found")
    
    # Format the template
    return templates[template].format(**kwargs)


def get_cognitive_bias_prompts() -> Dict[str, str]:
    """
    Get prompts for cognitive bias analysis
    
    Returns:
        Dictionary of prompt templates
    """
    return {
        "cognitive_bias_analysis": """
You are an expert smart contract security auditor specializing in identifying cognitive bias patterns in code.
Your task is to analyze the following contract for vulnerabilities resulting from {bias_description}

CONTRACT:
```solidity
{contract_code}
```

Technical patterns associated with this cognitive bias include:
{technical_patterns}

DETECTION INSTRUCTIONS:
{detection_prompt}

Analyze the code carefully, looking for instances of this cognitive bias. For each instance:
1. Identify the specific function or code section that demonstrates the bias
2. Explain how the developer's thinking exhibits this cognitive bias
3. Describe the potential security vulnerability that results from this bias
4. Rate the severity (Critical, High, Medium, Low, Informational)
5. Suggest how to fix the issue

Provide your analysis in a structured format with clear sections for each finding.
Be specific and reference line numbers or function names where possible.
Conclude with a summary of all findings related to this cognitive bias.
""",

        "developer_assumption_analysis": """
You are an expert in analyzing the cognitive biases and assumptions that lead to security vulnerabilities in smart contracts.
Analyze the following contract and the vulnerabilities that have been identified:

CONTRACT:
```solidity
{contract_code}
```

VULNERABILITIES IDENTIFIED:
{vulnerabilities}

Your task is to map each vulnerability to the developer assumption that created it.
Consider these common assumption categories:
{assumption_categories}

For each assumption category that applies:
1. Explain how this assumption manifests in the code
2. List which vulnerabilities resulted from this assumption
3. Describe why this assumption fails in real-world conditions
4. Suggest how to adjust the developer's mental model to avoid this assumption

Provide a detailed analysis for each relevant assumption category.
Be specific and reference the actual vulnerabilities listed above.
""",

        "cognitive_bias_remediation": """
You are a smart contract security remediation specialist focusing on cognitive bias-driven vulnerabilities.
You need to create a remediation plan for {bias_type} bias vulnerabilities in the following contract:

CONTRACT:
```solidity
{contract_code}
```

BIAS DESCRIPTION: {bias_description}

IDENTIFIED INSTANCES:
{instances}

For each instance, provide a detailed remediation plan that includes:

1. Specific code changes that address both the technical vulnerability and the underlying cognitive bias
2. Explanation of how the fix addresses the root cause, not just the symptom
3. Alternative approaches that could also work
4. Validation checks to ensure the fix is correct
5. Any potential side effects of the proposed fix that should be considered

Make your remediation advice concrete and actionable.
Include code examples that demonstrate proper implementation.
Ensure your fixes are gas-efficient and follow best practices.
""",

        "four_stage_analysis": """
You are a smart contract security analyzer using a structured four-stage approach to vulnerability detection.
Analyze the following contract using all four stages of the analysis framework:

CONTRACT:
```solidity
{contract_code}
```

STAGE 1: Pattern-Based Vulnerability Identification
First, scan the contract for these specific vulnerability patterns and catalog all instances:
- Pattern A: External calls followed by state changes
- Pattern B: Division without zero-checks
- Pattern C: Public state-changing functions without access control
- Pattern D: Single-step privilege operations
List each instance with line numbers.

STAGE 2: Context and Impact Analysis
For each pattern identified, answer these specific questions:
1. What is the DIRECT impact if exploited? (funds lost, parameters corrupted)
2. What is the WIDER impact on the protocol? (cascading failures, economic damage)
3. Who could EXPLOIT this vulnerability? (any user, specific roles, sophisticated attackers)
4. What CONDITIONS need to exist for successful exploitation?
Provide specific, concrete answers for each instance.

STAGE 3: Developer Assumption Analysis
For each vulnerability, identify which developer assumption created it:
A. "This will only be called in the expected order"
B. "This value will never be zero/extreme"
C. "Only authorized users would call this function"
D. "This interaction will always succeed"
E. "Users will use this as intended"
Explain exactly how this assumption fails under real-world conditions.

STAGE 4: Remediation and Validation
For each vulnerability:
1. Provide a SPECIFIC code fix (not just general advice)
2. Explain how the fix addresses the root cause
3. Add a validation check that would confirm the fix works
4. Suggest a test scenario that would verify security
Ensure fixes address both the technical vulnerability and the underlying assumption failure.

Present your analysis in a clear, organized format covering all four stages.
"""
    }


def get_business_flow_prompts() -> Dict[str, str]:
    """
    Get prompts for business flow extraction
    
    Returns:
        Dictionary of prompt templates
    """
    return {
        "function_analysis": """
You are an expert smart contract code analyzer with a deep understanding of business logic.
Analyze the following solidity function from the {contract_name} contract:

```solidity
{function_code}
```

Function signature: {function_signature}

Contract context:
{contract_context}

I need you to analyze this function's business purpose and role in the contract's overall functionality.
Provide your analysis in the following structure:

Function Type: [Categorize the function's primary type (e.g., setter, getter, transfer, withdraw, deposit, swap, mint, burn, etc.)]

Business Purpose: [One-line description of the function's business purpose]

Description: [Detailed description of what the function does in business terms]

State Changes: [List the state variables modified by this function and how they change]

External Calls: [List any external contract calls made by this function]

Security Considerations: [List any security concerns related to this function]

Business Flow Potential: [High/Medium/Low - indicate if this function is likely to be part of a critical business flow]

Your analysis should focus on the business logic and purpose, not just code mechanics.
""",

        "function_relationships": """
You are an expert in identifying business processes and workflows in smart contracts.
Analyze the functions in the {contract_name} {contract_type} to identify relationships and business flows.

Key Functions:
{functions}

Function Relationships:
{relationship_graph}

Identify the following:

1. Workflows: Sequences of function calls that together implement a business process
2. Key Functions: The most important functions that define the contract's core functionality
3. Function Groups: Sets of functions that work together to implement a specific feature

Provide your analysis in JSON format:

```json
{{
  "workflows": [
    {{
      "name": "Workflow Name",
      "functions": ["function1", "function2"],
      "description": "What this workflow accomplishes"
    }}
  ],
  "key_functions": ["function1", "function2"],
  "function_groups": [
    {{
      "name": "Group Name",
      "functions": ["function1", "function2"],
      "purpose": "What this group of functions does"
    }}
  ]
}}
```

Focus on identifying meaningful business flows rather than simple implementation details.
""",

        "business_flow_validation": """
You are an expert in smart contract business logic and security analysis.
Validate and enhance the following business flow extracted from the {contract_name} contract:

Flow Name: {flow_name}
Flow Type: {flow_type}
Description: {flow_description}

Code:
```solidity
{flow_code}
```

Context:
{flow_context}

Your task is to:
1. Validate if this is a meaningful business flow
2. Enhance the flow name, description, and type if needed
3. Add any missing context that would help in understanding this flow
4. Identify potential security considerations related to this flow

Provide your analysis in the following format:

Valid Business Flow: [Yes/No]

Enhanced Name: [Improved name for the flow]

Enhanced Type: [Improved type classification]

Enhanced Description: [More detailed and accurate description]

Additional Context: [Any missing contextual information]

Security Considerations:
- [Security consideration 1]
- [Security consideration 2]

Focus on how this flow contributes to the overall business logic of the contract and any security implications it may have.
""",

        "project_scan": """
You are analyzing a smart contract project for security vulnerabilities and potential business logic flaws.
Based on the identified business flows, you need to generate specific checks to perform.

Project Overview:
{project_overview}

Identified Business Flows:
{business_flows}

Your task is to generate a comprehensive set of security and business logic checks that should be performed
on these business flows. The checks should be specific to the identified flows, not generic checks.

For each business flow, provide:

1. Flow Name: [Name of the flow]
2. Critical Invariants:
   - [Invariant 1]: [Why it matters]
   - [Invariant 2]: [Why it matters]
3. Security Checks:
   - [Check 1]: [Description of what to verify]
   - [Check 2]: [Description of what to verify]
4. Edge Cases to Test:
   - [Edge case 1]
   - [Edge case 2]
5. Business Logic Verification:
   - [Verification point 1]
   - [Verification point 2]

Focus on the most critical aspects of each business flow, particularly concerning value handling,
access control, state transitions, and interactions with external contracts.
"""
    }


# Dynamic prompt functions that integrate with the database
async def get_dynamic_prompt(prompt_name: str, project_id: Optional[str] = None, **kwargs) -> str:
    """
    Get a dynamic prompt from the database (async version)
    
    Args:
        prompt_name: Name of the prompt
        project_id: Project ID (optional)
        **kwargs: Parameters for the prompt
        
    Returns:
        Rendered prompt string
    """
    global _prompt_service
    
    if _prompt_service is None:
        from finite_monkey.db.prompts import prompt_service
        _prompt_service = prompt_service
    
    return await _prompt_service.get_prompt(prompt_name, project_id, **kwargs)


def get_dynamic_prompt_sync(prompt_name: str, project_id: Optional[str] = None, **kwargs) -> str:
    """
    Get a dynamic prompt from the database (sync version)
    
    Args:
        prompt_name: Name of the prompt
        project_id: Project ID (optional)
        **kwargs: Parameters for the prompt
        
    Returns:
        Rendered prompt string
    """
    # Run the async function in a new event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(get_dynamic_prompt(prompt_name, project_id, **kwargs))


async def store_prompt_result(prompt_name: str, result: str, project_id: Optional[str] = None) -> bool:
    """
    Store a prompt result in the database
    
    Args:
        prompt_name: Name of the prompt
        result: Result text
        project_id: Project ID (optional)
        
    Returns:
        True if successful, False otherwise
    """
    global _prompt_service
    
    if _prompt_service is None:
        from finite_monkey.db.prompts import prompt_service
        _prompt_service = prompt_service
    
    return await _prompt_service.store_prompt_result(prompt_name, result, project_id)


def store_prompt_result_sync(prompt_name: str, result: str, project_id: Optional[str] = None) -> bool:
    """
    Store a prompt result in the database (sync version)
    
    Args:
        prompt_name: Name of the prompt
        result: Result text
        project_id: Project ID (optional)
        
    Returns:
        True if successful, False otherwise
    """
    # Run the async function in a new event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(store_prompt_result(prompt_name, result, project_id))