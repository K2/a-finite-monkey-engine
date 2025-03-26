"""
LlamaIndex validator agent implementation

This module provides integration with LlamaIndex's agent framework for
validating vulnerability findings.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Union

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama as LlamaOllama

from ..adapters.ollama import AsyncOllamaClient
from ..nodes_config import nodes_config


class ValidatorAgent:
    """
    Validator agent for verifying analysis results using LlamaIndex
    
    This agent is responsible for validating security findings and
    providing an independent assessment of vulnerabilities discovered
    during the research phase.
    
    This is part of the inner agent layer in the Finite Monkey architecture,
    which uses llama-index agents for structured tasks.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        timeout: int = 300,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the validator agent
        
        Args:
            model: LLM model to use
            timeout: Request timeout in seconds
            prompt_template: Custom system prompt template
        """
        # Get settings from config
        config = nodes_config()
        
        # Set up the model
        self.model = model or config.CONFIRMATION_MODEL or "phi4-mini:3.8b"
        self.timeout = timeout
        
        # Define system prompt
        self.system_prompt = prompt_template or (
            "You are an expert smart contract security validator specialized in critically "
            "assessing vulnerability reports. Your task is to carefully analyze security findings, "
            "distinguish between true and false positives, and provide an independent assessment. "
            "You should be skeptical but fair, focusing on evidence-based validation."
        )
        
        # Initialize the LlamaIndex agent
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the LlamaIndex agent with appropriate tools"""
        
        # Define validation tools
        def validate_finding(
            code: str, 
            finding: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Validate a security finding against the code.
            
            Args:
                code: Solidity code to analyze
                finding: Finding to validate with type, description, severity, etc.
                
            Returns:
                Validation results
            """
            # This is a stub - the actual validation is done by the LLM
            # We just define the tool interface here and the LLM will
            # decide what to return based on its analysis
            return {}
        
        def check_for_missed_vulnerabilities(
            code: str,
            existing_findings: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """
            Check for vulnerabilities that might have been missed in the initial analysis.
            
            Args:
                code: Solidity code to analyze
                existing_findings: List of already identified findings
                
            Returns:
                Any additional findings
            """
            # This is a stub - the actual analysis is done by the LLM
            # We just define the tool interface here and the LLM will
            # decide what to return based on its analysis
            return {}
        
        # Create tools
        tools = [
            FunctionTool.from_defaults(fn=validate_finding),
            FunctionTool.from_defaults(fn=check_for_missed_vulnerabilities),
        ]
        
        # Create LlamaIndex agent
        self.agent = FunctionAgent(
            name="VulnerabilityValidator",
            description="Validates smart contract security vulnerability findings",
            tools=tools,
            llm=LlamaOllama(model=self.model, request_timeout=self.timeout),
            system_prompt=self.system_prompt,
        )
    
    async def validate(
        self, 
        code: str, 
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate findings using the validator agent
        
        Args:
            code: Solidity code being analyzed
            findings: List of findings from the research phase
            
        Returns:
            Validation results
        """
        # Format findings for the prompt
        findings_str = ""
        for i, finding in enumerate(findings, 1):
            finding_type = finding.get("type", "Unknown")
            severity = finding.get("severity", "Medium")
            description = finding.get("description", "No description provided")
            location = finding.get("location", "Unknown location")
            
            findings_str += f"Finding #{i}: {finding_type} (Severity: {severity})\n"
            findings_str += f"Location: {location}\n"
            findings_str += f"Description: {description}\n\n"
        
        # Prepare the prompt
        prompt = f"""
        Validate the following security findings for this Solidity code:
        
        ```solidity
        {code}
        ```
        
        FINDINGS TO VALIDATE:
        {findings_str}
        
        For each finding, determine if it is:
        1. Confirmed (true positive)
        2. False positive
        3. Needs more context
        
        If confirmed, provide additional context or insights.
        If false positive, explain why with code references.
        Also check for any vulnerabilities that might have been missed in the initial analysis.
        """
        
        # Run the agent (use run() instead of arun() for compatibility)
        response = await self.agent.run(prompt)
        
        # Process response
        return {
            "raw_response": str(response),
            "structured_validation": self._extract_validation(str(response), findings)
        }
    
    def _extract_validation(
        self, 
        text: str, 
        original_findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract structured validation results from the agent's response
        
        Args:
            text: Raw response text
            original_findings: Original findings being validated
            
        Returns:
            Structured validation results
        """
        validation_results = {
            "confirmed": [],
            "false_positives": [],
            "needs_context": [],
            "missed_vulnerabilities": []
        }
        
        # Basic implementation - in practice, would need more robust parsing
        if "MISSED VULNERABILITIES:" in text:
            missed_section = text.split("MISSED VULNERABILITIES:")[1].strip()
            # Extract missed vulnerabilities
            # This is a simplified version - real implementation would be more robust
            for line in missed_section.split("\n"):
                if line.strip() and not line.startswith("-"):
                    validation_results["missed_vulnerabilities"].append({
                        "description": line.strip()
                    })
        
        # Extract validation status for each finding
        for i, finding in enumerate(original_findings, 1):
            finding_id = f"Finding #{i}"
            
            if finding_id in text:
                finding_section = text.split(finding_id)[1].split("Finding #")[0] if i < len(original_findings) else text.split(finding_id)[1]
                
                # Determine status
                status = "needs_context"  # Default
                
                if "confirmed" in finding_section.lower():
                    status = "confirmed"
                elif "false positive" in finding_section.lower():
                    status = "false_positives"
                
                # Add to appropriate list
                validation_results[status].append({
                    "original_finding": finding,
                    "validation_notes": finding_section.strip()
                })
        
        return validation_results