"""
Cognitive Bias Analyzer for Smart Contract Security

This agent analyzes smart contracts for vulnerabilities stemming from cognitive biases 
in developer thinking. It builds on the work of the DocumentationAnalyzer and 
CounterfactualGenerator to provide deeper insights into why vulnerabilities occur.
"""

from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging

from finite_monkey.adapters.ollama import AsyncOllamaClient
from finite_monkey.models.analysis import (
    CodeAnalysis, VulnerabilityReport, BiasAnalysisResult
)
from finite_monkey.utils.prompting import format_prompt

logger = logging.getLogger(__name__)

class CognitiveBiasAnalyzer:
    """
    Agent that analyzes smart contracts for vulnerabilities stemming from cognitive biases.
    """
    
    def __init__(self, llm_client: AsyncOllamaClient):
        """
        Initialize the CognitiveBiasAnalyzer agent.
        
        Args:
            llm_client: AsyncOllamaClient for generating analysis
        """
        self.llm_client = llm_client
        self.bias_categories = self._init_bias_categories()
        
    def _init_bias_categories(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the cognitive bias categories with their patterns and prompts."""
        return {
            "normalcy_bias": {
                "description": "Developers assume normal conditions will persist and fail to plan for extreme scenarios.",
                "technical_patterns": [
                    "Insufficient slippage protection",
                    "Missing circuit breakers",
                    "Inadequate liquidity checks",
                    "Lack of oracle failsafes"
                ],
                "detection_prompt": """
                Identify functions that handle financial calculations without:
                1. Bounds checking on external values
                2. Circuit breakers for extreme conditions
                3. Fallback mechanisms for component failure
                For each instance, explain what extreme scenario wasn't accounted for.
                """
            },
            "authority_bias": {
                "description": "Over-reliance on trusted roles without verification mechanisms.",
                "technical_patterns": [
                    "Privileged functions without timelocks",
                    "Missing validation of admin-supplied parameters",
                    "Centralized control of critical parameters"
                ],
                "detection_prompt": """
                Find all privileged functions that:
                1. Modify critical protocol parameters
                2. Lack validation on input values
                3. Don't use timelocks or multi-signature requirements
                Explain how each function assumes the privileged role will always act correctly.
                """
            },
            "confirmation_bias": {
                "description": "Developers focus on evidence supporting their security assumptions while ignoring contradicting scenarios.",
                "technical_patterns": [
                    "Incomplete validation checks",
                    "Edge cases handled inconsistently",
                    "Selective state verification"
                ],
                "detection_prompt": """
                Identify functions where validation is incomplete:
                1. Only some conditions are checked, while others are assumed
                2. Success paths are well-defined but failure modes are ignored
                3. Accounting changes before verifying operation success
                Explain what additional validations should be performed.
                """
            },
            "curse_of_knowledge": {
                "description": "Developers can't imagine how others might misunderstand or misuse their contracts.",
                "technical_patterns": [
                    "Unclear function semantics",
                    "Missing input validation",
                    "Ambiguous error handling"
                ],
                "detection_prompt": """
                Find functions where developer intention isn't enforced in code:
                1. Functions with unclear or ambiguous purposes
                2. Operations that assume particular timing or sequencing
                3. Methods that could be used in ways not intended
                Identify the gap between developer expectations and actual limitations.
                """
            },
            "hyperbolic_discounting": {
                "description": "Developers prioritize immediate benefits (gas savings, simpler code) over long-term security.",
                "technical_patterns": [
                    "Dangerous optimizations",
                    "Simplified validation",
                    "Skipped safety checks"
                ],
                "detection_prompt": """
                Identify code patterns where security appears to be sacrificed for efficiency:
                1. Missing validation checks that would increase gas costs
                2. Unbounded operations that save code complexity
                3. Optimizations that reduce safety margins
                Explain the security principle compromised for each optimization.
                """
            }
        }
    
    async def analyze_cognitive_biases(self, 
                                      contract_code: str, 
                                      contract_name: str,
                                      previous_analysis: Optional[CodeAnalysis] = None
                                     ) -> BiasAnalysisResult:
        """
        Analyze a smart contract for cognitive bias-related vulnerabilities.
        
        Args:
            contract_code: The smart contract source code
            contract_name: The name of the contract
            previous_analysis: Optional previous analysis results to build upon
            
        Returns:
            BiasAnalysisResult containing the bias analysis findings
        """
        logger.info(f"Analyzing cognitive biases in {contract_name}")
        
        # Initialize result
        result = BiasAnalysisResult(
            contract_name=contract_name,
            bias_findings={}
        )
        
        # Analyze each bias category in parallel
        tasks = []
        for bias_type, bias_info in self.bias_categories.items():
            task = self._analyze_specific_bias(
                contract_code=contract_code,
                bias_type=bias_type,
                bias_info=bias_info,
                previous_analysis=previous_analysis
            )
            tasks.append(task)
        
        # Gather results
        bias_results = await asyncio.gather(*tasks)
        
        # Compile results
        for bias_type, findings in bias_results:
            result.bias_findings[bias_type] = findings
            
        return result
    
    async def _analyze_specific_bias(self, 
                                    contract_code: str, 
                                    bias_type: str, 
                                    bias_info: Dict[str, Any],
                                    previous_analysis: Optional[CodeAnalysis]
                                   ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze a specific cognitive bias in the contract.
        
        Args:
            contract_code: The smart contract source code
            bias_type: The type of cognitive bias to analyze
            bias_info: Information about the bias
            previous_analysis: Optional previous analysis results
            
        Returns:
            Tuple of (bias_type, findings)
        """
        logger.info(f"Analyzing {bias_type}")
        
        # Construct prompt for the LLM
        prompt = format_prompt(
            template="cognitive_bias_analysis",
            contract_code=contract_code,
            bias_description=bias_info["description"],
            technical_patterns="\n".join([f"- {p}" for p in bias_info["technical_patterns"]]),
            detection_prompt=bias_info["detection_prompt"],
            previous_findings=previous_analysis.vulnerabilities if previous_analysis else None
        )
        
        # Get analysis from LLM
        response = await self.llm_client.completion(prompt=prompt)
        
        # Parse the response
        findings = self._parse_bias_findings(response, bias_type)
        
        return bias_type, findings
    
    def _parse_bias_findings(self, 
                           llm_response: str, 
                           bias_type: str
                          ) -> Dict[str, Any]:
        """
        Parse the LLM response into structured findings.
        
        Args:
            llm_response: The response from the LLM
            bias_type: The type of cognitive bias analyzed
            
        Returns:
            Dictionary of structured findings
        """
        # This is a simple parser that can be enhanced
        findings = {
            "instances": [],
            "summary": "",
            "severity": "medium",  # Default, can be updated based on content
        }
        
        # Extract instances (basic parsing - would be improved with actual implementation)
        lines = llm_response.split("\n")
        current_instance = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for instance headers
            if line.startswith("#") or line.startswith("Function:") or line.startswith("Issue:"):
                # Save previous instance if it exists
                if current_instance and current_instance.get("description"):
                    findings["instances"].append(current_instance)
                
                # Start new instance
                current_instance = {
                    "title": line.lstrip("#").strip(),
                    "description": "",
                    "location": "",
                    "suggestion": ""
                }
            elif current_instance:
                # Parse location information
                if line.startswith("Location:") or line.startswith("Line:"):
                    current_instance["location"] = line.split(":", 1)[1].strip()
                # Parse suggestion information
                elif line.startswith("Fix:") or line.startswith("Suggestion:") or line.startswith("Recommendation:"):
                    current_instance["suggestion"] = line.split(":", 1)[1].strip()
                # Otherwise add to description
                else:
                    current_instance["description"] += line + "\n"
        
        # Add the last instance if it exists
        if current_instance and current_instance.get("description"):
            findings["instances"].append(current_instance)
            
        # Extract summary (assume it's at the end or beginning)
        summary_markers = ["Summary:", "Overall:", "Conclusion:"]
        for marker in summary_markers:
            if marker in llm_response:
                summary_part = llm_response.split(marker, 1)[1].strip()
                findings["summary"] = summary_part.split("\n\n")[0].strip()
                break
                
        if not findings["summary"] and findings["instances"]:
            # Generate a simple summary
            findings["summary"] = f"Found {len(findings['instances'])} instances of {bias_type.replace('_', ' ')}."
            
        return findings
    
    async def generate_assumption_analysis(self, 
                                         contract_code: str,
                                         vulnerability_reports: List[VulnerabilityReport]
                                        ) -> Dict[str, Any]:
        """
        Generate an analysis of developer assumptions for identified vulnerabilities.
        
        Args:
            contract_code: The smart contract source code
            vulnerability_reports: List of vulnerability reports from previous analysis
            
        Returns:
            Dictionary mapping assumptions to vulnerability instances
        """
        logger.info("Generating developer assumption analysis")
        
        # Define assumption categories (from the Stage 3 in notes)
        assumption_categories = [
            "This will only be called in the expected order",
            "This value will never be zero/extreme",
            "Only authorized users would call this function",
            "This interaction will always succeed",
            "Users will use this as intended"
        ]
        
        # Format vulnerabilities for prompt
        vuln_sections = []
        for report in vulnerability_reports:
            vuln_sections.append(f"Title: {report.title}\n"
                               f"Description: {report.description}\n"
                               f"Location: {report.location}\n"
                               f"Severity: {report.severity}\n")
        
        vulnerabilities_text = "\n---\n".join(vuln_sections)
        
        # Construct prompt
        prompt = format_prompt(
            template="developer_assumption_analysis",
            contract_code=contract_code,
            vulnerabilities=vulnerabilities_text,
            assumption_categories="\n".join([f"- {a}" for a in assumption_categories])
        )
        
        # Get analysis from LLM
        response = await self.llm_client.completion(prompt=prompt)
        
        # Parse response (basic implementation)
        assumptions = {}
        current_assumption = None
        
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Check if line is an assumption category
            for category in assumption_categories:
                if line.startswith(f"### {category}") or line == category:
                    current_assumption = category
                    assumptions[current_assumption] = {"description": "", "vulnerabilities": []}
                    break
                    
            if current_assumption:
                # Check if line refers to a vulnerability
                for i, report in enumerate(vulnerability_reports):
                    if report.title in line or f"Vulnerability {i+1}" in line:
                        assumptions[current_assumption]["vulnerabilities"].append(report.title)
                    elif "description" in assumptions[current_assumption]:
                        # Add to description if not a vulnerability reference
                        assumptions[current_assumption]["description"] += line + "\n"
                        
        return assumptions
    
    async def generate_remediation_plan(self,
                                      contract_code: str,
                                      bias_analysis: BiasAnalysisResult
                                     ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate remediation plans for identified cognitive bias vulnerabilities.
        
        Args:
            contract_code: The smart contract source code
            bias_analysis: The bias analysis result
            
        Returns:
            Dictionary mapping bias types to lists of remediation steps
        """
        logger.info("Generating remediation plan")
        
        remediation_plans = {}
        
        for bias_type, findings in bias_analysis.bias_findings.items():
            # Skip empty findings
            if not findings.get("instances"):
                continue
                
            # Format instances for the prompt
            instances_text = ""
            for i, instance in enumerate(findings["instances"]):
                instances_text += f"Issue {i+1}: {instance['title']}\n"
                instances_text += f"Location: {instance['location']}\n"
                instances_text += f"Description: {instance['description']}\n\n"
            
            # Create prompt
            prompt = format_prompt(
                template="cognitive_bias_remediation",
                contract_code=contract_code,
                bias_type=bias_type.replace("_", " "),
                bias_description=self.bias_categories[bias_type]["description"],
                instances=instances_text
            )
            
            # Get remediation from LLM
            response = await self.llm_client.completion(prompt=prompt)
            
            # Parse remediation (simplified implementation)
            remediation_steps = []
            current_step = None
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("### ") or line.startswith("## ") or line.startswith("# "):
                    # Save previous step if it exists
                    if current_step and current_step.get("description"):
                        remediation_steps.append(current_step)
                    
                    # Start new step
                    current_step = {
                        "title": line.lstrip("#").strip(),
                        "description": "",
                        "code_example": "",
                        "validation": ""
                    }
                elif current_step:
                    # Check for code example
                    if "```" in line:
                        in_code_block = True
                        current_step["code_example"] = ""
                    elif in_code_block:
                        if "```" in line:
                            in_code_block = False
                        else:
                            current_step["code_example"] += line + "\n"
                    # Check for validation section
                    elif line.startswith("Validation:") or "To validate this fix" in line:
                        current_step["validation"] = line
                    # Otherwise add to description
                    else:
                        current_step["description"] += line + "\n"
            
            # Add the last step if it exists
            if current_step and current_step.get("description"):
                remediation_steps.append(current_step)
                
            remediation_plans[bias_type] = remediation_steps
            
        return remediation_plans