"""
Specialized guidance module for business flow analysis in smart contracts.
"""

import logging
from typing import Dict, List, Any, Optional, cast
from datetime import datetime

from .core import GuidanceManager
from .adapter import GuidanceAdapter
from .models import BusinessFlowData, SecurityAnalysisResult, Node, Link, SecurityFinding

logger = logging.getLogger(__name__)

class BusinessFlowAnalyzer:
    """
    Uses guidance to analyze business flows in smart contracts.
    Implements constrained generation for consistent outputs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the business flow analyzer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.manager = GuidanceManager(config)
        self.adapter = GuidanceAdapter(config)
    
    async def analyze_contract_flow(self, 
                                  contract_code: str, 
                                  model_identifier: Optional[str] = None) -> BusinessFlowData:
        """
        Analyze the business flow in a smart contract using guidance constraints.
        
        Args:
            contract_code: Smart contract source code
            model_identifier: Model to use for analysis
            
        Returns:
            BusinessFlowData Pydantic model representing the business flow
        """
        logger.info("Starting business flow analysis")
        # Build system prompt for contract analysis
        system_prompt = """
        You are a business flow analyzer examining smart contract functions.
        Follow these rules strictly:
        1. Identify all key steps in the function execution
        2. Map data flows between components
        3. Highlight security checkpoints
        4. Note state changes
        5. Include confidence scores where appropriate
        """
        
        # Add user prompt with contract code
        user_prompt = f"""
        Analyze the following smart contract and map its business flow:
        
        ```solidity
        {contract_code}
        ```
        
        Extract the nodes and links of the business flow. Be detailed in your analysis
        and include confidence scores for your assertions.
        """
        
        try:
            # Use the structured output approach
            logger.info(f"Executing structured prompt with BusinessFlowData output class")
            result = await self.manager.execute_structured_prompt(
                prompt_text=user_prompt,
                output_class=BusinessFlowData,
                system_prompt=system_prompt,
                model_identifier=model_identifier
            )
            
            logger.info(f"Analysis complete - received {len(result.nodes)} nodes and {len(result.links)} links")
            
            # Add default size to nodes based on type if not already set
            for node in result.nodes:
                if node.size is None:
                    node_type = node.type
                    if node_type == "function":
                        node.size = 15
                    elif node_type == "external":
                        node.size = 14
                    else:
                        node.size = 12
            
            # Add timestamp if not present
            if not result.timestamp:
                result.timestamp = datetime.now()
            
            # Add metadata about the analysis
            result.metadata["analyzer_version"] = "1.0.0"
            result.metadata["model_used"] = model_identifier or self.manager.default_model
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing contract flow: {e}", exc_info=True)
            # Return an empty but valid BusinessFlowData object
            return BusinessFlowData(
                nodes=[], 
                links=[],
                metadata={"error": str(e)}
            )
    
    async def generate_security_analysis(self,
                                      contract_code: str,
                                      flow_data: Optional[BusinessFlowData] = None,
                                      model_identifier: Optional[str] = None) -> SecurityAnalysisResult:
        """
        Generate security analysis for contract functions using guidance.
        
        Args:
            contract_code: Smart contract source code
            flow_data: Optional flow data from previous analysis
            model_identifier: Model to use for analysis
            
        Returns:
            SecurityAnalysisResult Pydantic model with findings
        """
        # Build system prompt
        system_prompt = """
        You are an expert smart contract security analyst.
        Analyze the given contract code and identify potential security vulnerabilities.
        Follow these rules strictly:
        1. Focus on common vulnerabilities like reentrancy, overflow/underflow, access control issues
        2. Provide specific line references when possible
        3. Assign a severity level (high, medium, low) to each finding
        4. Be specific about the potential impact of each vulnerability
        5. Include confidence levels for each finding
        6. Provide CWE IDs when applicable
        """
        
        # Build user prompt
        user_prompt = f"""
        Analyze the following smart contract for security vulnerabilities:
        
        ```solidity
        {contract_code}
        ```
        """
        
        if flow_data:
            # Use the structured flow data to provide context
            user_prompt += "\n\nHere's the business flow analysis to provide context for your security analysis:\n\n"
            
            # Add information about nodes by type
            functions = [node for node in flow_data.nodes if node.type == "function"]
            if functions:
                user_prompt += "\nFunctions identified:\n"
                for func in functions:
                    user_prompt += f"- {func.name}: {func.description or 'No description'}\n"
            
            validations = [node for node in flow_data.nodes if node.type == "validation"]
            if validations:
                user_prompt += "\nValidation points identified:\n"
                for val in validations:
                    user_prompt += f"- {val.name}: {val.description or 'No description'}\n"
            
            state_changes = [node for node in flow_data.nodes if node.type == "state"]
            if state_changes:
                user_prompt += "\nState changes identified:\n"
                for state in state_changes:
                    user_prompt += f"- {state.name}: {state.description or 'No description'}\n"
        
        try:
            # Use structured output approach
            result = await self.manager.execute_structured_prompt(
                prompt_text=user_prompt,
                output_class=SecurityAnalysisResult,
                system_prompt=system_prompt,
                model_identifier=model_identifier
            )
            
            # Ensure timestamp is set
            if not result.timestamp:
                result.timestamp = datetime.now()
                
            # Add metadata
            result.metadata["analyzer_version"] = "1.0.0"
            result.metadata["model_used"] = model_identifier or self.manager.default_model
            
            # Calculate overall risk score if not provided
            if result.risk_score is None and result.findings:
                # Simple scoring: average of findings with severity weights
                severity_weights = {"high": 9.0, "medium": 5.0, "low": 2.0}
                total_weight = 0
                for finding in result.findings:
                    total_weight += severity_weights.get(finding.severity, 0)
                
                if result.findings:
                    result.risk_score = min(total_weight / len(result.findings), 10.0)
                else:
                    result.risk_score = 0.0
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating security analysis: {e}")
            # Return empty but valid SecurityAnalysisResult
            return SecurityAnalysisResult(
                findings=[],
                summary=f"Error during analysis: {str(e)}",
                metadata={"error": str(e)}
            )
