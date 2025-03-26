"""
Business flow analyzer that uses LlamaIndex for structured output.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import BusinessFlowData, SecurityAnalysisResult
from .llama_index_adapter import LlamaIndexAdapter
from llama_index.core.llms import ChatMessage, MessageRole
import json
logger = logging.getLogger(__name__)

class BusinessFlowAnalyzer:
    """
    Business flow analyzer using LlamaIndex for structured output.
    """
    
    def __init__(self, model_name: str = "ollama:llama3", **kwargs):
        """
        Initialize the business flow analyzer.
        
        Args:
            model_name: Model identifier (e.g., "openai:gpt-4o", "ollama:llama3")
            **kwargs: Additional LLM configuration options
        """
        self.adapter = LlamaIndexAdapter(model_name=model_name, **kwargs)
    
    async def analyze_contract_flow(self,
                                  contract_code: str,
                                  contract_name: Optional[str] = "Contract") -> BusinessFlowData:
        """
        Analyze the business flow in a smart contract.
        
        Args:
            contract_code: Smart contract source code
            contract_name: Optional contract name
            
        Returns:
            BusinessFlowData object with the analysis
        """
        logger.info(f"Analyzing business flow for contract: {contract_name}")
        
        # Create analysis prompt
        prompt = f"""Analyze the business flow in the following Solidity contract:
        
        Contract: {contract_name}
        
        ```solidity
        {contract_code}
        ```
        
        Create a graph representation of the business flow with nodes and links.
        For each node, include:
        - A unique ID
        - A descriptive name
        - The type (function, validation, state, event, external)
        - A description of its purpose
        - A confidence score (0-1) of your analysis
        
        For each link, include:
        - Source node ID
        - Target node ID
        - A descriptive label
        - A confidence score (0-1) of your analysis
        
        Identify all key functions, state variables, events, and external interactions.
        Include confidence scores for your identifications.
        """
        
        system_prompt = "You are an expert smart contract analyzer focused on understanding business flows and function relationships."
        
        try:
            # Get structured LLM instance
            structured_llm = self.adapter.as_structured_llm(BusinessFlowData)
            
            # Create chat message
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            # Generate structured output
            result = json.loads((await structured_llm.achat(messages)).message.content)
            
            # Ensure we have a timestamp
            if not result.timestamp:
                result.timestamp = datetime.now()
                
            # Add metadata
            result.metadata["analyzer"] = "business_flow_analyzer"
            result.metadata["contract_name"] = contract_name
            result.metadata["model_used"] = self.adapter.model_name
            
            # Set default size for nodes if not provided
            for node in result.nodes:
                if node.size is None:
                    if node.type == "function":
                        node.size = 15
                    elif node.type == "external":
                        node.size = 14
                    else:
                        node.size = 12
            
            logger.info(f"Business flow analysis complete - found {len(result.nodes)} nodes and {len(result.links)} links")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing business flow: {e}", exc_info=True)
            # Return an empty but valid BusinessFlowData object
            return BusinessFlowData(
                nodes=[],
                links=[],
                metadata={"error": str(e), "contract_name": contract_name}
            )
    
    async def analyze_security(self, 
                             contract_code: str,
                             flow_data: Optional[BusinessFlowData] = None,
                             contract_name: Optional[str] = "Contract") -> SecurityAnalysisResult:
        """
        Analyze security issues in a smart contract.
        
        Args:
            contract_code: Smart contract source code
            flow_data: Optional business flow data from previous analysis
            contract_name: Optional contract name
            
        Returns:
            SecurityAnalysisResult object with the findings
        """
        logger.info(f"Analyzing security for contract: {contract_name}")
        
        # Create analysis prompt
        prompt = f"""Analyze the security of the following Solidity contract:
        
        Contract: {contract_name}
        
        ```solidity
        {contract_code}
        ```
        
        Identify any security vulnerabilities, including:
        - Reentrancy attacks
        - Access control issues
        - Integer overflow/underflow
        - Unprotected functions
        - Improper error handling
        - Any other security concerns
        
        For each finding, include:
        - A clear title
        - A detailed description
        - The severity (high, medium, low)
        - The location in the code
        - A recommendation to address the issue
        - A confidence score (0-1) for your finding        """
        
        # If we have flow data, add it to the prompt
        if flow_data and flow_data.nodes:
            prompt += "\n\nBased on the business flow analysis, these are the key components of the contract:\n\n"
            
            # Add information about functions
            functions = [node for node in flow_data.nodes if node.type == "function"]
            if functions:
                prompt += "Functions:\n"
                for func in functions:
                    prompt += f"- {func.name}: {func.description or 'No description'}\n"
            
            # Add information about state variables
            states = [node for node in flow_data.nodes if node.type == "state"]
            if states:
                prompt += "\nState variables:\n"
                for state in states:
                    prompt += f"- {state.name}: {state.description or 'No description'}\n"
            
            # Add information about external interactions
            externals = [node for node in flow_data.nodes if node.type == "external"]
            if externals:
                prompt += "\nExternal interactions:\n"
                for ext in externals:
                    prompt += f"- {ext.name}: {ext.description or 'No description'}\n"
        
        system_prompt = "You are an expert smart contract security analyst focused on identifying vulnerabilities and providing remediation advice."
        
        try:
            # Get structured LLM instance
            structured_llm = self.adapter.as_structured_llm(SecurityAnalysisResult)
            
            # Create chat message
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            # Generate structured output
            result = json.loads((await structured_llm.achat(messages)).message.content)
            
            # Ensure we have a timestamp
            if not result.timestamp:
                result.timestamp = datetime.now()
                
            # Add metadata
            result.metadata["analyzer"] = "security_analyzer"
            result.metadata["contract_name"] = contract_name
            result.metadata["model_used"] = self.adapter.model_name
            
            # Calculate risk score if not provided
            if result.risk_score is None and result.findings:
                severity_weights = {"high": 9.0, "medium": 5.0, "low": 2.0}
                total_weight = sum(severity_weights.get(finding.severity, 0) for finding in result.findings)
                
                if result.findings:
                    result.risk_score = min(total_weight / len(result.findings), 10.0)
                else:
                    result.risk_score = 0.0
            
            logger.info(f"Security analysis complete - found {len(result.findings)} issues")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing security: {e}", exc_info=True)
            # Return an empty but valid SecurityAnalysisResult object
            return SecurityAnalysisResult(
                findings=[],
                summary=f"Error during analysis: {str(e)}",
                metadata={"error": str(e), "contract_name": contract_name}
            )
