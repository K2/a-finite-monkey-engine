"""
LlamaIndex agent implementation for Finite Monkey framework

This module provides integration with LlamaIndex's agent framework for 
code analysis and vulnerability detection.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Union

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama as LlamaOllama
from llama_index.core.response_synthesizers import ResponseMode

from ..adapters.ollama import AsyncOllamaClient
from ..nodes_config import nodes_config


class ResearchAgent:
    """
    Research agent for code analysis using LlamaIndex
    
    This agent is responsible for performing initial code analysis
    and vulnerability detection using LlamaIndex's agent framework.
    
    This is the inner agent that follows the architecture outlined in README-EN.md,
    where llama-index agents form the inner core for structured analysis tasks.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        timeout: int = 300,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the research agent
        
        Args:
            model: LLM model to use
            timeout: Request timeout in seconds
            prompt_template: Custom system prompt template
        """
        # Get settings from config
        config = nodes_config()
        
        # Set up the model
        self.model = model or config.WORKFLOW_MODEL or "qwen2.5-coder:latest"
        self.timeout = timeout
        
        # Define system prompt
        self.system_prompt = prompt_template or (
            "You are an expert smart contract security researcher specialized in vulnerability detection. "
            "Your task is to thoroughly analyze Solidity code for security vulnerabilities, logical flaws, "
            "and potential exploits. Focus on identifying concrete issues with clear explanations and evidence."
        )
        
        # Initialize the LlamaIndex agent
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the LlamaIndex agent with appropriate tools"""
        
        # Define analysis tools
        def analyze_vulnerabilities(code: str, focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
            """
            Analyze smart contract code for security vulnerabilities.
            
            Args:
                code: Solidity code to analyze
                focus_areas: Optional list of vulnerability types to focus on
                
            Returns:
                Dictionary containing analysis results
            """
            # This is a stub - the actual analysis is done by the LLM
            # We just define the tool interface here and the LLM will
            # decide what to return based on its analysis
            return {}
        
        # Create tools
        tools = [
            FunctionTool.from_defaults(fn=analyze_vulnerabilities),
        ]
        
        # Create LlamaIndex agent
        self.agent = FunctionAgent(
            name="VulnerabilityResearcher",
            description="Analyzes smart contract code for security vulnerabilities",
            tools=tools,
            llm=LlamaOllama(model=self.model, request_timeout=self.timeout),
            system_prompt=self.system_prompt,
        )
    
    async def analyze(self, code: str, query: str) -> Dict[str, Any]:
        """
        Analyze code using the research agent
        
        Args:
            code: Solidity code to analyze
            query: Analysis query or focus
            
        Returns:
            Analysis results
        """
        # Prepare the prompt
        prompt = f"""
        Analyze the following Solidity code for security vulnerabilities:
        
        ```solidity
        {code}
        ```
        
        Focus on: {query}
        
        Provide a detailed analysis with specific vulnerability findings, 
        their severity, impact, and recommended fixes.
        """
        
        # Run the agent
        # Note: FunctionAgent uses run() not arun() in some versions
        response = await self.agent.run(prompt)
        
        # Process response
        return {
            "raw_response": str(response),
            "structured_findings": self._extract_findings(str(response))
        }
    
    def _extract_findings(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structured findings from the agent's response
        
        Args:
            text: Raw response text
            
        Returns:
            List of structured findings
        """
        # Basic implementation - in practice, would need more robust parsing
        findings = []
        
        # Simple parsing based on common formats
        if "FINDINGS:" in text:
            findings_section = text.split("FINDINGS:")[1].split("RECOMMENDATIONS:")[0].strip()
            items = findings_section.split("\n\n")
            
            for item in items:
                if not item.strip():
                    continue
                    
                lines = item.strip().split("\n")
                title = lines[0].strip()
                severity = "Medium"  # Default
                
                # Extract severity if present
                for line in lines:
                    if "severity" in line.lower():
                        parts = line.split(":")
                        if len(parts) > 1:
                            severity = parts[1].strip()
                
                description = "\n".join(lines[1:]).strip()
                
                findings.append({
                    "title": title,
                    "severity": severity, 
                    "description": description
                })
        
        return findings