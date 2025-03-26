"""
Counterfactual Generator for the Finite Monkey framework

This module implements a specialized agent that generates counterfactual scenarios
to help train human operators and enhance their understanding of potential security risks.
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from ..adapters import Ollama
from ..models import CodeAnalysis, ValidationResult, InconsistencyReport


class CounterfactualGenerator:
    """
    Agent that generates counterfactual scenarios for human operator training
    
    This agent specializes in taking security findings and generating alternative
    scenarios that help human operators understand:
    1. Why the security issue exists
    2. How it could be exploited
    3. How small changes to the code could create or prevent vulnerabilities
    4. What related vulnerabilities might also exist
    
    This enhances human operators' ability to detect similar issues in the future.
    """
    
    def __init__(
        self,
        llm_client: Optional[Ollama] = None,
        model_name: str = "llama3:70b-instruct-q6_K",
    ):
        """
        Initialize the Counterfactual Generator
        
        Args:
            llm_client: LLM client for generation
            model_name: Default model to use
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    async def generate_counterfactuals(
        self,
        finding: Dict[str, Any],
        code_snippet: str,
        vulnerability_type: str,
        num_scenarios: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual scenarios for a given finding
        
        Args:
            finding: The security finding to generate counterfactuals for
            code_snippet: The vulnerable code snippet
            vulnerability_type: Type of vulnerability
            num_scenarios: Number of counterfactual scenarios to generate
            
        Returns:
            List of counterfactual scenarios
        """
        prompt = f"""
        You are an expert smart contract security auditor specializing in {vulnerability_type} vulnerabilities.
        
        I'm going to show you a vulnerable code snippet and a description of a security finding.
        Your task is to generate {num_scenarios} counterfactual scenarios that will help a human auditor
        better understand this vulnerability.
        
        For each scenario:
        1. Create a slight variation of the code that would either:
           - Make the vulnerability more severe
           - Fix the vulnerability in a subtle way
           - Change the vulnerability to a similar but different type
        
        2. Explain exactly what changed and why it matters
        
        3. Provide a specific learning objective for the human auditor
        
        Vulnerable code:
        ```solidity
        {code_snippet}
        ```
        
        Finding:
        {json.dumps(finding, indent=2)}
        
        For each counterfactual, provide:
        1. A title describing the scenario
        2. The modified code
        3. An explanation of what changed
        4. Why this change matters (security impact)
        5. A specific learning objective
        
        Format your answer as a JSON array where each object has these fields:
        "title", "modified_code", "explanation", "security_impact", "learning_objective"
        
        Return just the JSON with no additional text.
        """
        
        try:
            # Get response from LLM
            response = await self.llm_client.acomplete(
                prompt=prompt,
                model=self.model_name
            )
            
            # Parse JSON
            json_str = self._extract_json(response)
            if not json_str:
                self.logger.error("Failed to extract JSON from response")
                return []
            
            counterfactuals = json.loads(json_str)
            return counterfactuals
            
        except Exception as e:
            self.logger.error(f"Error generating counterfactuals: {str(e)}")
            return []
    
    async def generate_exploitation_path(
        self,
        finding: Dict[str, Any],
        code_snippet: str,
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a detailed exploitation path for a vulnerability
        
        Args:
            finding: The security finding
            code_snippet: The vulnerable code snippet
            detailed: Whether to generate a detailed path with code examples
            
        Returns:
            Dictionary with exploitation path details
        """
        prompt = f"""
        You are an expert smart contract security auditor and offensive security researcher.
        
        I'm going to show you a vulnerable code snippet and a description of a security finding.
        Your task is to generate a detailed explanation of how this vulnerability could be exploited
        in a real-world scenario.
        
        Vulnerable code:
        ```solidity
        {code_snippet}
        ```
        
        Finding:
        {json.dumps(finding, indent=2)}
        
        Please provide:
        1. A step-by-step exploitation path
        2. {"Example attack code that could exploit this vulnerability" if detailed else "A high-level description of attack vectors"}
        3. Conditions necessary for successful exploitation
        4. Potential impact if exploited
        5. How difficult this would be to exploit in practice (on a scale from 1-5)
        
        Format your answer as a JSON object with these fields:
        "exploitation_steps", "attack_code", "conditions", "impact", "difficulty_rating", "reasoning"
        
        Return just the JSON with no additional text.
        """
        
        try:
            # Get response from LLM
            response = await self.llm_client.acomplete(
                prompt=prompt,
                model=self.model_name
            )
            
            # Parse JSON
            json_str = self._extract_json(response)
            if not json_str:
                self.logger.error("Failed to extract JSON from response")
                return {}
            
            exploitation_path = json.loads(json_str)
            return exploitation_path
            
        except Exception as e:
            self.logger.error(f"Error generating exploitation path: {str(e)}")
            return {}
    
    async def generate_training_scenarios(
        self,
        vulnerability_type: str,
        difficulty: str = "medium",
        num_scenarios: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Generate training scenarios for a human operator to practice on
        
        Args:
            vulnerability_type: Type of vulnerability to focus on
            difficulty: Difficulty level (easy, medium, hard)
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of training scenarios
        """
        prompt = f"""
        You are an expert smart contract security auditor and educator.
        
        Create {num_scenarios} {difficulty}-difficulty training scenarios focused on {vulnerability_type} vulnerabilities.
        These scenarios will be used to train human auditors to better identify and understand these issues.
        
        For each scenario:
        1. Create a realistic code snippet that contains a subtly vulnerable pattern
        2. Provide context about what the code is supposed to do
        3. Include a hint that points the auditor in the right direction
        4. Include a detailed explanation of the vulnerability (to be revealed after the exercise)
        
        The {difficulty} difficulty level means:
        - Easy: Vulnerability is straightforward and follows common patterns
        - Medium: Vulnerability requires understanding interactions between different parts of the code
        - Hard: Vulnerability is subtle, non-obvious, or involves complex interactions
        
        Format your answer as a JSON array where each object has these fields:
        "title", "code", "context", "hint", "explanation", "difficulty"
        
        Return just the JSON with no additional text.
        """
        
        try:
            # Get response from LLM
            response = await self.llm_client.acomplete(
                prompt=prompt,
                model=self.model_name
            )
            
            # Parse JSON
            json_str = self._extract_json(response)
            if not json_str:
                self.logger.error("Failed to extract JSON from response")
                return []
            
            scenarios = json.loads(json_str)
            return scenarios
            
        except Exception as e:
            self.logger.error(f"Error generating training scenarios: {str(e)}")
            return []
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text response
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON string or empty string
        """
        # Look for JSON markers - using a more specific pattern to match properly nested JSON
        # The pattern looks for balanced braces/brackets
        json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}|\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\])'
        json_match = re.search(json_pattern, text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            # Validate that it's parseable
            try:
                json.loads(json_str)
                return json_str
            except:
                pass
        
        # Try common patterns
        try:
            # Try to extract object/array from beginning to end
            if (text.strip().startswith('{') and text.strip().endswith('}')) or \
               (text.strip().startswith('[') and text.strip().endswith(']')):
                json_str = text.strip()
                json.loads(json_str)
                return json_str
                
            # Look for code blocks with json
            code_block_pattern = r'```(?:json)?\s*([\s\S]+?)\s*```'
            code_match = re.search(code_block_pattern, text, re.DOTALL)
            if code_match:
                json_str = code_match.group(1)
                json.loads(json_str)
                return json_str
        except:
            pass
            
        # If all else fails, try to extract any JSON-like structure as a last resort
        try:
            # Attempt to find outer-most brackets
            first_open_brace = text.find('{')
            first_open_bracket = text.find('[')
            
            # Determine which comes first (if any)
            if first_open_brace != -1 and (first_open_bracket == -1 or first_open_brace < first_open_bracket):
                # Object starts first
                last_close_brace = text.rfind('}')
                if last_close_brace > first_open_brace:
                    json_str = text[first_open_brace:last_close_brace+1]
                    json.loads(json_str)  # Test if valid
                    return json_str
            elif first_open_bracket != -1:
                # Array starts first
                last_close_bracket = text.rfind(']')
                if last_close_bracket > first_open_bracket:
                    json_str = text[first_open_bracket:last_close_bracket+1]
                    json.loads(json_str)  # Test if valid
                    return json_str
        except:
            pass
            
        # If absolutely everything fails, return empty string
        return ""