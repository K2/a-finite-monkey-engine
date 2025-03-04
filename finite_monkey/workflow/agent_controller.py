"""
Agent controller for the Finite Monkey framework

This module provides a controller for atomic agents using prompt-driven interactions.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple

from ..adapters import Ollama


class AgentController:
    """
    Controller for atomic agents through prompts
    
    This class manages the interactions between atomic agents,
    guiding their behavior through specialized prompting.
    """
    
    def __init__(
        self,
        llm_client: Optional[Ollama] = None,
        model_name: str = "llama3",
    ):
        """
        Initialize the agent controller
        
        Args:
            llm_client: Ollama client for LLM functions
            model_name: Default model name
        """
        self.llm_client = llm_client or Ollama(model=model_name)
        self.model_name = model_name
        self.state = "initialized"
        self.agent_states = {
            "researcher": "idle",
            "validator": "idle",
            "documentor": "idle",
        }
        
    async def generate_agent_prompt(
        self,
        agent_type: str,
        task: str,
        context: str,
    ) -> str:
        """
        Generate specialized prompts for atomic agents
        
        Args:
            agent_type: Type of agent (researcher, validator, documentor)
            task: Task description
            context: Additional context
            
        Returns:
            Prompt for the agent
        """
        system_prompts = {
            "researcher": (
                "You are a Researcher agent specialized in smart contract security analysis. "
                "Your job is to thoroughly analyze code and identify potential vulnerabilities "
                "and security issues. Focus on common smart contract vulnerabilities like reentrancy, "
                "access control issues, integer overflows, and other security concerns."
            ),
            "validator": (
                "You are a Validator agent specialized in verifying security findings in smart contracts. "
                "Your job is to carefully review analysis results and provide an independent assessment "
                "of identified vulnerabilities. You should confirm true positives, identify false positives, "
                "and potentially discover issues that were missed in the initial analysis."
            ),
            "documentor": (
                "You are a Documentor agent specialized in creating clear, comprehensive security reports. "
                "Your job is to synthesize analysis and validation results into a well-structured report "
                "that explains vulnerabilities, their potential impact, and recommendations for fixing them. "
                "The report should be accessible to both technical and non-technical stakeholders."
            ),
        }
        
        specific_instructions = {
            "researcher": (
                "Analyze the provided code thoroughly and identify any security issues or vulnerabilities. "
                "For each issue, specify:\n"
                "- The vulnerability type\n"
                "- The location in the code\n"
                "- The severity (Critical, High, Medium, Low, Informational)\n"
                "- A clear explanation of the issue\n"
                "- Potential impact if exploited\n\n"
                "Focus particularly on:\n"
                "- Reentrancy vulnerabilities\n"
                "- Access control issues\n"
                "- State management problems\n"
                "- Timestamp dependencies\n"
                "- Insecure randomness\n"
                "- Front-running opportunities\n"
                "- Other common smart contract vulnerabilities"
            ),
            "validator": (
                "Carefully validate the provided analysis results by:\n"
                "- Examining each reported issue in detail\n"
                "- Checking for false positives\n"
                "- Identifying any missed vulnerabilities\n"
                "- Providing an independent assessment of severity\n"
                "- Suggesting additional context or insights\n\n"
                "For each issue, provide:\n"
                "- Confirmation status (Confirmed, False Positive, Needs More Context)\n"
                "- Your reasoning with specific code references\n"
                "- Adjusted severity assessment if needed\n"
                "- Any additional context or insights"
            ),
            "documentor": (
                "Create a comprehensive security report based on the provided analysis and validation results.\n"
                "The report should include:\n"
                "- An executive summary of findings\n"
                "- Detailed explanations of each vulnerability\n"
                "- Code snippets highlighting issues\n"
                "- Impact assessments\n"
                "- Clear recommendations for remediation\n"
                "- Overall security assessment\n\n"
                "Format the report in clear, professional Markdown, suitable for both technical and non-technical audiences."
            ),
        }
        
        prompt = f"{system_prompts[agent_type]}\n\n"
        prompt += f"Task: {task}\n\n"
        prompt += f"Instructions: {specific_instructions[agent_type]}\n\n"
        prompt += f"Context:\n{context}\n\n"
        
        return prompt
    
    async def monitor_agent(
        self,
        agent_type: str,
        state: str,
        results: str,
    ) -> str:
        """
        Monitor agent progress and provide feedback
        
        Args:
            agent_type: Type of agent
            state: Current state of the agent
            results: Results from the agent
            
        Returns:
            Feedback for the agent
        """
        self.agent_states[agent_type] = state
        
        if state == "completed":
            # Generate feedback prompt
            feedback_prompt = (
                f"You are a supervisor overseeing a team of smart contract security specialists. "
                f"Review the following {agent_type} results and provide constructive feedback:\n\n"
                f"{results}\n\n"
                f"Provide specific feedback on:\n"
                f"1. Completeness of the analysis\n"
                f"2. Accuracy of the findings\n"
                f"3. Clarity of the explanations\n"
                f"4. Any missed areas or considerations\n"
                f"5. Suggestions for improvement\n\n"
                f"Your feedback will be used to guide further analysis and reporting."
            )
            
            feedback = await self.llm_client.acomplete(
                prompt=feedback_prompt,
                model=self.model_name,
            )
            return feedback
        
        return "Agent still working..."
    
    async def coordinate_workflow(
        self,
        research_results: Dict[str, Any],
        validation_results: Dict[str, Any],
    ) -> str:
        """
        Coordinate workflow between agents
        
        Args:
            research_results: Results from researcher
            validation_results: Results from validator
            
        Returns:
            Coordination instructions
        """
        # Convert results to strings if needed
        research_str = (
            json.dumps(research_results, indent=2) 
            if isinstance(research_results, dict) 
            else str(research_results)
        )
        
        validation_str = (
            json.dumps(validation_results, indent=2) 
            if isinstance(validation_results, dict) 
            else str(validation_results)
        )
        
        # Create coordination prompt
        prompt = (
            "You are the workflow coordinator for a smart contract security audit team. "
            "Review the research and validation results and provide coordination instructions "
            "for the documentation phase.\n\n"
            f"Research Results:\n{research_str}\n\n"
            f"Validation Results:\n{validation_str}\n\n"
            "Provide specific instructions for the documentation phase, including:\n"
            "1. Which findings should be prioritized in the report\n"
            "2. Any areas that need additional emphasis or explanation\n"
            "3. Specific recommendations to include\n"
            "4. Overall narrative or framing for the report\n\n"
            "Your instructions will guide the creation of the final audit report."
        )
        
        # Get coordination instructions
        instructions = await self.llm_client.acomplete(
            prompt=prompt,
            model=self.model_name,
        )
        
        return instructions