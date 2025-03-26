"""
Base Analyzer Interface for Finite Monkey Engine

This module provides the base classes and interfaces for all analyzers
to create a unified model for different analysis approaches.
"""

from abc import ABC, abstractmethod
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from loguru import logger


class BaseAnalyzer(ABC):
    """
    Base abstract class for all analyzers in the Finite Monkey Engine.
    
    This class defines the common interface that all analyzers must implement,
    allowing for a unified approach to different analysis methods.
    """
    
    @abstractmethod
    async def analyze_file(self, 
                        file_path: str, 
                        project_id: str = "default",
                        **kwargs) -> Dict[str, Any]:
        """
        Analyze a single file.
        
        Args:
            file_path: Path to the file to analyze
            project_id: Identifier for the project
            **kwargs: Additional analyzer-specific parameters
            
        Returns:
            Analysis results
        """
        pass
    
    @abstractmethod
    async def analyze_project(self, 
                           project_path: str, 
                           project_id: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Analyze a project directory.
        
        Args:
            project_path: Path to the project directory
            project_id: Optional identifier for the project (defaults to directory name)
            **kwargs: Additional analyzer-specific parameters
            
        Returns:
            Project analysis results
        """
        pass
    
    @staticmethod
    def get_default_project_id(project_path: str) -> str:
        """
        Get a default project ID from a project path.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Default project ID
        """
        return os.path.basename(os.path.normpath(project_path))
    
    @staticmethod
    def get_file_id(file_path: str) -> str:
        """
        Get a file identifier from a file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File identifier
        """
        return os.path.basename(file_path)
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get a formatted timestamp.
        
        Returns:
            Formatted timestamp
        """
        return datetime.now().isoformat()
    
    def log_progress(self, message: str, level: str = "info") -> None:
        """
        Log progress message with the appropriate level.
        
        Args:
            message: Message to log
            level: Log level (info, debug, warning, error, critical)
        """
        if level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "critical":
            logger.critical(message)
        else:
            logger.info(message)


class ChunkingBaseAnalyzer(BaseAnalyzer):
    """
    Base class for analyzers that use chunking strategies.
    
    This extends the BaseAnalyzer with functionality specific to
    analyzers that break down code into chunks for processing.
    """
    
    @abstractmethod
    async def get_chunks(self, 
                        file_path: str, 
                        max_chunk_size: int = 4000,
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Get chunks for a file.
        
        Args:
            file_path: Path to the file to chunk
            max_chunk_size: Maximum size of each chunk
            **kwargs: Additional chunking parameters
            
        Returns:
            List of chunks
        """
        pass
    
    @abstractmethod
    async def analyze_chunks(self, 
                          chunks: List[Dict[str, Any]],
                          file_path: str,
                          project_id: str = "default",
                          **kwargs) -> Dict[str, Any]:
        """
        Analyze chunks of a file.
        
        Args:
            chunks: List of chunks to analyze
            file_path: Path to the original file
            project_id: Identifier for the project
            **kwargs: Additional analyzer-specific parameters
            
        Returns:
            Analysis results
        """
        pass


class LLMAnalyzerMixin:
    """
    Mixin for analyzers that use LLMs for analysis.
    
    This mixin provides common functionality for LLM-based analyzers,
    such as prompt handling and LLM client management.
    """
    
    def create_analysis_prompt(self, 
                              source_code: str, 
                              query: str,
                              additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for LLM analysis.
        
        Args:
            source_code: Source code to analyze
            query: Query for the analysis
            additional_context: Additional context for the prompt
            
        Returns:
            Formatted prompt
        """
        # Basic prompt structure
        prompt = f"""
You are a smart contract security auditor analyzing Solidity code. 
Analyze the following contract for security vulnerabilities and issues.

QUERY: {query}

CONTRACT SOURCE CODE:
```solidity
{source_code}
```
"""
        
        # Add additional context if provided
        if additional_context:
            for context_name, context_content in additional_context.items():
                if isinstance(context_content, str):
                    prompt += f"\n{context_name.upper()}:\n{context_content}\n"
                elif isinstance(context_content, dict):
                    prompt += f"\n{context_name.upper()}:\n{json.dumps(context_content, indent=2)}\n"
                elif isinstance(context_content, list):
                    content_str = "\n".join([f"- {item}" for item in context_content])
                    prompt += f"\n{context_name.upper()}:\n{content_str}\n"
        
        # Add standard analysis instructions
        prompt += """
Analyze the contract for the following vulnerability categories:
1. Reentrancy
2. Access Control
3. Arithmetic Issues
4. Unchecked External Calls
5. Denial of Service
6. Front-Running
7. Transaction Ordering Dependence
8. Block Timestamp Manipulation
9. Unsafe Type Inference
10. Gas Optimization Issues

For each issue identified:
1. Provide a clear title
2. Describe the vulnerability in detail
3. Specify the severity (Critical, High, Medium, Low, Informational)
4. Include the exact location in the code (contract, function, line)
5. Explain the impact of the vulnerability
6. Recommend a specific fix with code examples where possible

Focus on actionable findings with clear impact, rather than theoretical or low-risk issues.
Format your response in Markdown with proper sections and code blocks.
"""
        
        return prompt
    
    def create_validation_prompt(self, 
                                source_code: str, 
                                findings: List[Dict[str, Any]],
                                additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for LLM validation of findings.
        
        Args:
            source_code: Source code that was analyzed
            findings: List of findings to validate
            additional_context: Additional context for the prompt
            
        Returns:
            Formatted prompt
        """
        # Format findings text
        findings_text = ""
        for i, finding in enumerate(findings, 1):
            findings_text += f"{i}. {finding.get('title', 'Unnamed Finding')} (Severity: {finding.get('severity', 'Medium')})\n"
            findings_text += f"   Description: {finding.get('description', 'No description')}\n"
            findings_text += f"   Location: {finding.get('location', 'Not specified')}\n\n"
        
        # Basic prompt structure
        prompt = f"""
You are a validator for smart contract security findings. Your job is to carefully review the findings from an initial analysis and provide an independent assessment.

SMART CONTRACT:
```solidity
{source_code}
```

FINDINGS TO VALIDATE:
{findings_text}
"""
        
        # Add additional context if provided
        if additional_context:
            for context_name, context_content in additional_context.items():
                if isinstance(context_content, str):
                    prompt += f"\n{context_name.upper()}:\n{context_content}\n"
                elif isinstance(context_content, dict):
                    prompt += f"\n{context_name.upper()}:\n{json.dumps(context_content, indent=2)}\n"
                elif isinstance(context_content, list):
                    content_str = "\n".join([f"- {item}" for item in context_content])
                    prompt += f"\n{context_name.upper()}:\n{content_str}\n"
        
        # Add validation instructions
        prompt += """
For each finding above:
1. Perform an independent verification using detailed code inspection
2. Provide your confirmation status: Is the finding valid? (Confirmed, False Positive, or Needs More Information)
3. Include your reasoning with specific code references and line numbers
4. Provide adjusted severity assessment if needed (Critical, High, Medium, Low, Informational)
5. Add any additional context or insights, especially related to control flow

Pay special attention to:
- The sequence of operations (state changes vs external calls)
- Path conditions that must be satisfied for vulnerabilities to be exploitable
- Cross-function and cross-contract interactions
- Value transfer flows and their security implications

Also note any important vulnerabilities that might have been missed in the initial analysis.

Approach this task methodically and provide evidence for your conclusions based on actual code examination.
"""
        
        return prompt