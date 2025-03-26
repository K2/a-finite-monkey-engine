"""
LLM-based code analyzer for Finite Monkey Engine

This module provides a standardized interface for analyzing code with different
LLM providers using a composable, functional approach.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
import time
from loguru import logger

#from ..utils.logging_utils import log_entry_exit,  AgentState
from ..pipeline.core import Context

# Type definitions
PromptFunc = Callable[[str, Dict[str, Any]], str]
AnalysisCallback = Callable[[Dict[str, Any], Dict[str, Any]], None]

class LLMAnalyzer:
    """
    LLM-based code analyzer
    
    This class provides a standardized interface for analyzing code with
    different LLM providers. It supports customizable prompting strategies
    and post-processing of results.
    """
    
    def __init__(
        self,
        llm_client: Any,
        prompt_templates: Optional[Dict[str, str]] = None,
        extract_json: bool = False,
        extract_findings: bool = True,
        model_name: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 2,
        retry_delay: int = 5,
        state: str = AgentState.IDLE
    ):
        """
        Initialize the LLM analyzer
        
        Args:
            llm_client: LLM client instance (Ollama, OpenAI, etc.)
            prompt_templates: Dictionary of prompt templates by type
            extract_json: Whether to extract JSON from responses
            extract_findings: Whether to extract findings from responses
            model_name: Name of the model used by the client
            timeout: Timeout in seconds for LLM requests
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            state: Initial state
        """
        self.llm_client = llm_client
        self.model_name = model_name or getattr(llm_client, 'model', 'unknown')
        self.prompt_templates = prompt_templates or {}
        self.extract_json = extract_json
        self.extract_findings = extract_findings
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.state = state
        
        # Load default templates if not provided
        if not self.prompt_templates:
            self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default prompt templates"""
        self.prompt_templates = {
            "security_audit": """
You are a smart contract security auditor analyzing Solidity code. 
Analyze the following contract for security vulnerabilities and issues.

CONTRACT SOURCE CODE:
```solidity
{code}
```

ANALYSIS QUERY: {query}

Analyze the contract for the following vulnerability categories:
1. Reentrancy
2. Access Control
3. Arithmetic Issues
4. Unchecked External Calls
5. Denial of Service
6. Front-Running
7. Timestamp Dependence
8. Gas Optimization

For each vulnerability, provide:
- Clear title
- Severity (Critical, High, Medium, Low, Informational)
- Description
- Location in the code
- Impact
- Recommendation for fixing

Also provide a summary of the overall security posture of the contract.
""",

            "contract_summary": """
You are a Solidity expert analyzing smart contracts.
Provide a concise summary of the following contract:

CONTRACT SOURCE CODE:
```solidity
{code}
```

Include:
1. Main purpose and functionality
2. Key functions and their purposes
3. State variables and their roles
4. External interactions
5. Access control mechanisms
""",

            "function_analysis": """
You are analyzing a specific function in a Solidity contract.
Provide a detailed analysis of this function:

FUNCTION CODE:
```solidity
{code}
```

CONTEXT: {context}

Include:
1. Function purpose and behavior
2. Input validation
3. Control flow
4. State changes
5. External calls
6. Return values
7. Potential edge cases and risks
"""
        }
    
    def format_prompt(self, prompt_type: str, code: str, **kwargs) -> str:
        """
        Format a prompt from a template
        
        Args:
            prompt_type: Type of prompt to use
            code: Code to analyze
            **kwargs: Additional values for template formatting
            
        Returns:
            Formatted prompt
        """
        if prompt_type not in self.prompt_templates:
            logger.warning(f"Unknown prompt type: {prompt_type}, using security_audit")
            prompt_type = "security_audit"
        
        template = self.prompt_templates[prompt_type]
        return template.format(code=code, **kwargs)
    
    def register_prompt_template(self, name: str, template: str) -> None:
        """
        Register a new prompt template
        
        Args:
            name: Name of the template
            template: Template string
        """
        self.prompt_templates[name] = template
    
    def register_prompt_templates(self, templates: Dict[str, str]) -> None:
        """
        Register multiple prompt templates
        
        Args:
            templates: Dictionary of name -> template
        """
        self.prompt_templates.update(templates)
    
    async def analyze_code(
        self, 
        code: str, 
        prompt_type: str = "security_audit", 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze code with the LLM
        
        Args:
            code: Code to analyze
            prompt_type: Type of prompt to use
            system_prompt: Optional system prompt
            **kwargs: Additional values for template formatting
            
        Returns:
            Analysis results
        """
        start_time = time.time()
        
        # Format the prompt
        prompt = self.format_prompt(prompt_type, code, **kwargs)
        
        # Track previous state and update to analyzing
        previous_state = self.state
        self.state = AgentState.ANALYZING
        
        # Try to analyze with retries
        retries = 0
        success = False
        llm_response = None
        error = None
        
        while not success and retries <= self.max_retries:
            try:
                # Call the LLM client
                if hasattr(self.llm_client, 'acomplete'):
                    # Async client
                    if system_prompt:
                        llm_response = await self.llm_client.acomplete(
                            prompt=prompt,
                            system_prompt=system_prompt
                        )
                    else:
                        llm_response = await self.llm_client.acomplete(prompt=prompt)
                else:
                    # Synchronous client - run in thread pool
                    loop = asyncio.get_event_loop()
                    if system_prompt:
                        llm_response = await loop.run_in_executor(
                            None,
                            lambda: self.llm_client.complete(prompt=prompt, system_prompt=system_prompt)
                        )
                    else:
                        llm_response = await loop.run_in_executor(
                            None,
                            lambda: self.llm_client.complete(prompt=prompt)
                        )
                
                success = True
                
            except Exception as e:
                error = str(e)
                retries += 1
                if retries <= self.max_retries:
                    logger.warning(f"LLM analysis failed (attempt {retries}/{self.max_retries+1}): {error}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"LLM analysis failed after {retries} attempts: {error}")
        
        # Process the response
        findings = []
        if success and llm_response:
            if self.extract_findings:
                findings = self._extract_findings(llm_response)
        
        # Calculate timing
        end_time = time.time()
        duration = end_time - start_time
        
        # Restore previous state if not completed
        self.state = AgentState.COMPLETED if success else previous_state
        
        # Return result
        result = {
            "success": success,
            "duration": duration,
            "prompt_type": prompt_type,
            "model": self.model_name,
            "raw_response": llm_response if success else None,
            "findings": findings,
            "error": error if not success else None,
            "timestamp": time.time()
        }
        
        # Extract JSON if requested
        if success and self.extract_json:
            json_data = self._extract_json(llm_response)
            if json_data:
                result["json_data"] = json_data
        
        return result
    
    async def analyze_chunk(
        self, 
        chunk: Dict[str, Any], 
        prompt_type: str = "security_audit", 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze a code chunk with the LLM
        
        Args:
            chunk: Chunk dictionary with content
            prompt_type: Type of prompt to use
            system_prompt: Optional system prompt
            **kwargs: Additional values for template formatting
            
        Returns:
            Analysis results including chunk metadata
        """
        # Get code from chunk
        code = chunk.get("content", "")
        
        # Add chunk context to kwargs
        chunk_kwargs = {
            "chunk_id": chunk.get("chunk_id", "unknown"),
            "chunk_type": chunk.get("chunk_type", "unknown"),
        }
        
        # Add contract/function specific context
        if chunk.get("chunk_type") == "contract":
            chunk_kwargs["contract_name"] = chunk.get("contract_name", "Unknown")
        elif chunk.get("chunk_type") == "function":
            chunk_kwargs["function_name"] = chunk.get("function_name", "Unknown")
            chunk_kwargs["contract_name"] = chunk.get("contract_name", "Unknown")
            
            # Add function metadata if available
            if "metadata" in chunk:
                metadata = chunk["metadata"]
                chunk_kwargs["visibility"] = metadata.get("visibility", "unknown")
                chunk_kwargs["mutability"] = metadata.get("mutability", "unknown")
                chunk_kwargs["parameters"] = metadata.get("parameters", "")
                chunk_kwargs["returns"] = metadata.get("returns", "")
        
        # Combine with user kwargs (user kwargs take precedence)
        analysis_kwargs = {**chunk_kwargs, **kwargs}
        
        # Analyze the chunk
        result = await self.analyze_code(code, prompt_type, system_prompt, **analysis_kwargs)
        
        # Add chunk metadata to result
        result["chunk"] = {
            "chunk_id": chunk.get("chunk_id"),
            "chunk_type": chunk.get("chunk_type"),
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
            "file_path": chunk.get("file_path")
        }
        
        return result

    async def analyze_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        prompt_type: str = "security_audit",
        system_prompt: Optional[str] = None,
        max_concurrency: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple chunks with controlled concurrency
        
        Args:
            chunks: List of chunks to analyze
            prompt_type: Type of prompt to use
            system_prompt: Optional system prompt
            max_concurrency: Maximum number of concurrent analyses
            **kwargs: Additional values for template formatting
            
        Returns:
            List of analysis results
        """
        # Track previous state and update to analyzing
        previous_state = self.state
        self.state = AgentState.ANALYZING
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def analyze_with_semaphore(chunk):
            async with semaphore:
                return await self.analyze_chunk(chunk, prompt_type, system_prompt, **kwargs)
        
        # Create tasks for each chunk
        tasks = []
        for chunk in chunks:
            task = analyze_with_semaphore(chunk)
            tasks.append(task)
        
        # Wait for all tasks to complete
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Restore state if completed
        self.state = AgentState.COMPLETED
        
        # Add summary
        summary = {
            "chunks_analyzed": len(chunks),
            "total_duration": end_time - start_time,
            "average_duration": (end_time - start_time) / max(1, len(chunks)),
            "prompt_type": prompt_type,
            "model": self.model_name,
            "timestamp": time.time()
        }
        
        # Count findings by severity
        if any(result.get("findings") for result in results):
            severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Informational": 0}
            
            for result in results:
                for finding in result.get("findings", []):
                    severity = finding.get("severity", "Unknown")
                    if severity in severity_counts:
                        severity_counts[severity] += 1
            
            summary["finding_counts"] = severity_counts
        
        return {
            "results": results,
            "summary": summary
        }
    
    def _extract_findings(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structured findings from LLM response
        
        Args:
            text: Raw LLM response
            
        Returns:
            List of findings
        """
        findings = []
        
        # Try to extract structured findings with regex
        import re
        
        # Look for patterns like "## Vulnerability 1: Title" or "### 1. Title"
        section_pattern = re.compile(
            r'(?:^|\n)(?:#+\s*\d+\.?\s*|#+\s*)([\w\s]+?)(?:\s*\(|\:|\n)', 
            re.MULTILINE
        )
        
        # Look for severity indicators
        severity_pattern = re.compile(
            r'(?:^|\n)(?:Severity|SEVERITY)\s*(?:\:|is|)\s*(Critical|High|Medium|Low|Informational)', 
            re.IGNORECASE
        )
        
        # Look for location indicators
        location_pattern = re.compile(
            r'(?:^|\n)(?:Location|Lines?|File)\s*(?:\:|at)\s*([^\n]+)', 
            re.IGNORECASE
        )
        
        # Look for description
        description_pattern = re.compile(
            r'(?:^|\n)(?:Description|Issue|Problem)\s*(?:\:|\n)\s*([^\n].+?)(?:\n\n|\n#)', 
            re.IGNORECASE | re.DOTALL
        )
        
        # Look for impact
        impact_pattern = re.compile(
            r'(?:^|\n)(?:Impact|Consequence)\s*(?:\:|\n)\s*([^\n].+?)(?:\n\n|\n#)', 
            re.IGNORECASE | re.DOTALL
        )
        
        # Look for recommendation
        recommendation_pattern = re.compile(
            r'(?:^|\n)(?:Recommendation|Fix|Mitigation|Solution)\s*(?:\:|\n)\s*([^\n].+?)(?:\n\n|\n#|$)', 
            re.IGNORECASE | re.DOTALL
        )
        
        # Find all section headers
        sections = section_pattern.finditer(text)
        
        for i, section in enumerate(sections):
            title = section.group(1).strip()
            
            # Skip if this doesn't look like a finding
            if not any(kw in title.lower() for kw in [
                'vulnerab', 'issue', 'bug', 'flaw', 'error', 'problem',
                'risk', 'exploit', 'attack', 'unsafe', 'danger'
            ]):
                continue
                
            # Calculate the section range
            start_pos = section.start()
            
            # Find the end by looking for the next section or end of text
            end_pos = len(text)
            match_list = list(section_pattern.finditer(text))
            for j, next_section in enumerate(match_list):
                if next_section.start() == start_pos and j < len(match_list) - 1:
                    end_pos = match_list[j+1].start()
                    break
            
            # Extract section text
            section_text = text[start_pos:end_pos]
            
            # Extract severity
            severity_match = severity_pattern.search(section_text)
            severity = severity_match.group(1) if severity_match else "Medium"
            
            # Extract location
            location_match = location_pattern.search(section_text)
            location = location_match.group(1).strip() if location_match else ""
            
            # Extract description
            description_match = description_pattern.search(section_text)
            description = description_match.group(1).strip() if description_match else section_text[:200] + "..."
            
            # Extract impact
            impact_match = impact_pattern.search(section_text)
            impact = impact_match.group(1).strip() if impact_match else ""
            
            # Extract recommendation
            recommendation_match = recommendation_pattern.search(section_text)
            recommendation = recommendation_match.group(1).strip() if recommendation_match else ""
            
            # Create finding
            finding = {
                "title": title,
                "severity": severity,
                "description": description,
                "impact": impact,
                "recommendation": recommendation
            }
            
            if location:
                finding["location"] = location
            
            findings.append(finding)
        
        # If no findings were found through regex, look for JSON blocks
        if not findings:
            findings = self._extract_json(text)
            
            # If JSON was found and it contains findings, use those
            if isinstance(findings, dict) and 'findings' in findings:
                findings = findings['findings']
        
        return findings
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text
        
        Args:
            text: Text to extract JSON from
            
        Returns:
            Extracted JSON data or None
        """
        import re
        
        # Look for JSON blocks
        json_pattern = re.compile(r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```|\{\s*"[^"]+"\s*:')
        json_matches = json_pattern.finditer(text)
        
        for match in json_matches:
            # Get the matched JSON text
            json_text = match.group(1) or match.group(2) or text[match.start():]
            
            # Clean it up
            json_text = json_text.strip()
            
            # If it doesn't start with {, wrap it
            if not json_text.startswith('{') and not json_text.startswith('['):
                continue
            
            # Try to parse it
            try:
                data = json.loads(json_text)
                return data
            except json.JSONDecodeError:
                continue
        
        return None


class ComposableLLMAnalyzer:
    """
    Composable LLM analyzer that combines multiple analysis steps
    
    This class allows for the composition of multiple analysis steps
    using a functional programming style with callbacks and transformations.
    """
    
    def __init__(
        self,
        llm_analyzer: LLMAnalyzer,
        prompt_type: str = "security_audit",
        system_prompt: Optional[str] = None,
        max_concurrency: int = 3
    ):
        """
        Initialize the composable analyzer
        
        Args:
            llm_analyzer: LLM analyzer to use
            prompt_type: Default prompt type to use
            system_prompt: Default system prompt to use
            max_concurrency: Maximum number of concurrent analyses
        """
        self.analyzer = llm_analyzer
        self.prompt_type = prompt_type
        self.system_prompt = system_prompt
        self.max_concurrency = max_concurrency
        self.pipeline = []
    
    def add_step(
        self, 
        analyzer_func: Callable,
        prompt_type: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> 'ComposableLLMAnalyzer':
        """
        Add an analysis step to the pipeline
        
        Args:
            analyzer_func: Function to call for this step
            prompt_type: Optional prompt type override
            system_prompt: Optional system prompt override
            **kwargs: Additional kwargs for the analyzer function
            
        Returns:
            Self for method chaining
        """
        step = {
            "func": analyzer_func,
            "prompt_type": prompt_type or self.prompt_type,
            "system_prompt": system_prompt or self.system_prompt,
            "kwargs": kwargs
        }
        self.pipeline.append(step)
        return self
    
    def add_transformation(
        self, 
        transform_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> 'ComposableLLMAnalyzer':
        """
        Add a transformation step to the pipeline
        
        Args:
            transform_func: Function that transforms results
            
        Returns:
            Self for method chaining
        """
        step = {
            "transform": transform_func
        }
        self.pipeline.append(step)
        return self
    
    def add_callback(
        self, 
        callback_func: Callable[[Dict[str, Any]], None]
    ) -> 'ComposableLLMAnalyzer':
        """
        Add a callback step to the pipeline
        
        Args:
            callback_func: Function to call with results
            
        Returns:
            Self for method chaining
        """
        step = {
            "callback": callback_func
        }
        self.pipeline.append(step)
        return self
    
    async def run(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the analysis pipeline on a single content string
        
        Args:
            content: Code content to analyze
            context: Optional context dictionary
            
        Returns:
            Analysis results
        """
        # Initialize state
        result = {
            "content": content,
            "context": context or {},
            "steps": []
        }
        
        # Run each step in sequence
        for i, step in enumerate(self.pipeline):
            step_name = f"step_{i+1}"
            
            try:
                if "func" in step:
                    # This is an analyzer function
                    analyzer_func = step["func"]
                    if analyzer_func == self.analyzer.analyze_code:
                        # Direct analyze_code call
                        step_result = await self.analyzer.analyze_code(
                            content,
                            prompt_type=step["prompt_type"],
                            system_prompt=step["system_prompt"],
                            **step["kwargs"],
                            **(context or {})
                        )
                    else:
                        # Custom analyzer function
                        step_result = await analyzer_func(
                            content,
                            prompt_type=step["prompt_type"],
                            system_prompt=step["system_prompt"],
                            **step["kwargs"],
                            **(context or {}),
                            result=result
                        )
                    
                    result["steps"].append({
                        "name": step_name,
                        "type": "analyzer",
                        "result": step_result
                    })
                    
                    # Update main result with findings
                    if "findings" not in result:
                        result["findings"] = []
                    
                    if "findings" in step_result:
                        result["findings"].extend(step_result["findings"])
                    
                elif "transform" in step:
                    # This is a transformation function
                    transform_func = step["transform"]
                    transformed = transform_func(result)
                    
                    result["steps"].append({
                        "name": step_name,
                        "type": "transform",
                        "transformed": True
                    })
                    
                    # Update result with transformation
                    if isinstance(transformed, dict):
                        result.update(transformed)
                    
                elif "callback" in step:
                    # This is a callback function
                    callback_func = step["callback"]
                    
                    if asyncio.iscoroutinefunction(callback_func):
                        await callback_func(result)
                    else:
                        callback_func(result)
                        
                    result["steps"].append({
                        "name": step_name,
                        "type": "callback",
                        "called": True
                    })
            
            except Exception as e:
                logger.error(f"Error in {step_name}: {e}")
                result["steps"].append({
                    "name": step_name,
                    "type": "error",
                    "error": str(e)
                })
        
        return result
    
    @()
    async def run_on_chunks(
        self, 
        chunks: List[Dict[str, Any]],
        combine_results: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run the analysis pipeline on multiple chunks
        
        Args:
            chunks: List of chunks to analyze
            combine_results: Whether to combine results into a single result
            
        Returns:
            Combined analysis results or list of individual results
        """
        # Create tasks for each chunk
        tasks = []
        for chunk in chunks:
            # Create context from chunk
            context = {
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "chunk_type": chunk.get("chunk_type", "unknown"),
                "file_path": chunk.get("file_path", "unknown"),
            }
            
            # Add contract/function specific context
            if chunk.get("chunk_type") == "contract":
                context["contract_name"] = chunk.get("contract_name", "Unknown")
            elif chunk.get("chunk_type") == "function":
                context["function_name"] = chunk.get("function_name", "Unknown")
                context["contract_name"] = chunk.get("contract_name", "Unknown")
            
            # Create task
            task = self.run(chunk.get("content", ""), context)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Return individual results if requested
        if not combine_results:
            return results
        
        # Combine results
        combined = {
            "chunks": len(chunks),
            "results": results,
            "findings": []
        }
        
        # Collect all findings
        for result in results:
            if "findings" in result:
                # Add chunk info to each finding
                for finding in result["findings"]:
                    # Add chunk context if not already present
                    if "context" not in finding and "context" in result:
                        finding["context"] = result["context"]
                
                combined["findings"].extend(result["findings"])
        
        # Count findings by severity
        severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Informational": 0}
        
        for finding in combined["findings"]:
            severity = finding.get("severity", "Unknown")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        combined["severity_counts"] = severity_counts
        combined["total_findings"] = len(combined["findings"])
        
        return combined