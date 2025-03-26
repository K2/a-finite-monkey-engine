"""
Researcher agent for code analysis

This agent is responsible for retrieving relevant code context and
generating initial code analysis using LlamaIndex and LLM integration.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any

from openai import project
from finite_monkey.nodes_config import config

class Researcher:
    """
    Researcher agent that analyzes code using LlamaIndex and LLM

    This agent is responsible for:
    1. Retrieving relevant code context using LlamaIndex
    2. Generating initial code analysis with LLM
    3. Identifying potential issues and recommendations
    """
    def __init__(
        self,
        query_engine: Any = None,
        llm_client: Optional[Any] = None,
        model_name: str = "llama3",
        max_context_items: int = 10,
    ):
        """
        Initialize the researcher agent

        Args:
            query_engine: LlamaIndex processor for querying code
            llm_client: Ollama client for analysis
            model_name: Model to use for analysis
            max_context_items: Maximum number of context items to retrieve
        """
        if not query_engine:
            from finite_monkey.llama_index.processor import AsyncIndexProcessor
            self.query_engine = AsyncIndexProcessor(project_id=nodes_config().id)
        else:
            self.query_engine = query_engine

        if not llm_client:
            from finite_monkey.adapters import Ollama
            self.llm_client = Ollama(model=model_name)
        else:
            self.llm_client = llm_client

        self.model_name = model_name
        self.max_context_items = max_context_items

    async def analyze_code_async(
        self,
        query: str,
        file_path: Optional[str] = None,
        code_snippet: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Analyze code asynchronously

        Args:
            query: Analysis query (e.g., "Check for reentrancy")
            file_path: Path to file to analyze
            code_snippet: Code snippet to analyze
            filters: Filters to apply to the query

        Returns:
            Code analysis results
        """
        # Get source code if not provided
        source_code = code_snippet
        if file_path and not source_code:
            # Read file asynchronously
            async with open(file_path, "r", encoding="utf-8") as f:
                source_code = await f.read()

        if not source_code:
            raise ValueError("Either file_path or code_snippet must be provided")

        # Retrieve relevant context
        context_result = await self.query_engine.aquery(
            query_text=query,
            filters=filters,
            top_k=self.max_context_items,
        )

        # Extract relevant nodes
        context_nodes = context_result.get("nodes", [])

        # Get related functions
        related_functions = await self.query_engine.get_related_functions(
            function_code=source_code,
            top_k=5,
        )
        # Build prompt for analysis
        from finite_monkey.utils.prompting import get_analysis_prompt
        analysis_prompt = get_analysis_prompt(
            query=query,
            code=source_code,
            context_nodes=context_nodes,
            related_functions=related_functions,
        )

        # Generate analysis using LLM
        analysis_text = await self.llm_client.acomplete(
            prompt=analysis_prompt,
            model=self.model_name,
        )

        # Parse the analysis
        from finite_monkey.models import CodeAnalysis
        analysis = self._parse_analysis(
            analysis_text=analysis_text,
            source_code=source_code,
            related_functions=related_functions,
        )

        return analysis

    def _parse_analysis(
        self,
        analysis_text: str,
        source_code: str,
        related_functions: List[Dict[str, Any]],
    ) -> Any:
        """
        Parse the analysis text into a structured format

        Args:
            analysis_text: Raw analysis text from LLM
            source_code: Source code analyzed
            related_functions: Related functions

        Returns:
            Structured code analysis
        """
        # Extract summary (first paragraph)
        lines = analysis_text.split("\n")
        summary_lines = []
        for line in lines:
            if line.strip():
                summary_lines.append(line.strip())
            elif summary_lines:
                break

        summary = " ".join(summary_lines) if summary_lines else "No summary available."

        # Extract findings
        findings = []
        findings_section = False
        current_finding = {}

        for line in lines:
            line = line.strip()

            # Check for findings section
            if "FINDINGS:" in line.upper() or "VULNERABILITIES:" in line.upper():
                findings_section = True
                continue

            # Check for recommendations section
            if "RECOMMENDATIONS:" in line.upper():
                findings_section = False
                continue

            # Process finding
            if findings_section and line:
                # Check for new finding
                if line.startswith("- ") or line.startswith("* "):
                    # Save previous finding
                    if current_finding and "title" in current_finding:
                        findings.append(current_finding)

                    # Start new finding
                    current_finding = {
                        "title": line[2:].strip(),
                        "description": "",
                        "severity": "Medium",
                    }

                # Check for severity indicator
                elif "severity:" in line.lower():
                    severity_parts = line.split(":")
                    if len(severity_parts) > 1:
                        severity = severity_parts[1].strip()
                        if severity in ["Critical", "High", "Medium", "Low", "Informational"]:
                            current_finding["severity"] = severity

                # Add to description
                elif current_finding and "title" in current_finding:
                    if current_finding["description"]:
                        current_finding["description"] += " " + line
                    else:
                        current_finding["description"] = line

        # Add last finding
        if current_finding and "title" in current_finding:
            findings.append(current_finding)

        # Extract recommendations
        recommendations = []
        recommendations_section = False

        for line in lines:
            line = line.strip()

            # Check for recommendations section
            if "RECOMMENDATIONS:" in line.upper():
                recommendations_section = True
                continue

            # Check for end of recommendations
            if recommendations_section and not line:
                continue

            # Process recommendation
            if recommendations_section and line:
                # Check for new recommendation
                if line.startswith("- ") or line.startswith("* "):
                    recommendations.append(line[2:].strip())

        # Create analysis object
        from finite_monkey.models import CodeAnalysis
        analysis = CodeAnalysis(
            source_code=source_code,
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            related_functions=[
                {
                    "id": func.get("id", ""),
                    "text": func.get("text", ""),
                    "score": func.get("score", 0.0),
                }
                for func in related_functions
            ],
            metadata={
                "query": analysis_text,
                "model": self.model_name,
            },
        )

        return analysis
        