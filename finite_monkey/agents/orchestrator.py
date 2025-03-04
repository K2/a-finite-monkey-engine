"""
Workflow orchestrator for the Finite Monkey framework

This module implements the central orchestrator that coordinates the
atomic agents in the workflow.
"""

import os
import re
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple

from ..llama_index import AsyncIndexProcessor
from ..adapters import Ollama
from ..models import AuditReport, CodeAnalysis, ValidationResult
from .researcher import Researcher
from .validator import Validator, TreeSitterAnalyzer
from .documentor import Documentor


class WorkflowOrchestrator:
    """
    Workflow orchestrator for audit pipeline

    Coordinates the atomic agents (Researcher, Validator, Documentor) to
    execute the complete audit workflow.
    """

    def __init__(
        self,
        llama_index: Optional[AsyncIndexProcessor] = None,
        ollama: Optional[Ollama] = None,
        researcher: Optional[Researcher] = None,
        validator: Optional[Validator] = None,
        documentor: Optional[Documentor] = None,
        model_name: str = "qwen2.5:14b-instruct-q6_K",
        base_dir: Optional[str] = None,
    ):
        """
        Initialize the workflow orchestrator

        Args:
            llama_index: LlamaIndex processor for queries
            ollama: Ollama client for LLM functions
            researcher: Researcher agent
            validator: Validator agent
            documentor: Documentor agent
            model_name: Default model name for LLM
            base_dir: Base directory for project files
        """
        # Set base directory
        self.base_dir = base_dir or os.getcwd()

        # Set up LLM client if not provided
        self.ollama = ollama or Ollama(model=model_name)

        # Set up LlamaIndex client if not provided
        project_id = f"project-{model_name}"
        self.llama_index = llama_index or AsyncIndexProcessor(
            project_id=project_id,
            base_dir=self.base_dir,
        )

        # Set up agents if not provided
        self.researcher = researcher or Researcher(
            query_engine=self.llama_index,
            llm_client=self.ollama,
            model_name=model_name,
        )

        tree_sitter_analyzer = TreeSitterAnalyzer()
        self.validator = validator or Validator(
            tree_sitter_analyzer=tree_sitter_analyzer,
            llm_client=self.ollama,
            model_name=model_name,
        )

        self.documentor = documentor or Documentor(
            llm_client=self.ollama,
            model_name=model_name,
        )

    async def run_audit(
        self,
        solidity_path: str,
        query: str,
        project_name: Optional[str] = None,
    ) -> AuditReport:
        """
        Run a complete audit workflow

        Args:
            solidity_path: Path to the Solidity file to audit
            query: Audit query (e.g., "Check for reentrancy vulnerabilities")
            project_name: Name of the project

        Returns:
            Audit report
        """
        # Set project name from file if not provided
        if project_name is None:
            project_name = os.path.basename(solidity_path).split(".")[0]

        # Index the code
        print(f"Indexing {solidity_path}...")
        await self.llama_index.load_and_index(
            file_paths=[solidity_path],
        )

        # Read the file content
        with open(solidity_path, "r", encoding="utf-8") as f:
            code_content = f.read()

        # Step 1: Analyze with Researcher
        print(f"Analyzing code...")
        analysis = await self.researcher.analyze_code_async(
            query=query,
            code_snippet=code_content,
        )

        # Step 2: Validate with Validator
        print(f"Validating analysis...")
        validation = await self.validator.validate_analysis(
            code=code_content,
            analysis=analysis,
        )

        # Step 3: Generate report with Documentor
        print(f"Generating report...")
        report = await self.documentor.generate_report_async(
            analysis=analysis,
            validation=validation,
            project_name=project_name,
            target_files=[solidity_path],
            query=query,
        )

        return report
    
    def run_atomic_agent_workflow(
        self,
        solidity_path: Union[str, List[str]],
        query: str,
        project_name: Optional[str] = None,
    ) -> AuditReport:
        # Initialize required variables
        summary = "Default summary"
        findings = []
        recommendations = []
        research_response = ""
        researcher_feedback = ""
        validation_response = ""
        validator_feedback = ""
        coordination_instructions = ""
        report_text = ""

        report = AuditReport(
            project_id=f"audit-{project_name}",
            project_name=project_name,
            target_files=solidity_path,  # Changed from [solidity_path] to solidity_paths
            query=query,
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            analysis_details={
                "summary": research_response[:500] + "...",
                "full_analysis": research_response,
                "feedback": researcher_feedback,
            },
            validation_results={
                "summary": validation_response[:500] + "...",
                "full_validation": validation_response,
                "feedback": validator_feedback,
            },
            metadata={
                "coordination": coordination_instructions,
                "full_report": report_text,
            },
        )
        return report

    # async def run_atomic_agent_workflow(
    #     self,
    #     solidity_path: Union[str, List[str]],
    #     query: str,
    #     project_name: Optional[str] = None,
    # ) -> AuditReport:
    #     """
    #     Run workflow with atomic agents monitoring each other

    #     Args:
    #         solidity_path: Path to the Solidity file to audit, or a list of files
    #         query: Audit query (e.g., "Check for reentrancy vulnerabilities")
    #         project_name: Name of the project

    #     Returns:
    #         Audit report
    #     """
    #     from ..workflow.agent_controller import AgentController

    #     # Handle single file or multiple files
    #     if isinstance(solidity_path, str):
    #         solidity_paths = [solidity_path]
    #         # Set project name from file if not provided
    #         if project_name is None:
    #             project_name = os.path.basename(solidity_path).split(".")[0]
    #     else:
    #         solidity_paths = solidity_path
    #         # Set project name from directory if not provided
    #         if project_name is None:
    #             # Try to find a common parent directory
    #             common_dir = os.path.commonpath([os.path.abspath(p) for p in solidity_paths])
    #             project_name = os.path.basename(common_dir)
    #             if not project_name or project_name == ".":
    #                 # Fall back to the first file name
    #                 project_name = os.path.basename(solidity_paths[0]).split(".")[0]

    #     # Initialize the agent controller
    #     controller = AgentController(self.ollama)

    #     # Index all code files
    #     print(f"Indexing {len(solidity_paths)} files...")
    #     await self.llama_index.load_and_index(
    #         file_paths=solidity_paths,
    #     )

    #     # Read all file contents and combine them
    #     code_contents = {}
    #     combined_code = ""

    #     for file_path in solidity_paths:
    #         try:
    #             with open(file_path, "r", encoding="utf-8") as f:
    #                 file_content = f.read()
    #                 code_contents[file_path] = file_content

    #                 # Add file separator for clear delineation in combined code
    #                 file_name = os.path.basename(file_path)
    #                 combined_code += f"\n\n// FILE: {file_name}\n{file_content}\n"
    #         except Exception as e:
    #             print(f"Warning: Could not read file {file_path}: {str(e)}")

    #     # For single file case, just use the content directly
    #     code_content = list(code_contents.values())[0] if len(solidity_paths) == 1 else combined_code

    #     # STEP 1: Generate Researcher prompt
    #     print(f"Generating researcher prompt...")
    #     researcher_prompt = await controller.generate_agent_prompt(
    #         agent_type="researcher",
    #         task=f"Analyze the {project_name} contract for security vulnerabilities",
    #         context=code_content,
    #     )

    #     # Use the prompt to generate a research analysis
    #     print(f"Researcher agent analyzing code...")
    #     research_response = await self.ollama.acomplete(
    #         prompt=researcher_prompt,
    #     )

    #     # Monitor and provide feedback
    #     print(f"Monitoring researcher results...")
    #     researcher_feedback = await controller.monitor_agent(
    #         agent_type="researcher",
    #         state="completed",
    #         results=research_response,
    #     )

    #     print(f"Researcher feedback received")

    #     # STEP 2: Generate Validator prompt with feedback
    #     print(f"Generating validator prompt...")
    #     validator_prompt = await controller.generate_agent_prompt(
    #         agent_type="validator",
    #         task=f"Validate the security analysis for the {project_name} contract",
    #         context=f"Code:\n```solidity\n{code_content}\n```\n\nResearch Results:\n{research_response}\n\nFeedback:\n{researcher_feedback}",
    #     )

    #     # Use the prompt to generate validation
    #     print(f"Validator agent validating analysis...")
    #     validation_response = await self.ollama.acomplete(
    #         prompt=validator_prompt,
    #     )

    #     # Monitor and provide feedback
    #     print(f"Monitoring validator results...")
    #     validator_feedback = await controller.monitor_agent(
    #         agent_type="validator",
    #         state="completed",
    #         results=validation_response,
    #     )

    #     print(f"Validator feedback received")

    #     # STEP 3: Get coordination instructions
    #     print(f"Coordinating workflow...")
    #     coordination_instructions = await controller.coordinate_workflow(
    #         research_results=research_response,
    #         validation_results=validation_response,
    #     )

    #     # STEP 4: Generate Documentor prompt with coordination
    #     print(f"Generating documentor prompt...")
    #     documentor_prompt = await controller.generate_agent_prompt(
    #         agent_type="documentor",
    #         task=f"Create a comprehensive security report for the {project_name} contract",
    #         context=(
    #             f"Code:\n```solidity\n{code_content}\n```\n\n"
    #             f"Research Results:\n{research_response}\n\n"
    #             f"Validation Results:\n{validation_response}\n\n"
    #             f"Coordination Instructions:\n{coordination_instructions}"
    #         ),
    #     )

    #     # Use the prompt to generate report
    #     print(f"Documentor agent generating report...")
    #     report_text = await self.ollama.acomplete(
    #         prompt=documentor_prompt,
    #     )

    #     # Create a structured report
    #     print(f"Creating structured report...")

    #     # Parse findings from research and validation
    #     findings = self._extract_findings_from_text(research_response, validation_response)
    #     recommendations = self._extract_recommendations_from_text(report_text)

    #     # Create summary from report text
    #     summary_lines = []
    #     for line in report_text.split("\n"):
    #         line = line.strip()
    #         if line and not line.startswith("#"):
    #             summary_lines.append(line)
    #         if len(summary_lines) >= 5:
    #             break

    #     summary = " ".join(summary_lines)

    #     # Create report
    #     report = AuditReport(
    #         project_id=f"audit-{project_name}",
    #         project_name=project_name,
    #         target_files=[solidity_path],
    #         query=query,
    #         summary=summary,
    #         findings=findings,
    #         recommendations=recommendations,
    #         analysis_details={
    #             "summary": research_response[:500] + "...",
    #             "full_analysis": research_response,
    #             "feedback": researcher_feedback,
    #         },
    #         validation_results={
    #             "summary": validation_response[:500] + "...",
    #             "full_validation": validation_response,
    #             "feedback": validator_feedback,
    #         },
    #         metadata={
    #             "coordination": coordination_instructions,
    #             "full_report": report_text,
    #         },
    #     )

    #     return report

    def _extract_findings_from_text(self, research_text, validation_text):
        """Extract findings from text"""
        findings = []

        # Look for patterns like "1. Finding name: description" or "## Finding name"
        finding_patterns = [
            # Section headers
            r"(?:^|\n)#{1,3}\s+([^\n]+?)\s*?\((?:Severity|Priority):\s*([^\)]+)\)",
            # Numbered findings
            r"(?:^|\n)(?:\d+\.|\*)\s+([^\n:]+)(?::|(?:\s+\((?:Severity|Priority):\s*([^\)]+)\)))",
            # Findings sections
            r"(?:^|\n)FINDINGS?:\s*\n+(?:\d+\.|\*)\s+([^\n:]+)(?::|(?:\s+\((?:Severity|Priority):\s*([^\)]+)\)))",
        ]

        for pattern in finding_patterns:
            for match in re.finditer(pattern, research_text, re.MULTILINE):
                title = match.group(1).strip()
                severity = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else "Medium"

                # Extract description - look for text after the title until the next section
                start_pos = match.end()
                end_pos = len(research_text)

                # Find the next finding or section
                next_finding = re.search(pattern, research_text[start_pos:], re.MULTILINE)
                if next_finding:
                    end_pos = start_pos + next_finding.start()

                description = research_text[start_pos:end_pos].strip()

                # Clean up description
                description = re.sub(r'^[^A-Za-z0-9]+', '', description)
                description = re.sub(r'\n+', ' ', description)

                # Extract location if mentioned
                location_match = re.search(r'(?:location|line|function):\s*([^\n]+)', description, re.IGNORECASE)
                location = location_match.group(1) if location_match else ""

                # Add finding
                findings.append({
                    "title": title,
                    "description": description,
                    "severity": severity,
                    "location": location,
                })

        # Look for validations in validation text
        for finding in findings:
            # Try to find validation for this finding
            validation_pattern = re.compile(
                r'(?:^|\n)(?:\d+\.|\*|-|#{1,3})\s+' + re.escape(finding["title"]) +
                r'.*?(?:confirmation|status|assessment):\s*([^\n.]+)',
                re.IGNORECASE | re.MULTILINE
            )

            validation_match = validation_pattern.search(validation_text)
            if validation_match:
                status = validation_match.group(1).strip().lower()

                # Update finding based on validation
                if "confirm" in status or "true" in status or "valid" in status:
                    finding["validated"] = True
                elif "false" in status or "invalid" in status or "rejected" in status:
                    finding["validated"] = False
                    finding["severity"] = "Informational"  # Downgrade false positives
                else:
                    finding["validated"] = None

        return findings

    def _extract_recommendations_from_text(self, text):
        """Extract recommendations from text"""
        recommendations = []

        # Look for recommendations section
        rec_section_match = re.search(
            r'(?:^|\n)#{1,3}\s+Recommendations?\s*\n+(.*?)(?:\n#{1,3}|\Z)',
            text,
            re.DOTALL | re.MULTILINE
        )

        if rec_section_match:
            rec_text = rec_section_match.group(1)

            # Extract bullet points
            for line in rec_text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line):
                    # Clean up the line
                    rec = re.sub(r'^[^A-Za-z0-9]+', '', line).strip()
                    if rec:
                        recommendations.append(rec)

        # If no specific recommendations section, try to find recommendations in the text
        if not recommendations:
            rec_patterns = [
                r'(?:^|\n)(?:recommended|recommendation|we recommend|should).*?(?::\s*)([^\n]+)',
                r'(?:^|\n)(?:\d+\.|\*|-)\s+(?:Fix|Implement|Add|Remove|Update)([^\n]+)'
            ]

            for pattern in rec_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    rec = match.group(1).strip()
                    if rec and len(rec) > 10:  # Skip too short recommendations
                        recommendations.append(rec)

        return recommendations

    async def analyze_code(
        self,
        code: str,
        query: str,
    ) -> CodeAnalysis:
        """
        Analyze code snippet without full audit

        Args:
            code: Code snippet to analyze
            query: Analysis query

        Returns:
            Code analysis results
        """
        return await self.researcher.analyze_code_async(
            query=query,
            code_snippet=code,
        )

    async def validate_analysis(
        self,
        code: str,
        analysis: CodeAnalysis,
    ) -> ValidationResult:
        """
        Validate analysis without full audit

        Args:
            code: Code snippet
            analysis: Analysis to validate

        Returns:
            Validation results
        """
        return await self.validator.validate_analysis(
            code=code,
            analysis=analysis,
        )

    async def generate_report(
        self,
        analysis: CodeAnalysis,
        validation: ValidationResult,
        project_name: str,
    ) -> AuditReport:
        """
        Generate report without full audit

        Args:
            analysis: Analysis results
            validation: Validation results
            project_name: Project name

        Returns:
            Audit report
        """
        return await self.documentor.generate_report_async(
            analysis=analysis,
            validation=validation,
            project_name=project_name,
        )