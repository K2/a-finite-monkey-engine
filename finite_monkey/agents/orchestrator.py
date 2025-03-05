"""
Workflow orchestrator for the Finite Monkey framework

This module implements the central orchestrator that coordinates the
atomic agents in the workflow.
"""

import os
import re
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path

from ..llama_index import AsyncIndexProcessor
from ..adapters import Ollama
from ..models import AuditReport, CodeAnalysis, ValidationResult, BiasAnalysisResult, AssumptionAnalysis
from .researcher import Researcher
from .validator import Validator, TreeSitterAnalyzer
from .documentor import Documentor
from .cognitive_bias_analyzer import CognitiveBiasAnalyzer
from .documentation_analyzer import DocumentationAnalyzer
from .counterfactual_generator import CounterfactualGenerator
from ..db.manager import TaskManager
from ..nodes_config import nodes_config


class WorkflowOrchestrator:
    """
    Workflow orchestrator for audit pipeline

    Coordinates the atomic agents (Researcher, Validator, Documentor) to
    execute the complete audit workflow. This class is the central component
    of the Finite Monkey framework that manages the execution of security
    audits on smart contracts.
    
    The orchestrator supports both direct synchronous execution and asynchronous
    task-based execution through the TaskManager. It handles the initialization
    of all required components including:
    
    1. LLM client (via Ollama)
    2. Vector store and embedding (via LlamaIndex)
    3. Specialized agents for different audit stages
    4. Task queue management for parallel execution
    
    Workflow Diagram:
    
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │             │      │             │      │             │      │             │
    │  Input      │─────▶│  Researcher │─────▶│  Validator  │─────▶│  Documentor │
    │  Code       │      │  Agent      │      │  Agent      │      │  Agent      │
    │             │      │             │      │             │      │             │
    └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
                               │                   │                     │
                               ▼                   ▼                     ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │             │      │             │      │             │      │             │
    │  Vector     │      │  Analysis   │      │  Validation │      │  Final      │
    │  Store      │      │  Results    │      │  Results    │      │  Report     │
    │             │      │             │      │             │      │             │
    └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
    
    Tasks can be executed:
    - Synchronously (via run_audit)
    - Asynchronously with background processing (via run_audit_workflow)
    
    The TaskManager handles parallelization and task dependencies automatically.
    
    TODO: Extension opportunities
    - Add more specialized agents for particular vulnerability classes
    - Support for custom analysis pipelines with configurable stages
    - Implement analysis caching for repeated audits
    - Add support for differential audits (comparing contract versions)
    - Integrate with external security tools and scanners
    - Implement a feedback loop for continuous improvement
    """

    def __init__(
        self,
        llama_index: Optional[AsyncIndexProcessor] = None,
        ollama: Optional[Ollama] = None,
        researcher: Optional[Researcher] = None,
        validator: Optional[Validator] = None,
        documentor: Optional[Documentor] = None,
        cognitive_bias_analyzer: Optional[CognitiveBiasAnalyzer] = None,
        documentation_analyzer: Optional[DocumentationAnalyzer] = None,
        counterfactual_generator: Optional[CounterfactualGenerator] = None,
        task_manager: Optional[TaskManager] = None,
        model_name: Optional[str] = None,
        base_dir: Optional[str] = None,
        db_url: Optional[str] = None,
        researcher_model: Optional[str] = None,
        validator_model: Optional[str] = None,
        documentor_model: Optional[str] = None,
        cognitive_bias_model: Optional[str] = None,
    ):
        """
        Initialize the workflow orchestrator

        Args:
            llama_index: LlamaIndex processor for queries
            ollama: Ollama client for LLM functions
            researcher: Researcher agent
            validator: Validator agent
            documentor: Documentor agent
            task_manager: Task manager for async workflows
            model_name: Default model name for LLM
            base_dir: Base directory for project files
            db_url: Database URL for task manager
            researcher_model: Specific model for researcher agent
            validator_model: Specific model for validator agent (typically Claude)
            documentor_model: Specific model for documentor agent
        """
        # Load config
        config = nodes_config()
        
        # Set model name from config if not provided
        self.model_name = model_name or config.WORKFLOW_MODEL or "qwen2.5:14b-instruct-q6_K"
        
        # Set specific agent models (with proper fallbacks)
        self.researcher_model = researcher_model or config.SCAN_MODEL or self.model_name
        self.validator_model = validator_model or config.CONFIRMATION_MODEL or "claude-3-5-sonnet"
        self.documentor_model = documentor_model or config.RELATION_MODEL or self.model_name
        self.cognitive_bias_model = cognitive_bias_model or config.COGNITIVE_BIAS_MODEL or "claude-3-5-sonnet"
        
        # Set base directory
        self.base_dir = base_dir or config.base_dir or os.getcwd()

        # Set up database URL from config if not provided
        self.db_url = db_url or config.ASYNC_DB_URL
        
        # Set up Task Manager if not provided
        self.task_manager = task_manager or TaskManager(db_url=self.db_url)
        
        # Set up LLM clients if not provided (with appropriate API bases)
        # Currently using the same Ollama instance for development, but the configuration 
        # supports different providers for different agents
        
        # Default client (compatibility with existing code)
        self.ollama = ollama or Ollama(model=self.model_name, api_base=config.OPENAI_API_BASE)
        
        # Agent-specific clients (for future differentiation)
        # Currently all using the same underlying client for development
        # but structure supports different providers
        self.researcher_client = self.ollama
        self.validator_client = self.ollama  # In the future this would be a Claude client
        self.documentor_client = self.ollama
        self.cognitive_bias_client = self.ollama
        self.documentation_analyzer_client = self.ollama
        self.counterfactual_generator_client = self.ollama
        
        # Set up the validator client based on configuration
        if self.validator_model.startswith("claude-") and config.CLAUDE_API_KEY:
            try:
                from ..adapters.claude import Claude
                self.validator_client = Claude(
                    api_key=config.CLAUDE_API_KEY, 
                    model=self.validator_model
                )
                print(f"Using Claude ({self.validator_model}) for validation")
            except (ImportError, ValueError) as e:
                print(f"Failed to initialize Claude adapter: {e}. Falling back to Ollama.")
                self.validator_client = self.ollama
        else:
            self.validator_client = self.ollama
            
        # Set up the cognitive bias client based on configuration
        if self.cognitive_bias_model.startswith("claude-") and config.CLAUDE_API_KEY:
            try:
                from ..adapters.claude import Claude
                self.cognitive_bias_client = Claude(
                    api_key=config.CLAUDE_API_KEY, 
                    model=self.cognitive_bias_model
                )
                print(f"Using Claude ({self.cognitive_bias_model}) for cognitive bias analysis")
            except (ImportError, ValueError) as e:
                print(f"Failed to initialize Claude adapter: {e}. Falling back to Ollama.")
                self.cognitive_bias_client = self.ollama
        else:
            self.cognitive_bias_client = self.ollama
            
        # Set up LlamaIndex client if not provided
        project_id = f"project-{self.model_name}"
        self.llama_index = llama_index or AsyncIndexProcessor(
            project_id=project_id,
            base_dir=self.base_dir,
        )

        # Set up agents if not provided
        self.researcher = researcher or Researcher(
            query_engine=self.llama_index,
            llm_client=self.researcher_client,
            model_name=self.researcher_model,
        )

        tree_sitter_analyzer = TreeSitterAnalyzer()
        self.validator = validator or Validator(
            tree_sitter_analyzer=tree_sitter_analyzer,
            llm_client=self.validator_client,
            model_name=self.validator_model,
        )

        self.documentor = documentor or Documentor(
            llm_client=self.documentor_client,
            model_name=self.documentor_model,
        )
        
        # Initialize specialized agents
        self.cognitive_bias_analyzer = cognitive_bias_analyzer or CognitiveBiasAnalyzer(
            llm_client=self.cognitive_bias_client
        )
        
        self.documentation_analyzer = documentation_analyzer or DocumentationAnalyzer(
            llm_client=self.documentation_analyzer_client
        )
        
        self.counterfactual_generator = counterfactual_generator or CounterfactualGenerator(
            llm_client=self.counterfactual_generator_client
        )
        
        # Tracking metrics for telemetry
        self.metrics = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "audit_start_time": None,
            "audit_end_time": None,
            "active_tasks": {},
        }
        
        # Start the task manager
        asyncio.create_task(self.task_manager.start())

    async def run_audit(
        self,
        solidity_path: Optional[str] = None,
        query: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> AuditReport:
        """
        Run a complete audit workflow with sensible defaults
        
        This method provides a "just works" approach that will use sensible defaults
        if parameters are not provided.

        Args:
            solidity_path: Path to the Solidity file to audit (default: examples/Vault.sol)
            query: Audit query (default: "Perform a comprehensive security audit")
            project_name: Name of the project (default: derived from file name)

        Returns:
            Audit report
        """
        # Set default file path if not provided
        if solidity_path is None:
            # Use the default example contract
            default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                        "examples", "Vault.sol")
            
            # Check if the default path exists
            if os.path.exists(default_path):
                solidity_path = default_path
            else:
                # Fallback to looking in the current directory
                example_files = [f for f in os.listdir(".") if f.endswith(".sol")]
                if example_files:
                    solidity_path = example_files[0]
                else:
                    raise ValueError("No Solidity file specified and no default example found")
        
        # Set default query if not provided
        if query is None:
            query = "Perform a comprehensive security audit"
            
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
    
    def _normalize_paths_and_project_name(
        self,
        file_paths: Union[str, List[str]],
        project_name: Optional[str] = None
    ) -> Tuple[List[str], str]:
        """
        Normalize file paths and derive project name if not provided
        
        Args:
            file_paths: Path or list of paths to process
            project_name: Optional project name
            
        Returns:
            Tuple of (normalized_paths, project_name)
        """
        # Handle single file or multiple files
        if isinstance(file_paths, str):
            normalized_paths = [file_paths]
            # Set project name from file if not provided
            if project_name is None:
                project_name = os.path.basename(file_paths).split(".")[0]
        else:
            normalized_paths = file_paths
            # Set project name from directory if not provided
            if project_name is None:
                try:
                    # Try to find a common parent directory
                    common_dir = os.path.commonpath([os.path.abspath(p) for p in normalized_paths])
                    project_name = os.path.basename(common_dir)
                    if not project_name or project_name == ".":
                        # Fall back to the first file name
                        project_name = os.path.basename(normalized_paths[0]).split(".")[0]
                except ValueError:
                    # Fallback for paths with no common parent
                    project_name = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
        return normalized_paths, project_name
        
    async def run_atomic_agent_workflow(
        self,
        solidity_path: Union[str, List[str]],
        query: str,
        project_name: Optional[str] = None,
    ) -> AuditReport:
        """
        Run workflow with atomic agents monitoring each other

        Args:
            solidity_path: Path to the Solidity file to audit, or a list of files
            query: Audit query (e.g., "Check for reentrancy vulnerabilities")
            project_name: Name of the project

        Returns:
            Audit report
        """
        from ..workflow.agent_controller import AgentController

        # Normalize paths and project name
        solidity_paths, project_name = self._normalize_paths_and_project_name(solidity_path, project_name)

        # Initialize the agent controller
        controller = AgentController(self.ollama)

        # Index all code files
        print(f"Indexing {len(solidity_paths)} files...")
        await self.llama_index.load_and_index(
            file_paths=solidity_paths,
        )

        # Read all file contents and combine them
        code_contents = {}
        combined_code = ""

        for file_path in solidity_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    code_contents[file_path] = file_content

                    # Add file separator for clear delineation in combined code
                    file_name = os.path.basename(file_path)
                    combined_code += f"\n\n// FILE: {file_name}\n{file_content}\n"
            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {str(e)}")

        # For single file case, just use the content directly
        code_content = list(code_contents.values())[0] if len(solidity_paths) == 1 else combined_code

        # STEP 1: Generate Researcher prompt
        print(f"Generating researcher prompt...")
        researcher_prompt = await controller.generate_agent_prompt(
            agent_type="researcher",
            task=f"Analyze the {project_name} contract for security vulnerabilities",
            context=code_content,
        )

        # Use the prompt to generate a research analysis
        print(f"Researcher agent analyzing code...")
        research_response = await self.ollama.acomplete(
            prompt=researcher_prompt,
        )

        # Monitor and provide feedback
        print(f"Monitoring researcher results...")
        researcher_feedback = await controller.monitor_agent(
            agent_type="researcher",
            state="completed",
            results=research_response,
        )

        print(f"Researcher feedback received")

        # STEP 2: Generate Validator prompt with feedback
        print(f"Generating validator prompt...")
        validator_prompt = await controller.generate_agent_prompt(
            agent_type="validator",
            task=f"Validate the security analysis for the {project_name} contract",
            context=f"Code:\n```solidity\n{code_content}\n```\n\nResearch Results:\n{research_response}\n\nFeedback:\n{researcher_feedback}",
        )

        # Use the prompt to generate validation
        print(f"Validator agent validating analysis...")
        validation_response = await self.ollama.acomplete(
            prompt=validator_prompt,
        )

        # Monitor and provide feedback
        print(f"Monitoring validator results...")
        validator_feedback = await controller.monitor_agent(
            agent_type="validator",
            state="completed",
            results=validation_response,
        )

        print(f"Validator feedback received")

        # STEP 3: Get coordination instructions
        print(f"Coordinating workflow...")
        coordination_instructions = await controller.coordinate_workflow(
            research_results=research_response,
            validation_results=validation_response,
        )

        # STEP 4: Generate Documentor prompt with coordination
        print(f"Generating documentor prompt...")
        documentor_prompt = await controller.generate_agent_prompt(
            agent_type="documentor",
            task=f"Create a comprehensive security report for the {project_name} contract",
            context=(
                f"Code:\n```solidity\n{code_content}\n```\n\n"
                f"Research Results:\n{research_response}\n\n"
                f"Validation Results:\n{validation_response}\n\n"
                f"Coordination Instructions:\n{coordination_instructions}"
            ),
        )

        # Use the prompt to generate report
        print(f"Documentor agent generating report...")
        report_text = await self.ollama.acomplete(
            prompt=documentor_prompt,
        )

        # Create a structured report
        print(f"Creating structured report...")

        # Parse findings from research and validation
        findings = self._extract_findings_from_text(research_response, validation_response)
        recommendations = self._extract_recommendations_from_text(report_text)

        # Create summary from report text
        summary_lines = []
        for line in report_text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                summary_lines.append(line)
            if len(summary_lines) >= 5:
                break

        summary = " ".join(summary_lines)

        # Create report
        report = AuditReport(
            project_id=f"audit-{project_name}",
            project_name=project_name,
            target_files=solidity_paths,  # Changed from [solidity_path] to solidity_paths
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
        
    async def analyze_cognitive_biases(
        self,
        solidity_path: str,
        project_name: Optional[str] = None,
        previous_analysis: Optional[CodeAnalysis] = None,
        include_remediation: bool = True,
        include_assumption_analysis: bool = True,
    ) -> Tuple[BiasAnalysisResult, Optional[Dict[str, List[Dict[str, str]]]], Optional[AssumptionAnalysis]]:
        """
        Analyze cognitive biases in smart contract code
        
        This method analyzes the cognitive biases present in the code that may lead to
        security vulnerabilities, providing a comprehensive cognitive security analysis.
        
        Args:
            solidity_path: Path to the Solidity file to analyze
            project_name: Name of the project (derived from filename if not provided)
            previous_analysis: Optional previous code analysis to build upon
            include_remediation: Whether to generate remediation plans
            include_assumption_analysis: Whether to perform assumption analysis
            
        Returns:
            Tuple containing:
            - BiasAnalysisResult: The cognitive bias analysis results
            - Optional remediation plans dictionary
            - Optional assumption analysis
        """
        # Set project name from file if not provided
        if project_name is None:
            project_name = os.path.basename(solidity_path).split(".")[0]
            
        # Read the file content
        with open(solidity_path, "r", encoding="utf-8") as f:
            code_content = f.read()
            
        # Step 1: Analyze cognitive biases
        print(f"Analyzing cognitive biases in {project_name}...")
        bias_analysis = await self.cognitive_bias_analyzer.analyze_cognitive_biases(
            contract_code=code_content,
            contract_name=project_name,
            previous_analysis=previous_analysis
        )
        
        # Step 2: Generate remediation plan if requested
        remediation_plan = None
        if include_remediation:
            print(f"Generating remediation plans...")
            remediation_plan = await self.cognitive_bias_analyzer.generate_remediation_plan(
                contract_code=code_content,
                bias_analysis=bias_analysis
            )
            
        # Step 3: Perform assumption analysis if requested and there are findings
        assumption_analysis = None
        if include_assumption_analysis and bias_analysis.bias_findings:
            print(f"Analyzing developer assumptions...")
            
            # Create vulnerability reports from bias findings
            vulnerability_reports = []
            for bias_type, findings in bias_analysis.bias_findings.items():
                for instance in findings.get("instances", []):
                    vulnerability_reports.append(
                        VulnerabilityReport(
                            title=instance.get("title", f"{bias_type} issue"),
                            description=instance.get("description", ""),
                            severity=instance.get("severity", "Medium"),
                            location=instance.get("location", ""),
                            vulnerability_type=bias_type.replace("_", " ").title()
                        )
                    )
                    
            # Skip if no vulnerabilities found
            if vulnerability_reports:
                assumption_results = await self.cognitive_bias_analyzer.generate_assumption_analysis(
                    contract_code=code_content,
                    vulnerability_reports=vulnerability_reports
                )
                
                # Create structured assumption analysis
                assumption_analysis = AssumptionAnalysis(
                    contract_name=project_name,
                    assumptions=assumption_results
                )
        
        return (bias_analysis, remediation_plan, assumption_analysis)
        
    async def analyze_documentation_consistency(
        self,
        solidity_path: str,
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze consistency between code and documentation/comments
        
        This method finds discrepancies between what the code does and what
        the documentation/comments claim it does.
        
        Args:
            solidity_path: Path to the Solidity file to analyze
            project_name: Name of the project (derived from filename if not provided)
            
        Returns:
            Documentation consistency analysis results
        """
        # Set project name from file if not provided
        if project_name is None:
            project_name = os.path.basename(solidity_path).split(".")[0]
            
        # Read the file content
        with open(solidity_path, "r", encoding="utf-8") as f:
            code_content = f.read()
            
        # Analyze documentation consistency
        print(f"Analyzing documentation consistency in {project_name}...")
        consistency_analysis = await self.documentation_analyzer.analyze_inconsistencies(
            contract_code=code_content,
            contract_name=project_name
        )
        
        # Generate linguistic heatmap
        print(f"Generating linguistic heatmap...")
        heatmap = await self.documentation_analyzer.generate_linguistic_heatmap(
            contract_code=code_content,
            inconsistencies=consistency_analysis
        )
        
        return {
            "consistency_analysis": consistency_analysis,
            "linguistic_heatmap": heatmap
        }
        
    async def generate_counterfactuals(
        self,
        solidity_path: str,
        vulnerability_reports: List[VulnerabilityReport],
        project_name: Optional[str] = None,
        scenario_count: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual scenarios for security training
        
        This method creates alternative scenarios that demonstrate how vulnerabilities
        could be exploited or prevented, useful for training security auditors.
        
        Args:
            solidity_path: Path to the Solidity file to analyze
            vulnerability_reports: List of vulnerability reports to generate counterfactuals for
            project_name: Name of the project (derived from filename if not provided)
            scenario_count: Number of scenarios to generate per vulnerability
            
        Returns:
            Counterfactual generation results
        """
        # Set project name from file if not provided
        if project_name is None:
            project_name = os.path.basename(solidity_path).split(".")[0]
            
        # Read the file content
        with open(solidity_path, "r", encoding="utf-8") as f:
            code_content = f.read()
            
        # Generate counterfactuals
        print(f"Generating counterfactual scenarios for {project_name}...")
        scenarios = await self.counterfactual_generator.generate_counterfactuals(
            contract_code=code_content,
            vulnerability_reports=vulnerability_reports,
            scenario_count=scenario_count
        )
        
        # Generate exploitation paths
        print(f"Generating exploitation paths...")
        exploitation_paths = await self.counterfactual_generator.generate_exploitation_paths(
            contract_code=code_content,
            vulnerability_reports=vulnerability_reports
        )
        
        # Generate training scenarios
        print(f"Generating training scenarios...")
        training_scenarios = await self.counterfactual_generator.generate_training_scenarios(
            contract_code=code_content,
            vulnerability_reports=vulnerability_reports
        )
        
        return {
            "counterfactual_scenarios": scenarios,
            "exploitation_paths": exploitation_paths,
            "training_scenarios": training_scenarios
        }
        
    async def run_audit_workflow(
        self,
        solidity_paths: Union[str, List[str]],
        query: str,
        project_name: Optional[str] = None,
        wait_for_completion: bool = True,
        timeout: Optional[float] = None,
    ) -> Union[Dict[str, Any], AuditReport]:
        """
        Run a complete audit workflow using the task manager
        
        This method uses the async task manager to run the workflow in the background,
        allowing for concurrent processing of multiple files.
        
        Args:
            solidity_paths: Path(s) to the Solidity file(s) to audit
            query: Audit query (e.g., "Check for reentrancy vulnerabilities")
            project_name: Name of the project
            wait_for_completion: Whether to wait for the workflow to complete
            timeout: Timeout in seconds for waiting (None for no timeout)
            
        Returns:
            If wait_for_completion is True, returns the final audit report.
            Otherwise, returns a dictionary of task IDs.
        """
        # Start telemetry tracking
        self.metrics["audit_start_time"] = datetime.now().isoformat()
        self.metrics["tasks_created"] = 0
        self.metrics["tasks_completed"] = 0 
        self.metrics["tasks_failed"] = 0
        self.metrics["active_tasks"] = {}
        
        # Normalize paths and project name
        solidity_paths, project_name = self._normalize_paths_and_project_name(solidity_paths, project_name)
        
        # Create workflow in the task manager
        try:
            task_ids = await self.task_manager.create_audit_workflow(
                project_id=project_name,
                file_paths=solidity_paths,
                query=query,
                model_name=self.model_name,
            )
            
            # Update telemetry with task IDs
            for file_path, file_tasks in task_ids.items():
                for task_type, task_id in file_tasks.items():
                    if task_id:
                        self.metrics["tasks_created"] += 1
                        self.metrics["active_tasks"][task_id] = {
                            "file": file_path,
                            "type": task_type,
                            "created_at": datetime.now().isoformat(),
                            "status": "pending"
                        }
            
            # If not waiting for completion, return task IDs
            if not wait_for_completion:
                return task_ids
            
            # Wait for all tasks to complete
            results = {}
            for file_path, file_tasks in task_ids.items():
                report_task_id = None
                report_pending = True
                
                # Check if there's already an analysis task
                if file_tasks["analysis"]:
                    # Wait for analysis to complete
                    analysis_result = await self.task_manager.wait_for_task(
                        file_tasks["analysis"], 
                        timeout=timeout
                    )
                    
                    # Update telemetry
                    if analysis_result["status"] == "completed":
                        self.metrics["tasks_completed"] += 1
                        self.metrics["active_tasks"][file_tasks["analysis"]]["status"] = "completed"
                        self.metrics["active_tasks"][file_tasks["analysis"]]["completed_at"] = datetime.now().isoformat()
                    elif analysis_result["status"] == "failed":
                        self.metrics["tasks_failed"] += 1
                        self.metrics["active_tasks"][file_tasks["analysis"]]["status"] = "failed"
                        self.metrics["active_tasks"][file_tasks["analysis"]]["failed_at"] = datetime.now().isoformat()
                        if "error" in analysis_result:
                            self.metrics["active_tasks"][file_tasks["analysis"]]["error"] = analysis_result["error"]
                    
                    # Check next task ID
                    if analysis_result["status"] == "completed" and "result" in analysis_result:
                        next_task_id = analysis_result["result"].get("next_task_id")
                        
                        if next_task_id:
                            # Update telemetry for validation task
                            self.metrics["active_tasks"][next_task_id] = {
                                "file": file_path,
                                "type": "validation",
                                "created_at": datetime.now().isoformat(),
                                "status": "pending"
                            }
                            
                            # Wait for validation to complete
                            validation_result = await self.task_manager.wait_for_task(
                                next_task_id,
                                timeout=timeout
                            )
                            
                            # Update telemetry
                            if validation_result["status"] == "completed":
                                self.metrics["tasks_completed"] += 1
                                self.metrics["active_tasks"][next_task_id]["status"] = "completed"
                                self.metrics["active_tasks"][next_task_id]["completed_at"] = datetime.now().isoformat()
                            elif validation_result["status"] == "failed":
                                self.metrics["tasks_failed"] += 1
                                self.metrics["active_tasks"][next_task_id]["status"] = "failed"
                                self.metrics["active_tasks"][next_task_id]["failed_at"] = datetime.now().isoformat()
                                if "error" in validation_result:
                                    self.metrics["active_tasks"][next_task_id]["error"] = validation_result["error"]
                            
                            # Check for report task
                            if validation_result["status"] == "completed" and "result" in validation_result:
                                report_task_id = validation_result["result"].get("next_task_id")
                                
                                if report_task_id:
                                    # Update telemetry for report task
                                    self.metrics["active_tasks"][report_task_id] = {
                                        "file": file_path,
                                        "type": "report",
                                        "created_at": datetime.now().isoformat(),
                                        "status": "pending"
                                    }
                                
                # If a report task was identified, wait for it
                if report_task_id:
                    report_result = await self.task_manager.wait_for_task(
                        report_task_id,
                        timeout=timeout
                    )
                    
                    # Update telemetry
                    if report_result["status"] == "completed":
                        self.metrics["tasks_completed"] += 1
                        self.metrics["active_tasks"][report_task_id]["status"] = "completed"
                        self.metrics["active_tasks"][report_task_id]["completed_at"] = datetime.now().isoformat()
                    elif report_result["status"] == "failed":
                        self.metrics["tasks_failed"] += 1
                        self.metrics["active_tasks"][report_task_id]["status"] = "failed"
                        self.metrics["active_tasks"][report_task_id]["failed_at"] = datetime.now().isoformat()
                        if "error" in report_result:
                            self.metrics["active_tasks"][report_task_id]["error"] = report_result["error"]
                    
                    if report_result["status"] == "completed" and "result" in report_result:
                        # Add to results
                        results[file_path] = report_result["result"]
                        report_pending = False
                
                # If report is still pending, check if there's a direct report task
                if report_pending and file_tasks.get("report"):
                    report_result = await self.task_manager.wait_for_task(
                        file_tasks["report"],
                        timeout=timeout
                    )
                    
                    # Update telemetry
                    if report_result["status"] == "completed":
                        self.metrics["tasks_completed"] += 1
                        self.metrics["active_tasks"][file_tasks["report"]]["status"] = "completed"
                        self.metrics["active_tasks"][file_tasks["report"]]["completed_at"] = datetime.now().isoformat()
                    elif report_result["status"] == "failed":
                        self.metrics["tasks_failed"] += 1
                        self.metrics["active_tasks"][file_tasks["report"]]["status"] = "failed"
                        self.metrics["active_tasks"][file_tasks["report"]]["failed_at"] = datetime.now().isoformat()
                        if "error" in report_result:
                            self.metrics["active_tasks"][file_tasks["report"]]["error"] = report_result["error"]
                    
                    if report_result["status"] == "completed" and "result" in report_result:
                        # Add to results
                        results[file_path] = report_result["result"]
            
            # Complete telemetry
            self.metrics["audit_end_time"] = datetime.now().isoformat()
            
            # Create a combined audit report
            audit_report = AuditReport(
                project_id=f"audit-{project_name}",
                project_name=project_name,
                target_files=solidity_paths,
                query=query,
                summary=f"Audit of {len(solidity_paths)} files for {project_name}",
                findings=[],
                recommendations=[],
                metadata={
                    "results": results,
                    "telemetry": self.metrics
                },
            )
            
            # Extract findings from all reports
            for file_path, result in results.items():
                if "report_data" in result:
                    if "findings" in result["report_data"]:
                        audit_report.findings.extend(result["report_data"]["findings"])
                    if "recommendations" in result["report_data"]:
                        audit_report.recommendations.extend(result["report_data"]["recommendations"])
            
            # Add report paths
            report_paths = []
            for file_path, result in results.items():
                if "report_path" in result:
                    report_paths.append(result["report_path"])
            
            if report_paths:
                audit_report.metadata["report_paths"] = report_paths
            
            return audit_report
            
        except Exception as e:
            # Update telemetry on error
            self.metrics["audit_end_time"] = datetime.now().isoformat()
            self.metrics["error"] = str(e)
            raise