"""
Documentor agent for generating reports

This agent is responsible for creating comprehensive reports based on
the analysis and validation results.
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from ..adapters import Ollama
from ..models import CodeAnalysis, ValidationResult, MarkdownReport, AuditReport
from ..utils.prompting import get_report_prompt


class Documentor:
    """
    Documentor agent that generates reports
    
    This agent is responsible for:
    1. Formatting analysis and validation results into readable reports
    2. Generating markdown documentation
    3. Creating visualization data for charts and graphs
    """
    
    def __init__(
        self,
        llm_client: Optional[Ollama] = None,
        model_name: str = "llama3",
    ):
        """
        Initialize the documentor agent
        
        Args:
            llm_client: Ollama client for report generation
            model_name: Model to use for report generation
        """
        self.llm_client = llm_client or Ollama(model=model_name)
        self.model_name = model_name
    
    async def generate_report_async(
        self,
        analysis: Union[CodeAnalysis, Dict[str, Any]],
        validation: Optional[Union[ValidationResult, Dict[str, Any]]] = None,
        project_name: str = "Smart Contract Audit",
        project_id: Optional[str] = None,
        target_files: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> AuditReport:
        """
        Generate a comprehensive audit report
        
        Args:
            analysis: Code analysis results
            validation: Validation results
            project_name: Project name
            project_id: Project identifier
            target_files: List of files analyzed
            query: Original query
            
        Returns:
            Audit report
        """
        # Convert to dict if needed
        if isinstance(analysis, CodeAnalysis):
            analysis_dict = analysis.model_dump()
        else:
            analysis_dict = analysis
        
        if validation is None:
            validation_dict = {}
        elif isinstance(validation, ValidationResult):
            validation_dict = validation.model_dump()
        else:
            validation_dict = validation
        
        # Set default project ID if not provided
        if project_id is None:
            project_id = f"audit-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Set default target files if not provided
        if target_files is None:
            target_files = ["Unknown"]
        
        # Set default query if not provided
        if query is None:
            query = "Security audit"
        
        # Get code
        source_code = analysis_dict.get("source_code", "")
        
        # Generate report content using LLM
        report_prompt = get_report_prompt(
            analysis=analysis_dict,
            validation=validation_dict,
            project_name=project_name,
        )
        
        # Generate report using LLM
        report_text = await self.llm_client.acomplete(
            prompt=report_prompt,
            model=self.model_name,
        )
        
        # Parse the report content
        report_sections = self._parse_report_content(report_text)
        
        # Extract summary
        summary = report_sections.get("executive_summary", "No summary available.")
        
        # Extract findings
        findings = []
        findings_section = report_sections.get("findings", "")
        if findings_section:
            # Parse findings
            findings = self._extract_findings_from_text(findings_section)
        else:
            # Use findings from analysis
            findings = analysis_dict.get("findings", [])
        
        # Extract recommendations
        recommendations = []
        recommendations_section = report_sections.get("recommendations", "")
        if recommendations_section:
            # Parse recommendations
            recommendations = self._extract_recommendations_from_text(recommendations_section)
        else:
            # Use recommendations from analysis
            recommendations = analysis_dict.get("recommendations", [])
        
        # Create analysis details
        analysis_details = {
            "summary": analysis_dict.get("summary", ""),
            "code_analysis": report_sections.get("code_analysis", ""),
            "related_functions": report_sections.get("related_functions", ""),
        }
        
        # Create validation results
        validation_results = {
            "summary": validation_dict.get("summary", ""),
            "validation_details": report_sections.get("validation", ""),
        }
        
        # Create audit report
        audit_report = AuditReport(
            project_id=project_id,
            project_name=project_name,
            target_files=target_files,
            query=query,
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            analysis_details=analysis_details,
            validation_results=validation_results,
            metadata={
                "model": self.model_name,
                "source_code_hash": hash(source_code),
                "generated_at": datetime.now(),
            },
        )
        
        return audit_report
    
    def _parse_report_content(self, report_text: str) -> Dict[str, str]:
        """
        Parse report content into sections
        
        Args:
            report_text: Raw report text
            
        Returns:
            Dict of section name to content
        """
        sections = {}
        current_section = "unknown"
        current_content = []
        
        # Parse the report by sections
        lines = report_text.split("\n")
        for line in lines:
            # Check for section headers
            if line.startswith("# "):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)
                    current_content = []
                
                # Start new section
                current_section = line[2:].strip().lower().replace(" ", "_")
            elif line.startswith("## "):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)
                    current_content = []
                
                # Start new section
                current_section = line[3:].strip().lower().replace(" ", "_")
            else:
                # Add to current section
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)
        
        # Handle common section names
        if "executive_summary" not in sections:
            if "summary" in sections:
                sections["executive_summary"] = sections["summary"]
            elif "overview" in sections:
                sections["executive_summary"] = sections["overview"]
        
        return sections
    
    def _extract_findings_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract findings from text
        
        Args:
            text: Findings section text
            
        Returns:
            List of findings
        """
        findings = []
        current_finding = None
        
        # Parse by line
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            
            # Check for finding headers
            if line.startswith("###") or line.startswith("####"):
                # Save previous finding
                if current_finding:
                    findings.append(current_finding)
                
                # Start new finding
                title_parts = line.strip("#").strip()
                
                # Extract severity if present
                severity = "Medium"
                if "Critical" in title_parts:
                    severity = "Critical"
                elif "High" in title_parts:
                    severity = "High"
                elif "Medium" in title_parts:
                    severity = "Medium"
                elif "Low" in title_parts:
                    severity = "Low"
                elif "Informational" in title_parts:
                    severity = "Informational"
                
                # Clean up title
                title = title_parts
                for sev in ["Critical", "High", "Medium", "Low", "Informational"]:
                    title = title.replace(f"({sev})", "").replace(f"[{sev}]", "")
                
                title = title.strip()
                
                current_finding = {
                    "title": title,
                    "description": "",
                    "severity": severity,
                    "location": "",
                }
            elif current_finding:
                # Check for description
                if "description:" in line.lower():
                    desc_parts = line.split(":", 1)
                    if len(desc_parts) > 1:
                        current_finding["description"] = desc_parts[1].strip()
                # Check for location
                elif "location:" in line.lower():
                    loc_parts = line.split(":", 1)
                    if len(loc_parts) > 1:
                        current_finding["location"] = loc_parts[1].strip()
                # Check for severity
                elif "severity:" in line.lower():
                    sev_parts = line.split(":", 1)
                    if len(sev_parts) > 1:
                        severity = sev_parts[1].strip()
                        if severity in ["Critical", "High", "Medium", "Low", "Informational"]:
                            current_finding["severity"] = severity
                # Add to description if not empty
                elif line and "description" in current_finding:
                    if current_finding["description"]:
                        current_finding["description"] += " " + line
                    else:
                        current_finding["description"] = line
        
        # Add last finding
        if current_finding:
            findings.append(current_finding)
        
        return findings
    
    def _extract_recommendations_from_text(self, text: str) -> List[str]:
        """
        Extract recommendations from text
        
        Args:
            text: Recommendations section text
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Parse by line
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            
            # Check for recommendation items
            if line.startswith("- ") or line.startswith("* "):
                recommendations.append(line[2:].strip())
            elif line.startswith("1. ") or line.startswith("2. "):
                # Extract number
                parts = line.split(". ", 1)
                if len(parts) > 1:
                    recommendations.append(parts[1].strip())
        
        return recommendations