"""
Report models for the Finite Monkey framework
"""

import os
import asyncio
import aiofiles
from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class MarkdownReport(BaseModel):
    """Markdown report model for code analysis outputs"""
    
    title: str = Field(..., description="Report title")
    summary: str = Field(..., description="Executive summary")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="Report sections")
    created_at: datetime = Field(default_factory=datetime.now, description="Report creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    async def save(self, file_path: str) -> None:
        """
        Save the report to a file asynchronously
        
        Args:
            file_path: Path to save the report to
            
        Raises:
            ValueError: If file_path is empty
        """
        if not file_path:
            raise ValueError("File path cannot be empty")
            
        # Create absolute path if relative
        file_path = os.path.abspath(file_path)
        
        # Create the directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Generate the markdown content
        content = self.to_markdown()
        
        # Write the content to the file
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)
    
    def to_markdown(self) -> str:
        """
        Convert the report to markdown format
        
        Returns:
            Markdown content
        """
        # Build the markdown content
        parts = []
        
        # Add title
        parts.append(f"# {self.title}")
        parts.append("")
        
        # Add timestamp
        parts.append(f"*Generated on: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        parts.append("")
        
        # Add summary
        parts.append("## Executive Summary")
        parts.append("")
        parts.append(self.summary)
        parts.append("")
        
        # Add sections
        for section in self.sections:
            # Add section title
            title = section.get("title", "")
            parts.append(f"## {title}")
            parts.append("")
            
            # Add section content
            content = section.get("content", "")
            parts.append(content)
            parts.append("")
            
            # Add subsections
            subsections = section.get("subsections", [])
            for subsection in subsections:
                # Add subsection title
                subtitle = subsection.get("title", "")
                parts.append(f"### {subtitle}")
                parts.append("")
                
                # Add subsection content
                subcontent = subsection.get("content", "")
                parts.append(subcontent)
                parts.append("")
        
        # Join parts and return
        return "\n".join(parts)


class AuditReport(BaseModel):
    """Comprehensive audit report model"""
    
    project_id: str = Field(..., description="Project identifier")
    project_name: str = Field(..., description="Project name")
    target_files: List[str] = Field(..., description="Files analyzed")
    query: str = Field(..., description="Original query or scope")
    summary: str = Field(..., description="Executive summary")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Audit findings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    analysis_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed analysis")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results")
    created_at: datetime = Field(default_factory=datetime.now, description="Report creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_markdown_report(self) -> MarkdownReport:
        """
        Convert to markdown report
        
        Returns:
            MarkdownReport object
        """
        # Build sections
        sections = []
        
        # Project details section
        project_section = {
            "title": "Project Details",
            "content": f"""
**Project Name:** {self.project_name}
**Project ID:** {self.project_id}
**Audit Query:** {self.query}
**Target Files:** {', '.join(self.target_files)}
**Analysis Date:** {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}
            """.strip(),
        }
        sections.append(project_section)
        
        # Findings section
        findings_content = []
        for i, finding in enumerate(self.findings):
            severity = finding.get("severity", "Medium")
            title = finding.get("title", f"Finding {i+1}")
            description = finding.get("description", "")
            location = finding.get("location", "")
            
            finding_text = f"""
#### {title} (Severity: {severity})

**Description:** {description}

**Location:** {location}
            """.strip()
            
            findings_content.append(finding_text)
        
        findings_section = {
            "title": "Audit Findings",
            "content": "\n\n".join(findings_content) if findings_content else "No findings identified.",
        }
        sections.append(findings_section)
        
        # Recommendations section
        recommendations_content = "\n".join([f"- {rec}" for rec in self.recommendations])
        recommendations_section = {
            "title": "Recommendations",
            "content": recommendations_content if recommendations_content else "No specific recommendations.",
        }
        sections.append(recommendations_section)
        
        # Analysis details section
        analysis_section = {
            "title": "Detailed Analysis",
            "content": self.analysis_details.get("summary", "No detailed analysis available."),
            "subsections": [],
        }
        
        # Add analysis subsections
        for key, value in self.analysis_details.items():
            if key != "summary" and isinstance(value, str):
                subsection = {
                    "title": key.replace("_", " ").title(),
                    "content": value,
                }
                analysis_section["subsections"].append(subsection)
        
        sections.append(analysis_section)
        
        # Validation results section
        validation_section = {
            "title": "Validation Results",
            "content": self.validation_results.get("summary", "No validation results available."),
            "subsections": [],
        }
        
        # Add validation subsections
        for key, value in self.validation_results.items():
            if key != "summary" and isinstance(value, str):
                subsection = {
                    "title": key.replace("_", " ").title(),
                    "content": value,
                }
                validation_section["subsections"].append(subsection)
        
        sections.append(validation_section)
        
        # Create markdown report
        markdown_report = MarkdownReport(
            title=f"Smart Contract Audit Report: {self.project_name}",
            summary=self.summary,
            sections=sections,
            created_at=self.created_at,
            metadata=self.metadata,
        )
        
        return markdown_report
    
    async def save(self, file_path: str) -> None:
        """
        Save the report to a file asynchronously
        
        Args:
            file_path: Path to save the report to
        """
        # Convert to markdown report
        markdown_report = self.to_markdown_report()
        
        # Save the report
        await markdown_report.save(file_path)