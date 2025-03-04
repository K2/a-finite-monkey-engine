"""
Analysis models for the Finite Monkey framework
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class CodeAnalysis(BaseModel):
    """Model for code analysis results"""
    
    source_code: str = Field(..., description="Source code analyzed")
    summary: str = Field(..., description="Analysis summary")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Analysis findings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    related_functions: List[Dict[str, Any]] = Field(default_factory=list, description="Related functions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the analysis"""
        summary_parts = [self.summary]
        
        if self.findings:
            summary_parts.append("\nFindings:")
            for i, finding in enumerate(self.findings):
                title = finding.get("title", f"Finding {i+1}")
                severity = finding.get("severity", "Medium")
                summary_parts.append(f"- {title} (Severity: {severity})")
        
        if self.recommendations:
            summary_parts.append("\nRecommendations:")
            for rec in self.recommendations:
                summary_parts.append(f"- {rec}")
        
        return "\n".join(summary_parts)


class ValidationIssue(BaseModel):
    """Model for validation issues"""
    
    title: str = Field(..., description="Issue title")
    description: str = Field(..., description="Issue description")
    severity: str = Field("Medium", description="Issue severity (Critical, High, Medium, Low, Informational)")
    location: Optional[str] = Field(None, description="Location in code")
    confidence: float = Field(0.5, description="Confidence score (0.0-1.0)")
    details: Optional[str] = Field(None, description="Additional details")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ValidationResult(BaseModel):
    """Model for validation results"""
    
    source_code: str = Field(..., description="Source code validated")
    summary: str = Field(..., description="Validation summary")
    issues: List[ValidationIssue] = Field(default_factory=list, description="Validation issues")
    has_critical_issues: bool = Field(False, description="Whether critical issues were identified")
    validation_methods: List[str] = Field(default_factory=list, description="Validation methods used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the validation"""
        summary_parts = [self.summary]
        
        if self.has_critical_issues:
            summary_parts.append("\n⚠️ CRITICAL ISSUES DETECTED ⚠️")
        
        if self.issues:
            summary_parts.append("\nIssues:")
            # Group issues by severity
            by_severity = {}
            for issue in self.issues:
                sev = issue.severity
                if sev not in by_severity:
                    by_severity[sev] = []
                by_severity[sev].append(issue)
            
            # Order by severity
            for severity in ["Critical", "High", "Medium", "Low", "Informational"]:
                if severity in by_severity:
                    for issue in by_severity[severity]:
                        summary_parts.append(f"- [{severity}] {issue.title}")
        else:
            summary_parts.append("\nNo issues detected.")
        
        summary_parts.append(f"\nValidation methods used: {', '.join(self.validation_methods)}")
        
        return "\n".join(summary_parts)