"""
Models for analysis results from various analyzers
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, model_validator


@dataclass
class BiasAnalysisResult:
    """Cognitive bias analysis results"""
    contract_name: str
    file_path: str
    biases: Dict[str, List[Dict[str, Any]]]  # bias_type -> instances
    analysis_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.analysis_date is None:
            self.analysis_date = datetime.now()


@dataclass
class AssumptionAnalysis:
    """Analysis of developer assumptions"""
    assumptions: List[Dict[str, str]]
    summary: str
    analysis_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.analysis_date is None:
            self.analysis_date = datetime.now()


@dataclass
class CounterfactualScenario:
    """A counterfactual scenario for smart contract analysis"""
    title: str
    category: str
    description: str
    impact: str
    likelihood: str
    severity: str
    mitigation: str
    function: Optional[str] = None
    location: Optional[str] = None


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


class DocumentationInconsistency(BaseModel):
    """Model for inconsistencies between documentation and code"""
    
    comment: Dict[str, Any] = Field(..., description="Comment information")
    code_snippet: str = Field(..., description="Related code snippet")
    inconsistency_type: str = Field(..., description="Type of inconsistency")
    description: str = Field(..., description="Description of the inconsistency")
    severity: str = Field("Medium", description="Severity (Critical, High, Medium, Low, Informational)")
    confidence: float = Field(0.5, description="Confidence score (0.0-1.0)")


class InconsistencyReport(BaseModel):
    """Report of inconsistencies between documentation and code"""
    
    total_comments: int = Field(..., description="Total number of comments analyzed")
    inconsistencies: List[Dict[str, Any]] = Field(default_factory=list, description="Found inconsistencies")
    code_language: str = Field("unknown", description="Detected language of the code")
    timestamp: Optional[str] = Field(None, description="Timestamp of the analysis")
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the inconsistencies"""
        summary_parts = [f"Documentation Analysis Report ({self.code_language})"]
        summary_parts.append(f"Analyzed {self.total_comments} comments and found {len(self.inconsistencies)} inconsistencies.")
        
        if self.inconsistencies:
            summary_parts.append("\nKey Inconsistencies:")
            
            # Group by severity
            by_severity = {}
            for inc in self.inconsistencies:
                sev = inc.get("severity", "Medium")
                if sev not in by_severity:
                    by_severity[sev] = []
                by_severity[sev].append(inc)
            
            # Show critical and high severity inconsistencies
            for severity in ["Critical", "High"]:
                if severity in by_severity:
                    for inc in by_severity[severity]:
                        desc = inc.get("description", "")
                        # Truncate long descriptions
                        if len(desc) > 100:
                            desc = desc[:97] + "..."
                        summary_parts.append(f"- [{severity}] {desc}")
            
            # Summary of others
            other_count = sum(len(by_severity.get(sev, [])) for sev in ["Medium", "Low", "Informational"])
            if other_count > 0:
                summary_parts.append(f"\nAdditionally, found {other_count} Medium/Low severity inconsistencies.")
        else:
            summary_parts.append("\nNo inconsistencies detected between documentation and code.")
        
        if self.timestamp:
            timestamp_obj = datetime.fromisoformat(self.timestamp)
            formatted_time = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
            summary_parts.append(f"\nAnalysis performed at: {formatted_time}")
        
        return "\n".join(summary_parts)


class VulnerabilityReport(BaseModel):
    """Model for vulnerability reports"""
    
    title: str = Field(..., description="Vulnerability title")
    description: str = Field(..., description="Vulnerability description")
    severity: str = Field("Medium", description="Severity (Critical, High, Medium, Low, Informational)")
    location: Optional[str] = Field(None, description="Location in code")
    vulnerability_type: Optional[str] = Field(None, description="Type of vulnerability")
    cwe_id: Optional[str] = Field(None, description="Common Weakness Enumeration ID")
    exploit_scenario: Optional[str] = Field(None, description="Example exploitation scenario")
    recommendation: Optional[str] = Field(None, description="Remediation recommendation")
    code_snippet: Optional[str] = Field(None, description="Related code snippet")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BiasAnalysisResult(BaseModel):
    """Model for cognitive bias analysis results"""
    
    contract_name: str = Field(..., description="Name of the analyzed contract")
    bias_findings: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Findings for each cognitive bias type"
    )
    has_critical_biases: bool = Field(False, description="Whether critical bias-related issues were identified")
    timestamp: Optional[str] = Field(None, description="Timestamp of the analysis")
    
    @model_validator(mode='after')
    def check_critical_biases(self):
        """Check if there are any critical bias-related issues"""
        findings = self.bias_findings
        
        for bias_type, bias_data in findings.items():
            instances = bias_data.get("instances", [])
            for instance in instances:
                if instance.get("severity", "").lower() == "critical":
                    self.has_critical_biases = True
                    break
        
        # Set timestamp if not provided
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
            
        return self
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the bias analysis"""
        summary_parts = [f"Cognitive Bias Analysis for {self.contract_name}"]
        
        if not self.bias_findings:
            summary_parts.append("\nNo cognitive bias-related issues detected.")
            return "\n".join(summary_parts)
        
        bias_types_found = list(self.bias_findings.keys())
        total_instances = sum(len(findings.get("instances", [])) for findings in self.bias_findings.values())
        
        summary_parts.append(f"\nDetected {total_instances} potential issues across {len(bias_types_found)} bias categories.")
        
        if self.has_critical_biases:
            summary_parts.append("\n⚠️ CRITICAL BIAS-RELATED ISSUES DETECTED ⚠️")
        
        # Summarize each bias type
        for bias_type, findings in self.bias_findings.items():
            instances = findings.get("instances", [])
            if not instances:
                continue
                
            bias_name = bias_type.replace("_", " ").title()
            summary = findings.get("summary", f"{len(instances)} instances found")
            
            summary_parts.append(f"\n{bias_name} ({len(instances)} issues):")
            summary_parts.append(f"  {summary}")
            
            # List critical and high severity instances
            critical_instances = [
                instance for instance in instances
                if instance.get("severity", "").lower() in ["critical", "high"]
            ]
            
            if critical_instances:
                for instance in critical_instances[:3]:  # Limit to 3
                    severity = instance.get("severity", "High")
                    title = instance.get("title", "Unnamed issue")
                    summary_parts.append(f"  - [{severity.upper()}] {title}")
                
                if len(critical_instances) > 3:
                    summary_parts.append(f"  - ...and {len(critical_instances) - 3} more critical/high issues")
        
        if self.timestamp:
            timestamp_obj = datetime.fromisoformat(self.timestamp)
            formatted_time = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
            summary_parts.append(f"\nAnalysis performed at: {formatted_time}")
        
        return "\n".join(summary_parts)


class AssumptionAnalysis(BaseModel):
    """Model for developer assumption analysis"""
    
    contract_name: str = Field(..., description="Name of the analyzed contract")
    assumptions: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Analysis of developer assumptions"
    )
    most_common_assumption: Optional[str] = Field(None, description="Most commonly occurring assumption")
    timestamp: Optional[str] = Field(None, description="Timestamp of the analysis")
    
    @model_validator(mode='after')
    def find_most_common(self):
        """Find the most common assumption"""
        assumptions = self.assumptions
        
        if assumptions:
            # Count vulnerabilities per assumption
            counts = {
                assumption: len(data.get("vulnerabilities", []))
                for assumption, data in assumptions.items()
            }
            
            # Find most common
            if counts:
                self.most_common_assumption = max(counts.items(), key=lambda x: x[1])[0]
        
        # Set timestamp if not provided
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
            
        return self
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the assumption analysis"""
        summary_parts = [f"Developer Assumption Analysis for {self.contract_name}"]
        
        if not self.assumptions:
            summary_parts.append("\nNo problematic developer assumptions identified.")
            return "\n".join(summary_parts)
        
        total_vulnerabilities = sum(
            len(data.get("vulnerabilities", [])) 
            for data in self.assumptions.values()
        )
        
        summary_parts.append(
            f"\nIdentified {len(self.assumptions)} problematic developer assumptions "
            f"leading to {total_vulnerabilities} vulnerabilities."
        )
        
        if self.most_common_assumption:
            summary_parts.append(f"\nMost common problematic assumption: \"{self.most_common_assumption}\"")
        
        # List each assumption and count
        for assumption, data in self.assumptions.items():
            vulns = data.get("vulnerabilities", [])
            if not vulns:
                continue
                
            summary_parts.append(f"\n• \"{assumption}\" ({len(vulns)} vulnerabilities)")
            
            # Optionally add a short description
            description = data.get("description", "").strip()
            if description:
                # Take first sentence or first 100 chars
                desc_brief = description.split(".")[0] if "." in description[:100] else description[:100]
                if len(desc_brief) < len(description):
                    desc_brief += "..."
                summary_parts.append(f"  {desc_brief}")
        
        if self.timestamp:
            timestamp_obj = datetime.fromisoformat(self.timestamp)
            formatted_time = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
            summary_parts.append(f"\nAnalysis performed at: {formatted_time}")
        
        return "\n".join(summary_parts)