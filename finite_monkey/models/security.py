from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class SecurityFinding(BaseModel):
    """Security vulnerability or concern in smart contract code"""
    title: str = Field(..., description="Brief title of the finding")
    description: str = Field(..., description="Detailed description of the issue")
    severity: Literal["high", "medium", "low"] = Field(..., description="Severity level of the finding")
    location: Optional[str] = Field(None, description="Location in code where the issue was found")
    recommendation: str = Field(..., description="Recommendation for addressing the issue")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0-1.0) of this finding's accuracy")
    cwe_id: Optional[str] = Field(None, description="Common Weakness Enumeration ID if applicable")
    impact: Optional[str] = Field(None, description="Impact description of the vulnerability")
    references: Optional[List[str]] = Field(default_factory=list, description="Reference links for more information")

class SecurityAnalysisResult(BaseModel):
    """Complete security analysis results"""
    findings: List[SecurityFinding] = Field(..., description="List of security findings")
    summary: Optional[str] = Field(None, description="Overall security assessment summary")
    risk_score: Optional[float] = Field(None, description="Overall risk score from 0-10")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="When the analysis was performed")
