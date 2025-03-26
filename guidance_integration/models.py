"""
Pydantic models for structured data used in A Finite Monkey Engine
"""

from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime

class Node(BaseModel):
    """Represents a node in a business flow graph"""
    name: str = Field(..., description="Display name for the node")
    flow_type: Literal["function", "validation", "state", "event", "external"] = Field(..., description="Type of node")
    description: Optional[str] = Field(None, description="Description of the node's purpose")
    size: Optional[int] = Field(None, description="Display size for visualization")
    variables_written: str = Field(..., description="Variables written by this node")
    variables_read: str = Field(..., description="Variables read by this node")
    Visibility: str = Field(...,description="one of public, external, internal, and private")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional properties")
    lines: List[str] = Field(default_factory=list, description="List range with the node")

class Link(BaseModel):
    """Represents a connection between nodes in a business flow graph"""
    source: str = Field(..., description="source node")
    target: str = Field(..., description="target node")
    label: Optional[str] = Field(None, description="Label for the connection")
    value: Optional[int] = Field(None, description="Strength or weight of the connection")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0-1.0) of this link's accuracy")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional properties")

class BusinessFlowData(BaseModel):
    """Complete business flow graph data"""
    flows: List[Node] = Field(..., description="List of nodes in the flow")
    links: List[Link] = Field(..., description="List of connections between nodes")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata about the analysis")
    notes: Optional[str] = Field(None, description="Additional notes about this node")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0-1.0) of this flow being accurate")
    f1: Optional[float] = Field(None, description="f1 score")
    rfi: Optional[float] = Field(None, description="request for information")

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

class ToolCall(BaseModel):
    """Represents an LLM tool call"""
    name: str = Field(..., description="Name of the tool called")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to the tool")
    id: Optional[str] = Field(None, description="Optional ID for the tool call")

class GenerationResult(BaseModel):
    """Result from a generation request"""
    response: str = Field(..., description="Text response from the LLM")
    toolCalls: List[ToolCall] = Field(default_factory=list, description="List of tool calls made")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata about the generation")
