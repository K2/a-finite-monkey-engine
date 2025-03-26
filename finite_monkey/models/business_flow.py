from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class FlowFunction(BaseModel):
    """A function call in a business flow"""
    call: str = Field(..., description="The name of the function being called")
    flow_vars: List[str] = Field(default_factory=list, description="Variables involved in this flow")

class AttackSurface(BaseModel):
    """A potential attack surface identified in the business flow"""
    name: str = Field(..., description="Name of the vulnerable element")
    type: Literal["function", "variable", "code_path", "privileged_operation"] = Field(..., description="Type of vulnerable element")
    description: Optional[str] = Field(None, description="Description of the vulnerability")
    related_to: Optional[str] = Field(None, description="Related function or component")

class BusinessFlow(BaseModel):
    """Business flow extracted from a function"""
    FlowFunctions: List[FlowFunction] = Field(default_factory=list, description="List of function calls in this flow")
    AttackSurfaces: List[AttackSurface] = Field(default_factory=list, description="List of potential attack surfaces")
    Confidence: float = Field(default=0.0, description="Confidence score for this flow (0.0-1.0)")
    Notes: Optional[str] = Field(None, description="Additional notes about this flow")
    Analysis: Optional[str] = Field(None, description="Analysis of the business flow")
