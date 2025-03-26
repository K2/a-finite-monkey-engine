from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class AttackSurfaceElement(BaseModel):
    """Represents an element that could be vulnerable to attacks"""
    name: str = Field(..., description="Name of the vulnerable element")
    type: Literal["function", "variable", "code_path", "privileged_operation"] = Field(..., description="Type of vulnerable element")
    description: Optional[str] = Field(None, description="Description of the vulnerability")
    related_to: Optional[str] = Field(None, description="Related function or component")
    impact: Optional[str] = Field(None, description="Potential impact if exploited")
    mitigation: Optional[str] = Field(None, description="Suggested mitigation strategy")

class AttackSurfaceSummary(BaseModel):
    """Summary of attack surfaces in a contract or function"""
    function_name: str = Field(..., description="Name of the analyzed function")
    contract_name: str = Field(..., description="Name of the contract")
    surfaces: List[AttackSurfaceElement] = Field(default_factory=list, description="List of attack surfaces")
    overall_severity: Literal["low", "medium", "high"] = Field(default="low", description="Overall severity rating")
    notes: Optional[str] = Field(None, description="Additional notes about the attack surfaces")
