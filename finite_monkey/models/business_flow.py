"""
Models for business flow analysis in smart contracts.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BusinessFlow(BaseModel):
    """
    Represents a business flow within a smart contract
    """
    name: str = Field(..., description="Descriptive name for the flow")
    description: str = Field(..., description="Brief description of what this flow does")
    steps: List[str] = Field(..., description="Sequence of operations that occur in this flow")
    functions: List[str] = Field(..., description="Main functions involved in this flow")
    actors: List[str] = Field(default_factory=list, description="Roles/addresses that participate in this flow")
    flow_type: str = Field(default="general", description="Type of business flow")
    contract_name: Optional[str] = Field(None, description="Name of the contract containing this flow")
    function_name: Optional[str] = Field(None, description="Name of the specific function if this is a function-level flow")


class BusinessFlowAnalysisResult(BaseModel):
    """
    Result of business flow analysis for a smart contract
    """
    flows: List[BusinessFlow] = Field(
        default_factory=list,
        description="List of business flows identified in the contract"
    )
    contract_summary: str = Field(
        default="",
        description="Summary of the contract's business logic"
    )
