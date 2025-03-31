"""
Models for structured output with Guidance.

These Pydantic models define the structured output formats used with
Guidance for ensuring consistently formatted LLM responses.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """Represents a decomposed sub-question from a complex query"""
    text: str = Field(..., description="The sub-question text")
    tool_name: str = Field(..., description="Name of the tool to use for answering")
    reasoning: Optional[str] = Field(None, description="Reasoning for this sub-question")


class QuestionDecompositionResult(BaseModel):
    """Structured result from question decomposition"""
    sub_questions: List[SubQuestion] = Field(
        default_factory=list,
        description="List of sub-questions derived from the main query"
    )
    reasoning: str = Field(
        default="",
        description="Reasoning behind the decomposition process"
    )


class BusinessFlow(BaseModel):
    """Represents a business flow within a smart contract"""
    name: str = Field(..., description="Descriptive name for the flow")
    description: str = Field(..., description="Brief description of what this flow does")
    steps: List[str] = Field(..., description="Sequence of operations that occur in this flow")
    functions: List[str] = Field(..., description="Main functions involved in this flow")
    actors: List[str] = Field(default_factory=list, description="Roles/addresses that participate in this flow")
    flow_type: str = Field(default="general", description="Type of business flow")
    contract_name: Optional[str] = Field(None, description="Name of the contract containing this flow")
    function_name: Optional[str] = Field(None, description="Name of the specific function if this is a function-level flow")


class BusinessFlowAnalysisResult(BaseModel):
    """Result of business flow analysis for a smart contract"""
    flows: List[BusinessFlow] = Field(
        default_factory=list,
        description="List of business flows identified in the contract"
    )
    contract_summary: str = Field(
        default="",
        description="Summary of the contract's business logic"
    )
