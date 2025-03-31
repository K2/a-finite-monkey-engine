"""
Models for structured query processing in the FLARE engine.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """Represents a decomposed sub-question from a complex query"""
    text: str = Field(..., description="The sub-question text")
    tool_name: str = Field(..., description="Name of the tool to use for answering")
    reasoning: Optional[str] = Field(None, description="Reasoning for this sub-question")


class QueryDecomposition(BaseModel):
    """Result of decomposing a complex query into sub-questions"""
    sub_questions: List[SubQuestion] = Field(
        default_factory=list,
        description="List of sub-questions derived from the main query"
    )
    reasoning: str = Field(
        default="",
        description="Reasoning behind the decomposition process"
    )
