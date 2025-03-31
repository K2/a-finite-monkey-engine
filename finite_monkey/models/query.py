"""
Models for structured query processing.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """
    Represents a decomposed sub-question from a complex query.
    """
    text: str = Field(..., description="The sub-question text")
    tool_name: str = Field(..., description="Name of the tool to use for answering")
    reasoning: Optional[str] = Field(None, description="Reasoning for this sub-question")


class QuestionDecompositionResult(BaseModel):
    """
    Structured result from question decomposition
    """
    sub_questions: List[SubQuestion] = Field(
        default_factory=list,
        description="List of sub-questions derived from the main query"
    )
    reasoning: str = Field(
        default="",
        description="Reasoning behind the decomposition process"
    )


class QueryResult(BaseModel):
    """
    Represents a query result from the query engine
    """
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The generated response")
    sub_questions: Optional[List[Dict[str, Any]]] = Field(None, description="Sub-questions if used")
    confidence: float = Field(0.0, description="Confidence score for the response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
