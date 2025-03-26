"""
Pydantic models for pipeline data.

This module defines Pydantic models to represent the data passed between
different stages of the analysis pipeline.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class FileData(BaseModel):
    """Represents a file in the codebase."""
    id: str = Field(..., description="Unique identifier for the file")
    path: str = Field(..., description="Path to the file")
    content: str = Field(..., description="Content of the file")
    is_solidity: bool = Field(False, description="Whether the file is a Solidity contract")
    name: Optional[str] = Field(None, description="Name of the contract (if applicable)")
    size: Optional[int] = Field(None, description="Size of the file in bytes")
    extension: Optional[str] = Field(None, description="File extension")

class ChunkData(BaseModel):
    """Represents a chunk of code."""
    id: str = Field(..., description="Unique identifier for the chunk")
    file_id: str = Field(..., description="ID of the file the chunk belongs to")
    content: str = Field(..., description="Content of the chunk")
    chunk_type: str = Field(..., description="Type of chunk (contract, function, etc.)")
    start_line: Optional[int] = Field(None, description="Start line number of the chunk")
    end_line: Optional[int] = Field(None, description="End line number of the chunk")

class FunctionData(BaseModel):
    """Represents a function in the codebase."""
    id: str = Field(..., description="Unique identifier for the function")
    contract_id: str = Field(..., description="ID of the contract the function belongs to")
    name: str = Field(..., description="Name of the function")
    visibility: str = Field(..., description="Visibility of the function (public, private, etc.)")
    parameters: List[str] = Field(..., description="List of function parameters")
    return_type: Optional[str] = Field(None, description="Return type of the function")
    modifiers: List[str] = Field(..., description="List of function modifiers")
    body: str = Field(..., description="Body of the function")

class DataFlowData(BaseModel):
    """Represents a data flow in the codebase."""
    id: str = Field(..., description="Unique identifier for the data flow")
    contract_id: str = Field(..., description="ID of the contract the data flow belongs to")
    function_name: str = Field(..., description="Name of the function the data flow originates from")
    path: List[Dict[str, Any]] = Field(..., description="Path of the data flow")
    description: Optional[str] = Field(None, description="Description of the data flow")

class VectorMatchData(BaseModel):
    """Represents a vector match in the codebase."""
    id: str = Field(..., description="Unique identifier for the vector match")
    contract_id: str = Field(..., description="ID of the contract the vector match belongs to")
    pattern: str = Field(..., description="The matched code pattern")
    score: float = Field(..., description="Similarity score of the match")

class ThreatData(BaseModel):
    """Represents a threat detected in the codebase."""
    id: str = Field(..., description="Unique identifier for the threat")
    contract_id: str = Field(..., description="ID of the contract the threat belongs to")
    threat_name: str = Field(..., description="Name of the threat")
    severity: str = Field(..., description="Severity of the threat")
    description: str = Field(..., description="Description of the threat")

class ProjectData(BaseModel):
    """Represents a project with its files, contracts, and functions."""
    file: FileData = Field(..., description="File data")
    contracts: List[Dict[str, Any]] = Field(default_factory=list, description="List of contracts in the file")
    functions: List[Dict[str, Any]] = Field(default_factory=list, description="List of functions in the file")
