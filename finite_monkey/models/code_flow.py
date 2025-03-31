"""
Models for code flow analysis, particularly source/sink relationships.
"""
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field


class FlowNodeType(str, Enum):
    """Types of nodes in a code flow graph"""
    SOURCE = "source"  # Data origins (e.g., user input, external calls)
    SINK = "sink"      # Data destinations (e.g., storage writes, external calls)
    TRANSFORM = "transform"  # Data transformations (e.g., calculations, conversions)
    CONDITION = "condition"  # Flow control (e.g., if statements, requires)
    STORAGE = "storage"  # State variables and storage access


class CodeLocation(BaseModel):
    """Represents a precise location in the code"""
    file_path: str = Field(..., description="Path to the file")
    start_line: int = Field(..., description="Starting line number (1-based)")
    start_column: int = Field(..., description="Starting column number (0-based)")
    end_line: int = Field(..., description="Ending line number (1-based)")
    end_column: int = Field(..., description="Ending column number (0-based)")
    code_snippet: Optional[str] = Field(None, description="The actual code snippet")
    
    def __str__(self) -> str:
        """String representation of location"""
        return f"{self.file_path}:{self.start_line}:{self.start_column}-{self.end_line}:{self.end_column}"


class FlowNode(BaseModel):
    """Represents a node in a code flow graph"""
    id: str = Field(..., description="Unique identifier for the node")
    name: str = Field(..., description="Name of the variable, function, or expression")
    node_type: FlowNodeType = Field(..., description="Type of node")
    location: Optional[CodeLocation] = Field(None, description="Location in code")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def __str__(self) -> str:
        """String representation of node"""
        return f"{self.node_type.value}: {self.name} at {self.location or 'unknown'}"


class FlowEdge(BaseModel):
    """Represents an edge in a code flow graph"""
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    edge_type: str = Field(..., description="Type of the edge (e.g., 'data_flow', 'control_flow')")
    label: Optional[str] = Field(None, description="Optional label for the edge")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CodeFlowGraph(BaseModel):
    """Represents a code flow graph with nodes and edges"""
    nodes: List[FlowNode] = Field(default_factory=list, description="List of nodes in the graph")
    edges: List[FlowEdge] = Field(default_factory=list, description="List of edges in the graph")
    function_name: Optional[str] = Field(None, description="Function this flow belongs to")
    contract_name: Optional[str] = Field(None, description="Contract this flow belongs to")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def add_node(self, node: FlowNode) -> None:
        """Add a node to the graph"""
        self.nodes.append(node)
    
    def add_edge(self, edge: FlowEdge) -> None:
        """Add an edge to the graph"""
        self.edges.append(edge)
        
    def get_sources(self) -> List[FlowNode]:
        """Get all source nodes in the graph"""
        return [n for n in self.nodes if n.node_type == FlowNodeType.SOURCE]
        
    def get_sinks(self) -> List[FlowNode]:
        """Get all sink nodes in the graph"""
        return [n for n in self.nodes if n.node_type == FlowNodeType.SINK]
