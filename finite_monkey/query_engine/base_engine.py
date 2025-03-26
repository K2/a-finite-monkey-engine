"""
Base query engine implementation that provides common functionality
for all specialized query engines in the finite-monkey system.
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from loguru import logger

from finite_monkey.pipeline.core import Context


class QueryResult(BaseModel):
    """Standard query result format for all query engines"""
    query: str = Field(..., description="The original query string")
    response: str = Field(..., description="The generated response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")
    confidence: float = Field(default=0.0, description="Confidence score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseQueryEngine:
    """
    Base class for all query engines in the finite-monkey system.
    Provides common functionality and a standard interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """z
        Initialize the base query engine with optional configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logger
    
    async def query(self, query_text: str, context: Optional[Context] = None) -> QueryResult:
        """
        Execute a query against the engine.
        
        Args:
            query_text: The query string
            context: Optional pipeline context with additional information
            
        Returns:
            StandardQueryResult with response and metadata
        """
        raise NotImplementedError("Subclasses must implement query method")
    
    async def batch_query(self, queries: List[str], context: Optional[Context] = None) -> List[QueryResult]:
        """
        Execute multiple queries in batch mode.
        
        Args:
            queries: List of query strings
            context: Optional pipeline context with additional information
            
        Returns:
            List of QueryResult objects
        """
        results = []
        for query in queries:
            result = await self.query(query, context)
            results.append(result)
        return results
    
    async def initialize(self) -> None:
        """
        Initialize the query engine, loading any necessary resources.
        """
        pass
    
    async def shutdown(self) -> None:
        """
        Clean up resources when the engine is no longer needed.
        """
        pass