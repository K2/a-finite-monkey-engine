"""
Implementation of a FLARE Instruction Query Engine for enhanced
reasoning capabilities in code analysis tasks.
"""
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import BaseModel, Field
from loguru import logger

from llama_index.core.query_engine import FLAREInstructQueryEngine
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

from finite_monkey.pipeline.core import Context
from finite_monkey.nodes_config import config
from .base_engine import BaseQueryEngine, QueryResult


class FlareQueryEngine(BaseQueryEngine):
    """
    FLARE (Forward-Looking Active REasoning) Query Engine implementation.
    
    This engine breaks down complex queries into step-by-step reasoning processes,
    and is particularly effective for complex code analysis tasks.
    """
    
    def __init__(
        self, 
        underlying_engine: Optional[Any] = None,
        vector_index: Optional[VectorStoreIndex] = None,
        max_iterations: int = 5,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the FLARE query engine.
        
        Args:
            underlying_engine: Optional existing query engine to use
            vector_index: Optional vector index for retrieval
            max_iterations: Maximum reasoning iterations
            verbose: Whether to log detailed reasoning steps
            config: Additional configuration
        """
        super().__init__(config)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.vector_index = vector_index
        self.underlying_engine = underlying_engine
        self.flare_engine = None  # Will be initialized later
    
    async def initialize(self) -> None:
        """
        Initialize the FLARE engine with the necessary components.
        """
        if self.flare_engine is not None:
            return  # Already initialized
        
        try:
            if self.underlying_engine is None and self.vector_index is not None:
                # Create a default query engine from the vector index
                retriever = VectorIndexRetriever(
                    index=self.vector_index,
                    similarity_top_k=5
                )
                
                # Add similarity score threshold
                similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
                
                self.underlying_engine = self.vector_index.as_query_engine(
                    retriever=retriever,
                    node_postprocessors=[similarity_postprocessor],
                )
            
            # Initialize the FLARE engine with the underlying engine
            if self.underlying_engine is not None:
                self.flare_engine = FLAREInstructQueryEngine(
                    query_engine=self.underlying_engine,
                    max_iterations=self.max_iterations,
                    verbose=self.verbose
                )
                self.logger.info("FLARE Instruction Query Engine initialized successfully")
            else:
                self.logger.error("Cannot initialize FLARE engine: No underlying engine or vector index provided")
        except Exception as e:
            self.logger.error(f"Error initializing FLARE engine: {e}")
            raise
    
    async def query(self, query_text: str, context: Optional[Context] = None) -> QueryResult:
        """
        Execute a query using the FLARE reasoning approach.
        
        Args:
            query_text: The query string
            context: Optional pipeline context with additional information
            
        Returns:
            QueryResult with response and metadata
        """
        await self.initialize()
        
        if self.flare_engine is None:
            return QueryResult(
                query=query_text,
                response="FLARE engine not properly initialized",
                confidence=0.0,
                metadata={"error": "Engine initialization failed"}
            )
        
        try:
            # Execute the query using the FLARE engine
            response = await asyncio.to_thread(
                self.flare_engine.query,
                query_text
            )
            
            # Extract sources if available
            sources = []
            if hasattr(response, 'source_nodes'):
                sources = [
                    {
                        "text": node.node.text,
                        "score": node.score if hasattr(node, 'score') else None,
                        "id": node.node.node_id if hasattr(node.node, 'node_id') else None,
                    }
                    for node in response.source_nodes
                ]
            
            # Extract the response text
            response_text = str(response)
            
            # Calculate a simple confidence score based on source relevance
            confidence = 0.0
            if sources:
                avg_score = sum(s.get('score', 0) or 0 for s in sources) / len(sources)
                confidence = min(avg_score, 1.0)  # Ensure it's between 0 and 1
            
            # Create metadata including FLARE reasoning steps if available
            metadata = {}
            if hasattr(response, 'metadata'):
                metadata = response.metadata
            
            # Enhance with context information if available
            if context and hasattr(context, 'state'):
                metadata['context_state'] = {
                    k: str(v) for k, v in context.state.items() 
                    if k not in ('files', 'chunks', 'contracts')  # Avoid large objects
                }
            
            return QueryResult(
                query=query_text,
                response=response_text,
                sources=sources,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error executing FLARE query: {e}")
            return QueryResult(
                query=query_text,
                response=f"Error occurred while processing query: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def shutdown(self) -> None:
        """
        Clean up resources when the engine is no longer needed.
        """
        self.flare_engine = None
        self.underlying_engine = None
        self.logger.info("FLARE engine resources released")