"""
Implementation of a compatibility layer for existing query engines.
"""
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine

from finite_monkey.pipeline.core import Context
from .base_engine import BaseQueryEngine, QueryResult

class ExistingQueryEngine(BaseQueryEngine):
    """
    Adapter for existing query engines in the system.
    
    This class wraps existing LlamaIndex query engines to make them compatible
    with our BaseQueryEngine interface.
    """
    
    def __init__(
        self, 
        vector_index: Optional[VectorStoreIndex] = None,
        similarity_cutoff: float = 0.7,
        similarity_top_k: int = 5,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the existing query engine adapter.
        
        Args:
            vector_index: Optional vector index for retrieval
            similarity_cutoff: Threshold for similarity filtering
            similarity_top_k: Number of top results to return
            config: Additional configuration options
        """
        super().__init__(config)
        self.vector_index = vector_index
        self.similarity_cutoff = similarity_cutoff
        self.similarity_top_k = similarity_top_k
        self.retriever_engine = None
    
    async def initialize(self) -> None:
        """
        Initialize the query engine with the necessary components.
        """
        if self.retriever_engine is not None:
            return  # Already initialized
        
        try:
            if self.vector_index is not None:
                # Create a retriever from the vector index
                retriever = VectorIndexRetriever(
                    index=self.vector_index,
                    similarity_top_k=self.similarity_top_k
                )
                
                # Add similarity score threshold
                similarity_postprocessor = SimilarityPostprocessor(
                    similarity_cutoff=self.similarity_cutoff
                )
                
                # Create the retriever query engine
                self.retriever_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    node_postprocessors=[similarity_postprocessor],
                )
                self.logger.info("Existing query engine initialized with vector index")
            else:
                self.logger.warning("No vector index provided for existing query engine")
        except Exception as e:
            self.logger.error(f"Error initializing existing query engine: {e}")
            raise
    
    async def query(self, query_text: str, context: Optional[Context] = None) -> QueryResult:
        """
        Execute a query using the existing query engine.
        
        Args:
            query_text: The query string
            context: Optional pipeline context with additional information
            
        Returns:
            QueryResult with response and metadata
        """
        await self.initialize()
        
        if self.retriever_engine is None:
            return QueryResult(
                query=query_text,
                response="Query engine not properly initialized",
                confidence=0.0,
                metadata={"error": "Engine initialization failed"}
            )
        
        try:
            import asyncio
            
            # Execute the query using the existing engine
            response = await asyncio.to_thread(
                self.retriever_engine.query,
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
            
            # Calculate confidence based on sources
            confidence = 0.0
            if sources:
                avg_score = sum(s.get('score', 0) or 0 for s in sources) / len(sources)
                confidence = min(avg_score, 1.0)
            
            return QueryResult(
                query=query_text,
                response=response_text,
                sources=sources,
                confidence=confidence,
                metadata={}
            )
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
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
        self.retriever_engine = None
        self.logger.info("Existing query engine resources released")