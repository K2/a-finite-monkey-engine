"""
Asynchronous LanceDB adapter for LlamaIndex

This module provides an async adapter for LanceDB to be used with LlamaIndex.
"""

import os
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.vector_stores.simple import SimpleVectorStore


class AsyncLanceDBAdapter:
    """
    Async adapter for LanceDB
    
    This class provides an asynchronous interface to interact with LanceDB.
    Since LanceDB doesn't have native async support, we use asyncio.to_thread
    to avoid blocking the event loop.
    """

    def __init__(
        self,
        uri: str = "./lancedb",
        table_name: str = "vectors",
        embed_dim: int = 384,
        create_table_if_not_exists: bool = True,
    ):
        """
        Initialize the LanceDB adapter
        
        Args:
            uri: URI for the LanceDB database
            table_name: Name of the table to use
            embed_dim: Embedding dimension
            create_table_if_not_exists: Whether to create the table if it doesn't exist
        """
        self.uri = uri
        self.table_name = table_name
        self.embed_dim = embed_dim
        self.create_table_if_not_exists = create_table_if_not_exists
        
        # Use SimpleVectorStore as an in-memory fallback
        self._fallback_store = SimpleVectorStore()
        
        # Flag to indicate whether to use the fallback store
        self._use_fallback = True  # Start with fallback enabled
    
    async def _init_lance_db(self):
        """Initialize LanceDB connection"""
        try:
            # Dynamically import lancedb to avoid hard dependency
            import lancedb
            
            # Create the database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.uri), exist_ok=True)
            
            # Connect to the database
            self._db = await asyncio.to_thread(lancedb.connect, self.uri)
            
            # Create or open the table
            if self.create_table_if_not_exists:
                # Define the schema
                import pyarrow as pa
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.embed_dim)),
                    pa.field("metadata", pa.string()),
                ])
                
                # Create the table if it doesn't exist
                tables = await asyncio.to_thread(self._db.table_names)
                if self.table_name not in tables:
                    # Create empty table with the schema
                    self._table = await asyncio.to_thread(
                        self._db.create_table,
                        self.table_name,
                        schema=schema,
                        mode="create"
                    )
                else:
                    # Open existing table
                    self._table = await asyncio.to_thread(self._db.open_table, self.table_name)
            else:
                # Just open the table
                self._table = await asyncio.to_thread(self._db.open_table, self.table_name)
            
            # Disable fallback
            self._use_fallback = False
            
        except (ImportError, Exception) as e:
            # Log the error
            print(f"Error initializing LanceDB: {str(e)}")
            print("Falling back to in-memory vector store")
            
            # Use fallback
            self._use_fallback = True
    
    async def add(self, nodes: List[BaseNode]) -> List[str]:
        """
        Add nodes to the vector store
        
        Args:
            nodes: List of nodes to add
            
        Returns:
            List of node IDs
        """
        if self._use_fallback:
            # Initialize LanceDB
            await self._init_lance_db()
        
        if self._use_fallback:
            # Use fallback store
            return self._fallback_store.add(nodes)
        
        try:
            # Prepare data for LanceDB
            records = []
            
            for node in nodes:
                # Extract node data
                node_id = node.id_
                embedding = node.embedding
                content = node.get_content(metadata_mode=MetadataMode.NONE)
                metadata = node.metadata
                
                # Convert metadata to JSON string
                metadata_str = json.dumps(metadata)
                
                # Add record
                records.append({
                    "id": node_id,
                    "text": content,
                    "vector": embedding,
                    "metadata": metadata_str,
                })
            
            # Add records to LanceDB
            if records:
                await asyncio.to_thread(self._table.add, records)
            
            # Return node IDs
            return [node.id_ for node in nodes]
            
        except Exception as e:
            # Log the error
            print(f"Error adding nodes to LanceDB: {str(e)}")
            
            # Fall back to in-memory store
            self._use_fallback = True
            return self._fallback_store.add(nodes)
    
    async def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """
        Query the vector store
        
        Args:
            query: Query parameters
            
        Returns:
            Query result
        """
        if self._use_fallback:
            # Initialize LanceDB
            await self._init_lance_db()
        
        if self._use_fallback:
            # Use fallback store
            return self._fallback_store.query(query)
        
        try:
            # Extract query parameters
            embedding = query.query_embedding
            similarity_top_k = query.similarity_top_k
            
            # Build LanceDB query
            lance_query = self._table.search(embedding)
            
            # Apply filters if specified
            if query.filters:
                # Convert metadata filters to LanceDB filters
                filter_str = self._convert_filters_to_lance(query.filters)
                if filter_str:
                    lance_query = lance_query.where(filter_str)
            
            # Limit results
            lance_query = lance_query.limit(similarity_top_k)
            
            # Execute query
            results = await asyncio.to_thread(lance_query.to_df)
            
            # Convert results to nodes
            nodes = []
            similarities = []
            
            for _, row in results.iterrows():
                # Extract data
                node_id = row["id"]
                content = row["text"]
                metadata = json.loads(row["metadata"])
                score = row["score"]
                
                # Create node
                node = TextNode(
                    id_=node_id,
                    text=content,
                    metadata=metadata,
                    embedding=row["vector"] if "vector" in row else None,
                )
                
                # Add to results
                nodes.append(node)
                similarities.append(score)
            
            # Return query result
            return VectorStoreQueryResult(
                nodes=nodes,
                similarities=similarities,
                ids=[node.id_ for node in nodes],
            )
            
        except Exception as e:
            # Log the error
            print(f"Error querying LanceDB: {str(e)}")
            
            # Fall back to in-memory store
            self._use_fallback = True
            return self._fallback_store.query(query)
    
    def _convert_filters_to_lance(self, filters):
        """
        Convert LlamaIndex filters to LanceDB filter strings
        
        This is a simplified implementation that doesn't handle all filter types.
        For a complete implementation, more logic would be needed.
        """
        if not filters:
            return None
        
        # Handle metadata filters
        if hasattr(filters, "filters"):
            conditions = []
            for filter_item in filters.filters:
                # Extract key and value
                key = filter_item.key
                value = filter_item.value
                
                # Build condition
                if isinstance(value, str):
                    conditions.append(f'json_extract(metadata, "$.{key}") = "{value}"')
                else:
                    conditions.append(f'json_extract(metadata, "$.{key}") = {value}')
            
            # Combine conditions
            if filters.condition.lower() == "and":
                return " AND ".join(conditions)
            else:
                return " OR ".join(conditions)
        
        return None