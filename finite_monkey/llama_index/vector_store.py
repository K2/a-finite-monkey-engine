"""
Asynchronous LanceDB adapter and vector store manager for LlamaIndex

This module provides an async adapter for LanceDB to be used with LlamaIndex
and a vector store manager for handling collections and embeddings.
"""

import os
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, cast, Union

import numpy as np

from llama_index.core.schema import BaseNode, MetadataMode, TextNode, Document
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from finite_monkey.nodes_config import nodes_config

logger = logging.getLogger(__name__)


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
            
            # Safely configure additional parameters if available
            try:
                # Some versions of LanceDB support nprobes for better search
                # Check both that the attribute exists AND it's not a LanceEmptyQueryBuilder
                if hasattr(lance_query, 'nprobes') and not type(lance_query).__name__ == 'LanceEmptyQueryBuilder':
                    lance_query = lance_query.nprobes(10)
            except Exception as e:
                print(f"Warning: Could not set additional search parameters: {e}")
            
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


class VectorStoreManager:
    """
    Manages vector store operations for document embeddings
    
    This class provides methods for creating and managing vector stores,
    adding documents, and performing semantic search.
    """
    
    def __init__(self, vector_store_path: Optional[str] = None):
        """
        Initialize the VectorStoreManager
        
        Args:
            vector_store_path: Path to the vector store directory
        """
        # Load config
        self.config = nodes_config()
        
        # Set vector store path
        self.vector_store_path = vector_store_path or os.path.join(
            os.getcwd(), "lancedb"
        )
        
        # Ensure vector store directory exists
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.config.EMBEDDING_MODEL_NAME or "BAAI/bge-small-en-v1.5"
        )
        
        # Initialize LanceDB vector store
        self._vector_store = None
        self._indices = {}
    
    async def get_or_create_collection(self, collection_name: str) -> Any:
        """
        Get or create a LanceDB collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            LanceDB collection
        """
        try:
            import lancedb
            
            # Connect to LanceDB
            db = await asyncio.to_thread(
                lancedb.connect, self.vector_store_path
            )
            
            # Get table names
            table_names = await asyncio.to_thread(db.table_names)
            
            # Get or create collection
            if collection_name in table_names:
                collection = await asyncio.to_thread(db.open_table, collection_name)
            else:
                # Create a new collection with schema
                embed_dim = len(self.embedding_model.get_text_embedding("test"))
                
                import pyarrow as pa
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), embed_dim)),
                    pa.field("metadata", pa.string()),
                ])
                
                collection = await asyncio.to_thread(
                    db.create_table,
                    collection_name,
                    schema=schema,
                    mode="create"
                )
            
            return collection
            
        except Exception as e:
            logger.error(f"Error getting or creating collection {collection_name}: {e}")
            raise
    
    async def add_documents(self, collection: Any, documents: List[Document]) -> bool:
        """
        Add documents to a vector store collection
        
        Args:
            collection: LanceDB collection
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse documents into nodes
            parser = SentenceSplitter(
                chunk_size=self.config.CHUNK_SIZE or 1024,
                chunk_overlap=self.config.CHUNK_OVERLAP or 200,
            )
            
            # Process each document
            for document in documents:
                # Parse document into nodes
                nodes = parser.get_nodes_from_documents([document])
                
                # Add document data
                records = []
                for node in nodes:
                    # Embed node text
                    embedding = self.embedding_model.get_text_embedding(node.text)
                    
                    # Create data entry
                    metadata = {
                        **node.metadata,
                        "node_id": node.id_,
                        "document_id": document.id_,
                    }
                    
                    records.append({
                        "id": node.id_,
                        "text": node.text,
                        "vector": embedding,
                        "metadata": json.dumps(metadata),
                    })
                
                # Add data to collection
                if records:
                    await asyncio.to_thread(collection.add, records)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to collection: {e}")
            return False
    
    async def search(
        self,
        collection: Any,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in a collection
        
        Args:
            collection: LanceDB collection
            query: Query string
            limit: Maximum number of results
            filters: Metadata filters
            
        Returns:
            List of search results
        """
        try:
            # Embed query
            query_embedding = self.embedding_model.get_text_embedding(query)
            
            # Build search query
            search_query = collection.search(query_embedding)
            
            # Add filters if provided
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(
                            f'json_extract(metadata, "$.{key}") = "{value}"'
                        )
                    else:
                        filter_conditions.append(
                            f'json_extract(metadata, "$.{key}") = {value}'
                        )
                
                if filter_conditions:
                    filter_str = " AND ".join(filter_conditions)
                    search_query = search_query.where(filter_str)
            
            # Limit results
            search_query = search_query.limit(limit)
            
            # Execute search
            results = await asyncio.to_thread(search_query.to_df)
            
            # Format results
            formatted_results = []
            for _, row in results.iterrows():
                try:
                    # Extract metadata
                    metadata = json.loads(row["metadata"]) if "metadata" in row and row["metadata"] else {}
                    
                    result = {
                        "id": row["id"] if "id" in row else "unknown",
                        "text": row["text"] if "text" in row else "",
                        "metadata": metadata,
                    }
                    
                    # Handle score which might be missing in some versions of LanceDB
                    if "score" in row:
                        result["score"] = row["score"]
                    else:
                        result["score"] = 0.0  # Provide default
                        
                    formatted_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing result row: {e}")
                    continue
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching collection: {e}")
            return []