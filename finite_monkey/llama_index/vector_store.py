"""
Vector store implementation for semantic search of code and vulnerabilities.
Provides an interface to store, query, and manage vector embeddings.
"""
import os
import json
from typing import List, Dict, Any, Optional, Union
import asyncio
from loguru import logger
from datetime import datetime

class VectorStore:
    """Vector store for code and vulnerability embeddings."""
    
    def __init__(
        self, 
        storage_dir: str = "./vector_store",
        collection_name: str = "default",
        embedding_dim: int = 1536,  # Default for OpenAI embeddings
        distance_metric: str = "cosine"
    ):
        """
        Initialize the vector store.
        
        Args:
            storage_dir: Directory to store vector indices
            collection_name: Name of the collection
            embedding_dim: Dimension of the embedding vectors
            distance_metric: Distance metric for similarity search
        """
        self.storage_dir = storage_dir
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self._index = None
        self._documents = []
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize the vector index
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize the vector index from storage or create a new one."""
        try:
            # Try to import LlamaIndex
            from llama_index.core import (
                VectorStoreIndex, 
                StorageContext,
                load_index_from_storage
            )
            from llama_index.core.schema import Document, TextNode
            
            # Check if index exists
            index_dir = os.path.join(self.storage_dir, self.collection_name)
            if os.path.exists(index_dir):
                logger.info(f"Loading existing vector index from {index_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                self._index = load_index_from_storage(storage_context)
                
                # Load document metadata
                metadata_path = os.path.join(index_dir, "document_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self._documents = json.load(f)
            else:
                logger.info(f"Creating new vector index at {index_dir}")
                # Create empty index
                self._index = VectorStoreIndex([])
                # Save the empty index
                self._index.storage_context.persist(persist_dir=index_dir)
                
                # Initialize empty document metadata
                self._documents = []
                self._save_document_metadata(index_dir)
                
            logger.info(f"Vector index initialized with {len(self._documents)} documents")
            
        except ImportError:
            logger.warning("LlamaIndex not available, using simplified vector store")
            self._index = None
        except Exception as e:
            logger.error(f"Error initializing vector index: {e}")
            self._index = None
    
    def _save_document_metadata(self, index_dir: str):
        """Save document metadata to disk."""
        metadata_path = os.path.join(index_dir, "document_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self._documents, f)
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            
        Returns:
            Success status
        """
        if not self._index:
            logger.warning("Vector store index not initialized")
            return False
            
        try:
            # Import LlamaIndex components
            from llama_index.core.schema import Document, TextNode
            from llama_index.core import Settings
            
            # Create document nodes
            nodes = []
            for doc in documents:
                # Create a unique ID
                doc_id = f"{doc.get('metadata', {}).get('contract_name', 'unknown')}_{len(self._documents)}"
                
                # Create a node
                node = TextNode(
                    text=doc['text'],
                    metadata=doc.get('metadata', {}),
                    id_=doc_id
                )
                nodes.append(node)
                
                # Add to document metadata store
                self._documents.append({
                    'id': doc_id,
                    'metadata': doc.get('metadata', {}),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Add nodes to index
            self._index.insert_nodes(nodes)
            
            # Save index and metadata
            index_dir = os.path.join(self.storage_dir, self.collection_name)
            self._index.storage_context.persist(persist_dir=index_dir)
            self._save_document_metadata(index_dir)
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    async def query(
        self, 
        text: str, 
        top_k: int = 5,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Args:
            text: Query text
            top_k: Number of results to return
            collection_name: Optional collection name to override default
            
        Returns:
            List of matching documents with similarity scores
        """
        if not self._index:
            logger.warning("Vector store index not initialized")
            return []
            
        try:
            # Use the provided collection or default
            coll_name = collection_name or self.collection_name
            
            # If collection differs from current, reload index
            if coll_name != self.collection_name:
                self.collection_name = coll_name
                self._initialize_index()
            
            # Create query engine
            query_engine = self._index.as_query_engine(similarity_top_k=top_k)
            
            # Execute query
            response = query_engine.query(text)
            
            # Extract nodes and scores
            results = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    results.append({
                        'text': node.text,
                        'metadata': node.metadata,
                        'score': node.score if hasattr(node, 'score') else 0.0
                    })
            
            logger.info(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return []