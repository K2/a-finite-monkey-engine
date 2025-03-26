"""
Asynchronous index processor for LlamaIndex integration.

This module provides the main async interface for using LlamaIndex functionality
in the Finite Monkey framework.
"""
from .loaders import AsyncCodeLoader
import asyncio
import fnmatch
import os
from typing import Any, Dict, List, Optional

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, VectorStoreQuery, VectorStoreQueryResult

from llama_index.vector_stores.lancedb import LanceDBVectorStore


class AsyncIndexProcessor:
    """
    Main async processor for creating and querying LlamaIndex indices.

    This class provides an asynchronous interface for the Finite Monkey framework's
    LlamaIndex integration.
    """

    def __init__(
        self,
        project_id: str,
        base_dir: Optional[str] = None,
        src_dir: Optional[str] = None,
        embed_dim: int = 384,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        embedding_model_name: str = "BAAI/bge-small-en",
        device_map: str = "auto",
    ):
        """
        Initialize the async index processor

        Args:
            project_id: Unique identifier for the project
            base_dir: Base directory for the project files
            src_dir: Source directory within the base directory
            embed_dim: Dimension of the embeddings
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            embedding_model_name: Name of the embedding model to use
        """
        # Store parameters
        self.project_id = project_id
        self.base_dir = base_dir or os.getcwd()
        self.src_dir = src_dir or ""
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.device_map = device_map

        # Default paths
        self.table_name = f"lance_{project_id}"
        self._setup_components()

        # Initialize components
    def _setup_components(self):
        """Set up the LlamaIndex components."""
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Create a HuggingFace embedding model with device mapping
        device = "cuda" if self.device_map == "cuda" else "cpu"
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            device=device  # Changed from device_map to device
        )
        
        # Configure LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        
        # Create vector store with explicit embedding function
        self.vector_store = LanceDBVectorStore(
            uri="./lancedb",
            table_name=self.table_name,
            embed_dim=self.embed_dim,
            embedding_function=self.embed_model.get_text_embedding,
        )
        
        # Initialize loader and parser
        self.code_loader = AsyncCodeLoader()
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    async def load_and_index(
        self,
        file_paths: Optional[List[str]] = None,
        dir_path: Optional[str] = None,
        recursive: bool = True,
        filters: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Asynchronously load and index files or directories.
        """
        # Load documents asynchronously
        documents = await self._load_documents_async(file_paths, dir_path, recursive, filters)

        # Parse nodes (run in thread to avoid blocking)
        nodes = await asyncio.to_thread(self.node_parser.get_nodes_from_documents, documents)

        # Pre-generate embeddings synchronously for each node
        for node in nodes:
            if not hasattr(node, 'embedding') or node.embedding is None:
                try:
                    # Generate embedding synchronously
                    embedding = self.embed_model.get_text_embedding(node.get_content())
                    setattr(node, 'embedding', embedding)
                except Exception as e:
                    print(f"Error generating embedding for node: {e}")
                    continue

        # Add nodes to vector store using thread to avoid blocking
        try:
            node_ids = await asyncio.to_thread(self.vector_store.add, nodes)
        except Exception as e:
            print(f"Error adding nodes to vector store: {e}")
            raise

        return node_ids

    async def _load_documents_async(
        self,
        file_paths: Optional[List[str]] = None,
        dir_path: Optional[str] = None,
        recursive: bool = True,
        filters: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Asynchronously load documents from files or directories.

        Args:
            file_paths: List of file paths to load
            dir_path: Directory path to load
            recursive: Whether to recursively search in subdirectories
            filters: File patterns to include (e.g., ["*.sol", "*.py"])

        Returns:
            List of loaded documents
        """
        # Default directory path is base_dir/src_dir
        if not file_paths and not dir_path:
            dir_path = os.path.join(self.base_dir, self.src_dir)

        # Initialize tasks list
        tasks = []

        # Load files
        if file_paths:
            for file_path in file_paths:
                tasks.append(self.code_loader.load_data(file_path=file_path))
        elif dir_path:
            tasks.append(self.code_loader.load_data(dir_path=dir_path, recursive=recursive))

        # Process all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_docs = [doc for result in results for doc in result]

        # Apply filters if specified
        if not filters or not all_docs:
            return all_docs
        filtered_docs = []
        for doc in all_docs:
            file_path = doc.metadata.get('file_path', "")
            if any((fnmatch.fnmatch(file_path, pattern) for pattern in filters)):
                filtered_docs.append(doc)
        return filtered_docs

    async def aquery(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Asynchronously query the index for relevant results.

        Args:
            query_text: Query text
            filters: Optional metadata filters
            top_k: Number of top results to return

        Returns:
            Dict containing query results
        """
        # Create metadata filters if specified
        metadata_filters = None
        if filters:
            filter_conditions = [
                MetadataFilter(key=key, value=value) for key, value in filters.items()
            ]

            metadata_filters = MetadataFilters(filters=filter_conditions, condition="and")

        # Create vector store query
        query = VectorStoreQuery(
            query_str=query_text,
            similarity_top_k=top_k,
            filters=metadata_filters,
        )

        # Execute query asynchronously
        try:
            query_result = await self.vector_store.query(query)
        except Exception as e:
            print(f"Error in vector store query: {e}")
            # Create a fallback empty result
            from llama_index.core.schema import NodeWithScore
            query_result = VectorStoreQueryResult(
                nodes=[],
                similarities=[],
                ids=[]
            )

        # Format results
        result = {
            "query": query_text,
            "nodes": [],
        }

        # Add source nodes
        for i, node in enumerate(query_result.nodes):
            node_info = {
                "id": node.id_,
                "text": node.get_content(),
                "score": query_result.similarities[i] if query_result.similarities else None,
                "metadata": node.metadata,
            }
            result["nodes"].append(node_info)

        return result

    async def similarity_search(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously perform a similarity search.

        Args:
            query_text: Query text
            filters: Optional metadata filters
            top_k: Number of top results to return

        Returns:
            List of search results
        """
        # Create metadata filters if specified
        metadata_filters = None
        if filters:
            filter_conditions = [
                MetadataFilter(key=key, value=value) for key, value in filters.items()
            ]

            metadata_filters = MetadataFilters(filters=filter_conditions, condition="and")

        # Create vector store query
        query = VectorStoreQuery(
            query_str=query_text,
            similarity_top_k=top_k,
            filters=metadata_filters,
        )

        # Execute query asynchronously
        try:
            query_result = await self.vector_store.query(query)
        except Exception as e:
            print(f"Error in vector store similarity search: {e}")
            # Create a fallback empty result
            from llama_index.core.schema import NodeWithScore
            query_result = VectorStoreQueryResult(
                nodes=[],
                similarities=[],
                ids=[]
            )

        # Format results
        results = []
        for i, node in enumerate(query_result.nodes):
            result = {
                "id": node.id_,
                "text": node.get_content(),
                "score": query_result.similarities[i] if query_result.similarities else None,
                "metadata": node.metadata,
            }
            results.append(result)

        return results

    async def get_related_functions(
        self,
        function_code: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously get related functions for a given function code.

        Args:
            function_code: Code of the function to find related functions for
            top_k: Number of top results to return

        Returns:
            List of related functions
        """
        try:
            # Filter to only include functions
            filters = {"node_type": "function"}

            # Perform similarity search asynchronously
            return await self.similarity_search(
                query_text=function_code,
                filters=filters,
                top_k=top_k,
            )
        except Exception as e:
            print(f"Error finding related functions: {e}")
            return []