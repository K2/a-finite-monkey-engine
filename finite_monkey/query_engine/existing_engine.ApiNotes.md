# Existing Query Engine Adapter
Implementation of a compatibility layer for existing query engines.
## Overview
from typing import Dict, List, Any, Optional, Union
The `ExistingQueryEngine` serves as a compatibility adapter that wraps LlamaIndex query engines to make them compatible with the Finite Monkey Engine's `BaseQueryEngine` interface. It provides a standardized way to integrate with existing retrieval-based systems.
from loguru import logger
## Key Components
from llama_index.core.indices import VectorStoreIndex
### 1. Adapter Patternre.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
The class implements the adapter pattern to bridge between:verQueryEngine
- The `BaseQueryEngine` interface used throughout the Finite Monkey system
- The existing LlamaIndex retrieval components
- The standardized `QueryResult` output format

### 2. Initialization Processclass ExistingQueryEngine(BaseQueryEngine):

The engine follows a lazy initialization pattern:    Adapter for existing query engines in the system.
- Created with minimal parameters initially
- Fully initialized when first query is madegines to make them compatible
- Supports vector index configuration for retrieval

### 3. Query Process
    def __init__(
When executing a query, the adapter:
1. Ensures initialization is complete        vector_index: Optional[VectorStoreIndex] = None,
2. Delegates to the underlying LlamaIndex retriever engine.7,
3. Extracts source information and calculates confidence
4. Formats the response with consistent metadata

### 4. Vector Index Integration

The adapter works primarily with vector indices:        
- Uses `VectorIndexRetriever` for semantic retrieval
- Applies `SimilarityPostprocessor` for filtering irrelevant results            vector_index: Optional vector index for retrieval
- Configurable similarity thresholds and result countilarity filtering
ults to return
## Integration Points

### With Pipeline Factory        super().__init__(config)
ndex = vector_index
The engine is typically created and managed by the PipelineFactory, which:        self.similarity_cutoff = similarity_cutoff
1. Creates the adapter when neededop_k = similarity_top_k
2. Provides the vector index populated with smart contract data        self.retriever_engine = None
3. Uses it as the underlying engine for more advanced engines like FLARE

### With FLARE Query Engine
sary components.
The adapter serves as a foundation for the FLARE engine:        """
- FLARE uses this engine for initial retrievaler_engine is not None:
- This engine provides context documents that FLARE reasons over            return  # Already initialized
- The adapter handles basic queries when advanced reasoning isn't needed        self.retriever_engine = None
        self.logger.info("Existing query engine resources released")