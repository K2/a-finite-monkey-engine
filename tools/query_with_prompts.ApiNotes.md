# Semantic Search with Context-Preserving Prompts

## Design Concept

This implementation addresses the need to leverage prompts associated with semantically similar documents to provide better context for queries. The approach follows these key principles:

1. **Two-Stage Retrieval**: 
   - First, find the top-k semantically similar documents using vector search
   - Then, extract the prompts associated with those documents
   
2. **Prompt Aggregation**: 
   - Collect prompts from the most relevant documents
   - Rank them according to the semantic similarity score
   - Combine them to create a comprehensive context prompt

3. **Multi-Retrieval Source**:
   - Prioritize prompts from the separate prompts file (`document_prompts.json`)
   - Fall back to prompts embedded in document metadata if necessary
   - Support both single prompts and multi-LLM prompts

## Implementation Details

### Vector Store Query Extension

The `query_with_prompts` method extends `SimpleVectorStore` with:

1. **Flexible Top-K**: Retrieves an adjustable number of semantically similar documents
2. **Prompt Extraction**: Collects prompts from the external prompts file or embedded metadata
3. **Result Ranking**: Orders prompts based on the semantic similarity of their documents
4. **Combined Context**: Creates a unified prompt from all relevant individual prompts

### Command-Line Utility

The accompanying utility provides:

1. **Interactive Testing**: Command-line interface to test the approach
2. **Rich Visualization**: Formatted display of results and prompts
3. **Export Capability**: Option to save results to a file for further analysis

## Usage Flow

1. User queries the vector store with a text query
2. System finds the most semantically similar documents
3. System extracts prompts associated with those documents
4. Prompts are ranked and combined into a unified context
5. This context can be used to generate more informed responses to the query

This approach creates a powerful knowledge retrieval system that not only finds relevant documents but also leverages the pre-generated prompts to provide richer context for understanding and processing queries.
