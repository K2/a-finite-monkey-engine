# Prompt Management System

## Architectural Design

This implementation separates prompts from the vector store to address several important concerns:

1. **Embedding Efficiency**: Prompts are not embedded in the vector database, which:
   - Reduces index size and memory usage
   - Improves search performance by focusing on the actual content
   - Allows independent updating of prompts without re-embedding

2. **Storage Separation**: Prompts are stored in a dedicated file (`document_prompts.json`), which:
   - Keeps the vector store focused on searchable content
   - Enables more efficient persistence and loading
   - Allows for easier backup, export, and analysis of prompts

3. **Flexible Retrieval**: The system maintains the connection between documents and their prompts via document IDs, which:
   - Preserves all functionality for using prompts during retrieval
   - Allows on-demand loading of prompts only when needed
   - Supports both in-memory and on-disk prompt management

## Implementation Details

### Vector Store Modifications

The `SimpleVectorStore` class was modified to:
1. Extract prompts from document metadata before saving
2. Store prompts in a separate JSON file
3. Load prompts asynchronously when needed
4. Provide methods to retrieve documents with their prompts attached

This creates a clean separation while maintaining all original functionality.

### Prompt Management Utility

The `manage_prompts.py` utility provides a complete prompt management system:

1. **Migration**: Convert from embedded prompts to separated storage
2. **Export/Import**: Move prompts between collections or environments
3. **Analysis**: Examine prompt statistics and content
4. **Validation**: Verify prompt integrity and coverage

## Migration Path

For existing vector stores with embedded prompts, the migration process:
1. Extracts prompts from document metadata
2. Stores them in a separate file
3. Removes prompts from the original metadata
4. Updates the vector store to use the new separated system

This allows for a smooth transition without losing any data.
