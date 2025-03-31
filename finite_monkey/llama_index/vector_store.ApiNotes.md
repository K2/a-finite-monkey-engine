# Vector Store API Notes

## Purpose and Architecture

The `VectorStore` provides semantic search capabilities for code fragments and vulnerability patterns:

1. **Storage**: Persists vector embeddings and document metadata on disk
2. **Querying**: Enables semantic search for similar code patterns
3. **Collections**: Supports multiple named collections (e.g., "vulnerabilities", "code_patterns")

This vector store is particularly valuable for vulnerability scanning as it enables:
- Finding similar vulnerability patterns across different contracts
- Learning from previously identified issues
- Improving detection by leveraging historical vulnerability data

## Integration with Vulnerability Scanner

The vulnerability scanner integrates with the vector store in three key ways:

1. **Initialization**: Sets up the vector store during pipeline initialization
   ```python
   await self._initialize_vector_store(context)
   ```

2. **Querying**: Finds similar vulnerability patterns before analysis
   ```python
   similar_vulns = await self._find_similar_vulnerabilities(content, contract_context)
   ```

3. **Updating**: Adds newly discovered vulnerabilities to improve future scans
   ```python
   await self._update_vector_store(context)
   ```

This bidirectional flow ensures the vulnerability scanner can both leverage existing knowledge and contribute new findings to the collective intelligence system.

## Implementation Notes

The `VectorStore` uses LlamaIndex as its underlying vector store implementation:

1. It loads or creates indices based on the collection name
2. It handles persistence of both vector embeddings and document metadata
3. It provides an async interface for adding and querying documents

If LlamaIndex is not available, it gracefully degrades to provide a minimal API surface that doesn't break callers.
