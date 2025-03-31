# Prompt Verification Utility

## Problem Context

When working with prompts in the vector store, it's crucial to verify that:
1. Prompts are properly generated during document processing
2. Prompts are correctly saved in the document metadata
3. Prompts persist across vector store operations

This utility provides a way to:
- Check if prompts are present in the metadata
- Generate statistics about prompt coverage
- Display sample prompts for review

## Implementation Details

The utility examines the document metadata JSON file directly to avoid any potential issues with vector store loading logic. It provides:

### Diagnostic Information
- File-level checks to ensure the metadata file exists and is valid
- Document-level analysis of metadata structure
- Counts of documents with different prompt types
- Statistics on prompt lengths

### Visualization
- Rich table display of prompt statistics
- Sample prompts for quick review
- Detailed logging of document structure (in verbose mode)

### Error Handling
- Graceful handling of missing or malformed metadata files
- JSON parsing error diagnostics
- File system access error handling

## Usage Examples

Basic check:
```bash
python tools/verify_prompts.py -c your_collection_name
```

Detailed diagnostics:
```bash
python tools/verify_prompts.py -c your_collection_name -v
```

Custom vector store location:
```bash
python tools/verify_prompts.py -c your_collection_name -d /path/to/vector_store
```

This utility complements the `add_documents` method in `SimpleVectorStore`, helping ensure that prompts are successfully persisted and available for use.
