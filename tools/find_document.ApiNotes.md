# Document Finder Utility

## Purpose

This utility helps locate and examine documents within the vector store, which is especially useful for:

1. **Debugging Prompt Generation**: Finding documents where prompts were successfully or unsuccessfully generated
2. **Examining Document Structure**: Inspecting the metadata and content of specific documents
3. **Verifying Data Processing**: Checking how raw data was processed into vector store documents

## Implementation Details

The finder supports multiple search methods:

### ID-based Lookup
- Exact match by document ID
- Partial match when exact ID is not found
- Handles special characters in IDs (like paths that were sanitized)

### Text-based Search
- Searches document content, metadata, and IDs
- Prioritizes ID matches, then content, then metadata
- Case-insensitive matching for better results

### Collection Management
- Lists all available collections with document counts
- Auto-selects the only available collection if not specified
- Verifies collection existence before searching

## Display Features

Rich, formatted display of document details:
- Basic information (ID, timestamp)
- Full metadata structure
- Generated prompts (both single and multi-LLM)
- Text content with syntax highlighting

## Usage Examples

Find document by ID:
```bash
python tools/find_document.py -c collection_name -i 102.json.jsonl_1_1
```

Search for documents containing text:
```bash
python tools/find_document.py -c collection_name -s "search term"
```

List all collections:
```bash
python tools/find_document.py -l
```

The tool is particularly useful for examining the log message you mentioned:
