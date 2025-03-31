# Document ID Path Sanitization

## Problem Context

Document IDs may contain embedded file paths (with '/' characters) due to how data is extracted from JSONL files, causing issues:

1. **File Path Issues**: Paths used directly in IDs can cause:
   - Path traversal security concerns
   - Serialization issues in some contexts
   - Problems when used in URLs or as directory names

2. **Consistency Issues**: Inconsistent ID formats make querying and management difficult

3. **Prompt Persistence Failures**: Embedded slashes may affect metadata storage

## Implementation Details

This utility provides both detection and remediation of document ID issues:

### Detection Logic

- Scans all document IDs for problematic characters ('/' and '\')
- Identifies IDs that contain full file paths rather than basenames
- Reports all issues in a clear tabular format

### Remediation Approach

- For IDs with slashes: Replaces them with underscores
- For IDs containing full file paths: Extracts the basename and maintains any suffixes
- Creates backups before making changes

### ID Generation Improvement

The accompanying changes to `vector_store_util.py` improve ID generation:

- `file_path` metadata is processed with `os.path.basename()` to extract just the filename
- Existing IDs are sanitized to replace slashes with underscores
- Added logging to track ID creation for debugging

## Usage Examples

```bash
# Check for document ID issues without fixing
python tools/fix_document_ids.py -c your_collection --dry-run

# Fix document ID issues
python tools/fix_document_ids.py -c your_collection
```

This tool can be run periodically to ensure document IDs remain clean and consistent.
