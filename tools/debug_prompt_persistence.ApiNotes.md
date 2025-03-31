# Prompt Persistence Debugging

## Issue Overview

The generated prompts may not be properly persisted in the vector store. This can happen due to:

1. **Referencing Issues**: Only some objects being updated with prompts
2. **Serialization Problems**: Non-serializable objects in prompts
3. **Metadata Synchronization**: Inconsistencies between documents and nodes
4. **Missing Verification**: No validation that prompts are properly saved

## Implementation Details

### Diagnostic Tool

The `debug_prompt_persistence.py` script provides:

1. **Comprehensive Analysis**: Examines all documents in a collection for prompts
2. **Multiple Collection Support**: Can scan all collections or a specific one
3. **Rich Statistics**: Shows counts, percentages, and prompt length analysis
4. **Sample Display**: Shows examples of saved prompts for quick inspection

### Save Method Enhancement

The enhanced `_save_document_metadata` method includes:

1. **Pre-Save Diagnostics**: Counts prompts and multi-prompts before saving
2. **Warning Logic**: Alerts when documents have no prompts despite generation being enabled
3. **Document Inspection**: Shows metadata keys for diagnostic purposes
4. **Custom Serialization**: Uses a safe serializer for any non-standard objects
5. **Post-Save Verification**: Loads the saved file to confirm prompts are persisted

### Add Documents Update

The improved prompt handling in `add_documents` ensures:

1. **Consistent Updates**: All objects (doc, node, doc_entry) get the same prompt data
2. **Sanitized Data**: Custom objects are converted to strings for reliable serialization
3. **Empty Prompt Detection**: Warns when a generated prompt is empty

## Usage

Run the diagnostic tool to check if prompts are being saved:

```bash
# Check a specific collection
python tools/debug_prompt_persistence.py -c your_collection_name

# Scan all collections
python tools/debug_prompt_persistence.py --scan-all
```

This will help determine exactly where any persistence issues might be occurring.
