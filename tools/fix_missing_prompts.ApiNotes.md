# Fix Missing Prompts Utility

## Problem Context

When working with the vector store, prompts may not be properly saved with documents due to:

1. **Reference Issues**: Using references instead of copies for document metadata
2. **Serialization Problems**: Non-serializable objects in prompts
3. **Dictionary Subclass Issues**: Using custom dict subclasses that don't serialize properly
4. **Metadata Propagation Failures**: Failing to update all locations where metadata is stored

This utility addresses these issues by:
1. Detecting documents with missing prompts
2. Regenerating prompts using the proper prompt generator
3. Saving the updated metadata with proper serialization

## Implementation Details

### Document Loading and Analysis
- Loads document metadata JSON directly, bypassing potential vector store loading issues
- Analyzes each document to identify those missing prompts
- Provides statistics on prompt coverage

### Prompt Generation
- Initializes a SimpleVectorStore with the correct settings
- Ensures prompt_generator is available and properly configured
- Regenerates prompts for all documents missing them
- Uses proper progress tracking for long-running operations

### Safe Document Updating
- Creates backups of metadata files before modifying them
- Deep copies documents to avoid reference issues
- Properly propagates generated prompts to all required locations

## Usage Examples

```bash
# Check without fixing (dry run)
python tools/fix_missing_prompts.py -c your_collection --dry-run

# Fix missing prompts
python tools/fix_missing_prompts.py -c your_collection

# Fix with custom vector store location
python tools/fix_missing_prompts.py -c your_collection -d /path/to/vector_store
```

This tool complements the `verify_prompts.py` utility, offering both detection and remediation of missing prompts.
