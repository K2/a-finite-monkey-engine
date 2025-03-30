# Vector Store Builder API Notes

## Purpose

The `build_vector_store.py` utility is designed to extract data from various datasets and build them into vector stores for similarity search. This tool is essential for:

1. **Populating Knowledge Bases**: Create searchable vector databases from existing datasets
2. **Building Vulnerability Databases**: Import known vulnerabilities for pattern matching
3. **Indexing Code Repositories**: Create searchable indices of Solidity code

## Data Source Handlers

The utility supports multiple data sources through specialized processors:

1. **JSON Datasets** (`process_json_dataset`):
   - Handles nested JSON structures
   - Extracts text and metadata
   - Works with both arrays and objects

2. **CSV Datasets** (`process_csv_dataset`):
   - Identifies common column patterns (title, description, etc.)
   - Creates structured documents from rows
   - Preserves all fields as metadata

3. **Solidity Files** (`process_solidity_files`):
   - Processes directories of .sol files
   - Extracts contract names, imports, and functions
   - Creates searchable code documents

4. **Vulnerability Databases** (`process_vulnerability_database`):
   - Imports known vulnerability patterns
   - Extracts severity, description, and code examples
   - Creates specialized vulnerability documents

5. **JSONL Datasets** (`process_jsonl_dataset`):
   - Each line in a JSONL file contains a single, valid JSON object
   - Processed line-by-line for memory efficiency with large datasets
   - Compatible with the same schema as dataset_text_analysis2.py
   - Supports the same options as JSON processing

## Integration with VectorStore

The tool uses the `VectorStore` class from `finite_monkey.llama_index.vector_store` to:

1. Initialize collections with appropriate names
2. Add processed documents with text and metadata
3. Store embeddings for similarity search

## Usage Examples

```bash
# Build vector store from JSON dataset
python tools/build_vector_store.py -t json -i data/issues.json -c github_issues

# Build vector store from Solidity files
python tools/build_vector_store.py -t solidity -i examples/src -c solidity_code

# Build vulnerability database
python tools/build_vector_store.py -t vulnerabilities -i data/vuln_db.json -c vulnerabilities

# Build from CSV with custom output location
python tools/build_vector_store.py -t csv -i data/audit_findings.csv -c audit_findings -o ./custom_vector_store

# Process a JSONL dataset
python tools/build_vector_store.py -t jsonl -i data/large_dataset.jsonl -c documents

# With custom field mapping
python tools/build_vector_store.py -t jsonl -i data/dataset.jsonl -c custom_collection \
  --id-field unique_id --text-fields body,description --title-fields heading,title
```

## Extension Points

The utility is designed for easy extension:

1. **New Data Sources**: Add new processor functions for different data types
2. **Custom Extraction Logic**: Modify the extraction functions for specialized document creation
3. **Vector Store Configuration**: Customize embedding dimensions or similarity metrics

## JSONL Support

JSONL (JSON Lines) format is now supported for efficient processing of large datasets:

- Each line in a JSONL file contains a single, valid JSON object
- Processed line-by-line for memory efficiency with large datasets
- Compatible with the same schema as dataset_text_analysis2.py
- Supports the same options as JSON processing

### Implementation Details

The `process_jsonl_dataset` function processes JSONL files with these key features:

1. **Streaming Processing**: Reads and processes one line at a time
2. **Progress Reporting**: Logs progress every 1000 lines
3. **Error Tolerance**: Skips invalid JSON lines with warnings
4. **Schema Compatibility**: Uses the same extraction logic as JSON format

## Ground Truth Support

The vector store builder now supports designating entries as "ground truth" data with quality metrics:

1. **Default Quality Markers**: All documents can be marked with `veryGood: true` and a quality score
2. **Quality Filtering**: Documents can be filtered based on a minimum quality threshold
3. **Flexible Configuration**: Quality field names and thresholds can be customized

### Implementation Details

Ground truth support is implemented in several ways:

1. **Metadata Fields**:
   - `veryGood`: Boolean flag for high-quality entries
   - `quality`: Numeric score (higher is better)

2. **Command Line Options**:
   ```bash
   # Mark all entries as ground truth
   --ground-truth
   
   # Only include entries above quality threshold
   --quality-threshold 7
   
   # Specify custom quality field
   --quality-field rating
   ```

3. **Quality Inheritance**:
   - If source data already contains quality metrics, they are preserved
   - Default values (`veryGood=True`, `quality=10`) can be overridden

## Usage Example

To build a vector store with only high-quality entries:

```bash
python tools/build_vector_store.py -t jsonl -i high_quality_data.jsonl -c ground_truth \
  --ground-truth --quality-threshold 7
```

This will create a vector store where all entries have `veryGood=True` and only include documents with a quality score of 7 or higher.

## Integration with dataset_text_analysis2.py

The tool is designed to be compatible with datasets processed by dataset_text_analysis2.py:

1. Uses the same field naming conventions
2. Supports the same clustering options
3. Handles the same input formats (with JSONL as an addition)
4. Preserves metadata fields in the same structure

This allows for seamless workflows where data can be:
1. Analyzed with dataset_text_analysis2.py
2. Imported into vector stores with build_vector_store.py
3. Queried through the LlamaIndex interface

## High-Quality Entry Extraction

The vector store builder now specifically targets high-quality entries and generates LLM guidance prompts:

1. **Quality Filtering**: Only extracts entries with `veryGood=True` and `quality>0`
2. **Guidance Prompt Generation**: Creates specialized prompts to guide LLMs at runtime
3. **Metadata Enhancement**: Stores quality metrics and guidance in metadata

### Guidance Prompt Architecture

Guidance prompts are designed to help an LLM make consistent determinations when a vector similarity match is found:

1. **Pattern Recognition**: Prompts identify the pattern the entry represents
2. **Analytical Framework**: Explains the correct approach to analyzing similar content
3. **Category-Specific Guidance**: Provides tailored advice based on entry category
4. **Ground Truth Reference**: Establishes the entry as a validated reference point

Example guidance prompt structure:

## JSON Schema Handling

The extraction process now handles the specific schema where:

```json
{
  "text": "...",
  "id": "...",
  "metadata": {
    "comments": [{"body": "..."}],
    "labels": [{"description": "..."}],
    "title": "...",
    "url": "...",
    "quality": ...,
    "veryGood": ...
  }
}
```

### Key Differences

1. **Nested Structure**: Quality indicators (`veryGood`, `quality`) are accessed under `metadata`
2. **Multi-field Content**: Content is primarily from `text` but falls back to `comments[].body` and `labels[].description`
3. **Rich Guidance**: Prompts incorporate information from comments and labels to provide context
4. **URL References**: Source URLs are preserved for attribution

### Extraction Logic

The extraction follows this priority sequence:

1. **Text Content**:
   - Primary: `item.text`
   - Fallback 1: `metadata.comments[].body` (combined)
   - Fallback 2: `metadata.labels[].description` (combined)

2. **Title Selection**:
   - Primary: `metadata.title`
   - Fallback 1: First line of first comment
   - Fallback 2: First label description
   - Fallback 3: Item ID

3. **Quality Filtering**:
   - Both `metadata.veryGood` must be true
   - `metadata.quality` must be greater than 0

### Guidance Prompt Enhancement

Guidance prompts are now enriched with:
- Label descriptions for contextual categories
- Comment excerpts to provide analytical insights
- Source URL references for attribution
- Structured analytical framework guidance
