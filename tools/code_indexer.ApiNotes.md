# Code Indexing and Retrieval System

## Design Philosophy

The code indexing system follows a dual-representation approach that addresses the limitations of traditional vector retrieval for code:

1. **Multi-level Representation**: 
   - Individual blocks (functions, methods, chunks) stored separately to allow precise retrieval
   - Amalgamated blocks combining related code for contextual understanding
   
2. **Structural Understanding**:
   - Language-specific parsing extracts meaningful code units (functions, classes)
   - Preserves code structure rather than arbitrary text chunks
   
3. **Metadata-rich Indexing**:
   - Extensive metadata tracks relationships between blocks
   - Supports filtering by language, block type, file, etc.

## Implementation Details

### Code Parsing Strategy

The `CodeParser` class employs a hierarchical extraction approach:

1. **Function Extraction (Primary)**
   - Uses language-specific regex patterns to identify meaningful code units
   - Extracts complete functions/methods with their signature and body
   - Includes class definitions for object-oriented languages

2. **Chunk Extraction (Fallback)**
   - Used when function extraction yields insufficient results
   - Creates overlapping chunks that preserve line boundaries
   - Ensures no code is broken mid-line for better readability

### Block Relationships

Blocks are connected through various metadata fields:

- `group_id`: Links individual blocks with their amalgamated version
- `block_index`: Preserves original order within source files
- `file_id`: Connects blocks back to their source file
- `block_type`: Differentiates between individual and amalgamated blocks

### Search and Retrieval

The system provides two primary search methods:

1. **Standard Search**: Fixed top-k retrieval with optional metadata filtering
2. **Grouped Results**: Results are organized by group_id to show related blocks together

## Usage Scenarios

### Source Code Indexing

Ideal for indexing codebases where understanding both specific functions and their context is important:

```bash
python code_indexer.py index /path/to/codebase -c project_code
```

### Documentation/Tutorial Indexing

Extracts and indexes code blocks from markdown files:

```bash
python code_indexer.py index /path/to/docs -c docs_code -e .md
```

### Code Search

Semantic search with filtering options:

```bash
python code_indexer.py search "file reading async function" -l python
```

### Machine Learning Applications

The dual representation approach is particularly valuable for:
- Code completion systems that need both precise and contextual understanding
- Code-to-text generation where amalgamated blocks provide better context
- Function re-implementation tasks where individual blocks provide focused examples
