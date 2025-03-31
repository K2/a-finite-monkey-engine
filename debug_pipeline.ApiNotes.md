# Debug Pipeline Entry Point

## Overview

The `debug_pipeline.py` script serves as a comprehensive debugging entry point for the Finite Monkey Engine. It creates a complete analysis pipeline with all available stages and processes smart contracts from the examples/src directory, generating a detailed markdown report.

## Key Components

### 1. Debug Logging

The script configures detailed logging with:
- Console output with colored formatting
- File logging with rotation
- Custom log format showing timestamps, log levels, and source locations

### 2. DebugStage Wrapper

The `DebugStage` class wraps standard pipeline stages with detailed logging:
- Logs when stages start and complete
- Tracks execution time
- Captures and summarizes context state before and after execution
- Provides detailed error information

### 3. Complete Pipeline

The script creates a comprehensive pipeline including all analysis stages:
1. Document loading
2. Contract extraction
3. Code chunking
4. Business flow extraction
5. Vulnerability scanning
6. Data flow analysis
7. Cognitive bias analysis
8. Documentation analysis
9. Result aggregation

### 4. FLARE Query Engine Testing

After the pipeline runs, the script tests the FLARE query engine with:
- A set of predefined debug queries
- Full integration with the analysis context
- Detailed logging of query execution

### 5. Markdown Report Generation

The `generate_markdown_report` function creates a comprehensive report including:
- Contract overview information
- Security vulnerability findings
- Business flow analysis
- Data flow analysis
- Cognitive bias detection
- FLARE query results
- Recommendations synthesized from all analysis stages
- Debug information

## Usage

The script is hardcoded to:
1. Take input from the examples/src directory
2. Place output in the debug_output directory
3. Create a timestamped markdown report

No command-line arguments are needed, simply run:
```bash
python debug_pipeline.py
```

# Debug Pipeline API Notes

## Overview

This file contains detailed notes about the `debug_pipeline.py` implementation and interactions with the `PipelineFactory` and related components.

## Key Components and Interactions

### Asynchronous File Enumeration

The `find_files_by_extension` function provides efficient asynchronous enumeration of files with specific extensions:

```python
async def find_files_by_extension(directory: str, extension: str = '.sol') -> List[Dict[str, Any]]:
    # ...implementation...
```

This function:
1. Uses `pathlib.Path.rglob` to find all matching files
2. Processes each file concurrently using `asyncio.gather`
3. Uses `aiofiles` for asynchronous file I/O operations
4. Returns detailed file information including path, name, content, and metadata

Key dependencies:
- `aiofiles`: For asynchronous file operations
- `asyncio`: For concurrent processing
- `pathlib`: For file system navigation

### Combined Document Processing Stage

The combined document processing stage optimizes cache efficiency by performing document loading, contract extraction, and code chunking in a single stage. This approach improves performance by keeping data "hot" in the cache between operations.

```python
combined_document_stage = await create_combined_document_processing_stage(factory)
```

The combined stage performs these operations in sequence:
1. Document Loading: Find and load Solidity files from input path
2. Contract Extraction: Extract contracts from the loaded files
3. Code Chunking: Chunk contracts into smaller processable segments

### PipelineFactory Methods

The `PipelineFactory` class provides several methods for creating pipeline stages:

- `load_documents`: Creates a stage for loading documents from a file path
- `extract_contracts_from_files`: Creates a stage for extracting contracts from loaded documents
- `create_business_flow_extractor`: Creates a business flow analysis stage
- `create_vulnerability_scanner`: Creates a vulnerability scanning stage
- `create_dataflow_analyzer`: Creates a data flow analysis stage
- `create_cognitive_bias_analyzer`: Creates a cognitive bias analysis stage
- `create_documentation_analyzer`: Creates a documentation analysis stage

## Performance Considerations

1. **Asynchronous I/O**: Using `aiofiles` for file operations improves throughput when processing multiple files by avoiding blocking I/O.

2. **Concurrent Processing**: Files are processed concurrently using `asyncio.gather` to maximize throughput.

3. **Error Handling**: Errors in individual file processing don't stop the entire enumeration process - they're logged and the file is marked with an error state.

4. **Cache Efficiency**: The combined document processing stage improves cache locality, reducing memory access latency.

## Pipeline Execution Pattern

The `Pipeline` class from `finite_monkey.pipeline.core` is not directly callable. Instead, it provides a `run()` method to execute the pipeline stages in sequence.

### Correct Usage Pattern:
```python
# Create pipeline
pipeline = Pipeline(stages=pipeline_stages)

# Execute pipeline
result = await pipeline.run(context)
```

### Incorrect Usage Pattern:
```python
# This will fail with: TypeError: 'Pipeline' object is not callable
result = await pipeline(context)
```

## Pipeline Stages
Each stage in the pipeline must be callable with an async signature and should accept and return a Context object:

```python
async def stage(context: Context) -> Context:
    # Process context
    return modified_context
```

The `DebugStage` wrapper correctly wraps each stage and maintains this contract.

## Contract Extraction Process

The core challenge in the document processing stage is properly extracting smart contract structures from Solidity files. The process works in multiple layers:

1. Files are discovered and loaded using `find_files_by_extension`
2. Each file is processed with `AsyncContractChunker.chunk_code` which:
   - Parses the Solidity code into an AST
   - Analyzes the AST to identify contracts and functions
   - Returns a hierarchical structure with contracts containing their functions

3. The resulting structure from `chunk_code` has this general form:
```json
{
  "chunk_id": "filename_full",
  "content": "full source code",
  "chunk_type": "file",
  "contracts": [
    {
      "name": "ContractName",
      "chunk_type": "contract",
      "chunk_id": "filename_ContractName",
      "content": "contract source",
      "functions": [
        {
          "name": "functionName",
          "chunk_type": "function",
          "chunk_id": "filename_ContractName_functionName",
          "content": "function source"
        }
      ]
    }
  ]
}
```

4. The pipeline needs to extract this hierarchical information and populate:
   - `context.contracts` - All contracts across all files
   - `context.functions` - All functions with identifiable keys

This approach enables subsequent analyzers to work with properly structured contract data.

## Contract Extraction Data Structures

When working with the contract extraction stage, it's important to understand the data structures involved:

### AsyncContractChunker.chunk_code() Return Values

The `chunk_code()` method can return either:

1. A list of chunk dictionaries
2. A single chunk dictionary

Each chunk can have the following structure depending on its type:

```python
# File-level chunk with contracts
{
    "chunk_id": "filename_full",
    "content": "full source code",
    "chunk_type": "file",
    "contracts": [
        {
            "name": "ContractName",
            "chunk_type": "contract",
            "chunk_id": "filename_ContractName",
            "content": "contract source",
            "functions": [
                {
                    "name": "functionName",
                    "chunk_type": "function",
                    "chunk_id": "filename_ContractName_functionName",
                    "content": "function source"
                }
            ]
        }
    ]
}

# Contract-level chunk
{
    "name": "ContractName",
    "chunk_type": "contract",
    "chunk_id": "filename_ContractName",
    "content": "contract source",
    "functions": [
        {
            "name": "functionName",
            "chunk_type": "function",
            "chunk_id": "filename_ContractName_functionName",
            "content": "function source"
        }
    ]
}
```

### Context Containers

The pipeline uses these key containers in the context:

1. `context.files` - Dictionary of file infos keyed by file path
2. `context.contracts` - List of all contracts extracted from all files
3. `context.functions` - Dictionary of functions keyed by function ID (typically ContractName_functionName)
4. `context.chunks` - Dictionary of all chunks keyed by chunk ID

## Debugging Contract Extraction

If contract extraction isn't working as expected:

1. Check the debug logs for the exact structure returned by `chunker.chunk_code()`
2. Verify if contracts are being correctly identified in each file
3. Examine how function extraction is working within each contract

The pipeline needs these correctly populated context objects for the subsequent analysis stages to work properly.

## Entity Type Handling

The pipeline needs to handle both dictionary and object representations of entities:

### Business Flow Access Patterns

Business flows can come in two forms:

1. **Dictionary Representation**:
   ```python
   flow_name = flow.get('name', 'Unnamed flow')
   flow_desc = flow.get('description', 'No description')
   ```

2. **Object Representation** (BusinessFlow class instances):
   ```python
   flow_name = getattr(flow, 'name', 'Unnamed flow')
   flow_desc = getattr(flow, 'description', 'No description')
   ```

The `generate_markdown_report` function now handles both formats by checking the instance type:

```python
if isinstance(flow, dict):
    # Use dictionary access methods (get)
    flow_name = flow.get('name', 'Unnamed flow')
else:
    # Use object attribute access methods (getattr)
    flow_name = getattr(flow, 'name', 'Unnamed flow')
```

This dual-access pattern should be used anywhere in the codebase that needs to handle
entities that might come in either form, particularly during report generation or when
working with data coming from different sources.