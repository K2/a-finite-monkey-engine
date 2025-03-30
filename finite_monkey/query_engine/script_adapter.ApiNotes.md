# Script Adapter for Query Enginesgines

## Overview

The `QueryEngineScriptAdapter` serves as a bridge between query engines and script generation capabilities. It leverages the reasoning power of the FLARE query engine to generate executable Python scripts based on code analysis, security findings, or specific requirements.quirements.

## Key Features

1. **Context-Aware Script Generation**
   - Uses code snippets and file paths as context for generating relevant scripts   - Uses code snippets and file paths as context for generating relevant scripts
   - Adapts to different script types (analysis, test, fix, generation)

2. **Configuration Loading**
   - Loads settings from a configuration file   - Loads settings from a configuration file
   - Supports commented JSON with special handlingth special handling
   - Falls back to sensible defaults when configuration is missing   - Falls back to sensible defaults when configuration is missing

3. **Script Extraction and Formatting**
   - Extracts code blocks from query responses
   - Handles different fence formats (markdown, XML)
   - Selects the most appropriate code block based on size and format   - Selects the most appropriate code block based on size and format

4. **File Management**4. **File Management**
   - Generates appropriate file paths based on script type and queryype and query
   - Creates necessary directories
   - Determines appropriate execution commandss

## Implementation Details

### Request Structure### Request Structure
The `ScriptGenerationRequest` object contains all necessary information for script generation:ject contains all necessary information for script generation:
- The query text that specifies what to generate- The query text that specifies what to generate
- Context snippets (code fragments)
- Relevant file paths
- Optional target path for the generated scriptnerated script
- Script type (analysis, test, fix, generation)
- Additional metadata

### Result Structure
The `ScriptGenerationResult` provides comprehensive information about the generated script:The `ScriptGenerationResult` provides comprehensive information about the generated script:
- The full script content
- The path where the script was saved- The path where the script was saved
- The command to execute the script
- Success/failure status
- Error information if generation failed
- Metadata about the generation process

### Query Construction
The adapter builds specialized queries for the FLARE engine that:The adapter builds specialized queries for the FLARE engine that:
1. Specify the purpose based on script type
2. Include relevant code context
3. List file paths
4. Establish requirements for the generated script
5. Incorporate the original user query5. Incorporate the original user query

### Error Handling### Error Handling
The adapter implements robust error handling:error handling:
- Safely loads configuration files with fallbacks- Safely loads configuration files with fallbacks
- Handles extraction failures gracefully
- Provides detailed error information in the result
- Ensures directories exist before writing files

## Integration Points

- **Query Engine**: Uses the `FlareQueryEngine` for generating script contentpt content
- **File System**: Interacts with the filesystem for configuration loading and script saving- **File System**: Interacts with the filesystem for configuration loading and script saving
- **Configuration**: Integrates with the GenAIScript configuration system: Integrates with the GenAIScript configuration system
- **Pipeline Context**: Uses the context from the pipeline when available- **Pipeline Context**: Uses the context from the pipeline when available

## Usage Patterns

### Basic Script Generation
```python
request = ScriptGenerationRequest(request = ScriptGenerationRequest(
    query="Create a security analysis script for the ERC20 contract",a security analysis script for the ERC20 contract",
    context_snippets=[contract_code],    context_snippets=[contract_code],
    file_paths=[contract_path],
    script_type="analysis"
)
result = await script_adapter.generate_script(request, context)t(request, context)


