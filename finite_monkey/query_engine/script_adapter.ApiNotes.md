# Query Engine Script Adapter

## Overview

The `QueryEngineScriptAdapter` connects the FLARE query engine with script generation capabilities. It translates analysis requests into executable Python scripts that can be used for security analysis, testing, and remediation of smart contracts.

## Key Components

### Models

1. **ScriptGenerationRequest**: A Pydantic model that encapsulates all parameters needed for script generation:
   - `query`: The specific task or analysis to perform
   - `context_snippets`: Code snippets providing context
   - `file_paths`: Relevant file paths for reference
   - `target_path`: Where to save the generated script
   - `script_type`: Type of script to generate (analysis, test, fix, generation)
   - `metadata`: Additional information

2. **ScriptGenerationResult**: A Pydantic model containing the results of script generation:
   - `script_content`: The actual generated script content
   - `script_path`: Where the script was saved
   - `execution_command`: Command to run the script
   - `success`: Whether generation was successful
   - `error`: Error message if generation failed
   - `metadata`: Additional information about the generation process

### Adapter Class

The `QueryEngineScriptAdapter` class is responsible for:

1. **Configuration Management**:
   - Loading GenAIScript configuration
   - Managing output directories
   - Setting up default paths

2. **Script Generation**:
   - Constructing appropriate queries
   - Using the FLARE engine to generate responses
   - Extracting code from responses
   - Saving scripts to disk

3. **Response Processing**:
   - Handling different code fence formats (markdown, XML)
   - Selecting the best/largest code block
   - Properly formatting scripts

## Integration with GenAIScript

The adapter integrates with the GenAIScript configuration through:

1. Loading the `genaiscript.config.json` file which contains:
   - Model aliases for different task types
   - Fence format preferences
   - Other configuration options

2. Using the configuration to determine:
   - Which model to use for reasoning
   - How to extract code from responses
   - Default paths and behaviors

## Workflow

The typical workflow is:

1. **Request Creation**: Create a `ScriptGenerationRequest` with query and context
2. **Query Construction**: Format an appropriate prompt for the FLARE engine
3. **Script Generation**: Execute the query and process the response
4. **Content Extraction**: Extract script content from the response
5. **Script Saving**: Save the script to the specified location
6. **Result Return**: Return a `ScriptGenerationResult` with all relevant information

## Usage Example

```python
# Initialize adapter with a FLARE query engine
adapter = QueryEngineScriptAdapter(flare_engine)

# Create a request
request = ScriptGenerationRequest(
    query="Create a script to detect reentrancy vulnerabilities",
    context_snippets=["contract SimpleStorage { ... }"],
    file_paths=["contracts/SimpleStorage.sol"],
    script_type="analysis"
)

# Generate the script
result = await adapter.generate_script(request)

# Check result and use the script
if result.success:
    print(f"Script saved to: {result.script_path}")
    print(f"Run with: {result.execution_command}")
else:
    print(f"Error: {result.error}")