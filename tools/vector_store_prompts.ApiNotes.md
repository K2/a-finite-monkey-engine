# Prompt Generation for Vector Store

## Overview

The `PromptGenerator` creates specialized prompts to guide LLMs in analyzing code for security vulnerabilities. It processes GitHub issue data to generate prompts that encourage LLMs to discover issues without explicitly revealing them.

## Key Design Considerations

1. **Issue Abstraction**: Prompts are designed to guide analysis without revealing the specific vulnerability, allowing LLMs to discover issues independently.

2. **Multi-perspective Analysis**: The system generates different prompts for specialized analysis types (security, code quality, optimization), enabling comprehensive code evaluation.

3. **Context Preservation**: The system extracts relevant context from GitHub issues while filtering out explicit mentions of vulnerabilities.

4. **Code Extraction**: Sophisticated extraction of code segments from markdown, supporting both code blocks and inline code.

## Main Components

1. **Prompt Generation**:
   - Basic prompts for general analysis
   - Specialized security prompts for different vulnerability types
   - Multi-LLM prompts for different analysis perspectives

2. **Text Processing**:
   - Code segment extraction from markdown
   - Non-code text extraction for context
   - Issue context extraction with vulnerability filtering

3. **LLM Integration**:
   - Async communication with Ollama API
   - Fallback to rule-based prompts when LLM is unavailable

## Usage Workflow

1. GitHub issue data is provided as input
2. Code segments are extracted for embedding
3. Remaining text is processed to extract context
4. Prompts are generated to guide future analysis
5. Both code and prompts are stored in the vector database
6. When similar code is found during analysis, stored prompts guide LLMs

## Example

For a GitHub issue about integer overflow:
- The code is extracted and embedded for similarity search
- The description is processed to create a prompt like "Analyze this code for arithmetic vulnerabilities, focusing on boundary conditions and edge cases"
- When similar code is found in a new project, this prompt helps the LLM discover potential integer overflow issues

This approach allows the system to leverage known vulnerabilities to detect similar issues in new code without explicitly revealing the vulnerability patterns.

# Ollama Completion Call Fix

## Issue Fixed
The `_call_ollama_completion` method in `PromptGenerator` was incomplete, missing crucial parts:
1. The completion of the official Python client try block
2. Empty exception handlers
3. Missing payload definition for the direct API call

## Implementation Details
The fixed implementation:

1. **Dual Strategy Approach**
   - First tries the official Python client (`import ollama`)
   - Falls back to direct API calls if the client is unavailable or fails

2. **Error Handling**
   - Properly handles ImportError when Ollama client isn't installed
   - Gracefully handles exceptions from both approaches
   - Includes detailed logging for troubleshooting

3. **API Compatibility** 
   - Uses the correct message format for the Ollama API:
   ```json
   {
     "model": "model_name",
     "messages": [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "..."}
     ]
   }
   ```

## Testing
The fixed code can be verified using the `test_ollama_connection.py` script, which tests both:
- The official Python client
- Direct API calls via aiohttp

Both approaches follow the same message format and should work with Ollama v0.1.14+.

# Prompt Generation Logging

## Enhancement Details

This update adds comprehensive logging for generated prompts, which helps with:
1. Debugging prompt generation issues
2. Understanding prompt quality
3. Tracking prompt generation performance

## Implementation Details

### PromptGenerator Class

Added explicit logging in:
- `generate_prompt`: Logs the generated prompt with length and preview
- `generate_multi_llm_prompts`: Logs each type of prompt with length and preview

Example log output:
```
Generated prompt: "Analyze this code for security vulnerabilities..." (Length: 120)
Generated multi-LLM prompt for optimization: "Optimize this code for performance..." (Length: 95)
```