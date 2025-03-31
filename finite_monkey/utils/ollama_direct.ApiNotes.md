# Direct Ollama Integration API Notes

## Background

The `ollama_direct.py` module was created to address a critical issue where the standard LlamaIndex Ollama integration was not making any network calls to the Ollama server. This prevented the business flow extractor and other components from getting responses from local LLMs.

## Key Issues Identified

1. **Silent Failures**: The LlamaIndex Ollama integration sometimes fails silently without making network calls
2. **Configuration Problems**: Base URL configuration may not be properly passed through layers of abstraction
3. **LLM Availability**: The configured model might not be available in the Ollama server
4. **Network Connectivity**: Basic connectivity to the Ollama server might be blocked

## Implementation Strategy

The direct Ollama client provides a minimal, dependency-free way to interact with Ollama API directly:

1. **Health Checks**: Verify the Ollama server is running and responsive
2. **Model Listing**: Check if the specified model is available
3. **Direct Generation**: Make API calls directly without going through LlamaIndex
4. **Structured Output**: Format requests to always return JSON results

## Integration Points

1. **Pre-check**: Before creating LlamaIndex Ollama LLMs, verify connectivity
2. **Fallback**: When LlamaIndex Ollama integration returns None, try direct API call
3. **Diagnostics**: Use the test script to diagnose Ollama issues separate from LlamaIndex

## Usage Notes

### Common Error Patterns

- **No Response**: When LlamaIndex Ollama integration makes no network calls
- **None Result**: When the LLM call completes but returns None instead of a response
- **Connectivity Errors**: When the Ollama server is unreachable

### Testing Hierarchy

When troubleshooting Ollama issues:

1. First run `test_ollama_direct.py` to verify basic Ollama functionality
2. If that works, the issue is in LlamaIndex integration, not Ollama itself
3. Check logs for specific errors in the LlamaIndex to Ollama interaction

## Implementation Details

The DirectOllamaClient uses `httpx` for async HTTP requests with these key methods:

- `check_health()`: Verifies the Ollama server is running
- `list_models()`: Lists available models to verify the target model exists
- `generate()`: Makes a direct generation request with full control over parameters

This implementation purposely avoids using LlamaIndex or Guidance to diagnose issues with those layers.

# Ollama Direct API Notes

## Efficient HTTP and JSON Handling

The DirectOllamaClient implements several performance optimizations:

1. **Direct JSON Parsing**: 
   - Parses JSON directly from `response.text` instead of calling `response.json()`
   - This avoids redundant parsing and handles cases where the response might contain valid JSON but have incorrect content-type headers

2. **Error Handling**:
   - Specifically catches `json.JSONDecodeError` to handle malformed JSON responses
   - Includes portions of the raw response in error reports for debugging
   - Uses specific error types (ConnectError vs generic Exception) for better diagnosis

3. **Resource Management**:
   - Uses a single `httpx.AsyncClient` instance for all requests
   - Explicitly provides `close()` method to ensure resources are released
   - Sets reasonable timeouts to prevent hanging on unresponsive servers

## JSON Response Handling

### Double Encoding Issues

The Ollama API sometimes returns responses that can lead to double encoding problems:

1. **Response Format Variations**:
   - Sometimes the `response` field contains plain text
   - Sometimes it contains valid JSON as a string
   - In rare cases, it may return escaped JSON strings

2. **Detection Mechanisms**:
   - The `generate()` method now includes logic to detect if the response contains JSON
   - It sets a `response_is_json` flag in the result when it detects JSON in the response
   - This helps callers handle the response appropriately without double parsing

3. **String Escaping Issues**:
   - JSON strings with lots of backslashes (`\\`) are often a sign of double encoding
   - The client includes special handling to detect and clean these cases

### Robust JSON Parsing

The `_analyze_contract_standard` method includes multiple layers of JSON parsing fallbacks:

1. **Standard Parsing**: First attempt to parse the response directly
2. **Unescaping**: If that fails, attempt to unescape potentially escaped JSON strings
3. **Regex Extraction**: As a last resort, try to find valid JSON objects within larger text
   using regular expressions

This multi-layered approach handles the various ways that LLMs might format their JSON responses.

### Debugging Assistance

The code now includes detailed logging to help diagnose JSON parsing issues:

- Logs the raw response type and preview
- Logs when it detects and cleans double-encoded strings
- Provides information about regex JSON extraction attempts

When troubleshooting JSON parsing issues, enable DEBUG level logging to see these details.

## Ollama REST API Endpoints

The DirectOllamaClient provides a thin wrapper around the Ollama REST API endpoints:

### 1. `/api/generate` (POST)

Used for text generation with an Ollama model.

**Request:**
```json
{
  "model": "llama2",
  "prompt": "Your prompt text here",
  "stream": false
}
```

**Response:**
```json
{
  "model": "llama2",
  "created_at": "2023-11-06T15:00:00.000000Z",
  "response": "Generated text response...",
  "done": true
}
```

### 2. `/api/tags` (GET)

Lists all available models in the Ollama server.

**Response:**
```json
{
  "models": [
    {
      "name": "llama2",
      "modified_at": "2023-11-06T15:00:00.000000Z",
      "size": 3791730475
    },
    {
      "name": "dolphin3:8b-llama3.1-q8_0",
      "modified_at": "2023-11-07T15:00:00.000000Z",
      "size": 4889529090
    }
  ]
}
```

### 3. `/api/version` (GET)

Returns version information about the Ollama server.

**Response:**
```json
{
  "version": "0.1.14"
}
```

## Troubleshooting Common Issues

1. **JSON Parsing Errors**: 
   - Check if Ollama is returning valid JSON
   - Examine the raw response in the error details
   - May indicate Ollama version mismatch or server issues

2. **Connection Errors**:
   - Verify Ollama is running (`ollama serve`)
   - Check network connectivity and firewall rules
   - Ensure correct base URL (default: http://localhost:11434)

3. **Timeout Issues**:
   - For large models or complex queries, increase the client timeout
   - Consider using a more powerful machine for running Ollama
   - Check CPU/RAM usage on the Ollama server

## Client Implementation Details

1. **DirectOllamaClient** uses `httpx.AsyncClient` to make asynchronous HTTP requests
2. All methods return dictionaries with consistent structure:
   - Success responses: Original API response fields
   - Error responses: `{"error": "error message", "status": "error_type"}`
3. Timeout is set to 30 seconds by default for all requests
4. The client handles connection errors separately to help diagnose network problems
5. All methods are async to maintain consistency with the rest of the application

## Common Issues & Troubleshooting

1. **Connection Errors**: If you receive connection errors, ensure the Ollama server is running with `ollama serve`
2. **Model Not Found**: If a model isn't available, pull it using `ollama pull <model_name>`
3. **Invalid Response Format**: Ensure you're using the correct API endpoints - they may change with Ollama versions
4. **Timeout Errors**: For large models, consider increasing the client timeout (`timeout=60.0`)
5. **HTTP Error Codes**: 
   - 404: API endpoint not found (check Ollama version)
   - 500: Server error (check Ollama logs)
   - 400: Invalid request (check request format)

## Relationship to LlamaIndex

This DirectOllamaClient is a fallback mechanism when the LlamaIndex integration with Ollama fails. It provides:

1. More detailed error information
2. Direct diagnostics of the Ollama server
3. Model availability checking
4. A last-resort generation capability

By bypassing LlamaIndex, we can determine whether issues are with Ollama itself or with the LlamaIndex integration.
