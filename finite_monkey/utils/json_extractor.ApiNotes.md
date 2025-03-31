# JSON Extractor Utility API Notes

## Purpose

The `json_extractor.py` module provides specialized functionality for extracting valid JSON from complex, nested response formats that may include:

1. JSON embedded in markdown code blocks
2. Double-encoded JSON strings
3. Escaped characters in JSON strings
4. JSON objects embedded in larger text responses
5. Malformed but recoverable JSON

## Key Functionality

### `extract_json_from_complex_response()`

This function implements a multi-layered approach to JSON extraction using progressively more aggressive techniques:

1. **Direct Parsing**: First attempts to parse the input as valid JSON
2. **String Unescaping**: Handles JSON strings that have been wrapped in quotes and escaped
3. **Backslash Cleaning**: Fixes common escaping issues with backslashes
4. **Markdown Extraction**: Identifies and extracts JSON from markdown code blocks
5. **Regex Pattern Matching**: Uses regular expressions to find JSON-like structures
6. **Targeted Extraction**: Looks specifically for expected object structures (e.g., with `flows` key)

## Implementation Details

### Debug Information

The function returns both the extracted JSON and detailed debug information about what was tried. This is valuable for diagnosing extraction failures and understanding the response format.

### Error Handling

The extractor handles various error conditions gracefully:
- JSONDecodeError from parsing attempts
- Nested structures with mismatched brackets
- Malformed markdown code blocks
- Various string encoding issues

### Use Cases

This utility is particularly useful when:
1. Working with LLMs that produce inconsistent response formats
2. Dealing with nested JSON structures (JSON containing JSON strings)
3. Processing markdown responses that contain code blocks
4. Extracting structured data from partially formatted responses

## Integration with Business Flow Extractor

The business flow extractor uses this utility to reliably extract JSON data from LLM responses, regardless of how nested or escaped the response might be. This provides resilience against changes in LLM output formatting.
