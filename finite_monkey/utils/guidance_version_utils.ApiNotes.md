# Guidance Version Utils API Notes

## Overview
This module provides compatibility utilities for working with different versions of the Guidance library, which has undergone significant API changes between versions. The module detects the installed version and adapts accordingly.

## Code Corruption Awareness

This module has experienced code corruption issues, particularly with:

1. **Unbalanced brackets and parentheses** - Missing closing `}`, `)`, etc.
2. **Truncated code blocks** - Functions cut off mid-implementation
3. **Missing control flow endings** - `if` statements without proper closure

When editing this code, special care should be taken to:

1. Make forward-only edits to maintain code flow
2. Check for balanced delimiters like `()`, `{}`, and `[]`
3. Complete any truncated code blocks
4. Ensure all control flows (`if`/`else`) are properly closed
5. Test after each edit to verify integrity

## Key Components

### GuidanceProgramWrapper
A wrapper class that provides a consistent interface for Guidance programs regardless of the underlying Guidance version. Handles execution, error recovery, and result processing.

- `__init__`: Initializes with a program, output class, and optional parameters
- `__call__`: Main entry point for executing the program with arguments
- `_execute_program`: Handles the actual execution with sync/async detection
- `_process_result`: Converts results into the expected output format

## DirectTemplateHandler Implementation

The `DirectTemplateHandler` class provides a complete template processing engine with specialized handling for various template elements:

### Key Features

1. **Variable Substitution**: Replaces `{{variable}}` with values from the data context
2. **Conditional Blocks**: Processes `{{#if var}}...{{/if}}` blocks
3. **Loop Processing**: Handles `{{#each array}}...{{/each}}` blocks with proper nesting
4. **Property Access**: Special handling for `{{this.property}}` references in loops

### Property Access Pattern

A critical aspect of the implementation is handling property access correctly:

```python
# Pattern for this.property references
self.this_property_pattern = re.compile(r'{{this\.([^}]+)}}')
```

This regex pattern specifically matches references like `{{this.name}}` and extracts the property name.

### Property Access Processing

The implementation uses explicit property access handling:

```python
# Handle this.property references explicitly before recursion
item_template = inner_template
if isinstance(item, dict):
    def replace_this_prop(prop_match):
        prop_name = prop_match.group(1)
        if prop_name in item:
            return str(item[prop_name])
        return f"{{{{this.{prop_name}}}}}"
    
    item_template = self.this_property_pattern.sub(replace_this_prop, item_template)
```

This approach:
1. Handles property access before recursive processing
2. Directly replaces `{{this.property}}` references with their values
3. Preserves references to missing properties

### Processing Order

The template processing follows this order:
1. Process conditionals first
2. Process loops next (with property access handling)
3. Replace variables last

This ensures all control structures are resolved before variable substitution.

### Template Processing Algorithm

The implementation optimizes processing based on template complexity:

```python
def _process_template(self, data):
    processed = self.template
    
    # Direct replacement for simple templates (fast path)
    for key, value in data.items():
        placeholder = "{{" + key + "}}"
        if placeholder in processed:
            processed = processed.replace(placeholder, str(value))
    
    # Only use recursive processing if control structures exist
    if "{{#if" in processed or "{{#each" in processed:
        processed = self._recursive_process(processed, data)
    
    return processed
```

This optimization ensures that simple templates like `"Hello {{name}}!"` are processed efficiently while still supporting complex nested structures when needed.

### String Handling in Loops

A critical aspect of the implementation is handling strings properly:

```python
# Check if the variable exists and is iterable
if var_name in data and hasattr(data[var_name], '__iter__') and not isinstance(data[var_name], str):
```

This ensures that strings aren't treated as iterables in `{{#each}}` loops, avoiding infinite recursion or incorrect processing.

### Recursive Processing

The recursive approach works by:

1. Processing conditionals first
2. Then processing loops (with their own nested processing)
3. Finally substituting variables

This order ensures correct handling of complex templates with nested structures.

## Resilient Method Resolution System

The DirectTemplateHandler now implements dynamic method resolution to avoid breaking when method names change or are missing:

1. **Dynamic Method Resolution**: Uses `_resolve_method()` to try multiple alternative names
2. **Fallback Implementations**: Provides robust fallbacks for critical methods
3. **Self-Healing Design**: Adapts to method name changes without breaking

This approach prevents AttributeErrors like:

### CRITICAL IMPLEMENTATION DETAILS

### 1. Always Await Async Methods

The `DirectTemplateHandler.__call__` method is async and MUST be awaited:

```python
# CORRECT USAGE
result = await handler(**kwargs)  # ALWAYS use await

# INCORRECT USAGE - Will cause "coroutine never awaited" warning
result = handler(**kwargs)  # Missing await - THIS WILL FAIL
```

### 2. Required Methods

The `DirectTemplateHandler` class MUST implement these methods:

- `__call__`: Main async entry point
- `_process_template`: Handles template variable substitution
- `_call_llm`: Async method to call the LLM
- `_process_response`: Processes the LLM response
- `_extract_text_from_response`: Extracts text from response object
- `_extract_structured_data`: Extracts structured data from text

### 3. Async Priority

Always prioritize async interfaces over sync:

- Use `acomplete`, `agenerate` methods when available
- Use `complete`, `generate` with `await` when they're coroutines
- Only fall back to sync methods when no async option exists

## Async Method Priority

When interacting with LLMs, use this async method priority order:

1. Explicitly named async methods (prefixed with "a"):
   - `acomplete`
   - `agenerate`
   - `achat`

2. Standard methods that are coroutines:
   - `complete` (if `inspect.iscoroutinefunction(llm.complete)` is True)
   - `generate` (if `inspect.iscoroutinefunction(llm.generate)` is True)

3. Only fall back to synchronous methods when no async option exists

```python
# CORRECT USAGE
if hasattr(llm, "acomplete"):
    response = await llm.acomplete(prompt)
elif hasattr(llm, "agenerate"):
    response = await llm.agenerate(prompt)
elif inspect.iscoroutinefunction(llm.complete):
    response = await llm.complete(prompt)
```

Always follow this pattern to ensure we're using the most appropriate async methods available in the LLM implementation.

## CRITICAL: _process_response and _extract_structured_data methods

The `_process_response` method references `_extract_structured_data` which MUST be defined in the `DirectTemplateHandler` class. This has been a recurring issue.

```python
# CORRECT IMPLEMENTATION - CRITICAL METHODS REQUIRED
def _process_response(self, response):
    """Process the response into structured data"""
    # Extract text from response
    response_text = self._extract_text_from_response(response)
    
    # Extract structured data
    return self._extract_structured_data(response_text)

def _extract_structured_data(self, text):
    """Extract structured data from the response text"""
    if not text:
        return {}
    
    try:
        import json
        
        # Try to find JSON objects
        matches = self.json_pattern.findall(text if isinstance(text, str) else str(text))
        
        for match in matches:
            try:
                parsed = json.loads(match)
                # Make sure it's a dict before returning
                if isinstance(parsed, dict):
                    logger.debug(f"Parsed JSON: {parsed}")
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # If we couldn't find JSON, look for key-value pairs
        lines = text.split("\n") if isinstance(text, str) else str(text).split("\n")
        result = {}
        
        for line in lines:
            # Try to match "key: value" pattern
            import re
            kv_match = re.match(r'^\s*"?([^":]+)"?\s*:\s*(.+)$', line)
            if kv_match:
                key, value = kv_match.groups()
                result[key.strip()] = value.strip()
        
        if result:
            return result
        
        # Return the text as a simple result if nothing else worked
        return {"result": text if isinstance(text, str) else str(text)}
    except Exception as e:
        logger.error(f"Error extracting structured data: {e}")
        return {"error": str(e)}
```

## Handling Malformed JSON

LLMs occasionally produce malformed JSON, with common issues including:

1. **Unbalanced quotes**: Missing closing quotes in strings
2. **Truncated arrays**: Arrays that don't close properly
3. **Mixed data structures**: Array elements turning into object properties
4. **Unbalanced brackets**: Missing closing braces or brackets

The enhanced `_extract_structured_data` method addresses these issues with:

1. **Robust JSON parsing**: Attempts to fix common formatting issues
2. **Flow validation**: Ensures all flows have required fields
3. **Manual extraction**: Falls back to regex-based extraction when JSON parsing fails

Example of a malformed JSON response that can now be handled:

```json
"actors": ["Actor1", "Actor2", "key": "value"]
```

The method will transform this into:

```json
"actors": ["Actor1", "Actor2"], "key": "value"
```

### Debugging Flow Extraction

Set log level to DEBUG to see detailed information about JSON parsing:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

If you see "Attempting manual flow extraction" in logs, it means the JSON was too malformed for standard parsing and the system is using regex-based extraction as a fallback.

## Runtime Warning: Coroutine Never Awaited

The warning "coroutine 'DirectTemplateHandler.__call__' was never awaited" indicates that somewhere the `__call__` method is being called without `await`.

This must be fixed in ALL locations that call the DirectTemplateHandler:

1. `GuidanceProgramWrapper.__call__` - Already has correct `await self.program(**kwargs)` 
2. `GuidanceProgramWrapper._execute_program` - Has duplicate functionality and should be removed to avoid confusion
3. Any direct usage of DirectTemplateHandler outside these classes must use `await`

## JSON Extraction

Parses JSON from LLM responses using a robust regex pattern that correctly handles nested structures:

```python
self.json_pattern = re.compile(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}')
```

This pattern can extract JSON objects even when they are embedded in other text.

## Regex Patterns

Three main regex patterns are used:

1.  **If Pattern**: `r'{{#if\s+([^}]+)}}(.*?){{\/if}}'` with DOTALL flag
    *   Matches conditional blocks and captures the variable name and content

2.  **Each Pattern**: `r'{{#each\s+([^}]+)}}(.*?){{\/each}}'` with DOTALL flag
    *   Matches loop blocks and captures the array name and loop body template

3.  **JSON Pattern**: `r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'`
    *   Complex pattern to match JSON objects, even when nested

4.  **this.property Pattern**: `r'{{this\.([^}]+)}}'`
    *   Matches `{{this.property}}` references

**Important**: Double-check the `this.property` regex pattern to ensure it's matching correctly.

## Sync/Async Compatibility:

One of the key features is proper handling of both synchronous and asynchronous LLM APIs:

```python
if inspect.iscoroutinefunction(self.llm.complete):
    response = await self.llm.complete(template)  # Async
else:
    response = self.llm.complete(template)  # Sync
```

This ensures compatibility with various LLM implementations like Ollama (synchronous) and others that might be asynchronous.

## Version Detection
- `GUIDANCE_AVAILABLE`: Boolean indicating if Guidance is installed
- `GUIDANCE_VERSION`: String containing the detected version
- `GUIDANCE_LLMS_AVAILABLE`: Boolean indicating if the guidance.llms module is available

## Template Processing Features

The template processing now supports:
1. Variable replacement: `{{variable}}`
2. Conditional blocks: `{{#if var}}...{{/if}}`
3. Loop blocks: `{{#each array}}...{{/each}}` with `{{this}}` or `{{this.property}}` references
4. Schema block removal: `{{#schema}}...{{/schema}}`

## API Compatibility Challenges

The module handles complex compatibility between:
1. Different Guidance versions (0.1.16 vs newer)
2. Synchronous vs asynchronous LLM APIs
3. Various response formats from different LLM providers

## Usage with Guidance 0.1.16

This template engine serves as a replacement for Guidance's built-in template processing when using version 0.1.16, which has issues with the decorator pattern.

The `create_guidance_program` function automatically uses this template engine when it detects Guidance 0.1.16:

```python
if guidance_version == "0.1.16":
    logger.info("Using direct template handler for Guidance 0.1.16")
    program = DirectTemplateHandler(template, llm)
    return GuidanceProgramWrapper(program, output_cls, verbose=verbose, fallback_fn=fallback_fn)
```

## Guidance 0.1.16 Specifics

For Guidance 0.1.16:
- The decorator approach is problematic and prone to internal errors
- `DirectTemplateHandler` is more reliable and should be the primary approach
- Internal module access is available as a fallback but may be fragile

## LLM API Compatibility

The code handles various LLM implementations:

1. **Sync vs Async Detection**: Uses `inspect.iscoroutinefunction` to determine if methods are asynchronous
2. **Method Discovery**: Tries `complete`, `generate`, and direct calling in order
3. **Response Extraction**: Handles various response structures with attribute or dictionary access
4. **Structured Data Parsing**: Extract JSON or key-value pairs from text responses

## Debugging Ollama Integration

The DirectTemplateHandler's interaction with Ollama can fail in several ways:

1. **Connection Failures**: Ollama server not running or unreachable
2. **Model Loading Failures**: Requested model not available or loaded
3. **Request/Response Failures**: Errors during the request/response cycle
4. **Parsing Failures**: Unable to parse the LLM's response into structured data

### Diagnostic Flow

When diagnosing Ollama integration issues, trace the execution path:

1. `GuidanceProgramWrapper.__call__` - Ensures proper awaiting of the DirectTemplateHandler
2. `DirectTemplateHandler.__call__` - Main entry point that processes templates and calls LLM 
3. `DirectTemplateHandler._call_llm` - Handles sync/async LLM API calls
4. `DirectTemplateHandler._process_response` - Processes the LLM response

Each method has detailed logging to track the progression and identify failures.

### Common Failure Points

1. **Silent LLM Failures**: The LLM call might appear to succeed but return empty responses
2. **Coroutine Not Awaited**: Async methods must be awaited; check for RuntimeWarnings
3. **Response Parsing**: The LLM response may not match the expected structure

### Testing with Direct Ollama Client

If the standard integration fails, use the DirectOllamaClient for testing:

```python
from finite_monkey.utils.ollama_direct import DirectOllamaClient

async def test_ollama_directly(prompt):
    client = DirectOllamaClient()
    health = await client.check_health()
    print(f"Ollama health: {health}")
    
    if health.get("status") == "healthy":
        response = await client.generate("dolphin3:8b-llama3.1-q8_0", prompt)
        print(f"Response: {response.get('response', '')[:100]}...")
    
    await client.close()
```

This bypasses the Guidance and LlamaIndex layers to test Ollama directly.

## LLM Integration Approach

This module uses official LlamaIndex interfaces rather than direct API clients:

1. **Provider-specific LLM classes** from LlamaIndex are used to interact with different LLM providers
2. **Proper async/sync handling** ensures all coroutines are properly awaited
3. **Standardized interfaces** make it easier to switch between different LLM providers

### Official LlamaIndex LLM Interfaces

```python
# OpenAI via LlamaIndex
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# Ollama via LlamaIndex
from llama_index.llms.ollama import Ollama
llm = Ollama(model="dolphin3:8b-llama3.1-q8_0", temperature=0.1)
```

### Coroutine Handling

The most common issue with LLM integration is incorrect coroutine handling. To ensure proper usage:

1. All async methods must be awaited with `await`
2. Detect if methods are coroutines using `inspect.iscoroutinefunction()`
3. Handle both sync and async interfaces consistently

```python
import inspect

if inspect.iscoroutinefunction(llm.complete):
    response = await llm.complete(template)  # Async
else:
    response = llm.complete(template)  # Sync
```

This pattern ensures compatibility with different LLM provider implementations.

## Usage Examples

### Basic Usage
```python
program = await create_guidance_program(
    output_cls=MyOutputClass,
    prompt_template="Hello {{name}}!",
    model="llama3",
    provider="ollama"
)
result = await program(name="World")
```

### With Conditional Blocks
```python
template = """
{{#if question}}
Q: {{question}}
{{/if}}
{{#if context}}
Context: {{context}}
{{/if}}
"""
program = await create_guidance_program(output_cls=AnswerClass, prompt_template=template)
result = await program(question="What is the capital of France?", context="France is in Europe.")
```

### Template with All Features
```handlebars
{{#if intro}}
Introduction: {{intro}}
{{/if}}

Items:
{{#each items}}
- {{this}}
{{/each}}

People:
{{#each people}}
- {{this.name}} ({{this.age}})
{{/each}}
```

### Data Structure
```python
data = {
    "intro": "Sample list",
    "items": ["Apple", "Banana", "Cherry"],
    "people": [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ]
}
```

## Error Handling
The implementation includes comprehensive error handling:
- Graceful handling of missing LLM methods
- Recovery from response parsing failures
- Structured logging for debugging
- Multiple fallback approaches if primary methods fail

## Error Handling and Retries

When dealing with LLM API calls, several types of errors can occur:

1. **HTTP 500 errors**: Often caused by server-side issues, including tokenizer failures
2. **Timeout errors**: When the server takes too long to respond
3. **Connection errors**: Network-related issues

The code implements robust error handling with exponential backoff:

```python
retry_count = 0
max_retries = 3

while retry_count <= max_retries:
    try:
        # API call here
    except Exception as e:
        retry_count += 1
        if retry_count <= max_retries:
            # Exponential backoff with jitter
            backoff_time = (2 ** retry_count) + (random.random() * 0.5)
            await asyncio.sleep(backoff_time)
```

## Tiktoken Tokenizer Issues

Tiktoken is OpenAI's tokenizer library. Issues with it can happen when:

1. **Inputs are too large**: Approaching or exceeding the model's context window
2. **Invalid UTF-8 sequences**: Characters that cause tokenization errors
3. **Extremely long tokens**: Unusual token patterns that cause internal errors

The code detects potential tokenizer issues with:
1. Conservative token count estimation (`_check_for_tokenizer_issue`)
2. Input size reduction when needed (`_reduce_input_size`)

## Empty Result Detection

When an LLM returns empty results or just code block markers (```` or ````), the code detects these cases and returns an empty flows list instead of trying to treat them as valid flow objects. This detection is done using:

```python
is_empty_result = ('result' in result and (
    not result['result'] or 
    isinstance(result['result'], str) and (
        result['result'].strip() == '```' or
        result['result'].strip() == '```\n```' or
        '```' in result['result'] and len(result['result'].strip()) < 10
    )
))
```

## Testing Recommendations

To verify changes and prevent corruption:
1. Run basic template processing tests without LLM calls
2. Test with actual LLM calls using both sync and async providers
3. Verify response handling with different response formats

## Testing Code Integrity

To verify the integrity of this module:

1. Run the integrity test: `python -m tests.test_guidance_version_utils_integrity`
2. Check for warnings or errors in the logs
3. Verify functionality with a simple template example

These tests will help ensure the module remains stable and functions correctly after changes.
