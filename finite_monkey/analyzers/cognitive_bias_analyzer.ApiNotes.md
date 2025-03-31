# Cognitive Bias Analyzer API Notes

## Error Handling for String Formatting

When handling error messages, especially those that might contain JSON or other formatted text, be careful with Python's string formatting:

1. The `%` operator in strings is treated as a format specifier
2. If your error message contains `%`, it will be interpreted as a format specifier
3. This can cause errors like: `Invalid format specifier '...' for object of type 'str'`

### Solution

Always escape percent signs in error messages or texts that might contain formatting characters:

```python
# Incorrect - might cause formatting errors
logger.error(f"Error analyzing: {str(e)}")

# Correct - escapes % signs to avoid formatting errors
error_msg = str(e).replace("%", "%%")
logger.error(f"Error analyzing: {error_msg}")
```

## Bias Detection Patterns

The analyzer uses regular expressions to detect cognitive biases in code:

1. **Overconfidence Bias**: Detects calls to external contracts without proper error handling
   - Pattern: `transfer/call/send` without `require/assert/revert/if(success)`

2. **Authority Bias**: Detects centralized control patterns
   - Pattern: `onlyOwner` modifiers or `require(msg.sender == owner)`

3. **Availability Bias**: Detects comments highlighting specific concerns
   - Pattern: Comments with `TODO/FIXME/WARNING/DANGER/SECURITY`

These patterns can be extended or modified to detect additional cognitive biases.
