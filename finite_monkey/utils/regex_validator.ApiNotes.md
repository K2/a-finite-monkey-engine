# Regex Validator API Notes

## Purpose
This utility provides regex validation capabilities to prevent runtime errors from invalid regex patterns. It's particularly useful when working with user-defined patterns or dynamically constructed patterns.

## Common Invalid Patterns
- `$\w*` - End-of-line anchor followed by characters (unreachable pattern)
- `*abc` - Quantifier without a preceding character
- `[abc` - Unclosed character class
- `(xyz` - Unclosed parentheses
- `a{2,1}` - Invalid quantifier range (min > max)

## Usage Examples
```python
from finite_monkey.utils.regex_validator import validate_regex, test_regex_match

# Validate a pattern
is_valid, error = validate_regex(r"\w+\d{3}")
if not is_valid:
    print(f"Error: {error}")

# Test a pattern against a string
matched, matches, error = test_regex_match(r"(\w+)_(\d+)", "test_123")
if matched:
    print(f"Matches: {matches}")
else:
    print(f"No match or error: {error}")
```

## Integration with Other Components
This validator can be integrated with component parsers, contract analyzers, and any other modules that utilize regex patterns to prevent runtime errors.
