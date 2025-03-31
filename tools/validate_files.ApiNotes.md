# Python File Validation Tool

## Purpose

This validation tool helps identify Python file corruption issues, including:

1. **Syntax Errors**: Using Python's built-in AST parser to catch invalid syntax
2. **Indentation Issues**: Detecting mixed tabs/spaces and other indentation problems
3. **Incomplete Code**: Finding truncated control structures or methods

## Technical Details

The validator uses Python's standard libraries to analyze files:

- `ast.parse()` - Checks for valid Python syntax
- Tokenizer analysis - Looks for inconsistent indentation
- Line-by-line scanning - Finds specific problematic patterns

## Common Corruption Patterns

1. **Indentation Corruption**
   - Mixed tabs and spaces
   - Inconsistent indentation levels
   - Truncated indentation blocks

2. **Incomplete Conditionals/Loops**
   - `if` statements without bodies
   - Incomplete `try/except` blocks
   - Loop constructs with missing bodies

3. **Method/Class Definition Issues**
   - Incorrectly nested class definitions
   - Methods with misaligned indentation
   - Incomplete method signatures

## Usage

Run the validator to identify potential corruption:

```bash
$ python tools/validate_files.py
```

When problems are found, use the output to locate and fix the issues.
