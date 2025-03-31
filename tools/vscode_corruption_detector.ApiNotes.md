# VSCode Corruption Detector

## Problem Context

VSCode sometimes introduces corruption in Python files, particularly with large files or when multiple instances are editing the same file. Common corruption patterns include:

1. **Code Interleaving**: Lines from different functions get mixed together
2. **Method Declaration Issues**: Methods appear without proper spacing or indentation
3. **Error Handling Corruption**: Exception handling blocks get malformed
4. **Unmatched Delimiters**: Quotes, parentheses, brackets, or braces left unmatched

These corruptions are often subtle and can pass initial visual inspection but cause syntax errors or runtime behavior issues.

## Implementation Details

The detector uses multiple approaches to identify corruption:

### 1. Syntax Error Detection
Uses Python's AST parser to identify files with syntax errors, providing line numbers and error descriptions.

### 2. Pattern-Based Detection
Looks for specific patterns of corruption:
- Except blocks without matching try blocks
- Method definitions without preceding blank lines
- Improperly formatted logger.error statements
- Unmatched quotes, parentheses, braces, and brackets

### 3. Auto-fixing Capabilities
For certain types of corruption:
- Fixes logger.error formatting by adding proper string formatting
- Adds missing quotes at the end of lines
- Creates backups before applying any fixes

## Usage Examples

```bash
# Check a single file
python tools/vscode_corruption_detector.py tools/vector_store_util.py

# Check and automatically fix issues
python tools/vscode_corruption_detector.py tools/vector_store_util.py --fix

# Recursively check an entire directory
python tools/vscode_corruption_detector.py tools --recursive
```

This tool can be integrated into CI/CD pipelines to catch corruption early or run manually when corruption is suspected.
