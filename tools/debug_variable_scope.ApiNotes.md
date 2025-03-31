# Variable Scope Debugging Utility

## Problem Context

The error `cannot access local variable 'os' where it is not associated with a value` is a common scope-related issue in Python that occurs when:

1. A module is imported at the top level of a file
2. The same module name is reused/shadowed in a function/method scope
3. Later references to the module name within that function are ambiguous

This pattern is particularly problematic because:
- It creates subtle bugs that are hard to detect by visual inspection
- The error only manifests when the problematic code path is executed
- It's a common mistake when refactoring or combining code from multiple sources

## Implementation Details

This utility provides a static analyzer that detects potential variable scope issues by:

1. **Parsing Python Code**: Uses the `ast` module to parse Python code into an abstract syntax tree
2. **Tracking Import Contexts**: Identifies imports at both the global and function/method levels
3. **Finding Shadowed Names**: Detects when the same module name is used at multiple scope levels
4. **Common Module Shadowing**: Specifically looks for shadowing of commonly used modules like `os`, `sys`, etc.

The analyzer focuses on two types of issues:
- **Double Import**: When a module is imported both globally and inside a function
- **Module Shadowing**: When a variable name overrides a common module name

## Usage Examples

Check a single file:
```bash
python tools/debug_variable_scope.py tools/vector_store_util.py
```

Check an entire directory:
```bash
python tools/debug_variable_scope.py tools --recursive
```

This tool can help prevent the specific issue that occurred in the `add_documents` method where the `os` module was inadvertently shadowed.
