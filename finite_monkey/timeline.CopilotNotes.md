# Development Timeline and Copilot Interactions

## Guidance Version Compatibility

### July 15, 2023 - Initial Guidance 0.1.16 compatibility issues

**Problem**: Guidance integration was failing with error "unexpected keyword argument 'llm'" when trying to create a Guidance program with version 0.1.16.

**Investigation**: 
- Examined the `vars()` output to understand the context and error
- Inspected the Guidance 0.1.16 source code to identify the correct parameter names
- Found that Guidance 0.1.16 uses `model` parameter instead of `llm` or `api`

**Solution**:
- Updated `create_guidance_program` function to handle Guidance 0.1.16 specifically
- Added proper error handling and logging for all attempts
- Created comprehensive documentation in ApiNotes.md

**Time Spent**:
- Investigation: ~30 minutes
- Implementation: ~45 minutes
- Testing: ~20 minutes
- Documentation: ~15 minutes

### July 16, 2023 - Deeper Guidance 0.1.16 integration issue

**Problem**: Deeper issues with Guidance 0.1.16 integration discovered. Error message: "module, class, method, function, traceback, frame, or code object was expected, got str" when trying to call `guidance(template, model=llm)`.

**Investigation**:
- Examined the Guidance 0.1.16 source code in more detail
- Discovered that `guidance` module is actually a class instance with a `__call__` method designed to be used as a decorator
- The decorator expects a function, not a string template

**Solution**:
- Implemented multiple approaches to create a Guidance program for version 0.1.16:
  1. Create a function that returns the template and decorate it with `guidance`
  2. Try to use internal module functions if necessary
  3. Create a dynamic callable object or minimal fallback program as a last resort
- Enhanced error handling with detailed logging
- Updated ApiNotes.md with comprehensive details about Guidance 0.1.16 structure

**Time Spent**:
- Investigation: ~45 minutes
- Implementation: ~1 hour
- Testing: ~30 minutes
- Documentation: ~20 minutes

### July 20, 2023 - Import issues with Guidance 0.1.16 Program class

**Problem**: Attempt to directly import `Program` class from Guidance 0.1.16 failed as this class is not directly exposed at the top level of the module.

**Investigation**:
- Examined the Guidance 0.1.16 `__init__.py` file
- Confirmed that there is no `Program` class imported or defined at the top level

**Solution**:
- Completely removed the direct import of `Program`
- Fixed the program creation method to use the guidance decorator pattern properly
- Added a dynamic callable object implementation for cases where the decorator approach fails
- Improved the fallback mechanism with better JSON extraction

**Time Spent**:
- Investigation: ~20 minutes
- Implementation: ~45 minutes
- Testing: ~25 minutes
- Documentation: ~15 minutes

### July 21, 2023 - Closure function issues with Guidance 0.1.16

**Problem**: New error when using the decorator approach: "You currently must use @guidance(dedent=False) for closure functions (function nested within other functions that reference the outer functions variables)!"

**Investigation**:
- The error message clearly indicated we need to use `dedent=False` with the decorator
- This is a specific requirement for closure functions (functions that reference variables from their outer scope)
- The error happens because our function references the `template_str` variable from the outer scope

**Solution**:
- Updated the decorator usage to `@guidance(dedent=False)` for closure functions
- Added alternative approaches that don't use closures as fallbacks
- Updated documentation to highlight this specific requirement

**Time Spent**:
- Investigation: ~15 minutes
- Implementation: ~30 minutes
- Testing: ~20 minutes
- Documentation: ~10 minutes

### July 22, 2023 - Internal errors in Guidance 0.1.16 decorator

**Problem**: Even with `dedent=False`, encountered internal error `IndexError: pop from empty list` in the Guidance library when using the decorator.

**Investigation**:
- The error appears to be an internal bug in the Guidance 0.1.16 implementation
- Occurs specifically when using the decorator with parameters

**Solution**:
- Completely abandoned the decorator approach due to internal bugs
- Implemented direct approaches that bypass the decorator completely:
  1. Try to use internal guidance modules directly (`_grammar`, `_guidance`)
  2. Create a direct template handler that works with the LLM
  3. Build a robust fallback LLM handler that extracts structured data
- Enhanced JSON extraction logic to better handle different response formats

**Time Spent**:
- Investigation: ~25 minutes
- Implementation: ~50 minutes
- Testing: ~30 minutes
- Documentation: ~15 minutes

### July 23, 2023 - Async/Sync API compatibility issues

**Problem**: Discovered that the Ollama LLM implementation returns synchronous objects rather than awaitable coroutines, causing an error: "object CompletionResponse can't be used in 'await' expression".

**Investigation**:
- The test script revealed that the LLM's `complete` method returns a synchronous `CompletionResponse` object
- Our implementation was assuming all LLM methods are async and using `await`
- Different LLM providers might have different API patterns (sync vs async)

**Solution**:
- Enhanced both the test script and the main implementation to handle both sync and async LLM APIs
- Added detection of coroutine functions using `inspect.iscoroutinefunction`
- Implemented robust response extraction that works with various response formats
- Updated documentation to explain the LLM API compatibility challenges

**Time Spent**:
- Investigation: ~20 minutes
- Implementation: ~40 minutes
- Testing: ~25 minutes
- Documentation: ~15 minutes

### July 25, 2023 - Template processing enhancement - Nested loops

**Problem**: The template processing implementation could not handle nested loops properly, causing tests with complex templates to fail.

**Investigation**:
- Analysis of test failures showed that nested structures like `{{#each sections}}{{#each items}}...` weren't being processed correctly
- The original implementation processes loops but not nested structures

**Solution**:
- Implemented a recursive template processing algorithm that correctly handles nested structures
- Modified the processing order to handle conditionals before loops
- Created a context management system for loop iterations
- Added proper handling for the `{{this}}` reference in nested contexts
- Extended test suite with a complex template test that verifies nested loops

**Time Spent**:
- Investigation: ~15 minutes
- Implementation: ~50 minutes
- Testing: ~20 minutes
- Documentation: ~15 minutes

**Lessons Learned**:
1. When implementing template engines, recursive processing is essential for proper nesting
2. Context management is crucial for correct variable resolution in nested scopes
3. The processing order (conditionals → loops → variables) affects the template outcome
4. Comprehensive testing with complex templates helps catch edge cases

### July 25, 2023 - Property Access in Templates

**Problem**: Templates with `{{this.property}}` notation in loops weren't being processed correctly, causing test failures.

**Investigation**:
- Debug output showed that while basic loops were working, object property access was not
- The issue was in how `{{this.property}}` references were being handled within loops
- The template engine didn't have specific handling for the property access notation

**Solution**:
- Added a dedicated regex pattern to match property access syntax: `{{this.property}}`
- Implemented explicit handling for property access before recursive processing
- Added a replacement function that directly substitutes property values
- Added tests with complex templates that include property access

**Time Spent**:
- Investigation: ~15 minutes
- Implementation: ~30 minutes
- Testing: ~15 minutes
- Documentation: ~10 minutes

**Lessons Learned**:
1. Property access within templates needs explicit handling
2. Pre-processing `{{this.property}}` references before recursive processing yields cleaner results
3. Comprehensive testing of all template features is essential

## Template Processing Debugging

### Initial State

*   `test_complex_template` failing with `AssertionError`.
*   Suspected issue with template processing logic.

### Iteration 1

*   Identified potential issue with `_recursive_process` method.
*   Added logging to inspect `item` and `item_template`.
*   Estimated time spent: 15 minutes.

### Iteration 2

*   Discovered test script was shadowing the imported `DirectTemplateHandler` class.
*   Removed the class definition from the test script.
*   Estimated time spent: 10 minutes.

### Next Steps

*   Run the test again with the corrected test script and logging.
*   Analyze the debug output to identify the root cause of the failure.
*   Estimated time to completion: 30 minutes.

## Table of Contents

| Date | Issue | Solution | Time to Resolve |
|------|-------|----------|-----------------|
| July 15, 2023 | Guidance 0.1.16 API parameter naming | Fixed parameter naming (`model` vs `llm`/`api`) | ~2 hours |
| July 16, 2023 | Guidance 0.1.16 decorator vs direct call | Implemented multiple creation approaches with fallbacks | ~2.5 hours |
| July 20, 2023 | Import issues with Guidance Program class | Removed direct import, used decorator pattern properly | ~1.75 hours |
| July 21, 2023 | Closure function issues with Guidance | Added `dedent=False` parameter to decorator | ~1.25 hours |
| July 22, 2023 | Internal errors in Guidance decorator | Abandoned decorator approach, implemented direct handlers | ~2 hours |
| July 23, 2023 | Async/Sync API compatibility | Enhanced implementation to handle both API patterns | ~1.75 hours |
| July 25, 2023 | Nested template processing | Implemented recursive template algorithm | ~1.67 hours |
| July 25, 2023 | Property access in templates | Added explicit property access handling | ~1.17 hours |
