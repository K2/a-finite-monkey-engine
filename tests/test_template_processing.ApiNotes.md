# Test Template Processing API Notes

## Debugging `test_complex_template`

The `test_complex_template` function is failing because the complex template isn't being processed correctly. The assertion `assert "- Item 1.1" in processed` is failing.

### Debugging Steps

1.  Added logging to the `_recursive_process` method in `DirectTemplateHandler` to inspect the `item` dictionary and the `item_template` before and after the property replacement.
2.  Removed the `DirectTemplateHandler` class definition from `tests/test_template_processing.py` to ensure that the test script is using the correct implementation of the class from `finite_monkey/utils/guidance_version_utils.py`.
3.  Run the test again and analyze the debug output.

### Important Notes

*   **Do not redefine the `DirectTemplateHandler` class in the test script.** The test script should only import the class and use it directly.

### Potential Issues

1.  The `item` dictionary might not contain the expected keys (`id` and `name`).
2.  The `this_property_pattern` regex might not be matching the `{{this.id}}` and `{{this.name}}` references correctly.
3.  The replacement logic might be faulty.