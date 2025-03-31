# Flow Joiner API Notes

## Purpose

The `FlowJoiner` class serves as the final normalization stage in the pipeline, ensuring that data from all previous stages follows a consistent structure for reporting and visualization.

## Implementation Requirements

As a pipeline stage, the `FlowJoiner` must:

1. **Implement `__call__**: An async method that takes and returns a `Context` object
2. **Handle Errors Gracefully**: Catch exceptions and add them to the context
3. **Normalize Data Consistently**: Convert all data to a standardized format

## Data Normalization

The joiner normalizes several data types:

1. **Business Flows**: Makes sure all flows have required fields (name, description, steps, etc.)
2. **Vulnerabilities**: Standardizes security issue representation
3. **Documentation Quality**: Normalizes quality metrics and suggestions
4. **Cognitive Biases**: Ensures bias information has consistent structure
5. **Data Flows**: Standardizes data flow representations

## Handling Different Data Types

A key challenge is handling both dictionary and object representations of the same data:

```python
# For dictionary-style data
normalized_flow = {
    'name': flow.get('name', 'Unnamed Flow'),
    # ...other fields
}

# For object-style data
normalized_flow = {
    'name': getattr(flow, 'name', 'Unnamed Flow'),
    # ...other fields
}
```

This ensures that regardless of the input format, the output is always a consistent dictionary structure.
