# API Helpers Notes

## Entity Access Pattern

The `api_helpers.py` module provides utilities for accessing entity attributes regardless of data format:

### Problem Context

Throughout the codebase, entities can be represented as either:

1. **Dictionaries**: Access via `entity.get('name', default)`
2. **Objects**: Access via `getattr(entity, 'name', default)`

This leads to bugs when code assumes one format but gets the other. For example:

```python
# This will fail if flow is an object, not a dictionary
flow_name = flow.get('name', 'Unnamed flow') 
```

### Unified Access Solution

Use the `get_entity_attribute` function to handle both formats:

```python
from finite_monkey.utils.api_helpers import get_entity_attribute

# Works with both dictionaries and objects
flow_name = get_entity_attribute(flow, 'name', 'Unnamed flow')
```

For multiple attributes, use `get_entity_attributes`:

```python
attrs = get_entity_attributes(flow, ['name', 'description', 'steps'], 
                             {'name': 'Unnamed', 'steps': []})
flow_name = attrs['name']
flow_steps = attrs['steps']
```

### Standardized Pattern

When processing entities that could be either dictionaries or objects:

1. Prefer using the helper functions from `api_helpers.py`
2. If not possible, use explicit type checking:

```python
if isinstance(entity, dict):
    name = entity.get('name', default)
else:
    name = getattr(entity, 'name', default)
```

This prevents `AttributeError` exceptions when code encounters an unexpected entity type.
