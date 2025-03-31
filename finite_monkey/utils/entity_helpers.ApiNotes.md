# Entity Helpers API Notes

## Problem Context

The codebase encounters issues when it interchangeably uses two different styles of entity representation:

1. **Dictionary-style entities**: Accessed using `entity.get('key', default)`
2. **Object-style entities**: Accessed using `getattr(entity, 'key', default)`

This leads to errors like:
