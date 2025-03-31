"""
Utilities for handling entity access regardless of data structure (dict or object).
"""
from typing import Any, Dict, List, Optional

def get_entity_value(entity: Any, key: str, default: Any = None) -> Any:
    """
    Get a value from an entity, whether it's a dictionary or an object.
    
    Args:
        entity: The entity to get the value from
        key: The key or attribute name
        default: Default value to return if not found
        
    Returns:
        The value, or default if not found
    """
    if entity is None:
        return default
        
    # Dictionary-style access
    if isinstance(entity, dict):
        return entity.get(key, default)
        
    # Object-style access
    return getattr(entity, key, default)

def ensure_entity_value(entity: Any, key: str, default: Any = None) -> None:
    """
    Ensure an entity has a value for the given key, setting a default if not.
    
    Args:
        entity: The entity to check/update
        key: The key or attribute name
        default: Default value to set if not present
    """
    if entity is None:
        return
        
    # Dictionary-style access
    if isinstance(entity, dict):
        if key not in entity or entity[key] is None:
            entity[key] = default
        return
        
    # Object-style access
    if not hasattr(entity, key) or getattr(entity, key) is None:
        setattr(entity, key, default)
