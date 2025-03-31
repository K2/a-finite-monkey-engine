"""
Helper utilities for API and entity access that work across different data formats.
"""
from typing import Any, Dict, List, Optional, Union, TypeVar, Type
import inspect

T = TypeVar('T')

def get_entity_attribute(entity: Any, attribute: str, default: Any = None) -> Any:
    """
    Get an attribute from an entity regardless of whether it's a dictionary or object.
    
    Args:
        entity: The entity to get the attribute from (dict or object)
        attribute: The attribute name to retrieve
        default: Default value to return if attribute not found
        
    Returns:
        The attribute value or default if not found
    """
    if entity is None:
        return default
        
    # Handle dictionary-style entities
    if isinstance(entity, dict):
        return entity.get(attribute, default)
    
    # Handle object-style entities
    return getattr(entity, attribute, default)

def get_entity_attributes(entity: Any, attributes: List[str], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get multiple attributes from an entity in a single call.
    
    Args:
        entity: The entity to get attributes from (dict or object)
        attributes: List of attribute names to retrieve
        defaults: Optional dictionary mapping attribute names to default values
        
    Returns:
        Dictionary of attribute name -> value
    """
    if defaults is None:
        defaults = {}
        
    result = {}
    for attr in attributes:
        result[attr] = get_entity_attribute(entity, attr, defaults.get(attr))
        
    return result

def entity_to_dict(entity: Any) -> Dict[str, Any]:
    """
    Convert any entity (dict or object) to a dictionary.
    
    Args:
        entity: Entity to convert
        
    Returns:
        Dictionary representation of the entity
    """
    if entity is None:
        return {}
        
    if isinstance(entity, dict):
        return entity
    
    # Handle Pydantic models
    if hasattr(entity, "model_dump"):
        return entity.model_dump()
    
    # Handle dataclasses
    if hasattr(entity, "__dataclass_fields__"):
        import dataclasses
        return dataclasses.asdict(entity)
    
    # Fallback to __dict__ for normal objects
    # Get all attributes that don't start with underscore
    result = {}
    for attr in dir(entity):
        if not attr.startswith('_') and not inspect.ismethod(getattr(entity, attr)):
            try:
                result[attr] = getattr(entity, attr)
            except (AttributeError, Exception):
                pass
    
    return result
