"""
Utility functions for Guidance integration.
"""
import re
from typing import Tuple, Optional
from loguru import logger

# Check for guidance availability
try:
    import guidance
    GUIDANCE_AVAILABLE = True
except ImportError:
    GUIDANCE_AVAILABLE = False


def is_guidance_available() -> bool:
    """Check if Guidance library is available"""
    return GUIDANCE_AVAILABLE


def get_llamaindex_version() -> Tuple[int, int, int]:
    """
    Get the LlamaIndex version as a tuple.
    
    Returns:
        Version tuple (major, minor, patch) or (0, 0, 0) if not found
    """
    try:
        import llama_index
        version_str = getattr(llama_index, "__version__", "0.0.0")
        parts = version_str.split(".")
        
        # Ensure we have at least 3 parts
        while len(parts) < 3:
            parts.append("0")
            
        # Convert to integers
        return tuple(int(part) for part in parts[:3])
    except (ImportError, AttributeError):
        try:
            # Try core module
            import llama_index.core
            version_str = getattr(llama_index.core, "__version__", "0.0.0")
            parts = version_str.split(".")
            
            # Ensure we have at least 3 parts
            while len(parts) < 3:
                parts.append("0")
                
            # Convert to integers
            return tuple(int(part) for part in parts[:3])
        except (ImportError, AttributeError):
            return (0, 0, 0)


def ensure_handlebars_format(template: str) -> str:
    """
    Ensure a template is in handlebars format.
    
    Args:
        template: Template string (Python format or handlebars format)
        
    Returns:
        Template in handlebars format
    """
    # Already in handlebars format
    if "{{" in template and "}}" in template:
        return template
        
    # Try to use LlamaIndex converter
    try:
        # Try different import paths based on version
        try:
            from llama_index.core.prompts.guidance_utils import convert_to_handlebars
            return convert_to_handlebars(template)
        except ImportError:
            from llama_index.prompts.guidance_utils import convert_to_handlebars
            return convert_to_handlebars(template)
    except ImportError:
        logger.warning("Could not import guidance utils, using simple regex conversion")
        
    # Simple fallback implementation
    return re.sub(r'\{([^{}]*)\}', r'{{\1}}', template)
