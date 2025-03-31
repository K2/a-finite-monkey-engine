"""
Utility module for validating and testing regular expressions.
"""
import re
from typing import Optional, List, Tuple, Any

def validate_regex(pattern: str) -> Tuple[bool, Optional[str]]:
    """
    Validates if a regex pattern is valid.
    
    Args:
        pattern: The regex pattern to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        re.compile(pattern)
        
        # Additional logical validation
        if pattern.startswith('$') and len(pattern) > 1:
            return False, "Pattern starts with $ (end anchor), making subsequent characters unreachable"
        
        return True, None
    except re.error as e:
        return False, f"Invalid regex pattern: {str(e)}"

def test_regex_match(pattern: str, test_string: str) -> Tuple[bool, Optional[List[Any]], Optional[str]]:
    """
    Tests if a regex pattern matches a string and returns the matches.
    
    Args:
        pattern: The regex pattern to test
        test_string: The string to match against
        
    Returns:
        Tuple of (is_match, matches or None, error_message or None)
    """
    is_valid, error = validate_regex(pattern)
    if not is_valid:
        return False, None, error
    
    try:
        matches = re.findall(pattern, test_string)
        return len(matches) > 0, matches, None
    except Exception as e:
        return False, None, f"Error matching pattern: {str(e)}"
