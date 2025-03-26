"""
Utilities for repairing and extracting JSON from text
"""

import re
import json
from loguru import logger
from typing import Any, Dict, List


def repair_json(json_str: str) -> str:
    """
    Attempt to repair common JSON formatting issues
    
    Args:
        json_str: JSON string to repair
        
    Returns:
        Repaired JSON string
    """
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')
    
    # Remove trailing commas in arrays
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Remove trailing commas in objects
    json_str = re.sub(r',\s*}', '}', json_str)
    
    # Fix unquoted property names
    def quote_properties(match):
        return f'"{match.group(1)}":'
    
    json_str = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', quote_properties, json_str)
    
    # Fix missing commas between properties - this is the key issue in the current error
    json_str = re.sub(r'(["}\]])(\s*)(["{\[]|\w+\s*:)', r'\1,\2\3', json_str)
    
    # Fix multiline strings - common in LLM outputs
    json_str = re.sub(r'"\s*\n\s*"', '', json_str)
    json_str = re.sub(r'"\s*\r\n\s*"', '', json_str)
    
    # Fix line breaks within strings
    json_str = re.sub(r'(?<!\\)\\n', '\\\\n', json_str)
    
    # Fix dangling quotation marks
    open_quotes = json_str.count('"')
    if open_quotes % 2 != 0:
        # Add closing quote at the end of the last unclosed string
        pattern = r'("[^"]*?)(\s*[,}\]]|$)'
        match = re.search(pattern, json_str[::-1])
        if match:
            pos = len(json_str) - match.start()
            json_str = json_str[:pos] + '"' + json_str[pos:]
    
    return json_str


def safe_parse_json(json_str: str, default=None) -> Any:
    """
    Safely parse JSON with repair attempts
    
    Args:
        json_str: JSON string to parse
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try simple repair first
        try:
            repaired = repair_json(json_str)
            return json.loads(repaired)
        except json.JSONDecodeError as e2:
            # If repair fails, try more aggressive approaches
            try:
                # Try line-by-line repair
                lines = json_str.split('\n')
                repaired_lines = []
                for i, line in enumerate(lines):
                    # Skip lines that might be causing problems
                    if i == e.lineno - 1 and "Expecting ',' delimiter" in str(e):
                        # Insert missing comma before the problematic character
                        pos = e.colno - 1 if e.colno < len(line) else 0
                        line = line[:pos] + "," + line[pos:]
                    repaired_lines.append(line)
                repaired = repair_json('\n'.join(repaired_lines))
                return json.loads(repaired)
            except Exception:
                # Last resort: try to parse as much as we can
                try:
                    # Extract the JSON structure without the problematic part
                    json_str = re.sub(r'([^,{}\[\]]+)(}|])', r'"fixed_value"\2', json_str)
                    repaired = repair_json(json_str)
                    return json.loads(repaired)
                except:
                    return default


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from text that may contain other content
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        Extracted JSON object or empty dict if not found
    """
    # Try to find JSON object
    patterns = [
        r'(\{[\s\S]*\})',  # Match object
        r'(\[[\s\S]*\])'   # Match array
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            result = safe_parse_json(match)
            if result:
                return result
    
    # If no valid JSON found, try to construct one from the most promising fragment
    try:
        # Look for JSON-like structure with balanced braces
        start_idx = text.find('{')
        if start_idx >= 0:
            # Find the matching closing brace
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        # Found complete JSON object
                        json_candidate = text[start_idx:i+1]
                        result = safe_parse_json(json_candidate)
                        if result:
                            return result
    except Exception:
        pass
    
    return {}
