"""
Utilities for extracting structured JSON from LLM responses
"""

import json
import re
import asyncio
from typing import Dict, Any, Optional, List
from loguru import logger

from finite_monkey.utils.json_repair import extract_json_from_text, safe_parse_json

async def extract_json_with_schema(llm, prompt: str, schema: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """
    Extract structured JSON from an LLM response that conforms to a schema
    
    Args:
        llm: LLM instance to use for generating responses
        prompt: Prompt to send to the LLM
        schema: JSON schema that the response should conform to
        max_retries: Maximum number of retries for malformed JSON
        
    Returns:
        Parsed JSON response that matches the schema
    """
    # Add schema instructions to the prompt if not already included
    if "json" not in prompt.lower() or "format" not in prompt.lower():
        schema_prompt = "\nRespond with a JSON object that conforms to this schema:\n"
        schema_prompt += json.dumps(schema, indent=2)
        prompt = prompt + schema_prompt
    
    # Add a reminder to use proper JSON format
    prompt += "\n\nMake sure your response is valid JSON and follows the provided schema exactly."
    
    for attempt in range(max_retries):
        try:
            # Get response from LLM
            response = await llm.acomplete(prompt)
            response_text = response.text
            
            # Extract JSON from the response
            json_str = extract_json_from_text(response_text)
            
            # Parse and validate against schema
            result = safe_parse_json(json_str)
            
            if result is None or not isinstance(result, dict):
                logger.warning(f"LLM returned invalid JSON (attempt {attempt+1}/{max_retries})")
                if attempt == max_retries - 1:
                    # Last attempt, create empty result with schema structure
                    result = _create_empty_from_schema(schema)
            else:
                # Basic schema validation
                is_valid = _validate_against_schema(result, schema)
                if not is_valid and attempt < max_retries - 1:
                    # Add a more specific instruction about the validation error
                    prompt += f"\n\nYour previous response didn't match the required schema. Please try again and ensure your JSON matches the schema exactly."
                    logger.warning(f"JSON failed schema validation (attempt {attempt+1}/{max_retries})")
                    continue
                
                # Return validated result
                return result
            
        except Exception as e:
            logger.error(f"Error extracting JSON (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                # Last attempt, create empty result with schema structure
                result = _create_empty_from_schema(schema)
    
    # Return the best we could get after all retries
    return result

def _validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Perform basic validation of data against a simplified JSON schema
    
    Args:
        data: Data to validate
        schema: Schema to validate against
        
    Returns:
        True if valid, False otherwise
    """
    # Check for required top-level properties
    if "properties" in schema:
        for prop, prop_schema in schema["properties"].items():
            if "required" in schema and prop in schema["required"] and prop not in data:
                logger.warning(f"Missing required property: {prop}")
                return False
    
    # For arrays, check their items against the schema if specified
    if "type" in schema and schema["type"] == "array" and "items" in schema:
        if not isinstance(data, list):
            logger.warning(f"Expected array but got {type(data)}")
            return False
            
        # Check array items against schema (we'll do basic type checking)
        if "type" in schema["items"]:
            expected_type = schema["items"]["type"]
            for i, item in enumerate(data):
                if expected_type == "object" and not isinstance(item, dict):
                    logger.warning(f"Item {i} should be an object but is {type(item)}")
                    return False
                elif expected_type == "string" and not isinstance(item, str):
                    logger.warning(f"Item {i} should be a string but is {type(item)}")
                    return False
                elif expected_type == "number" and not isinstance(item, (int, float)):
                    logger.warning(f"Item {i} should be a number but is {type(item)}")
                    return False
    
    # For objects, recursively check properties
    if "type" in schema and schema["type"] == "object" and "properties" in schema:
        if not isinstance(data, dict):
            logger.warning(f"Expected object but got {type(data)}")
            return False
        
        # Recursively check properties that exist in the data
        for prop_name, prop_value in data.items():
            if prop_name in schema["properties"]:
                prop_schema = schema["properties"][prop_name]
                if isinstance(prop_value, dict) and isinstance(prop_schema, dict):
                    if not _validate_against_schema(prop_value, prop_schema):
                        return False
    
    return True

def _create_empty_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an empty object that matches the schema structure
    
    Args:
        schema: JSON schema
        
    Returns:
        Empty object that matches the schema
    """
    result = {}
    
    # Handle different schema types
    if "type" in schema:
        schema_type = schema["type"]
        
        if schema_type == "object" and "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if "type" in prop_schema:
                    if prop_schema["type"] == "object":
                        result[prop] = _create_empty_from_schema(prop_schema)
                    elif prop_schema["type"] == "array":
                        result[prop] = []
                    elif prop_schema["type"] == "string":
                        result[prop] = ""
                    elif prop_schema["type"] == "number" or prop_schema["type"] == "integer":
                        result[prop] = 0
                    elif prop_schema["type"] == "boolean":
                        result[prop] = False
                    else:
                        result[prop] = None
        
        elif schema_type == "array":
            return []
    
    return result
