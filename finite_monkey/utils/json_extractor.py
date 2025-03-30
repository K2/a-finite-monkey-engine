"""
Utilities for extracting structured JSON from LLM responses
"""

import json
import re
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple
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
            json_obj, debug_info = extract_json_from_complex_response(response_text)
            logger.debug(debug_info)
            
            if json_obj is None or not isinstance(json_obj, dict):
                logger.warning(f"LLM returned invalid JSON (attempt {attempt+1}/{max_retries})")
                if attempt == max_retries - 1:
                    # Last attempt, create empty result with schema structure
                    json_obj = _create_empty_from_schema(schema)
            else:
                # Basic schema validation
                is_valid = _validate_against_schema(json_obj, schema)
                if not is_valid and attempt < max_retries - 1:
                    # Add a more specific instruction about the validation error
                    prompt += f"\n\nYour previous response didn't match the required schema. Please try again and ensure your JSON matches the schema exactly."
                    logger.warning(f"JSON failed schema validation (attempt {attempt+1}/{max_retries})")
                    continue
                
                # Return validated result
                return json_obj
            
        except Exception as e:
            logger.error(f"Error extracting JSON (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                # Last attempt, create empty result with schema structure
                json_obj = _create_empty_from_schema(schema)
    
    # Return the best we could get after all retries
    return json_obj

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

def extract_json_from_complex_response(response: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract a JSON object from a potentially complex response format.
    Handles JSON embedded in markdown, double encoding, and other edge cases.
    
    Args:
        response: The response string to parse
        
    Returns:
        Tuple of (parsed JSON object or None if extraction failed, debug info)
    """
    debug_info = []
    debug_info.append(f"Input type: {type(response)}")
    debug_info.append(f"Input length: {len(response)}")
    debug_info.append(f"Input preview: {response[:100]}...")
    
    if not isinstance(response, str):
        debug_info.append("Not a string, returning None")
        return None, "\n".join(debug_info)
    
    # Strip whitespace
    response = response.strip()
    debug_info.append(f"After strip length: {len(response)}")
    
    # 1. Try direct JSON parsing first
    try:
        result = json.loads(response)
        debug_info.append("Direct JSON parsing successful")
        if isinstance(result, dict):
            return result, "\n".join(debug_info)
    except json.JSONDecodeError as e:
        debug_info.append(f"Direct JSON parsing failed: {e}")
    
    # 2. Try unescaping if it looks like an escaped JSON string
    if response.startswith('"') and response.endswith('"'):
        try:
            unescaped = json.loads(response)
            debug_info.append("Unescaped double-quoted string")
            if isinstance(unescaped, str):
                try:
                    result = json.loads(unescaped)
                    debug_info.append("Successfully parsed unescaped content as JSON")
                    if isinstance(result, dict):
                        return result, "\n".join(debug_info)
                except json.JSONDecodeError as e:
                    debug_info.append(f"Failed to parse unescaped content: {e}")
        except json.JSONDecodeError as e:
            debug_info.append(f"Failed to unescape string: {e}")
    
    # 3. Try cleaning up common escaping issues
    if '\\' in response:
        cleaned = response.replace('\\\\', '\\').replace('\\"', '"')
        debug_info.append("Cleaned backslashes")
        try:
            result = json.loads(cleaned)
            debug_info.append("Successfully parsed cleaned string")
            if isinstance(result, dict):
                return result, "\n".join(debug_info)
        except json.JSONDecodeError as e:
            debug_info.append(f"Failed to parse cleaned string: {e}")
    
    # 4. Look for markdown code blocks containing JSON
    code_block_matches = re.findall(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
    if code_block_matches:
        debug_info.append(f"Found {len(code_block_matches)} markdown code blocks")
        for i, code_block in enumerate(code_block_matches):
            try:
                result = json.loads(code_block)
                debug_info.append(f"Successfully parsed code block {i+1}")
                if isinstance(result, dict):
                    return result, "\n".join(debug_info)
            except json.JSONDecodeError as e:
                debug_info.append(f"Failed to parse code block {i+1}: {e}")
    
    # 5. Try to extract JSON-like patterns using regex
    json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    matches = re.findall(json_pattern, response)
    if matches:
        debug_info.append(f"Found {len(matches)} potential JSON objects using regex")
        for i, match in enumerate(matches):
            try:
                result = json.loads(match)
                debug_info.append(f"Successfully parsed regex match {i+1}")
                if isinstance(result, dict) and "flows" in result:
                    return result, "\n".join(debug_info)
            except json.JSONDecodeError as e:
                debug_info.append(f"Failed to parse regex match {i+1}: {e}")
    
    # 6. Last resort: Find anything that looks like our expected structure
    if '"flows"' in response or "'flows'" in response:
        debug_info.append("Found 'flows' key, attempting targeted extraction")
        
        # Try to extract a substring from 'flows' to the end of what looks like a dict
        try:
            flows_start = response.find('"flows"')
            if flows_start == -1:
                flows_start = response.find("'flows'")
            
            if flows_start != -1:
                # Find the opening bracket before "flows"
                obj_start = response.rfind('{', 0, flows_start)
                if obj_start != -1:
                    # Find the matching closing bracket
                    depth = 1
                    obj_end = -1
                    
                    for i in range(obj_start + 1, len(response)):
                        if response[i] == '{':
                            depth += 1
                        elif response[i] == '}':
                            depth -= 1
                            if depth == 0:
                                obj_end = i + 1
                                break
                    
                    if obj_end != -1:
                        potential_json = response[obj_start:obj_end]
                        try:
                            result = json.loads(potential_json)
                            debug_info.append("Successfully extracted JSON object containing 'flows'")
                            return result, "\n".join(debug_info)
                        except json.JSONDecodeError as e:
                            debug_info.append(f"Failed to parse extracted JSON: {e}")
        except Exception as e:
            debug_info.append(f"Error in targeted extraction: {e}")
    
    # Nothing worked
    debug_info.append("All extraction methods failed")
    return None, "\n".join(debug_info)
