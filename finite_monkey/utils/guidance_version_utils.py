"""
Utilities for detecting and working with different versions of LlamaIndex Guidance integration.
"""
import re
import asyncio
import json
import random
from typing import Tuple, Optional, Any, Type, Callable, Union, Dict
from pydantic import BaseModel
from loguru import logger
import httpx

# Check for guidance availability
try:
    import guidance
    import inspect
    GUIDANCE_AVAILABLE = True
    
    # Instead of just checking for llms attribute, inspect what's actually available
    # in the installed version
    GUIDANCE_VERSION = getattr(guidance, "__version__", "0.0.0")
    logger.info(f"Found Guidance version: {GUIDANCE_VERSION}")
    
    # Guidance 0.1.x structure is different from newer versions
    if hasattr(guidance, "llms"):
        GUIDANCE_LLMS_AVAILABLE = True
        logger.info("Using guidance.llms module")
    elif hasattr(guidance, "LLM"):
        # Newer pattern for guidance
        GUIDANCE_LLMS_AVAILABLE = False
        logger.info("Using newer guidance.LLM pattern")
    else:
        GUIDANCE_LLMS_AVAILABLE = False
        logger.info("No recognized Guidance LLM structure found")
except ImportError:
    GUIDANCE_AVAILABLE = False
    GUIDANCE_LLMS_AVAILABLE = False
    logger.warning("Guidance library not available. Install with 'pip install guidance>=0.1.16'")

# Semaphore to limit concurrent Ollama calls
OLLAMA_SEMAPHORE = asyncio.Semaphore(2)  # Allow max 2 concurrent calls to Ollama

class GuidanceProgramWrapper:
    """Wrapper for Guidance programs with consistent interface"""
    
    def __init__(
        self, 
        program: Any, 
        output_cls: Type[BaseModel],
        verbose: bool = False,
        fallback_fn: Optional[Callable] = None
    ):
        """Initialize with a Guidance program"""
        self.program = program
        self.output_cls = output_cls
        self.verbose = verbose
        self.fallback_fn = fallback_fn
    
    async def __call__(self, **kwargs) -> Union[BaseModel, Dict[str, Any]]:
        """Call the Guidance program with arguments"""
        if self.verbose:
            logger.debug(f"Executing Guidance program with parameters: {kwargs}")
            
        # Track retries with exponential backoff
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Critical fix: Ensure DirectTemplateHandler is properly awaited
                if isinstance(self.program, DirectTemplateHandler):
                    result = await self.program(**kwargs)  # Use await here
                elif hasattr(self.program, "__call__") and inspect.iscoroutinefunction(self.program.__call__):
                    result = await self.program(**kwargs)  # Use await for any coroutine
                else:
                    result = self.program(**kwargs)  # Regular synchronous call
                    
                # Add debug logging to see what's being returned
                logger.debug(f"Raw LLM result type: {type(result)}")
                logger.debug(f"Raw LLM result: {str(result)[:500]}...")
                    
                # Process the result
                if self.output_cls and not isinstance(result, self.output_cls):
                    try:
                        if isinstance(result, dict):
                            logger.debug(f"Converting dict result to {self.output_cls.__name__}")
                            # Log available keys in the result dict
                            logger.debug(f"Result keys: {result.keys() if hasattr(result, 'keys') else 'no keys'}")
                            
                            # Improved check to detect empty or just code markers
                            is_empty_result = ('result' in result and (
                                not result['result'] or 
                                isinstance(result['result'], str) and (
                                    result['result'].strip() == '```' or
                                    result['result'].strip() == '```\n```' or
                                    '```' in result['result'] and len(result['result'].strip()) < 10
                                )
                            ))
                            
                            # Enhanced flow mapping:
                            # 1. If result looks like a single flow, wrap it in a flows array
                            if 'name' in result and 'description' in result and 'flows' not in result:
                                logger.info("Converting single flow result to flows array")
                                return self.output_cls(flows=[result])
                                
                            # 2. If we have a 'flows' key, use it directly
                            elif 'flows' in result:
                                return self.output_cls(**result)
                                
                            # 3. Return empty flows for empty results
                            elif is_empty_result:
                                logger.warning(f"Detected empty code block result, returning empty flows")
                                return self.output_cls(flows=[])
                                
                            # 4. Wrap any other result structure
                            else:
                                # Create a safe fallback
                                return self.output_cls(flows=[]) if 'error' in result else self.output_cls(flows=[result])
                        else:
                            return result
                    except Exception as e:
                        logger.error(f"Error converting result to {self.output_cls.__name__}: {e}")
                        if self.fallback_fn:
                            return self.fallback_fn(**kwargs)
                        return result
                return result
                
            except httpx.HTTPStatusError as e:
                # Handle HTTP status errors (like 500)
                status_code = getattr(e.response, 'status_code', None)
                logger.warning(f"HTTP error during LLM call (attempt {retry_count+1}/{max_retries+1}): {status_code} - {str(e)}")
                last_error = e
                
                # Check if it might be a tokenizer issue
                if status_code == 500 and _check_for_tokenizer_issue(kwargs):
                    logger.warning("Possible tiktoken tokenizer issue detected. Reducing input size.")
                    kwargs = _reduce_input_size(kwargs)
                
                retry_count += 1
                if retry_count <= max_retries:
                    # Exponential backoff with jitter
                    backoff_time = (2 ** retry_count) + (random.random() * 0.5)
                    logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded.")
                    if self.fallback_fn:
                        try:
                            return self.fallback_fn(**kwargs)
                        except Exception as fallback_e:
                            logger.error(f"Fallback function also failed: {fallback_e}")
                    return {}
            
            except Exception as e:
                logger.error(f"Error executing guidance program: {e}")
                last_error = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    # Exponential backoff with jitter
                    backoff_time = (2 ** retry_count) + (random.random() * 0.5)
                    logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded.")
                    if self.fallback_fn:
                        try:
                            return self.fallback_fn(**kwargs)
                        except Exception as fallback_e:
                            logger.error(f"Fallback function also failed: {fallback_e}")
                    return {}
        
        # If we got here, all retries failed
        logger.error(f"All retries failed. Last error: {last_error}")
        return {}

# Add these utility functions for handling tokenizer issues

def _check_for_tokenizer_issue(data: Dict[str, Any]) -> bool:
    """
    Check if the input data might cause tokenizer issues (likely too large)
    
    Args:
        data: The input data to check
        
    Returns:
        True if there might be a tokenizer issue
    """
    # Try to estimate the total size of the input
    total_size = 0
    
    for key, value in data.items():
        if isinstance(value, str):
            # Rough estimate: 1 token per 4 characters for English text
            total_size += len(value) // 4
        elif isinstance(value, (list, dict)):
            # Rough estimate for complex objects
            total_size += len(str(value)) // 4
    
    # Most models have context limits around 8K-32K tokens
    # If we're getting close to that, we might have issues
    if total_size > 7000:  # Conservative threshold
        logger.warning(f"Large input detected: ~{total_size} tokens. May cause tokenizer issues.")
        return True
    
    return False

def _reduce_input_size(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce the input size to avoid tokenizer issues
    
    Args:
        data: The input data to reduce
        
    Returns:
        Reduced input data
    """
    reduced_data = {}
    
    for key, value in data.items():
        if isinstance(value, str) and len(value) > 4000:
            # Truncate long strings (but keep the beginning and some of the end)
            beginning = value[:3000]
            end = value[-1000:]
            reduced_data[key] = f"{beginning}\n\n...[content truncated to avoid tokenizer issues]...\n\n{end}"
            logger.info(f"Truncated input field '{key}' from {len(value)} to {len(reduced_data[key])} characters")
        else:
            reduced_data[key] = value
    
    return reduced_data

class DirectTemplateHandler:
    """Direct handler for templates with LLM integration"""
    
    def __init__(self, template, llm):
        """Initialize with a template and LLM"""
        self.template = template
        self.llm = llm
        # Pre-compile regex patterns for better performance
        import re
        self.if_pattern = re.compile(r'{{#if\s+([^}]+)}}(.*?){{\/if}}', re.DOTALL)
        self.each_pattern = re.compile(r'{{#each\s+([^}]+)}}(.*?){{\/each}}', re.DOTALL)
        self.json_pattern = re.compile(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}')
        self.this_property_pattern = re.compile(r'{{this\.([^}]+)}}')

    async def __call__(self, **kwargs):
        """Main entry point for processing templates and generating results - MUST BE AWAITED"""
        retry_count = 0
        max_retries = 3
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # Process template with variables and conditionals
                filled_template = self._process_template(kwargs)
                
                # Dynamic method resolution - will adapt to method name changes
                call_method = self._resolve_method('_call_llm', ['call_llm', 'complete', 'generate', 'call_model'])
                if call_method:
                    logger.debug(f"Using resolved method: {call_method.__name__}")
                    response = await call_method(filled_template)
                else:
                    # Fallback implementation if method not found
                    logger.warning("Method '_call_llm' not found, using fallback implementation")
                    response = await self._fallback_call_llm(filled_template)
                
                # Dynamic resolution for response processing
                process_method = self._resolve_method('_process_response', ['process_response', 'extract_data'])
                if process_method:
                    return process_method(response)
                else:
                    logger.warning("Method '_process_response' not found, using fallback implementation")
                    return self._fallback_process_response(response)
                
            except httpx.HTTPStatusError as e:
                # Handle HTTP status errors (like 500)
                status_code = getattr(e.response, 'status_code', None)
                logger.warning(f"HTTP error from LLM service (attempt {retry_count+1}/{max_retries+1}): {status_code} - {str(e)}")
                last_error = e
                
                retry_count += 1
                if retry_count <= max_retries:
                    # Exponential backoff with jitter
                    backoff_time = (2 ** retry_count) + (random.random() * 0.5)
                    logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded.")
                    return {"error": f"Max retries exceeded: {str(e)}"}
            
            except Exception as e:
                logger.error(f"Error in template handling: {e}")
                logger.exception("Stack trace:")
                last_error = e
                
                retry_count += 1
                if retry_count <= max_retries:
                    # Exponential backoff with jitter
                    backoff_time = (2 ** retry_count) + (random.random() * 0.5)
                    logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded.")
                    return {"error": f"Max retries exceeded: {str(e)}"}
        
        # If we get here, all retries failed
        return {"error": f"All retries failed. Last error: {str(last_error)}"}

    def _resolve_method(self, primary_name, alternative_names=None):
        """
        Dynamically resolve a method by name with fallbacks.
        
        Args:
            primary_name: The preferred method name
            alternative_names: List of alternative method names to try
            
        Returns:
            Method object or None if not found
        """
        # Try the primary method name first
        if hasattr(self, primary_name) and callable(getattr(self, primary_name)):
            return getattr(self, primary_name)
        
        # Try alternative names if provided
        if alternative_names:
            for name in alternative_names:
                if hasattr(self, name) and callable(getattr(self, name)):
                    logger.info(f"Using alternative method '{name}' instead of '{primary_name}'")
                    return getattr(self, name)
        
        # Nothing found
        return None

    async def _fallback_call_llm(self, template):
        """
        Fallback implementation for calling the LLM if _call_llm method is not found.
        
        Args:
            template: The processed template to send to the LLM
            
        Returns:
            LLM response
        """
        logger.warning("Using fallback LLM call implementation")
        
        # Try common LLM interface patterns
        try:
            # Try acomplete first (async complete)
            if hasattr(self.llm, "acomplete") and callable(self.llm.acomplete):
                logger.debug("Using LLM.acomplete method")
                return await self.llm.acomplete(template)
                
            # Try complete with inspection for async
            elif hasattr(self.llm, "complete") and callable(self.llm.complete):
                logger.debug("Using LLM.complete method")
                if inspect.iscoroutinefunction(self.llm.complete):
                    return await self.llm.complete(template)
                else:
                    return self.llm.complete(template)
                    
            # Try generate methods
            elif hasattr(self.llm, "agenerate") and callable(self.llm.agenerate):
                logger.debug("Using LLM.agenerate method")
                return await self.llm.agenerate(template)
            elif hasattr(self.llm, "generate") and callable(self.llm.generate):
                logger.debug("Using LLM.generate method")
                if inspect.iscoroutinefunction(self.llm.generate):
                    return await self.llm.generate(template)
                else:
                    return self.llm.generate(template)
                    
            # Try direct callable
            elif callable(self.llm):
                logger.debug("Using LLM as direct callable")
                if inspect.iscoroutinefunction(self.llm):
                    return await self.llm(template)
                else:
                    return self.llm(template)
                    
            # Last resort - return empty response with error
            else:
                logger.error(f"No usable method found on LLM: {type(self.llm)}")
                return {"error": f"No method to call LLM of type {type(self.llm)}"}
        
        except Exception as e:
            logger.error(f"Error in fallback LLM call: {e}")
            return {"error": f"LLM call failed: {str(e)}"}

    def _fallback_process_response(self, response):
        """
        Fallback implementation for processing responses if _process_response method is not found.
        
        Args:
            response: The raw LLM response
            
        Returns:
            Processed response dictionary
        """
        logger.warning("Using fallback response processing implementation")
        
        try:
            # Extract text content from various response formats
            if response is None:
                return {"result": ""}
                
            if hasattr(response, "text"):
                text = response.text
            elif hasattr(response, "content"):
                text = response.content
            elif hasattr(response, "completion"):
                text = response.completion
            elif hasattr(response, "message") and hasattr(response.message, "content"):
                text = response.message.content
            elif isinstance(response, dict):
                for key in ["text", "content", "response", "result", "output", "generated_text"]:
                    if key in response:
                        text = response[key]
                        break
                else:
                    text = str(response)
            elif isinstance(response, str):
                text = response
            else:
                text = str(response)
            
            # Try to extract JSON if present
            try:
                import json
                import re
                
                # Simple pattern to find JSON objects
                json_pattern = re.compile(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}')
                match = json_pattern.search(text)
                
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict):
                        return parsed
            except Exception:
                pass
            
            # Return as simple result if no structured data found
            return {"result": text}
        
        except Exception as e:
            logger.error(f"Error in fallback response processing: {e}")
            return {"error": str(e), "result": str(response) if response else ""}

    def _process_template(self, data):
        """Process a template with variables and control structures"""
        # Start with basic variable replacement for simple templates
        processed = self.template
        
        # Replace any variables directly for simple cases
        for key, value in data.items():
            placeholder = "{{" + key + "}}"
            # Only replace if the placeholder exists in the template
            if placeholder in processed:
                processed = processed.replace(placeholder, str(value))
        
        # Check if we have any control structures that need processing
        if "{{#if" in processed or "{{#each" in processed:
            # Use recursive processing for complex templates
            processed = self._recursive_process(processed, data)
            
        logger.debug(f"Processed template: {processed[:100]}...")
        return processed
    
    def _recursive_process(self, template, data):
        """Recursively process a template with all control structures"""
        # First, process all conditionals
        template = self._process_conditionals(template, data)
        
        # Then process all loops
        # This will recursively process nested loops and conditionals
        def replace_each(match):
            var_name = match.group(1).strip()
            inner_template = match.group(2)
            
            # Check if the variable exists and is iterable
            if var_name in data and hasattr(data[var_name], '__iter__') and not isinstance(data[var_name], str):
                result = []
                for item in data[var_name]:
                    # Create a context for this iteration
                    context = {"this": item}
                    
                    # For dict items, add all properties to the context
                    if isinstance(item, dict):
                        context.update(item)
                    
                    # Handle this.property references explicitly before recursion
                    item_template = inner_template
                    if isinstance(item, dict):
                        def replace_this_prop(prop_match):
                            prop_name = prop_match.group(1)
                            if prop_name in item:
                                return str(item[prop_name])
                            return f"{{{{this.{prop_name}}}}}"
                        
                        # Add logging here
                        logger.debug(f"Item before property replacement: {item}")
                        logger.debug(f"Item template before property replacement: {item_template}")
                        
                        item_template = self.this_property_pattern.sub(replace_this_prop, item_template)
                        
                        # Add logging here
                        logger.debug(f"Item template after property replacement: {item_template}")
                    
                    # Process the inner template with this context
                    processed_item = self._recursive_process(item_template, context)
                    result.append(processed_item)
                
                return "".join(result)
            return ""
        
        # Process all loops in the template
        template = self.each_pattern.sub(replace_each, template)
        
        # Finally, replace all variables
        for key, value in data.items():
            # Handle all value types for variable replacement
            placeholder = "{{" + key + "}}"
            if placeholder in template:
                template = template.replace(placeholder, str(value))
                
        # Special handling for {{this}} reference
        if "this" in data:
            template = template.replace("{{this}}", str(data["this"]))
            
        return template
    
    def _process_conditionals(self, template, data):
        """Process conditional blocks in the template"""
        def replace_if(match):
            condition = match.group(1).strip()
            inner_template = match.group(2)
            
            # Evaluate the condition
            try:
                # Attempt to access the variable directly from the data
                if condition in data and data[condition]:
                    return inner_template
                # If not directly present, evaluate as a boolean expression
                elif eval(condition, {}, data):
                    return inner_template
                else:
                    return ""
            except Exception as e:
                logger.warning(f"Could not evaluate condition '{condition}': {e}")
                return ""
        
        # Process all conditionals in the template
        template = self.if_pattern.sub(replace_if, template)
        return template
    
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from a text response, with robust handling for malformed JSON.
        
        Args:
            text (str): The text response to extract data from.
        
        Returns:
            Dict[str, Any]: A dictionary containing the extracted data.
        """
        if not text:
            return {}
        
        try:
            import json
            import re
            
            # Log the raw text for debugging
            logger.debug(f"Extracting structured data from text: {text[:200]}...")
            
            # Clean up common JSON formatting errors
            text_to_parse = text
            
            # Look for JSON objects with regex first
            json_matches = self.json_pattern.findall(text if isinstance(text, str) else str(text))
            
            for match in json_matches:
                # Attempt to fix common JSON errors before parsing
                fixed_match = self._fix_json_formatting(match)
                
                try:
                    parsed = json.loads(fixed_match)
                    # Make sure it's a dict before returning
                    if isinstance(parsed, dict):
                        logger.debug(f"Successfully parsed JSON with length {len(fixed_match)}")
                        
                        # Verify and fix any malformed flow objects
                        if 'flows' in parsed and isinstance(parsed['flows'], list):
                            fixed_flows = []
                            for flow in parsed['flows']:
                                if isinstance(flow, dict):
                                    # Fix any malformed actors arrays or other fields
                                    if 'actors' in flow and not isinstance(flow['actors'], list):
                                        logger.warning(f"Fixing malformed actors in flow: {flow.get('name', 'unnamed')}")
                                        flow['actors'] = ["Unknown actor"]
                                    
                                    # Ensure all required fields are present
                                    for field in ['name', 'description', 'steps', 'functions']:
                                        if field not in flow or not flow[field]:
                                            logger.warning(f"Adding missing {field} in flow: {flow.get('name', 'unnamed')}")
                                            if field in ['steps', 'functions', 'actors']:
                                                flow[field] = []
                                            else:
                                                flow[field] = f"Unknown {field}"
                                    
                                    fixed_flows.append(flow)
                            
                            parsed['flows'] = fixed_flows
                        
                        return parsed
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error: {e} for match: {fixed_match[:50]}...")
                    continue
            
            # If no valid JSON was found, try to extract flows manually
            if "flows" in text and ("name" in text or "description" in text):
                logger.info("Attempting manual flow extraction from text")
                return self._manual_flow_extraction(text)
            
            # If we couldn't find JSON, look for key-value pairs
            lines = text.split("\n") if isinstance(text, str) else str(text).split("\n")
            result = {}
            
            for line in lines:
                # Try to match "key: value" pattern
                kv_match = re.match(r'^\s*"?([^":]+)"?\s*:\s*(.+)$', line)
                if kv_match:
                    key, value = kv_match.groups()
                    result[key.strip()] = value.strip()
            
            if result:
                return result
            
            # Return the text as a simple result if nothing else worked
            return {"result": text if isinstance(text, str) else str(text)}
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            return {"error": str(e)}

    def _fix_json_formatting(self, json_str: str) -> str:
        """
        Fix common JSON formatting issues.
        
        Args:
            json_str: The JSON string to fix
        
        Returns:
            Fixed JSON string
        """
        import re
        
        # Get only the first JSON-like object if there are multiple
        first_brace_idx = json_str.find('{')
        if first_brace_idx > 0:
            json_str = json_str[first_brace_idx:]
        
        # Fix unbalanced quotes
        quote_count = json_str.count('"')
        if quote_count % 2 != 0:
            logger.warning(f"Unbalanced quotes detected in JSON: {quote_count} quotes")
            # Find unterminated strings and fix them
            pattern = r'"[^"]*$'
            json_str = re.sub(pattern, '', json_str)
        
        # Fix arrays with improper formatting that include colons inside
        # This pattern detects ["item1", "item2", "key": "value", ...] issues
        actors_pattern = r'(\["[^"]*"(?:\s*,\s*"[^"]*")*)\s*,\s*("[^"]*"\s*:\s*)'
        json_str = re.sub(actors_pattern, r'\1],\2', json_str)
        
        # Fix arrays mixing with objects (common in malformed actors arrays)
        # Find actors arrays that don't close properly
        actors_fix_pattern = r'"actors"\s*:\s*\[(.*?),\s*"([^"]+)"\s*:'
        json_str = re.sub(actors_fix_pattern, r'"actors": [\1], "\2":', json_str)
        
        # Fix unbalanced brackets if needed
        open_brace = json_str.count('{')
        close_brace = json_str.count('}')
        if open_brace > close_brace:
            json_str += "}" * (open_brace - close_brace)
        elif close_brace > open_brace:
            json_str = json_str[:json_str.rfind('}')] + "}" * (open_brace)
        
        # Fix unbalanced square brackets
        open_bracket = json_str.count('[')
        close_bracket = json_str.count(']')
        if open_bracket > close_bracket:
            json_str += "]" * (open_bracket - close_bracket)
        
        return json_str

    def _manual_flow_extraction(self, text: str) -> Dict[str, Any]:
        """
        Manually extract flows from text when JSON parsing fails.
        
        Args:
            text: Text containing flow data
        
        Returns:
            Dict with flows array
        """
        import re
        
        flows = []
        
        # Try to find flow objects based on common patterns
        flow_blocks = re.findall(r'(?:"name"\s*:\s*"([^"]+)".*?"description"\s*:\s*"([^"]+)".*?(?:"steps"|"functions"|"actors").*?)(?:"name"|$)', text, re.DOTALL)
        
        for i, (name, description) in enumerate(flow_blocks):
            try:
                # Extract steps if present
                steps = []
                steps_match = re.search(rf'"steps"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                if steps_match:
                    steps_text = steps_match.group(1)
                    steps = [s.strip(' "') for s in re.findall(r'"([^"]+)"', steps_text)]
                
                # Extract functions if present
                functions = []
                functions_match = re.search(rf'"functions"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                if functions_match:
                    functions_text = functions_match.group(1)
                    functions = [f.strip(' "') for f in re.findall(r'"([^"]+)"', functions_text)]
                
                # Create a flow object
                flow = {
                    "name": name,
                    "description": description,
                    "steps": steps,
                    "functions": functions,
                    "actors": ["Unknown actor"],  # Default value since actors are often problematic
                    "flow_type": "unknown"  # Default value
                }
                
                # Add to flows list
                flows.append(flow)
                
            except Exception as e:
                logger.error(f"Error in manual flow extraction for flow {i}: {e}")
        
        # If we found any flows, return them; otherwise, return original text
        if flows:
            return {"flows": flows}
        else:
            return {"result": text}

async def create_guidance_program(
    output_cls: Type[BaseModel],
    prompt_template: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    fallback_fn: Optional[Callable] = None,
    verbose: bool = False,
    **kwargs
) -> Optional[GuidanceProgramWrapper]:
    """Create a Guidance program for structured output."""
    if not GUIDANCE_AVAILABLE:
        logger.warning("Guidance library not available. Cannot create Guidance program.")
        return None
        
    try:
        # Create LLM
        llm = await _create_llm(model, provider)
        if not llm:
            logger.error("Failed to create LLM for Guidance")
            return None
            
        # Log the LLM we're using
        llm_info = f"provider={getattr(llm, 'provider', 'unknown')}, model={getattr(llm, 'model', getattr(llm, 'model_name', 'unknown'))}"
        logger.info(f"Creating Guidance program with {llm_info}")
        
        # Ensure the template is in the right format
        template = ensure_handlebars_format(prompt_template)
        
        # For Guidance 0.1.x (like 0.1.16), we need to use a different approach
        guidance_version = GUIDANCE_VERSION
        logger.debug(f"Guidance version: {guidance_version}")
        
        # Initialize program to None
        program = None
        
        # Special handling for Guidance 0.1.16 based on its specific structure
        if guidance_version == "0.1.16":
            logger.info("Using direct template handler for Guidance 0.1.16")
            program = DirectTemplateHandler(template, llm)
            return GuidanceProgramWrapper(program, output_cls, verbose=verbose, fallback_fn=fallback_fn)
            
        # For other versions, try different approaches
        # ... (rest of the function)
            
    except Exception as e:
        logger.error(f"Error creating Guidance program: {e}")
        logger.exception("Full traceback:")
        return None

def get_llama_index_version() -> Tuple[int, int, int]:
    """
    Get the LlamaIndex version as a tuple.
    
    Returns:
        Version tuple (major, minor, patch) or (0, 0, 0) if not found
    """
    try:
        # Try direct import
        import llama_index
        version_str = getattr(llama_index, "__version__", "0.0.0")
        # Extract version numbers
        parts = version_str.split(".")
        while len(parts) < 3:
            parts.append("0")
        return tuple(int(p) for p in parts[:3])
    except (ImportError, AttributeError):
        try:
            # Try core module
            import llama_index.core
            version_str = getattr(llama_index.core, "__version__", "0.0.0")
            # Extract version numbers
            parts = version_str.split(".")
            while len(parts) < 3:
                parts.append("0")
            return tuple(int(p) for p in parts[:3])
        except (ImportError, AttributeError):
            return (0, 0, 0)


def ensure_handlebars_format(template: str) -> str:
    """
    Ensure a template is in handlebars format for Guidance.
    
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
            try:
                from llama_index.prompts.guidance_utils import convert_to_handlebars
                return convert_to_handlebars(template)
            except ImportError:
                pass  # Fall through to fallback implementation
    except Exception as e:
        logger.warning(f"Could not import guidance utils: {e}")
        
    # Simple fallback implementation
    return re.sub(r'\{([^{}]*)\}', r'{{\1}}', template)


async def _create_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None
) -> Optional[Any]:
    """Create an LLM instance using official LlamaIndex interfaces"""
    from ..nodes_config import config
    
    # Use defaults if not provided
    model = model or getattr(config, "ANALYSIS_MODEL", config.DEFAULT_MODEL)
    provider = provider or getattr(config, "ANALYSIS_MODEL_PROVIDER", config.DEFAULT_PROVIDER)
    provider = provider.lower() if provider else "openai"
    
    try:
        # Get the timeout from nodes_config - ensure it's properly passed
        request_timeout = getattr(config, "REQUEST_TIMEOUT", 300.0)  # Default to 5 minutes if not specified
        logger.info(f"Creating LLM using official LlamaIndex interface: provider={provider}, model={model}, timeout={request_timeout}")
        
        # Use official LlamaIndex interfaces based on provider
        if provider == "openai":
            try:
                # Try the official OpenAI interface from LlamaIndex
                try:
                    from llama_index.llms.openai import OpenAI
                    return OpenAI(model=model, temperature=0.1, request_timeout=request_timeout)
                except ImportError:
                    from llama_index.core.llms.openai import OpenAI
                    return OpenAI(model=model, temperature=0.1, request_timeout=request_timeout)
            except ImportError as e:
                logger.error(f"Could not import OpenAI LLM: {e}")
                return None
                
        elif provider == "ollama":
            try:
                # Use the official Ollama interface from LlamaIndex
                try:
                    from llama_index.llms.ollama import Ollama
                    logger.info(f"Using official LlamaIndex Ollama interface with model={model}")
                    return Ollama(
                        model=model, 
                        temperature=0.1,
                        request_timeout=request_timeout,  # Ensure timeout is set
                        base_url=getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
                    )
                except ImportError:
                    from llama_index.core.llms.ollama import Ollama
                    logger.info(f"Using official LlamaIndex Core Ollama interface with model={model}")
                    return Ollama(
                        model=model, 
                        temperature=0.1,
                        request_timeout=request_timeout,  # Ensure timeout is set
                        base_url=getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
                    )
            except ImportError as e:
                logger.error(f"Could not import Ollama LLM: {e}")
                return None
        
        # Add more official interfaces as needed for other providers
        else:
            logger.error(f"Unsupported provider: {provider}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating LLM: {e}")
        return None