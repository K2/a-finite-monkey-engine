"""
Adapter module to interface between guidance and A Finite Monkey Engine.
Converts between the data formats and APIs of both systems.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar
from pathlib import Path

from .core import GuidanceManager
from .models import GenerationResult, ToolCall

logger = logging.getLogger(__name__)

class GuidanceAdapter:
    """
    Adapter class that bridges guidance library with A Finite Monkey Engine.
    Converts between data formats and APIs of both systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.guidance_manager = GuidanceManager(config)
    
    def convert_tool_to_guidance(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a GenAIScript/Finite Monkey Engine tool to a guidance-compatible tool.
        
        Args:
            tool: Tool definition from FiniteMonkey
            
        Returns:
            Guidance-compatible tool definition
        """
        # Create a wrapper function that will handle the tool execution
        def tool_wrapper(args):
            try:
                # Call the original handler with the provided arguments
                if callable(tool.get("handler")):
                    result = tool["handler"](args)
                    return {"result": result}
                else:
                    logger.warning(f"Tool {tool['name']} has no handler function")
                    return {"error": "Tool handler not found"}
            except Exception as e:
                logger.error(f"Error executing tool {tool['name']}: {e}")
                return {"error": str(e)}
        
        # Return the guidance-compatible tool definition
        return {
            "name": tool["name"],
            "description": tool["description"],
            "function": tool_wrapper
        }
    
    def convert_tools_to_guidance(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert multiple tools to guidance format.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            List of guidance-compatible tool definitions
        """
        return [self.convert_tool_to_guidance(tool) for tool in tools]
    
    def execute_prompt(self,
                      prompt_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a prompt using guidance based on the configuration.
        
        Args:
            prompt_config: Configuration for the prompt execution
            
        Returns:
            Response from guidance
        """
        # Extract parameters from the config
        prompt_text = prompt_config.get("prompt", "")
        system = prompt_config.get("system", "")
        variables = prompt_config.get("variables", {})
        model = prompt_config.get("model", None)
        
        # Extract and convert constraints
        constraints = {}
        if "regex" in prompt_config:
            constraints["regex"] = prompt_config["regex"]
        if "grammar" in prompt_config:
            constraints["grammar"] = prompt_config["grammar"]
        if "select" in prompt_config:
            # Convert select options to grammar format
            select_options = prompt_config["select"]
            if isinstance(select_options, list):
                constraints["grammar"] = {
                    "type": "string",
                    "enum": select_options
                }
        
        # Extract and convert tools
        tools = []
        if "tools" in prompt_config:
            tools = self.convert_tools_to_guidance(prompt_config["tools"])
        
        try:
            # Execute the prompt using guidance using structured output
            result = self.guidance_manager.execute_prompt(
                prompt_text=prompt_text,
                variables=variables,
                system_prompt=system,
                constraints=constraints,
                tools=tools,
                model_identifier=model
            )
            
            return result
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            return {
                "error": str(e),
                "response": "",
                "raw_result": {},
                "toolCalls": []
            }
    
    def create_grammar_from_schema(self, schema: Dict[str, Any]) -> str:
        """
        Convert a JSON schema to a guidance grammar string.
        
        Args:
            schema: JSON schema definition
            
        Returns:
            Guidance grammar string
        """
        # For guidance, we can just use the JSON schema directly
        return json.dumps(schema)
    
    def convert_finite_monkey_prompt_to_guidance(self, 
                                               prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a Finite Monkey Engine prompt to guidance format.
        
        Args:
            prompt: Finite Monkey Engine prompt definition
            
        Returns:
            Guidance-compatible prompt configuration
        """
        guidance_prompt = {
            "prompt": "",
            "variables": {},
            "tools": []
        }
        
        # Extract system message
        system_messages = [m for m in prompt.get("messages", []) if m.get("role") == "system"]
        if system_messages:
            guidance_prompt["system"] = "\n\n".join(m.get("content", "") for m in system_messages)
        
        # Extract user messages
        user_messages = [m for m in prompt.get("messages", []) if m.get("role") == "user"]
        if user_messages:
            guidance_prompt["prompt"] = "\n\n".join(m.get("content", "") for m in user_messages)
        
        # Extract variables
        if "variables" in prompt:
            guidance_prompt["variables"] = prompt["variables"]
        
        # Extract tools
        if "tools" in prompt:
            guidance_prompt["tools"] = self.convert_tools_to_guidance(prompt["tools"])
        
        # Extract model information
        if "model" in prompt:
            guidance_prompt["model"] = prompt["model"]
        
        return guidance_prompt
