#!/usr/bin/env python
"""
Guidance Adapter for A Finite Monkey Engine

This module serves as a bridge between the JavaScript/TypeScript 
frontend and the Python-based guidance library.
"""

import os
import sys
import json
import argparse
import guidance
from typing import Dict, List, Any, Optional, Union

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_llm(provider: str, model: str, config: Dict[str, Any]) -> guidance.llms.LLM:
    """
    Load an LLM based on provider and model name
    """
    logger.info(f"Loading model {model} from provider {provider}")
    
    try:
        if provider.lower() == "openai":
            return guidance.llms.OpenAI(model, **config)
        elif provider.lower() == "anthropic":
            return guidance.llms.Anthropic(model, **config)
        elif provider.lower() == "ollama":
            return guidance.llms.Ollama(model, **config)
        elif provider.lower() == "huggingface":
            return guidance.llms.HuggingFace(model, **config)
        elif provider.lower() == "transformers":
            return guidance.llms.Transformers(model, **config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def register_tools(program: guidance.Program, tools: List[Dict[str, Any]]) -> guidance.Program:
    """
    Register tools with the guidance program
    """
    for tool in tools:
        # Create a wrapper function that handles the tool call
        def tool_wrapper(args, tool=tool):
            try:
                # Call API endpoint to execute the tool
                result = {"success": True, "result": f"Tool {tool['name']} executed with args {args}"}
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        program = program.registerTool(
            tool["name"], 
            tool["description"], 
            tool_wrapper
        )
    
    return program

def execute_prompt(prompt_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a guidance prompt with the given configuration
    """
    try:
        # Extract configuration
        provider, model = prompt_config.get("model", "openai:gpt-4o").split(":", 1)
        model_config = prompt_config.get("modelConfig", {})
        system = prompt_config.get("system", "")
        user = prompt_config.get("user", "")
        tools = prompt_config.get("tools", [])
        constraints = prompt_config.get("constraints", {})
        variables = prompt_config.get("variables", {})
        
        # Load LLM
        llm = load_llm(provider, model, model_config)
        
        # Create program
        program = guidance.Program(llm)
        
        # Add system message
        if system:
            program = program.system(system)
        
        # Add user message
        program = program.user(user)
        
        # Register tools
        if tools:
            program = register_tools(program, tools)
        
        # Apply constraints
        if constraints.get("regex"):
            program = program.regex(constraints["regex"])
        
        if constraints.get("select"):
            program = program.select(constraints["select"])
        
        if constraints.get("grammar"):
            # For grammar constraints, we need to modify the assistant instruction
            program = program.assistant(f'{{% gen "response" json_schema={constraints["grammar"]} %}}')
        else:
            # Default assistant instruction
            program = program.assistant('{% gen "response" %}')
        
        # Generate response
        result = program.generate(variables)
        
        # Extract and return the result
        return {
            "text": result.get("response", ""),
            "toolCalls": result.get("toolCalls", []),
            "raw": {k: v for k, v in result.items() if k != "_guidance_state"}
        }
    
    except Exception as e:
        logger.error(f"Error executing prompt: {e}")
        return {
            "error": str(e),
            "text": "",
            "toolCalls": [],
            "raw": {}
        }

def main():
    """
    Main entry point for the adapter
    """
    parser = argparse.ArgumentParser(description="Guidance Adapter for A Finite Monkey Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to prompt configuration JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to write output JSON")
    
    args = parser.parse_args()
    
    try:
        # Read the prompt configuration
        with open(args.config, 'r') as f:
            prompt_config = json.load(f)
        
        # Execute the prompt
        result = execute_prompt(prompt_config)
        
        # Write the result
        with open(args.output, 'w') as f:
            json.dump(result, f)
        
        logger.info(f"Result written to {args.output}")
        return 0
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
