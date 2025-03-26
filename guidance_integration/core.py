"""
Core integration module for guidance library with A Finite Monkey Engine.
Provides base classes and utilities for guided LLM interactions.
"""

import guidance
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar, Generic
from pathlib import Path
from pydantic import BaseModel
import json
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type variable for generic structured output
T = TypeVar('T', bound=BaseModel)

class GuidanceManager:
    """
    Core manager class for guidance integration with A Finite Monkey Engine.
    Handles model loading, program creation, and execution.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GuidanceManager with optional model configuration.
        
        Args:
            model_config: Configuration dictionary for the model
        """
        self.model_config = model_config or {}
        self.default_model = self.model_config.get("default_model", "openai:gpt-4o")
        self.llm_cache = {}
        
    def get_llm(self, model_identifier: str) -> guidance.llm:
        """
        Get or create an LLM instance based on the model identifier.
        Uses caching to avoid redundant model loading.
        
        Args:
            model_identifier: String identifier for the model, formatted as 'provider:model_name'
            
        Returns:
            A guidance LLM instance
        """
        if model_identifier in self.llm_cache:
            return self.llm_cache[model_identifier]
            
        try:
            provider, model_name = model_identifier.split(':', 1)
        except ValueError:
            logger.warning(f"Invalid model identifier format: {model_identifier}. Using default.")
            provider, model_name = self.default_model.split(':', 1)
        
        # Load the appropriate LLM based on provider
        try:
            if provider.lower() == "openai":
                llm = guidance.llms.OpenAI(model_name, **self.model_config.get("openai", {}))
            elif provider.lower() == "anthropic":
                llm = guidance.llms.Anthropic(model_name, **self.model_config.get("anthropic", {}))
            elif provider.lower() == "ollama":
                # Add json_mode=True for Ollama models
                ollama_config = self.model_config.get("ollama", {}).copy()
                ollama_config["json_mode"] = True
                logger.info(f"Creating Ollama LLM with json_mode=True: {model_name}")
                llm = guidance.llms.Ollama(model_name, **ollama_config)
            elif provider.lower() == "transformers":
                llm = guidance.llms.Transformers(model_name, **self.model_config.get("transformers", {}))
            else:
                logger.warning(f"Unsupported provider: {provider}. Using default.")
                provider, model_name = self.default_model.split(':', 1)
                return self.get_llm(self.default_model)
                
            # Cache the LLM instance
            self.llm_cache[model_identifier] = llm
            return llm
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_program(self, 
                       model_identifier: Optional[str] = None, 
                       system_prompt: Optional[str] = None) -> guidance.Program:
        """
        Create a guidance program with the specified model and system prompt.
        
        Args:
            model_identifier: String identifier for the model
            system_prompt: Optional system prompt to initialize the program
            
        Returns:
            A guidance Program instance
        """
        model_id = model_identifier or self.default_model
        llm = self.get_llm(model_id)
        
        program = guidance.Program(llm)
        if system_prompt:
            program = program.system(system_prompt)
            
        return program
    
    async def execute_structured_prompt(self,
                               prompt_text: str,
                               output_class: Type[T],
                               variables: Dict[str, Any] = None,
                               system_prompt: Optional[str] = None,
                               model_identifier: Optional[str] = None) -> T:
        """
        Execute a prompt with guidance and return a structured Pydantic object
        
        Args:
            prompt_text: The main user prompt text
            output_class: Pydantic model class for structured output
            variables: Variables to insert into the prompt
            system_prompt: Optional system prompt
            model_identifier: Model to use for this execution
            
        Returns:
            Pydantic model instance with the structured data
        """
        logger.info(f"Executing structured prompt with output class: {output_class.__name__}")
        program = self.create_program(model_identifier, system_prompt)
        variables = variables or {}
        
        # Add user prompt
        program = program.user(prompt_text)
        
        try:
            # Get the LLM instance
            model_id = model_identifier or self.default_model
            llm = self.get_llm(model_id)
            
            # Use as_structured_llm for structured output
            logger.info(f"Converting LLM to structured LLM for class: {output_class.__name__}")
            structured_llm = llm.as_structured_llm(output_class)
            
            # Generate the structured output directly
            logger.info("Generating structured output with messages")
            result = json.loads((await structured_llm.acomplete(program.messages)).message.content)
            logger.info(f"Structured output generated successfully: {type(result)}")
            
            return result
        except Exception as e:
            logger.error(f"Error generating structured response: {e}", exc_info=True)
            # Create a minimal valid instance as fallback
            logger.info(f"Falling back to default approach with JSON schema")
            try:
                # Fall back to using the JSON schema approach
                schema = output_class.model_json_schema()
                program = program.assistant(f'{{% gen "response" json_schema={schema} %}}')
                
                result = await program.agenerate(variables)
                response_data = result.get("response", {})
                logger.info(f"Generated response with fallback: {response_data}")
                
                # Parse the response into the Pydantic model
                return output_class.model_validate(response_data)
            except Exception as fallback_error:
                logger.error(f"Error in fallback approach: {fallback_error}", exc_info=True)
                raise fallback_error
        
    def execute_prompt(self,
                      prompt_text: str,
                      variables: Dict[str, Any] = None,
                      system_prompt: Optional[str] = None,
                      constraints: Optional[Dict[str, Any]] = None,
                      tools: Optional[List[Dict[str, Any]]] = None,
                      model_identifier: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a prompt with guidance, applying constraints and tools as needed.
        
        Args:
            prompt_text: The main user prompt text
            variables: Variables to insert into the prompt
            system_prompt: Optional system prompt
            constraints: Dictionary of constraints (regex, grammar, select)
            tools: List of tools to make available
            model_identifier: Model to use for this execution
            
        Returns:
            Dictionary with generation results
        """
        program = self.create_program(model_identifier, system_prompt)
        variables = variables or {}
        constraints = constraints or {}
        
        # Add user prompt
        program = program.user(prompt_text)
        
        # Register tools if provided
        if tools:
            for tool in tools:
                program = program.registerTool(
                    tool["name"],
                    tool["description"],
                    tool["function"]
                )
        
        # Apply constraints to the assistant response
        response_gen = "{{gen 'response'"
        
        # Add constraint parameters if provided
        if "regex" in constraints:
            response_gen += f" regex='{constraints['regex']}'"
        
        if "grammar" in constraints:
            response_gen += f" json_schema={constraints['grammar']}"
            
        response_gen += "}}"
        
        # Add assistant response with constraints
        program = program.assistant(response_gen)
        
        # Generate and return results
        try:
            result = program.generate(variables)
            return {
                "response": result.get("response", ""),
                "raw_result": result,
                "toolCalls": result.get("toolCalls", [])
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "error": str(e),
                "response": "",
                "raw_result": {}
            }

    def generate_with_tools(self,
                           prompt_text: str,
                           tools: List[Dict[str, Any]],
                           variables: Dict[str, Any] = None,
                           system_prompt: Optional[str] = None,
                           model_identifier: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response with automatic tool calling capabilities.
        
        Args:
            prompt_text: The main user prompt text
            tools: List of tools with name, description, and function
            variables: Variables to insert into the prompt
            system_prompt: Optional system prompt
            model_identifier: Model to use for this execution
            
        Returns:
            Dictionary with generation results and tool calls
        """
        program = self.create_program(model_identifier, system_prompt)
        variables = variables or {}
        
        # Add user prompt
        program = program.user(prompt_text)
        
        # Register tools
        for tool in tools:
            program = program.registerTool(
                tool["name"],
                tool["description"],
                tool["function"]
            )
            
        # Let the model decide when to call tools
        try:
            result = program.generate(variables)
            return {
                "response": result.get("response", ""),
                "raw_result": result,
                "toolCalls": result.get("toolCalls", [])
            }
        except Exception as e:
            logger.error(f"Error generating response with tools: {e}")
            return {
                "error": str(e),
                "response": "",
                "raw_result": {},
                "toolCalls": []
            }
