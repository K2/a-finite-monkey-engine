"""
Core integration module for guidance library with A Finite Monkey Engine.
Provides base classes and utilities for guided LLM interactions.
"""

import guidance
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type, TypeVar, Generic
from pathlib import Path
import ollama
from pydantic import BaseModel
import json
import inspect

# Import nodes_config for configuration values
from finite_monkey.nodes_config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type variable for generic structured output
T = TypeVar('T', bound=BaseModel)

# Define GuidancePydanticProgram here to avoid circular import
class GuidancePydanticProgram:
    """
    Pydantic-enabled guidance program for structured output.
    """
    
    def __init__(self, llm, system_prompt=None):
        """Initialize with LLM and optional system prompt"""
        self.llm = llm
        self.messages = []
        if system_prompt:
            self.system(system_prompt)
    
    def system(self, content):
        """Add a system message"""
        self.messages.append({"role": "system", "content": content})
        return self
    
    def user(self, content):
        """Add a user message"""
        self.messages.append({"role": "user", "content": content})
        return self
    
    def assistant(self, content):
        """Add an assistant message"""
        self.messages.append({"role": "assistant", "content": content})
        return self
    
    async def agenerate(self, variables=None):
        """Generate response asynchronously"""
        variables = variables or {}
        # Use the LLM to generate a response
        try:
            if hasattr(self.llm, "achat") and inspect.iscoroutinefunction(self.llm.achat):
                response = await self.llm.achat(messages=self.messages)
                return {"response": response.message.content, "messages": self.messages}
            else:
                # Fallback to non-async method if needed
                logger.warning("Using non-async chat method - consider using async version")
                if hasattr(self.llm, "chat") and callable(self.llm.chat):
                    response = self.llm.chat(messages=self.messages)
                    return {"response": response.message.content, "messages": self.messages}
                else:
                    raise ValueError("LLM does not support chat methods")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e), "messages": self.messages}

class GuidanceManager:
    """
    Core manager class for guidance integration with A Finite Monkey Engine.
    Handles model loading, program creation, and execution.
    """
    
    def __init__(self):
        """
        Initialize the GuidanceManager using configuration from nodes_config.
        """
        self.default_model = getattr(config, "DEFAULT_MODEL", "gpt-4o")
        self.default_provider = getattr(config, "DEFAULT_PROVIDER", "openai").lower()
        self.llm_cache = {}
        
    def get_llm(self, model_identifier: str) -> guidance.models:
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
            if ":" in model_identifier:
                provider, model_name = model_identifier.split(':', 1)
            else:
                # If provider not specified, use default provider
                provider = self.default_provider
                model_name = model_identifier
        except ValueError:
            logger.warning(f"Invalid model identifier format: {model_identifier}. Using default.")
            provider = self.default_provider
            model_name = self.default_model
        
        # Load the appropriate LLM based on provider
        try:
            # Get parameters from nodes_config
            temperature = getattr(config, "TEMPERATURE", 0.1)
            request_timeout = getattr(config, "REQUEST_TIMEOUT", 60)
            max_tokens = getattr(config, "MAX_TOKENS", 1024)
            
            if provider.lower() == "openai":
                # Get OpenAI-specific config from nodes_config
                api_key = getattr(config, "OPENAI_API_KEY", None)
                openai_params = {
                    "temperature": temperature,
                    "request_timeout": request_timeout,
                    "max_tokens": max_tokens
                }
                if api_key:
                    openai_params["api_key"] = api_key
                
                llm = guidance.llms.OpenAI(model_name, **openai_params)
            
            elif provider.lower() == "anthropic":
                # Get Anthropic-specific config
                api_key = getattr(config, "ANTHROPIC_API_KEY", None)
                anthropic_params = {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if api_key:
                    anthropic_params["api_key"] = api_key
                
                llm = guidance.llms.Anthropic(model_name, **anthropic_params)
            
            elif provider.lower() == "ollama":
                # Get Ollama-specific config
                base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
                ollama_params = {
                    "temperature": temperature,
                    "base_url": base_url,
                    "json_mode": True
                }
                
                logger.info(f"Creating Ollama LLM with json_mode=True: {model_name}")
                llm = guidance.llms.Ollama(model_name, **ollama_params)
            
            elif provider.lower() == "transformers":
                # Get Transformers-specific config
                device = getattr(config, "TRANSFORMERS_DEVICE", "cuda" if getattr(config, "USE_GPU", False) else "cpu")
                transformers_params = {
                    "device": device
                }
                
                llm = guidance.llms.Transformers(model_name, **transformers_params)
            
            else:
                logger.warning(f"Unsupported provider: {provider}. Using default.")
                return self.get_llm(f"{self.default_provider}:{self.default_model}")
                
            # Cache the LLM instance
            self.llm_cache[model_identifier] = llm
            return llm
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_program(self,        
                       model_identifier: Optional[str] = None, 
                       system_prompt: Optional[str] = None) -> GuidancePydanticProgram:
        """
        Create a guidance program with the specified model and system prompt.
        
        Args:
            model_identifier: String identifier for the model
            system_prompt: Optional system prompt to initialize the program
            
        Returns:
            A GuidancePydanticProgram instance
        """
        model_id = model_identifier or f"{self.default_provider}:{self.default_model}"
        llm = self.get_llm(model_id)
        
        program = GuidancePydanticProgram(
            llm=llm,
            system_prompt=system_prompt
        )
            
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
            model_id = model_identifier or f"{self.default_provider}:{self.default_model}"
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
