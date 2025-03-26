#!/usr/bin/env python
"""
Guidance Manager for A Finite Monkey Engine

This module provides a comprehensive interface for using guidance
to manage LLM interactions and tool usage in pipeline stages.
"""

import os
import json
import logging
import guidance
from typing import Dict, List, Any, Optional, Union, Callable
import llama_index
import llama_index.llms
import llama_index.question_gen
from llama_index.question_gen import guidance


logger = logging.getLogger(__name__)

class GuidanceManager:
    """
    Manager class for Guidance-based LLM interactions and pipeline stages
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        """
        Initialize the GuidanceManager with a model configuration
        
        Args:
            model_config: Configuration dictionary with provider, model name, and options
        """
        self.model_config = model_config or {}
        self.llm = None
        self.tools_registry = {}
        self._load_llm()
    
    def _load_llm(self):
        """
        Load the LLM based on the model configuration
        """
        provider = self.model_config.get('provider', 'openai')
        model_name = self.model_config.get('model', 'gpt-4o')
        options = self.model_config.get('options', {})
        
        logger.info(f"Loading LLM: {provider}:{model_name}")
        
        
        
        try:
            if provider == 'openai':
                self.llm = guidance.llms.OpenAI(model_name, **options)
            elif provider == 'anthropic':
                self.llm = guidance.llms.Anthropic(model_name, **options)
            elif provider == 'huggingface':
                self.llm = guidance.llms.HuggingFace(model_name, **options)
            elif provider == 'ollama':
                self.llm = guidance.llms.Ollama(model_name, **options)
            elif provider == 'transformers':
                self.llm = guidance.llms.Transformers(model_name, **options)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            logger.info(f"LLM loaded successfully: {provider}:{model_name}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise
    
    def register_tool(self, name: str, description: str, callback: Callable):
        """
        Register a tool for use with guidance programs
        
        Args:
            name: Tool name
            description: Tool description
            callback: Tool implementation function
        """
        self.tools_registry[name] = {
            'name': name,
            'description': description,
            'callback': callback
        }
        logger.info(f"Registered tool: {name}")
        
    def create_program(self) -> guidance.Program:
        """
        Create a new guidance program with the configured LLM
        
        Returns:
            A guidance Program instance
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
            
        return guidance.Program(self.llm)
    
    def register_all_tools(self, program: guidance.Program) -> guidance.Program:
        """
        Register all available tools with the provided program
        
        Args:
            program: Guidance program to register tools with
            
        Returns:
            The program with tools registered
        """
        for tool_name, tool_info in self.tools_registry.items():
            program = program.registerTool(
                tool_info['name'],
                tool_info['description'],
                tool_info['callback']
            )
            
        return program
    
    def execute_stage(self, stage_config: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a pipeline stage using guidance
        
        Args:
            stage_config: Configuration for the pipeline stage
            context: Context data to include in the generation
            
        Returns:
            Result of the stage execution
        """
        context = context or {}
        
        try:
            # Create a program
            program = self.create_program()
            
            # Add system message if provided
            if system_message := stage_config.get('system'):
                program = program.system(system_message)
            
            # Add user message
            if user_message := stage_config.get('user'):
                program = program.user(user_message)
            
            # Register tools if needed
            if stage_config.get('use_tools', False):
                program = self.register_all_tools(program)
            
            # Apply constraints
            constraints = stage_config.get('constraints', {})
            if regex := constraints.get('regex'):
                program = program.regex(regex)
                
            if select := constraints.get('select'):
                program = program.select(select)
                
            if grammar := constraints.get('grammar'):
                program = program.assistant(f'{{% gen "response" json_schema={grammar} %}}')
            else:
                program = program.assistant('{% gen "response" %}')
            
            # Generate the result
            result = program.generate(context)
            
            # Process and return the result
            return {
                'text': result.get('response', ''),
                'toolCalls': result.get('toolCalls', []),
                'metadata': {
                    'stage': stage_config.get('name', 'unnamed_stage'),
                    'model': f"{self.model_config.get('provider', 'unknown')}:{self.model_config.get('model', 'unknown')}"
                },
                'raw': {k: v for k, v in result.items() if k != '_guidance_state'}
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline stage execution: {e}")
            return {
                'error': str(e),
                'text': '',
                'metadata': {
                    'stage': stage_config.get('name', 'unnamed_stage'),
                    'status': 'error'
                }
            }
    
    def execute_pipeline(self, pipeline_config: List[Dict[str, Any]], initial_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a series of pipeline stages
        
        Args:
            pipeline_config: List of stage configurations
            initial_context: Starting context data
            
        Returns:
            List of results from each stage
        """
        context = initial_context or {}
        results = []
        
        for stage_config in pipeline_config:
            stage_name = stage_config.get('name', 'unnamed_stage')
            logger.info(f"Executing pipeline stage: {stage_name}")
            
            # Execute the stage
            stage_result = self.execute_stage(stage_config, context)
            results.append(stage_result)
            
            # Update context with stage result for next stage
            if not stage_result.get('error'):
                context.update({
                    f"{stage_name}_result": stage_result.get('text', ''),
                    f"{stage_name}_raw": stage_result.get('raw', {})
                })
            
            # Stop pipeline if stop_on_error is True and there was an error
            if stage_config.get('stop_on_error', False) and stage_result.get('error'):
                logger.warning(f"Stopping pipeline execution due to error in stage {stage_name}")
                break
        
        return results
