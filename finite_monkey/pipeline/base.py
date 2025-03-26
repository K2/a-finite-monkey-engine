"""
Base pipeline classes for Finite Monkey Engine

This module provides the core abstractions for creating
analysis pipelines within the Finite Monkey framework.
"""

import abc
import asyncio
from typing import Dict, List, Any, Optional, Callable, Awaitable

class PipelineStep:
    """
    Base class for pipeline steps
    
    A pipeline step represents a single unit of work in a pipeline.
    Steps can be chained together to form a complete analysis workflow.
    """
    
    def __init__(self, name: str):
        """
        Initialize the pipeline step
        
        Args:
            name: Name of the step
        """
        self.name = name
        self.next_step = None
        
    @abc.abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline step
        
        Args:
            context: Input context containing data from previous steps
            
        Returns:
            Updated context with results from this step
        """
        pass
        
    def set_next(self, step: 'PipelineStep') -> 'PipelineStep':
        """
        Set the next step in the pipeline
        
        Args:
            step: Next pipeline step
            
        Returns:
            The next step for chaining
        """
        self.next_step = step
        return step
        
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process this step and continue to the next
        
        Args:
            context: Input context
            
        Returns:
            Final context after this and all subsequent steps
        """
        # Execute this step
        result = await self.execute(context)
        
        # Continue to next step if available
        if self.next_step:
            return await self.next_step.process(result)
        
        # Otherwise return the result
        return result


class Pipeline:
    """
    Pipeline for processing steps in sequence
    
    A pipeline is a sequence of steps that are executed in order,
    with each step receiving the output of the previous step.
    """
    
    def __init__(self, name: str):
        """
        Initialize the pipeline
        
        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.steps: List[PipelineStep] = []
        self.first_step = None
        self.last_step = None
        
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """
        Add a step to the pipeline
        
        Args:
            step: Pipeline step to add
            
        Returns:
            Self for chaining
        """
        self.steps.append(step)
        
        # Set up the linked list of steps
        if not self.first_step:
            self.first_step = step
            self.last_step = step
        else:
            self.last_step.set_next(step)
            self.last_step = step
            
        return self
        
    async def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the pipeline with an initial context
        
        Args:
            initial_context: Initial context to pass to the first step
            
        Returns:
            Final context after all steps have executed
        """
        if not self.first_step:
            raise ValueError("Pipeline has no steps")
            
        # Create empty context if none provided
        context = initial_context or {}
        
        # Add pipeline metadata to context
        context["pipeline"] = {
            "name": self.name,
            "step_count": len(self.steps),
            "current_step": 0
        }
        
        # Execute the pipeline
        result = await self.first_step.process(context)
        
        return result
        
    @staticmethod
    def create_functional_step(name: str, func: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> PipelineStep:
        """
        Create a pipeline step from a function
        
        Args:
            name: Step name
            func: Async function that takes a context dict and returns a context dict
            
        Returns:
            PipelineStep wrapping the function
        """
        class FunctionalStep(PipelineStep):
            async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
                return await func(context)
                
        return FunctionalStep(name)
