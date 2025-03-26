"""
Pipeline executor for Finite Monkey Engine

This module provides the executor that runs pipelines with
error handling, logging, and performance tracking.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple

from .base import Pipeline, PipelineStep
from ..utils.logger import logger

class PipelineExecutor:
    """
    Executor for running pipelines with error handling and logging
    
    This class provides a way to execute pipelines with additional
    features like error handling, logging, and performance tracking.
    """
    
    def __init__(self, 
                 pipeline: Pipeline, 
                 error_handlers: Optional[Dict[str, callable]] = None):
        """
        Initialize the executor
        
        Args:
            pipeline: Pipeline to execute
            error_handlers: Optional mapping of step names to error handlers
        """
        self.pipeline = pipeline
        self.error_handlers = error_handlers or {}
        self.step_timings: Dict[str, float] = {}
        
    async def execute(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the pipeline with error handling and timing
        
        Args:
            initial_context: Initial context to pass to the pipeline
            
        Returns:
            Final context after pipeline execution
        """
        context = initial_context or {}
        start_time = time.time()
        
        try:
            # Run the pipeline
            logger.info(f"Starting pipeline: {self.pipeline.name}")
            
            # Add hooks for timing and logging
            for i, step in enumerate(self.pipeline.steps):
                original_execute = step.execute
                
                # Create a wrapped execute function
                async def wrapped_execute(original_fn, step_name, step_index, ctx):
                    # Update pipeline metadata
                    if "pipeline" in ctx:
                        ctx["pipeline"]["current_step"] = step_index
                    
                    # Log step start
                    logger.info(f"Starting step {step_index+1}/{len(self.pipeline.steps)}: {step_name}")
                    step_start = time.time()
                    
                    try:
                        # Execute the original step
                        result = await original_fn(ctx)
                        
                        # Track timing
                        step_end = time.time()
                        elapsed = step_end - step_start
                        self.step_timings[step_name] = elapsed
                        logger.info(f"Completed step {step_name} in {elapsed:.2f}s")
                        
                        return result
                        
                    except Exception as e:
                        # Log the error
                        logger.error(f"Error in step {step_name}: {str(e)}")
                        
                        # Try to handle the error if a handler exists
                        if step_name in self.error_handlers:
                            logger.info(f"Attempting to recover from error in step {step_name}")
                            return await self.error_handlers[step_name](e, ctx)
                        
                        # Otherwise re-raise
                        raise
                
                # Replace the execute method with our wrapped version
                step.execute = lambda ctx, fn=original_execute, name=step.name, idx=i: wrapped_execute(fn, name, idx, ctx)
            
            # Run the pipeline
            result = await self.pipeline.run(context)
            
            # Calculate total time
            total_time = time.time() - start_time
            logger.info(f"Pipeline {self.pipeline.name} completed in {total_time:.2f}s")
            
            # Add timing information to result
            if "pipeline" in result:
                result["pipeline"]["timings"] = self.step_timings
                result["pipeline"]["total_time"] = total_time
                
            return result
            
        except Exception as e:
            # Log the error
            logger.error(f"Pipeline {self.pipeline.name} failed: {str(e)}")
            
            # Add error information to context
            context["error"] = {
                "message": str(e),
                "type": type(e).__name__,
                "pipeline": self.pipeline.name,
                "timings": self.step_timings,
                "total_time": time.time() - start_time
            }
            
            # Re-raise the exception
            raise
