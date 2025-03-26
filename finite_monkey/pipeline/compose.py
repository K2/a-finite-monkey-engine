"""
Pipeline composition utilities for Finite Monkey Engine

This module provides functional composition utilities for pipeline stages,
enabling the creation of complex pipelines from simple, reusable components.
"""

import asyncio
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

from .core import Context, PipelineStage
from .logging import PipelineLogger

# Type definitions
T = TypeVar('T')
StageResult = TypeVar('StageResult')


def compose(*funcs: PipelineStage) -> PipelineStage:
    """
    Compose multiple pipeline stages into a single stage
    
    Args:
        *funcs: Pipeline stages to compose
        
    Returns:
        Composed pipeline stage
    """
    async def composed_stage(context: Context, *args: Any, **kwargs: Any) -> Context:
        """Composed pipeline stage"""
        result = context
        for func in funcs:
            if inspect.iscoroutinefunction(func):
                result = await func(result, *args, **kwargs)
            else:
                result = func(result, *args, **kwargs)
        return result
    
    # Set name and docstring
    composed_stage.__name__ = f"composed_{'_'.join(f.__name__ for f in funcs)}"
    composed_stage.__doc__ = "Composed pipeline stage: " + " -> ".join(f.__name__ for f in funcs)
    
    return composed_stage


def conditional(
    condition: Callable[[Context], bool],
    if_true: PipelineStage,
    if_false: Optional[PipelineStage] = None,
) -> PipelineStage:
    """
    Create a conditional pipeline stage
    
    Args:
        condition: Condition function that returns True or False
        if_true: Stage to execute if condition is True
        if_false: Optional stage to execute if condition is False
        
    Returns:
        Conditional pipeline stage
    """
    async def conditional_stage(context: Context, *args: Any, **kwargs: Any) -> Context:
        """Conditional pipeline stage"""
        if condition(context):
            if inspect.iscoroutinefunction(if_true):
                return await if_true(context, *args, **kwargs)
            else:
                return if_true(context, *args, **kwargs)
        elif if_false:
            if inspect.iscoroutinefunction(if_false):
                return await if_false(context, *args, **kwargs)
            else:
                return if_false(context, *args, **kwargs)
        else:
            return context
    
    # Set name and docstring
    conditional_stage.__name__ = f"conditional_{if_true.__name__}"
    conditional_stage.__doc__ = f"Conditional pipeline stage: {condition.__name__} -> {if_true.__name__}"
    
    return conditional_stage


def parallel(*funcs: PipelineStage) -> PipelineStage:
    """
    Execute multiple pipeline stages in parallel
    
    Args:
        *funcs: Pipeline stages to execute in parallel
        
    Returns:
        Parallel pipeline stage
    """
    async def parallel_stage(context: Context, *args: Any, **kwargs: Any) -> Context:
        """Parallel pipeline stage"""
        # Create individual context copies for each stage
        contexts = [context.copy() for _ in funcs]
        
        # Run stages in parallel
        tasks = []
        for i, func in enumerate(funcs):
            if inspect.iscoroutinefunction(func):
                tasks.append(func(contexts[i], *args, **kwargs))
            else:
                # Wrap sync function in async task
                async def run_sync(f, ctx):
                    return f(ctx, *args, **kwargs)
                tasks.append(run_sync(func, contexts[i]))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        for result in results:
            if isinstance(result, Exception):
                context.add_error("parallel_stage", f"Error in parallel stage: {str(result)}", result)
                continue
                
            # Merge metrics
            for key, value in result.metrics.items():
                if key in context.metrics:
                    context.metrics[key] += value
                else:
                    context.metrics[key] = value
                    
            # Merge findings
            for finding in result.findings:
                context.add_finding(finding)
                
            # Merge errors
            for error in result.errors:
                context.errors.append(error)
                
            # Merge files
            for file_id, file_data in result.files.items():
                if file_id not in context.files:
                    context.files[file_id] = file_data
                else:
                    # Merge file data
                    for key, value in file_data.items():
                        if key not in context.files[file_id]:
                            context.files[file_id][key] = value
                        elif key == "chunks" and isinstance(value, list):
                            # Combine chunks
                            context.files[file_id]["chunks"].extend(value)
                            
        return context
    
    # Set name and docstring
    parallel_stage.__name__ = f"parallel_{'_'.join(f.__name__ for f in funcs)}"
    parallel_stage.__doc__ = "Parallel pipeline stage: " + " + ".join(f.__name__ for f in funcs)
    
    return parallel_stage


def retry(
    stage: PipelineStage,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    should_retry: Callable[[Exception], bool] = lambda _: True,
) -> PipelineStage:
    """
    Create a pipeline stage with retry logic
    
    Args:
        stage: Pipeline stage to retry
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        should_retry: Function that decides if an exception should trigger a retry
        
    Returns:
        Pipeline stage with retry logic
    """
    async def retry_stage(context: Context, *args: Any, **kwargs: Any) -> Context:
        """Pipeline stage with retry logic"""
        logger = PipelineLogger(f"pipeline.retry.{stage.__name__}")
        retries = 0
        
        while True:
            try:
                if inspect.iscoroutinefunction(stage):
                    result = await stage(context, *args, **kwargs)
                else:
                    result = stage(context, *args, **kwargs)
                return result
                
            except Exception as e:
                retries += 1
                
                if retries > max_retries or not should_retry(e):
                    context.add_error(
                        f"retry_{stage.__name__}", 
                        f"Failed after {retries} attempts: {str(e)}", 
                        e
                    )
                    if context.config.get("raise_exceptions", False):
                        raise
                    return context
                
                logger.warning(
                    f"Retry {retries}/{max_retries} for {stage.__name__}: {str(e)}",
                    {
                        "stage": stage.__name__,
                        "retry_count": retries,
                        "max_retries": max_retries,
                        "error": str(e),
                    }
                )
                
                # Wait before retrying
                await asyncio.sleep(retry_delay)
    
    # Set name and docstring
    retry_stage.__name__ = f"retry_{stage.__name__}"
    retry_stage.__doc__ = f"Retry pipeline stage: {stage.__name__} (max {max_retries} times)"
    
    return retry_stage


def timed(
    stage: PipelineStage,
    name: Optional[str] = None,
) -> PipelineStage:
    """
    Create a timed pipeline stage
    
    Args:
        stage: Pipeline stage to time
        name: Optional custom name for timing metrics
        
    Returns:
        Timed pipeline stage
    """
    import time
    
    async def timed_stage(context: Context, *args: Any, **kwargs: Any) -> Context:
        """Timed pipeline stage"""
        stage_name = name or stage.__name__
        logger = PipelineLogger(f"pipeline.timed.{stage_name}")
        
        start_time = time.time()
        
        # Log stage start
        logger.info(
            f"Starting timed stage: {stage_name}",
            {"stage": stage_name, "action": "start"}
        )
        
        try:
            # Execute the stage
            if inspect.iscoroutinefunction(stage):
                result = await stage(context, *args, **kwargs)
            else:
                result = stage(context, *args, **kwargs)
                
            # Calculate duration
            duration = time.time() - start_time
            
            # Update timing metrics
            metric_name = f"time_{stage_name}"
            if metric_name not in result.metrics:
                result.metrics[metric_name] = duration
            else:
                result.metrics[metric_name] += duration
                
            # Log stage completion
            logger.info(
                f"Completed timed stage: {stage_name} in {duration:.4f}s",
                {
                    "stage": stage_name,
                    "action": "complete",
                    "duration": duration,
                }
            )
            
            return result
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log stage failure
            logger.error(
                f"Failed timed stage: {stage_name} in {duration:.4f}s: {str(e)}",
                {
                    "stage": stage_name,
                    "action": "failed",
                    "duration": duration,
                    "error": str(e),
                },
                exc_info=e,
            )
            
            # Re-raise the exception
            raise
    
    # Set name and docstring
    timed_stage.__name__ = f"timed_{stage.__name__}"
    timed_stage.__doc__ = f"Timed pipeline stage: {stage.__name__}"
    
    return timed_stage


def branch(
    stages: Dict[str, PipelineStage],
    combiner: Optional[Callable[[Dict[str, Context], Context], Context]] = None,
) -> PipelineStage:
    """
    Create a branching pipeline stage
    
    Args:
        stages: Dictionary of branch name to pipeline stage
        combiner: Optional function to combine branch results
        
    Returns:
        Branching pipeline stage
    """
    async def branch_stage(context: Context, *args: Any, **kwargs: Any) -> Context:
        """Branching pipeline stage"""
        # Create a copy of the context for each branch
        branch_contexts = {name: context.copy() for name in stages}
        
        # Execute each branch
        branch_results = {}
        for name, stage in stages.items():
            try:
                if inspect.iscoroutinefunction(stage):
                    branch_results[name] = await stage(branch_contexts[name], *args, **kwargs)
                else:
                    branch_results[name] = stage(branch_contexts[name], *args, **kwargs)
                    
            except Exception as e:
                context.add_error(
                    f"branch_{name}", 
                    f"Error in branch {name}: {str(e)}", 
                    e
                )
                branch_results[name] = branch_contexts[name]
        
        # Combine results
        if combiner:
            return combiner(branch_results, context)
        else:
            # Default combination: merge metrics, findings, and errors
            for name, result in branch_results.items():
                # Merge metrics
                for key, value in result.metrics.items():
                    if key in context.metrics:
                        context.metrics[key] += value
                    else:
                        context.metrics[key] = value
                        
                # Merge findings
                for finding in result.findings:
                    context.add_finding(finding)
                    
                # Merge errors
                for error in result.errors:
                    context.errors.append(error)
                    
                # Store branch result in context state
                if "branch_results" not in context.state:
                    context.state["branch_results"] = {}
                context.state["branch_results"][name] = result
        
        return context
    
    # Set name and docstring
    branch_stage.__name__ = f"branch_{'_'.join(stages.keys())}"
    branch_stage.__doc__ = "Branching pipeline stage: " + ", ".join(stages.keys())
    
    return branch_stage


def pipeline(*stages: PipelineStage) -> PipelineStage:
    """
    Create a pipeline from multiple stages
    
    This is a convenience function for compose(*stages)
    
    Args:
        *stages: Pipeline stages
        
    Returns:
        Composed pipeline stage
    """
    return compose(*stages)