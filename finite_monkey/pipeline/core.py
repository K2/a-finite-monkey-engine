"""
Core pipeline module for Finite Monkey Engine

This module provides the fundamental components for building functional
pipelines with composable stages and standardized logging.
"""

import asyncio
import inspect
import json
import time
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from pathlib import Path
import uuid
from datetime import datetime
import functools

from loguru import logger

from finite_monkey.utils.logger import setup_logger

# Make sure to import the PipelineLogger
from .logging import PipelineLogger

from .transformers import AgentState, agent_workflow, WorkflowContext

# Add at the top with other imports
from enum import Enum, auto

class PipelineStageState(Enum):
    """
    Enum representing the current processing stage of a contract
    """
    INIT = auto()              # Initial state
    FILES_LOADED = auto()      # Raw files loaded
    CONTRACTS_EXTRACTED = auto() # Contract code extracted from files
    FUNCTIONS_EXTRACTED = auto() # Functions extracted from contracts
    ANALYSIS_COMPLETE = auto() # Analysis completed
    ERROR = auto()             # Processing error

# Type definitions
T = TypeVar('T')
StageFunc = Callable[["Context", Any], "Context"]
AsyncStageFunc = Callable[["Context", Any], "Context"]
PipelineStage = Union[StageFunc, AsyncStageFunc]

class Context:
    """
    Pipeline context for passing data between stages
    
    Context is the primary data structure that flows through the pipeline.
    It contains the state, files, metrics, findings, and errors from each stage.
    """
    
    def __init__(
        self, 
        project_id: Optional[str] = None, 
        config: Optional[Dict[str, Any]] = None,
        input_path: Optional[str] = None
    ):
        """
        Initialize a pipeline context
        
        Args:
            project_id: Unique identifier for the project
            config: Configuration dictionary
            input_path: Path to the input files
        """
        # Generate ID if not provided
        self.project_id = project_id or f"project_{uuid.uuid4().hex[:8]}"
        
        # Store input path
        self.input_path = input_path
        
        # Initialize empty containers
        self.state: Dict[str, Any] = {
            "project_id": self.project_id,
            "created_at": datetime.now().isoformat(),
            "current_time": datetime.now().isoformat(),
            "pipeline_stages": [],
            "current_stage": PipelineStageState.INIT.name,
        }
        
        # Processing hierarchy: files -> contracts -> functions
        self.files: Dict[str, Dict[str, Any]] = {}
        self.contracts: Dict[str, Dict[str, Any]] = {}  # Renamed from chunks
        self.functions: Dict[str, Dict[str, Any]] = {}  # Functions extracted from contracts
        
        # Metadata for processing information
        self.meta: Dict[str, Any] = {
            "input_path": input_path,
            "file_count": 0,
            "contract_count": 0,
            "function_count": 0,
            "processing_stats": {}
        }
        
        self.metrics: Dict[str, Any] = {
            "processed_items": 0,
            "total_findings": 0,
            "error_count": 0,
            "stage_timings": {},
            "start_time": time.time(),
        }
        self.findings: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, str]] = []
        
        # Store configuration
        self.config: Dict[str, Any] = config or {}
    
    def add_finding(self, finding: Dict[str, Any]) -> None:
        """
        Add a finding to the context
        
        Args:
            finding: Finding dictionary with title, description, etc.
        """
        # Add timestamp if not present
        if "timestamp" not in finding:
            finding["timestamp"] = datetime.now().isoformat()
        
        # Add finding ID if not present
        if "id" not in finding:
            finding["id"] = f"finding_{uuid.uuid4().hex[:8]}"
        
        self.findings.append(finding)
        self.metrics["total_findings"] += 1
        logger.info(f"Added finding: {finding['title']} ({finding.get('severity', 'Unknown')})")
    
    def add_error(
        self, 
        stage: str, 
        message: str, 
        exception: Optional[Exception] = None
    ) -> None:
        """
        Add an error to the context
        
        Args:
            stage: Pipeline stage where the error occurred
            message: Error message
            exception: Optional exception object
        """
        error = {
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "exception": str(exception) if exception else None,
            "id": f"error_{uuid.uuid4().hex[:8]}",
        }
        
        self.errors.append(error)
        self.metrics["error_count"] += 1
        logger.error(f"Error in {stage}: {message}")
        if exception:
            logger.error(f"Exception: {exception}")
    
    def update_metrics(self, stage_name: str, duration: float) -> None:
        """
        Update metrics with stage timing information
        
        Args:
            stage_name: Name of the pipeline stage
            duration: Duration in seconds
        """
        if "stage_timings" not in self.metrics:
            self.metrics["stage_timings"] = {}
            
        if stage_name not in self.metrics["stage_timings"]:
            self.metrics["stage_timings"][stage_name] = {
                "count": 0,
                "total_duration": 0,
                "average_duration": 0,
                "min_duration": float('inf'),
                "max_duration": 0,
            }
            
        timing = self.metrics["stage_timings"][stage_name]
        timing["count"] += 1
        timing["total_duration"] += duration
        timing["average_duration"] = timing["total_duration"] / timing["count"]
        timing["min_duration"] = min(timing["min_duration"], duration)
        timing["max_duration"] = max(timing["max_duration"], duration)
        
        # Update total time
        current_time = time.time()
        self.metrics["total_duration"] = current_time - self.metrics["start_time"]
        self.state["current_time"] = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context to dictionary
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "project_id": self.project_id,
            "state": self.state,
            "metrics": self.metrics,
            "findings": self.findings,
            "errors": self.errors,
            "file_count": len(self.files),
            "config": self.config,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert context to JSON string
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation of the context
        """
        context_dict = self.to_dict()
        
        # Omit file contents to avoid huge JSON strings
        context_dict["files"] = {
            file_id: {
                "id": file_data["id"], 
                "path": file_data["path"],
                "name": file_data["name"],
                "size": file_data["size"],
                "extension": file_data["extension"],
                "is_solidity": file_data.get("is_solidity", False),
                "has_contracts": any(c.get("file_id") == file_id for c in self.contracts.values()),
            }
            for file_id, file_data in self.files.items()
        }
        
        # Include contract metadata (previously chunks)
        context_dict["contracts"] = {
            contract_id: {
                "id": contract_data["id"],
                "name": contract_data.get("name", "Unknown"),
                "file_id": contract_data.get("file_id", "Unknown"),
                "size": len(contract_data.get("content", "")),
                "has_functions": any(fn.get("contract_id") == contract_id for fn in self.functions.values()),
            }
            for contract_id, contract_data in self.contracts.items()
        }
        
        # Include function metadata
        context_dict["functions"] = {
            function_id: {
                "id": function_data["id"],
                "name": function_data.get("name", "Unknown"),
                "contract_id": function_data.get("contract_id", "Unknown"),
                "visibility": function_data.get("visibility", "Unknown"),
                "is_constructor": function_data.get("is_constructor", False),
                "line_count": function_data.get("line_count", 0),
            }
            for function_id, function_data in self.functions.items()
        }
        
        return json.dumps(context_dict, indent=indent)
    
    def update_stage(self, stage: PipelineStageState) -> None:
        """
        Update the current pipeline stage state
        
        Args:
            stage: New pipeline stage state
        """
        prev_stage = self.state.get("current_stage", PipelineStageState.INIT.name)
        self.state["current_stage"] = stage.name
        self.state["stage_transition"] = {
            "from": prev_stage,
            "to": stage.name,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Pipeline stage transition: {prev_stage} -> {stage.name}")

    def set_files_loaded(self) -> None:
        """Mark files as loaded and update metadata"""
        self.update_stage(PipelineStageState.FILES_LOADED)
        self.meta["file_count"] = len(self.files)
        self.meta["processing_stats"]["files_loaded_at"] = datetime.now().isoformat()

    def set_contracts_extracted(self) -> None:
        """Mark contracts as extracted and update metadata"""
        self.update_stage(PipelineStageState.CONTRACTS_EXTRACTED)
        self.meta["contract_count"] = len(self.contracts)
        self.meta["processing_stats"]["contracts_extracted_at"] = datetime.now().isoformat()

    def set_functions_extracted(self) -> None:
        """Mark functions as extracted and update metadata"""
        self.update_stage(PipelineStageState.FUNCTIONS_EXTRACTED)
        self.meta["function_count"] = len(self.functions)
        self.meta["processing_stats"]["functions_extracted_at"] = datetime.now().isoformat()

    def add_contract(self, contract_id: str, contract_data: Dict[str, Any]) -> None:
        """
        Add a contract to the context
        
        Args:
            contract_id: Unique contract identifier
            contract_data: Contract data dictionary
        """
        self.contracts[contract_id] = contract_data
        logger.debug(f"Added contract: {contract_id}")
        
    def add_function(self, function_id: str, function_data: Dict[str, Any]) -> None:
        """
        Add a function to the context
        
        Args:
            function_id: Unique function identifier
            function_data: Function data dictionary
        """
        self.functions[function_id] = function_data
        logger.debug(f"Added function: {function_id}")

class Stage(Generic[T]):
    """
    Pipeline stage
    
    A stage is a function that takes a context and returns a modified context.
    It can be synchronous or asynchronous.
    """
    
    def __init__(
        self,
        func: PipelineStage,
        name: Optional[str] = None,
        description: Optional[str] = None,
        required: bool = True,  # Added required parameter
        retry_count: int = 0    # Added retry_count parameter
    ):
        """
        Initialize a pipeline stage
        
        Args:
            func: Stage function
            name: Stage name
            description: Stage description
            required: Whether the stage is required
            retry_count: Number of retries for the stage
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self.is_async = asyncio.iscoroutinefunction(func)
        self.required = required
        self.retry_count = retry_count
    
    async def __call__(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        """
        Call the stage function
        
        Args:
            context: Pipeline context
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Updated pipeline context
        """
        # Record stage start
        logger.info(f"Starting pipeline stage: {self.name}")
        start_time = time.time()
        
        # Add stage to context history
        context.state["pipeline_stages"].append({
            "name": self.name,
            "started_at": datetime.now().isoformat(),
        })
        
        retries = 0
        while True:
            try:
                # Call the stage function
                #if self.is_async:
                result = await self.func(context, *args, **kwargs)
                #else:
                    #result = self.func(context, *args, **kwargs)
                
                # Calculate duration
                end_time = time.time()
                duration = end_time - start_time
                
                # Update stage history and metrics
                context.state["pipeline_stages"][-1]["completed_at"] = datetime.now().isoformat()
                context.state["pipeline_stages"][-1]["duration"] = duration
                context.state["pipeline_stages"][-1]["status"] = "completed"
                context.update_metrics(self.name, duration)
                
                logger.info(f"Completed pipeline stage: {self.name} (took {duration:.2f}s)")
                return result
                
            except Exception as e:
                retries += 1
                if retries <= self.retry_count:
                    logger.warning(f"Stage {self.name} failed, retrying ({retries}/{self.retry_count})")
                    continue
                
                # Calculate duration
                end_time = time.time()
                duration = end_time - start_time
                
                # Update stage history and metrics
                context.state["pipeline_stages"][-1]["completed_at"] = datetime.now().isoformat()
                context.state["pipeline_stages"][-1]["duration"] = duration
                context.state["pipeline_stages"][-1]["status"] = "error"
                context.state["pipeline_stages"][-1]["error"] = str(e)
                context.update_metrics(self.name, duration)
                
                # Add error to context
                context.add_error(self.name, f"Stage '{self.name}' failed", e)
                
                if self.required:
                    logger.error(f"Required stage {self.name} failed: {str(e)}")
                    raise
                else:
                    logger.warning(f"Optional stage {self.name} failed: {str(e)}")
                    return context


def stage(name: Optional[str] = None, description: Optional[str] = None):
    """
    Decorator to define a pipeline stage
    
    Args:
        name: Stage name
        description: Stage description
        
    Returns:
        Decorator function
    """
    def decorator(func: PipelineStage) -> Stage:
        """Decorator implementation"""
        return Stage(
            func=func,
            name=name or func.__name__,
            description=description or func.__doc__ or "",
        )
    return decorator


from ..utils.structured_logging import get_logger
from ..utils.progress_tracker import ProgressTracker

class Pipeline:
    """
    Pipeline for processing documents
    """
    
    def __init__(self, stages: List):
        """
        Initialize the pipeline
        
        Args:
            stages: List of pipeline stages
        """
        self.stages = stages
        self.name = "solidity_analysis"  # Add default name attribute
        self.logger = get_logger("pipeline")
        
        # Create a progress tracker with estimated prompts per stage
        estimated_prompts = {
            "document_loading": 0,
            "contract_chunking": 0,
            "function_extraction": 1,
            "business_flow_extraction": 10,
            "data_flow_analysis": 15,
            "vulnerability_scan": 20,
            "cognitive_bias_analysis": 12,
            "documentation_analysis": 8,
            "documentation_inconsistency_analysis": 15,
            "counterfactual_analysis": 20,
            "validation_stage": 5,
            "report_generation": 1
        }
        
        self.progress_tracker = ProgressTracker(
            total_stages=len(stages),
            estimated_prompts_per_stage=estimated_prompts,
            name=self.name
        )
    
    def get_progress_tracker(self) -> ProgressTracker:
        """Get the progress tracker"""
        return self.progress_tracker
    
    async def run(self, context: Context) -> Context:
        """
        Run the pipeline with the given context
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context after pipeline execution
        """
        # Add analysis start timestamp
        context.analysis_started_at = datetime.now()
        self.logger.start_task(
            task_id=f"pipeline_{self.name}",
            task_name=f"Pipeline: {self.name}",
            total_items=len(self.stages)
        )
        
        # Track stage completion status
        if not hasattr(context, 'completed_stages'):
            context.completed_stages = []
        
        # Make sure we have a place to store output
        if not hasattr(context, 'output'):
            context.output = {}
        
        self.logger.info(
            f"Starting pipeline with {len(self.stages)} stages",
            pipeline_name=self.name,
            stage_count=len(self.stages)
        )
        
        # Run all stages
        completed_count = 0
        for i, stage in enumerate(self.stages):
            try:
                stage_name = getattr(stage, 'name', f"stage_{i}")
                
                # Log detailed stage start with structured data
                self.logger.info(
                    f"Starting stage: {stage_name}",
                    stage_name=stage_name,
                    stage_index=i,
                    stage_number=i+1,
                    total_stages=len(self.stages),
                    progress_percent=completed_count/len(self.stages)*100
                )
                
                # Mark stage start in progress tracker
                self.progress_tracker.start_stage(stage_name)
                
                # Run the stage
                new_context = await stage(context)
                
                # Check if the stage returned None (which is an error)
                if new_context is None:
                    self.logger.error(
                        f"Pipeline stage {stage_name} returned None instead of context",
                        stage_name=stage_name,
                        stage_index=i,
                        error_type="null_context"
                    )
                    context.add_error(
                        stage=stage_name,
                        message=f"Stage returned None instead of context"
                    )
                    # Continue with the original context
                    self.logger.info(
                        f"Continuing with previous context after stage {stage_name} returned None",
                        stage_name=stage_name,
                        recovery_action="use_previous_context"
                    )
                else:
                    # Update the context with the result of this stage
                    context = new_context
                
                # Record successful completion
                context.completed_stages.append(stage_name)
                completed_count += 1
                
                # Log detailed stage completion with structured data
                self.logger.info(
                    f"Completed stage: {stage_name}",
                    stage_name=stage_name,
                    stage_index=i,
                    stage_number=i+1,
                    total_stages=len(self.stages),
                    progress_percent=completed_count/len(self.stages)*100
                )
                
                # Add stage result to output summary
                if hasattr(context, stage_name.lower()):
                    stage_result = getattr(context, stage_name.lower())
                    context.output[stage_name] = {
                        "status": "completed",
                        "result_summary": f"{type(stage_result).__name__} with {len(stage_result) if hasattr(stage_result, '__len__') else 'N/A'} items"
                    }
                else:
                    context.output[stage_name] = {
                        "status": "completed",
                        "result_summary": "No specific output captured"
                    }
                
                # Mark stage complete in progress tracker
                self.progress_tracker.complete_stage(stage_name)
                
            except Exception as e:
                self.logger.error(
                    f"Error in pipeline stage {i} ({stage_name}): {str(e)}",
                    stage_name=stage_name,
                    stage_index=i,
                    error=str(e),
                    error_type=type(e).__name__
                )
                context.add_error(
                    stage=f"pipeline_stage_{i}",
                    message=f"Pipeline stage error: {str(e)}",
                    exception=e
                )
                
                # Add to output summary
                context.output[stage_name] = {
                    "status": "error",
                    "error": str(e)
                }
                
                # Mark stage as failed in progress tracker
                if hasattr(self.progress_tracker, 'current_stage'):
                    self.progress_tracker.complete_stage(self.progress_tracker.current_stage)
                
                # Don't halt the pipeline on errors, continue to next stage
                continue
        
        # Add analysis end timestamp
        context.analysis_completed_at = datetime.now()
        
        # Add progress stats to context
        context.progress_stats = self.progress_tracker.get_progress_stats()
        
        # Generate summary of completed stages
        total_stages = len(self.stages)
        completed_stages = len(context.completed_stages)
        
        # Finish progress tracking
        self.progress_tracker.finish(
            status="completed",
            completed_stages=completed_stages,
            total_stages=total_stages,
            success_rate=completed_stages/total_stages*100 if total_stages > 0 else 0
        )
        
        # Log completion with structured data
        self.logger.info(
            f"Pipeline completed: {completed_stages}/{total_stages} stages finished successfully",
            completed_stages=completed_stages,
            total_stages=total_stages,
            success_rate=completed_stages/total_stages*100 if total_stages > 0 else 0
        )
        
        # List errors if any occurred
        if hasattr(context, 'errors') and context.errors:
            self.logger.error(
                f"Pipeline had {len(context.errors)} errors",
                error_count=len(context.errors),
                first_errors=[e.get('message', 'Unknown error') for e in context.errors[:5]]
            )
            
        # End the task
        self.logger.end_task(
            task_id=f"pipeline_{self.name}",
            status="completed",
            completed_stages=completed_stages,
            total_stages=total_stages
        )
        
        return context


def compose(*funcs: Callable[[T], T]) -> Callable[[T], T]:
    """
    Compose multiple functions into a single function
    
    Args:
        *funcs: Functions to compose
        
    Returns:
        Composed function
    """
    def compose_two(f: Callable[[T], T], g: Callable[[T], T]) -> Callable[[T], T]:
        return lambda x: g(f(x))
    
    return functools.reduce(compose_two, funcs, lambda x: x)


async def parallel_execute(
    context: Context,
    funcs: List[Callable[[Context], Context]],
    max_concurrency: Optional[int] = None
) -> Context:
    """
    Execute multiple functions in parallel
    
    Args:
        context: Pipeline context
        funcs: Functions to execute
        max_concurrency: Maximum number of concurrent executions
        
    Returns:
        Updated pipeline context
    """
    # Create semaphore if max_concurrency is specified
    semaphore = None
    if max_concurrency is not None:
        semaphore = asyncio.Semaphore(max_concurrency)
    
    # Wrap the function execution to use the semaphore
    async def execute_with_semaphore(func):
        if semaphore:
            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(context)
                return func(context)
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(context)
            return func(context)
    
    # Create tasks for all functions
    tasks = [asyncio.create_task(execute_with_semaphore(func)) for func in funcs]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            context.add_error(
                "parallel_execute", 
                f"Function {funcs[i].__name__} failed",
                result
            )
    
    return context