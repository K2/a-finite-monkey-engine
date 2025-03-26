"""
Pipeline orchestrator with effect handling
"""

import asyncio
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar, Generic
from pathlib import Path
import uuid
from dataclasses import dataclass, field
from enum import Enum
import time

from loguru import logger
from ..utils.functional import AsyncPipeline, amap, async_pipe
from .core import Context
from .functional_pipeline import FunctionalPipeline

# Type for representing effects
T = TypeVar('T')
Effect = Callable[[], Union[T, asyncio.Future[T]]]

class FlowState(str, Enum):
    """States for tracking pipeline workflow"""
    PREPARING = "preparing"
    LOADING = "loading_data"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    VALIDATING = "validating"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowEvent:
    """Event for tracking state transitions in workflow"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "generic"
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EffectResult(Generic[T]):
    """Result of an effect with metadata"""
    value: T
    events: List[WorkflowEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def map(self, func: Callable[[T], Any]) -> 'EffectResult[Any]':
        """Apply a function to the value"""
        return EffectResult(
            value=func(self.value),
            events=self.events.copy(),
            metadata=self.metadata.copy()
        )
    
    async def map_async(self, func: Callable[[T], Any]) -> 'EffectResult[Any]':
        """Apply an async function to the value"""
        if asyncio.iscoroutinefunction(func):
            return EffectResult(
                value=await func(self.value),
                events=self.events.copy(),
                metadata=self.metadata.copy()
            )
        else:
            return EffectResult(
                value=func(self.value),
                events=self.events.copy(),
                metadata=self.metadata.copy()
            )
    
    def add_event(self, event: WorkflowEvent) -> 'EffectResult[T]':
        """Add an event to the result"""
        events = self.events.copy()
        events.append(event)
        return EffectResult(
            value=self.value,
            events=events,
            metadata=self.metadata.copy()
        )

class EffectOrchestrator:
    """
    Orchestrator that handles effects and state transitions
    
    This orchestrator uses functional programming patterns to manage
    pipeline execution with proper effect handling.
    """
    
    def __init__(self, name: str = "EffectOrchestrator"):
        """Initialize the orchestrator"""
        self.name = name
        self.state = FlowState.PREPARING
        self.events: List[WorkflowEvent] = []
        self.start_time = time.time()
    
    def create_event(self, event_type: str, description: str, **metadata) -> WorkflowEvent:
        """Create a workflow event"""
        return WorkflowEvent(
            event_type=event_type,
            description=description,
            metadata=metadata
        )
    
    def record_event(self, event: WorkflowEvent) -> None:
        """Record a workflow event"""
        self.events.append(event)
        logger.info(f"Event: {event.event_type} - {event.description}")
    
    def transition_state(self, new_state: FlowState) -> WorkflowEvent:
        """Transition to a new state and return the event"""
        old_state = self.state
        self.state = new_state
        
        event = self.create_event(
            event_type="state_change",
            description=f"Workflow state change: {old_state} → {new_state}",
            from_state=old_state.value,
            to_state=new_state.value
        )
        
        self.record_event(event)
        return event
    
    async def run_with_effects(
        self, 
        pipeline: FunctionalPipeline[Context],
        initial_context: Context
    ) -> EffectResult[Context]:
        """Run a pipeline with effect handling"""
        # Start effect execution
        self.transition_state(FlowState.PROCESSING)
        start_time = time.time()
        
        try:
            # Run the pipeline
            result = await pipeline.run(initial_context)
            
            # Record completion
            execution_time = time.time() - start_time
            completion_event = self.create_event(
                event_type="pipeline_complete",
                description=f"Pipeline {pipeline.name} completed in {execution_time:.2f}s",
                execution_time=execution_time
            )
            
            self.transition_state(FlowState.COMPLETED)
            
            # Return result with events
            return EffectResult(
                value=result,
                events=self.events.copy(),
                metadata={"execution_time": execution_time}
            )
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            error_event = self.create_event(
                event_type="pipeline_error",
                description=f"Pipeline {pipeline.name} failed: {str(e)}",
                error=str(e),
                execution_time=execution_time
            )
            
            self.record_event(error_event)
            self.transition_state(FlowState.FAILED)
            
            # Re-raise the exception
            raise
