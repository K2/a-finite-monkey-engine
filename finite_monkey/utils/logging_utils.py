"""
Logging utilities for Finite Monkey Engine

This module provides standardized logging functions and decorators that can be
used across all components of the engine for consistent logging.
"""

import functools
from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass
import time
from .logger import setup_logger

# Set up module logger
logger = setup_logger(__name__)

class AgentState(Enum):
    """Enum for agent workflow states"""
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    ERROR = "ERROR"
    FAILED = "FAILED"
    COMPLETED = "COMPLETE"

@dataclass
class WorkflowContext:
    """Context for workflow tracking"""
    state: AgentState
    start_time: float
    metadata: dict

def log_entry_exit(func: Callable) -> Callable:
    """Decorator to log function entry and exit"""
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug(f"Entering {func.__name__}")
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Exiting {func.__name__} (duration: {duration:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def log_agent_state(state: AgentState) -> Callable:
    """Decorator to log agent state changes"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            logger.info(f"Agent {self.name} state change: {state.value}")
            try:
                return await func(self, *args, **kwargs)
            finally:
                logger.debug(f"Agent {self.name} completed state: {state.value}")
        return wrapper
    return decorator

def agent_workflow(cls: Optional[type] = None, *, initial_state: AgentState = AgentState.IDLE) -> Callable:
    """Class decorator to add workflow management to an agent class"""
    def wrap(cls):
        original_init = cls.__init__
        
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.workflow_context = WorkflowContext(
                state=initial_state,
                start_time=time.time(),
                metadata={}
            )
        
        async def _set_state(self, state: AgentState):
            self.state = state
            self.workflow_context.state = state
            logger.debug(f"Agent {self.name} state changed to: {state.value}")
        
        cls.__init__ = __init__
        cls._set_state = _set_state
        
        if not hasattr(cls, 'name'):
            cls.name = cls.__name__
        if not hasattr(cls, 'state'):
            cls.state = initial_state
            
        return cls
    
    return wrap if cls is None else wrap(cls)