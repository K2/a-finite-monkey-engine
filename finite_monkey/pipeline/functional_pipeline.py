"""
Enhanced functional pipeline with composable stages
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic, Union, Tuple
from dataclasses import dataclass
import functools

from loguru import logger
from .core import Context, Stage

# Type definitions
T = TypeVar('T')
StageFunction = Callable[[T], Union[T, asyncio.Future[T]]]

class FunctionalPipeline(Generic[T]):
    """
    A functional pipeline that supports composition and async execution
    
    This pipeline uses functional programming patterns to compose stages,
    with full support for both synchronous and asynchronous functions.
    """
    
    def __init__(self, name: str):
        """Initialize the pipeline"""
        self.name = name
        self.stages: List[Tuple[str, StageFunction[T]]] = []
    
    def add_stage(self, name: str, func: StageFunction[T]) -> 'FunctionalPipeline[T]':
        """Add a stage to the pipeline"""
        self.stages.append((name, func))
        return self
        
    def compose(self, other: 'FunctionalPipeline[T]') -> 'FunctionalPipeline[T]':
        """Compose with another pipeline"""
        result = FunctionalPipeline[T](f"{self.name}+{other.name}")
        result.stages = list(self.stages) + list(other.stages)
        return result
    
    async def run(self, initial: T) -> T:
        """Run the pipeline with an initial value"""
        result = initial
        for name, func in self.stages:
            logger.info(f"Running stage: {name}")
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(result)
                else:
                    result = func(result)
            except Exception as e:
                logger.error(f"Error in stage {name}: {e}")
                raise
        return result
    
    @staticmethod
    def of(func: StageFunction[T], name: Optional[str] = None) -> 'FunctionalPipeline[T]':
        """Create a single-stage pipeline from a function"""
        pipeline = FunctionalPipeline[T](name or func.__name__)
        pipeline.add_stage(name or func.__name__, func)
        return pipeline

# Adapters to convert between traditional Pipeline and FunctionalPipeline
def adapt_functional_pipeline(pipeline: FunctionalPipeline[Context]) -> 'Pipeline':
    """Adapt a FunctionalPipeline to traditional Pipeline"""
    from .core import Pipeline, Stage
    
    traditional = Pipeline(name=pipeline.name)
    
    for name, func in pipeline.stages:
        stage = Stage(
            name=name,
            func=func,
            description=f"Functional stage: {name}"
        )
        traditional.add_stage(stage)
    
    return traditional

def adapt_traditional_pipeline(pipeline: 'Pipeline') -> FunctionalPipeline[Context]:
    """Adapt a traditional Pipeline to FunctionalPipeline"""
    functional = FunctionalPipeline[Context](pipeline.name)
    
    for stage in pipeline.stages:
        functional.add_stage(stage.name, stage.func)
    
    return functional
