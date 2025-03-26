"""
Functional programming utilities for data transformation and composition
"""

import asyncio
import functools
from typing import Any, Callable, Dict, List, TypeVar, Generic, Iterable, AsyncIterable, Optional, Union

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

def compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Compose multiple functions into a single function
    
    Args:
        *funcs: Functions to compose from left to right (executed in reverse order)
        
    Returns:
        Composed function
    """
    def compose_two(f: Callable[[Any], Any], g: Callable[[Any], Any]) -> Callable[[Any], Any]:
        return lambda x: g(f(x))
    
    return functools.reduce(compose_two, funcs, lambda x: x)

def pipe(value: T, *funcs: Callable[[Any], Any]) -> Any:
    """
    Pipe a value through a series of functions
    
    Args:
        value: Initial value
        *funcs: Functions to apply in sequence
        
    Returns:
        Result after applying all functions
    """
    result = value
    for func in funcs:
        result = func(result)
    return result

async def async_pipe(value: T, *funcs: Callable[[Any], Any]) -> Any:
    """
    Pipe a value through a series of functions, supporting async functions
    
    Args:
        value: Initial value
        *funcs: Functions to apply in sequence (can be sync or async)
        
    Returns:
        Result after applying all functions
    """
    result = value
    for func in funcs:
        if asyncio.iscoroutinefunction(func):
            result = await func(result)
        else:
            result = func(result)
    return result

def map_async(func: Callable[[T], U], iterable: Iterable[T]) -> List[U]:
    """
    Apply a function to each item in an iterable (functional equivalent of map())
    
    Args:
        func: Function to apply
        iterable: Items to process
        
    Returns:
        List of results
    """
    return [func(item) for item in iterable]

async def amap(func: Callable[[T], Union[U, asyncio.Future[U]]], iterable: Iterable[T], 
          max_concurrency: Optional[int] = None) -> List[U]:
    """
    Asynchronously apply a function to each item in an iterable with concurrency control
    
    Args:
        func: Function to apply (can be sync or async)
        iterable: Items to process
        max_concurrency: Maximum number of concurrent tasks (None for unlimited)
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    
    async def process_item(item):
        if semaphore:
            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(item)
                else:
                    return func(item)
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                return func(item)
    
    return await asyncio.gather(*[process_item(item) for item in iterable])

async def afilter(func: Callable[[T], Union[bool, asyncio.Future[bool]]], iterable: Iterable[T]) -> List[T]:
    """
    Asynchronously filter items from an iterable
    
    Args:
        func: Predicate function (can be sync or async)
        iterable: Items to filter
        
    Returns:
        Filtered list
    """
    results = []
    for item in iterable:
        if asyncio.iscoroutinefunction(func):
            if await func(item):
                results.append(item)
        else:
            if func(item):
                results.append(item)
    return results

async def areduce(func: Callable[[U, T], Union[U, asyncio.Future[U]]], iterable: Iterable[T], 
            initial: U) -> U:
    """
    Asynchronously reduce an iterable to a single value
    
    Args:
        func: Reducer function (can be sync or async)
        iterable: Items to reduce
        initial: Initial accumulator value
        
    Returns:
        Reduced result
    """
    result = initial
    for item in iterable:
        if asyncio.iscoroutinefunction(func):
            result = await func(result, item)
        else:
            result = func(result, item)
    return result

def curry(func: Callable[..., V]) -> Callable[..., Union[V, Callable[..., V]]]:
    """
    Curry a function to allow partial application
    
    Args:
        func: Function to curry
        
    Returns:
        Curried function
    """
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(*(args + more_args), **{**kwargs, **more_kwargs})
    return curried

class AsyncPipeline:
    """
    Functional pipeline for asynchronous processing
    
    This class implements a monadic-style pipeline for chaining
    async operations with proper error handling.
    """
    
    def __init__(self, initial_value: Any = None):
        """Initialize with an optional initial value"""
        self.value = initial_value
        self.error = None
        self.has_error = False
    
    @classmethod
    def of(cls, value: Any) -> 'AsyncPipeline':
        """Create a new pipeline with an initial value"""
        return cls(value)
    
    async def map(self, func: Callable[[Any], Any]) -> 'AsyncPipeline':
        """Apply a function to the value if no error has occurred"""
        if self.has_error:
            return self
            
        try:
            if asyncio.iscoroutinefunction(func):
                self.value = await func(self.value)
            else:
                self.value = func(self.value)
        except Exception as e:
            self.error = e
            self.has_error = True
            
        return self
    
    async def flat_map(self, func: Callable[[Any], 'AsyncPipeline']) -> 'AsyncPipeline':
        """Apply a function that returns a pipeline to the value"""
        if self.has_error:
            return self
            
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(self.value)
            else:
                result = func(self.value)
                
            if result.has_error:
                self.error = result.error
                self.has_error = True
            else:
                self.value = result.value
                
        except Exception as e:
            self.error = e
            self.has_error = True
            
        return self
    
    async def recover(self, func: Callable[[Exception], Any]) -> 'AsyncPipeline':
        """Recover from an error by applying a function to the error"""
        if not self.has_error:
            return self
            
        try:
            if asyncio.iscoroutinefunction(func):
                self.value = await func(self.error)
            else:
                self.value = func(self.error)
                
            self.error = None
            self.has_error = False
            
        except Exception as e:
            self.error = e
            # Still in error state
            
        return self
    
    def get(self) -> Any:
        """Get the current value or raise the error"""
        if self.has_error:
            raise self.error
        return self.value
    
    def get_or_else(self, default: Any) -> Any:
        """Get the current value or a default if an error occurred"""
        if self.has_error:
            return default
        return self.value
