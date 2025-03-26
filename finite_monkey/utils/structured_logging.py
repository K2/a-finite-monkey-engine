"""
Structured logging utilities for Finite Monkey Engine.

This module provides structured, type-expressive logging capabilities
that complement the standard logging system with additional metadata.
"""

import time
import json
from enum import Enum
from typing import Any, Dict, Optional, Union, List, Tuple
from datetime import datetime, timedelta
import threading
from loguru import logger

# Constants for log types
class LogType(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
    PROGRESS = "PROGRESS"
    METRIC = "METRIC"
    ETA = "ETA"
    COMPLETION = "COMPLETION"
    START = "START"
    END = "END"
    RESULT = "RESULT"

class StructuredLogger:
    """
    Enhanced logger that adds structured data to log messages.
    
    This class wraps the loguru logger to add structured data that can be
    easily parsed by downstream tools while maintaining human-readable logs.
    """
    
    def __init__(self, module_name: str):
        """
        Initialize the structured logger.
        
        Args:
            module_name: Name of the module using this logger
        """
        self.module_name = module_name
        self._start_times: Dict[str, float] = {}
        self._progress_data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def _add_metadata(self, message: str, metadata: Dict[str, Any], log_type: LogType) -> str:
        """Add metadata to a log message in a structured way"""
        # Create a structured log with metadata
        structured_data = {
            "timestamp": datetime.now().isoformat(),
            "module": self.module_name,
            "type": log_type,
            "message": message,
            **metadata
        }
        
        # Add metadata string to the end of the log message
        metadata_str = f" [METADATA: {json.dumps(structured_data)}]"
        return message + metadata_str
    
    def info(self, message: str, **metadata) -> None:
        """Log an info message with structured metadata"""
        logger.info(self._add_metadata(message, metadata, LogType.INFO))
    
    def warning(self, message: str, **metadata) -> None:
        """Log a warning message with structured metadata"""
        logger.warning(self._add_metadata(message, metadata, LogType.WARNING))
    
    def error(self, message: str, **metadata) -> None:
        """Log an error message with structured metadata"""
        logger.error(self._add_metadata(message, metadata, LogType.ERROR))
    
    def debug(self, message: str, **metadata) -> None:
        """Log a debug message with structured metadata"""
        logger.debug(self._add_metadata(message, metadata, LogType.DEBUG))
    
    def start_task(self, task_id: str, task_name: str, total_items: Optional[int] = None) -> None:
        """
        Log the start of a task and track its start time.
        
        Args:
            task_id: Unique identifier for the task
            task_name: Human-readable name for the task
            total_items: Optional total number of items to process
        """
        with self._lock:
            self._start_times[task_id] = time.time()
            self._progress_data[task_id] = {
                "name": task_name,
                "total": total_items,
                "completed": 0,
                "start_time": time.time(),
                "last_update_time": time.time()
            }
        
        metadata = {"task_id": task_id, "task_name": task_name}
        if total_items is not None:
            metadata["total_items"] = total_items
            
        logger.info(self._add_metadata(f"Starting task: {task_name}", metadata, LogType.START))
    
    def update_progress(self, task_id: str, completed: int, total: Optional[int] = None) -> None:
        """
        Update and log the progress of a task.
        
        Args:
            task_id: Unique identifier for the task
            completed: Number of items completed
            total: Optional updated total number of items
        """
        with self._lock:
            if task_id not in self._progress_data:
                self.start_task(task_id, task_id, total)
                
            task_data = self._progress_data[task_id]
            task_data["completed"] = completed
            if total is not None:
                task_data["total"] = total
                
            current_time = time.time()
            task_data["last_update_time"] = current_time
            
            # Calculate ETA
            eta = None
            elapsed = current_time - task_data["start_time"]
            
            if task_data["total"] and completed > 0:
                # Linear projection
                items_per_sec = completed / elapsed
                remaining_items = task_data["total"] - completed
                
                if items_per_sec > 0:
                    remaining_secs = remaining_items / items_per_sec
                    eta = datetime.now() + timedelta(seconds=remaining_secs)
            
            # Calculate percentage
            percentage = (completed / task_data["total"] * 100) if task_data["total"] else None
        
        # Prepare message and metadata
        message = f"Progress: {completed}"
        if task_data["total"]:
            message += f"/{task_data['total']}"
            
        if percentage is not None:
            message += f" ({percentage:.1f}%)"
            
        if eta:
            message += f" - ETA: {eta.strftime('%H:%M:%S')}"
            
        metadata = {
            "task_id": task_id,
            "task_name": task_data["name"],
            "completed": completed,
            "total": task_data["total"],
            "percentage": percentage,
            "eta_timestamp": eta.isoformat() if eta else None,
            "elapsed_seconds": elapsed
        }
            
        logger.info(self._add_metadata(message, metadata, LogType.PROGRESS))
    
    def end_task(self, task_id: str, status: str = "completed", **result_data) -> None:
        """
        Log the end of a task and its results.
        
        Args:
            task_id: Unique identifier for the task
            status: Final status of the task (completed, failed, etc.)
            **result_data: Additional result data to include
        """
        with self._lock:
            if task_id in self._start_times:
                start_time = self._start_times[task_id]
                duration = time.time() - start_time
                
                task_data = self._progress_data.get(task_id, {})
                task_name = task_data.get("name", task_id)
                
                # Remove from tracking
                self._start_times.pop(task_id, None)
                self._progress_data.pop(task_id, None)
            else:
                duration = None
                task_name = task_id
        
        # Prepare metadata
        metadata = {
            "task_id": task_id,
            "status": status,
            "duration_seconds": duration,
            **result_data
        }
        
        # Log completion
        message = f"Task {task_name} {status}"
        if duration:
            message += f" in {duration:.2f}s"
            
        logger.info(self._add_metadata(message, metadata, LogType.COMPLETION))
    
    def metric(self, name: str, value: Union[int, float, str], unit: Optional[str] = None, **metadata) -> None:
        """
        Log a metric value with structured metadata.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            unit: Optional unit of measurement
            **metadata: Additional metadata to include
        """
        metric_metadata = {"metric_name": name, "metric_value": value}
        
        if unit:
            metric_metadata["metric_unit"] = unit
            
        metric_metadata.update(metadata)
        
        # Format message based on value type
        if isinstance(value, (int, float)):
            message = f"Metric {name}: {value}"
            if unit:
                message += f" {unit}"
        else:
            message = f"Metric {name}: {value}"
            
        logger.info(self._add_metadata(message, metric_metadata, LogType.METRIC))
    
    def result(self, name: str, value: Any, **metadata) -> None:
        """
        Log an operation result with structured metadata.
        
        Args:
            name: Name of the result
            value: Result value
            **metadata: Additional metadata to include
        """
        result_metadata = {"result_name": name, "result_value": value}
        result_metadata.update(metadata)
        
        message = f"Result {name}: {value}"
        logger.info(self._add_metadata(message, result_metadata, LogType.RESULT))


def get_logger(module_name: str) -> StructuredLogger:
    """
    Get a structured logger for a module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        A structured logger instance
    """
    return StructuredLogger(module_name)
