"""
Pipeline logging for Finite Monkey Engine

This module provides logging utilities for pipeline components, ensuring
consistent logging format and behavior throughout the pipeline.
"""

import logging
import sys
import time
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from loguru import logger

# Configure default logging format
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class PipelineLogFormatter(logging.Formatter):
    """Custom formatter for pipeline logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with additional pipeline metadata
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message
        """
        # Get base formatted message
        message = super().format(record)
        
        # Add extra fields as JSON if they exist
        if hasattr(record, "extra") and record.extra:
            try:
                extra_json = json.dumps(record.extra)
                message = f"{message} | {extra_json}"
            except (TypeError, ValueError):
                # If JSON serialization fails, add as string
                message = f"{message} | {str(record.extra)}"
                
        return message


class JsonLogFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log message
        """
        # Build base log object
        log_dict = {
            "timestamp": self.formatTime(record, self.datefmt or DEFAULT_DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if exists
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields
        if hasattr(record, "extra") and record.extra:
            for key, value in record.extra.items():
                if key not in log_dict:
                    log_dict[key] = value
                    
        # Add pipeline stage info if available
        if hasattr(record, "stage"):
            log_dict["stage"] = record.stage
            
        if hasattr(record, "action"):
            log_dict["action"] = record.action
            
        if hasattr(record, "duration") and record.duration is not None:
            log_dict["duration"] = record.duration
            
        # Convert to JSON
        return json.dumps(log_dict)


def setup_pipeline_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    json_output: bool = False,
) -> None:
    """
    Set up pipeline logging
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        log_format: Log format string
        date_format: Date format string
        json_output: Whether to use JSON output format
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = JsonLogFormatter() if json_output else PipelineLogFormatter(
        fmt=log_format,
        datefmt=date_format,
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    

class PipelineLogger:
    """Logger for pipeline operations"""
    
    def __init__(self, name: str):
        """Initialize the logger"""
        self.name = name
        self.logger = logger.bind(pipeline=name)
    
    def start(self):
        """Log pipeline start"""
        self.logger.info(f"Starting pipeline: {self.name}")
    
    def end(self, status: str):
        """Log pipeline end"""
        self.logger.info(f"Pipeline '{self.name}' {status}")
    
    def step_start(self, step_name: str):
        """Log step start"""
        self.logger.info(f"Starting step: {step_name}")
    
    def step_end(self, status: str):
        """Log step end"""
        self.logger.info(f"Step {status}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error message"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}")
        else:
            self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)


# Performance logging
class StagePerformanceTracker:
    """Track performance of pipeline stages"""
    
    def __init__(self, logger: Optional[PipelineLogger] = None):
        """
        Initialize stage performance tracker
        
        Args:
            logger: Optional logger instance
        """
        self.stage_timings: Dict[str, List[float]] = {}
        self.current_stages: Dict[str, float] = {}
        self.logger = logger or PipelineLogger("pipeline.performance")
        
    def start_stage(self, stage_name: str) -> None:
        """
        Start timing a stage
        
        Args:
            stage_name: Name of the stage
        """
        self.current_stages[stage_name] = time.time()
        
    def end_stage(self, stage_name: str) -> float:
        """
        End timing a stage
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Duration in seconds
        """
        if stage_name not in self.current_stages:
            return 0.0
            
        start_time = self.current_stages.pop(stage_name)
        duration = time.time() - start_time
        
        if stage_name not in self.stage_timings:
            self.stage_timings[stage_name] = []
            
        self.stage_timings[stage_name].append(duration)
        
        # Log stage completion
        self.logger.info(
            f"Stage {stage_name} completed in {duration:.4f}s",
            {
                "stage": stage_name,
                "duration": duration,
                "action": "complete"
            }
        )
        
        return duration
        
    def get_stage_stats(self, stage_name: str) -> Dict[str, Union[float, int]]:
        """
        Get statistics for a stage
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Dictionary of statistics
        """
        if stage_name not in self.stage_timings or not self.stage_timings[stage_name]:
            return {
                "count": 0,
                "total": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            
        timings = self.stage_timings[stage_name]
        return {
            "count": len(timings),
            "total": sum(timings),
            "mean": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
        }
        
    def get_all_stats(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Get statistics for all stages
        
        Returns:
            Dictionary of stage statistics
        """
        return {
            stage_name: self.get_stage_stats(stage_name)
            for stage_name in self.stage_timings
        }
        
    def log_all_stats(self) -> None:
        """Log statistics for all stages"""
        stats = self.get_all_stats()
        
        # Log overall statistics
        total_time = sum(stat["total"] for stat in stats.values())
        stage_count = sum(stat["count"] for stat in stats.values())
        
        self.logger.info(
            f"Pipeline completed: {stage_count} stages in {total_time:.4f}s",
            {
                "total_time": total_time,
                "stage_count": stage_count,
                "stats": {k: v for k, v in stats.items()}
            }
        )
        
        # Log individual stage statistics
        for stage_name, stat in stats.items():
            self.logger.info(
                f"Stage {stage_name}: {stat['count']} executions, "
                f"avg {stat['mean']:.4f}s, total {stat['total']:.4f}s",
                {
                    "stage": stage_name,
                    "count": stat["count"],
                    "mean": stat["mean"],
                    "total": stat["total"],
                    "min": stat["min"],
                    "max": stat["max"],
                }
            )