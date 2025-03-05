"""
Metrics models for the Finite Monkey framework

This module defines data classes for tracking metrics related to agent performance,
tool usage, and workflow efficiency.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgentMetrics:
    """Metrics for tracking agent performance"""
    
    name: str
    success_rate: float
    avg_response_time: float
    calls: int
    last_called: Optional[str] = None
    
    errors: Optional[Dict[str, int]] = None
    tool_usage: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "calls": self.calls,
            "last_called": self.last_called,
            "errors": self.errors,
            "tool_usage": self.tool_usage,
        }


@dataclass
class ToolUsageMetrics:
    """Metrics for tracking tool usage"""
    
    name: str
    calls: int
    success: int
    failures: int
    avg_latency: float
    last_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "calls": self.calls,
            "success": self.success,
            "failures": self.failures,
            "avg_latency": self.avg_latency,
            "last_used": self.last_used,
            "success_rate": self.success / self.calls if self.calls > 0 else 0,
        }


@dataclass
class WorkflowMetrics:
    """Metrics for tracking workflow performance"""
    
    workflow_id: str
    start_time: str
    end_time: Optional[str] = None
    duration: Optional[float] = None
    completion_status: str = "pending"
    
    # Task statistics
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    
    # Resource usage
    total_tokens: int = 0
    total_api_calls: int = 0
    peak_memory_mb: Optional[float] = None
    
    # Performance breakdown by stage
    stage_durations: Optional[Dict[str, float]] = None
    stage_token_usage: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "completion_status": self.completion_status,
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_tokens": self.total_tokens,
            "total_api_calls": self.total_api_calls,
            "peak_memory_mb": self.peak_memory_mb,
            "stage_durations": self.stage_durations,
            "stage_token_usage": self.stage_token_usage,
        }


@dataclass
class TelemetryRecord:
    """Complete telemetry record including all metrics"""
    
    timestamp: str
    workflow_id: str
    agent_metrics: Dict[str, AgentMetrics]
    tool_metrics: Dict[str, ToolUsageMetrics]
    workflow_metrics: WorkflowMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "workflow_id": self.workflow_id,
            "agent_metrics": {name: metrics.to_dict() for name, metrics in self.agent_metrics.items()},
            "tool_metrics": {name: metrics.to_dict() for name, metrics in self.tool_metrics.items()},
            "workflow_metrics": self.workflow_metrics.to_dict(),
        }