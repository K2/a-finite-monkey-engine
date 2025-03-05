"""
Telemetry tracking for the Finite Monkey framework.

This module extends the TaskManager with telemetry tracking capabilities
for monitoring task performance and collecting metrics.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from .manager import TaskManager


async def start_telemetry(
    self: TaskManager,
    task_id: str,
    metrics: Optional[Dict[str, Any]] = None,
    status_message: Optional[str] = None,
) -> str:
    """
    Start telemetry tracking for a task
    
    Args:
        task_id: Task ID
        metrics: Initial metrics to track
        status_message: Status message
        
    Returns:
        Telemetry ID
    """
    # Generate telemetry ID
    telemetry_id = str(uuid.uuid4())
    
    # Check if task exists
    if task_id not in self.tasks:
        raise ValueError(f"Task with ID {task_id} not found")
    
    # Create telemetry record
    telemetry = {
        "id": telemetry_id,
        "task_id": task_id,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "metrics": metrics or {},
        "status_message": status_message,
        "checkpoints": [],
    }
    
    # Store in memory
    self.tasks[task_id]["telemetry_id"] = telemetry_id
    self.tasks[task_id]["telemetry"] = telemetry
    
    # Store in database
    async with self.async_session() as session:
        # Store telemetry in database
        # This would typically be implemented with proper ORM models
        
        return telemetry_id

async def update_telemetry(
    self: TaskManager,
    telemetry_id: str,
    metrics: Optional[Dict[str, Any]] = None,
    status_message: Optional[str] = None,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Update telemetry for a task
    
    Args:
        telemetry_id: Telemetry ID
        metrics: Updated metrics
        status_message: Status message
        checkpoint: New checkpoint to add
        
    Returns:
        True if successful, False otherwise
    """
    # Find task with this telemetry ID
    task_id = None
    for tid, task in self.tasks.items():
        if task.get("telemetry_id") == telemetry_id:
            task_id = tid
            break
    
    if not task_id:
        return False
    
    # Get telemetry
    telemetry = self.tasks[task_id]["telemetry"]
    
    # Update metrics
    if metrics:
        telemetry["metrics"].update(metrics)
    
    # Update status message
    if status_message:
        telemetry["status_message"] = status_message
    
    # Add checkpoint
    if checkpoint:
        checkpoint["timestamp"] = datetime.utcnow().isoformat()
        telemetry["checkpoints"].append(checkpoint)
    
    # Update timestamp
    telemetry["updated_at"] = datetime.utcnow().isoformat()
    
    # Update in database
    async with self.async_session() as session:
        # Update telemetry in database
        # This would typically be implemented with proper ORM models
        
        return True

async def get_telemetry(
    self: TaskManager,
    task_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get telemetry for a task
    
    Args:
        task_id: Task ID
        
    Returns:
        Telemetry data or None if not found
    """
    # Check if task exists
    if task_id not in self.tasks:
        return None
    
    # Check if telemetry exists
    if "telemetry" not in self.tasks[task_id]:
        return None
    
    # Return telemetry
    return self.tasks[task_id]["telemetry"]

async def get_system_performance(self: TaskManager) -> Dict[str, Any]:
    """
    Get system performance metrics
    
    Returns:
        System performance metrics
    """
    # Gather basic statistics
    total_tasks = len(self.tasks)
    completed_tasks = sum(1 for t in self.tasks.values() if t.get("status") == "completed")
    running_tasks = sum(1 for t in self.tasks.values() if t.get("status") == "running")
    pending_tasks = sum(1 for t in self.tasks.values() if t.get("status") == "pending")
    failed_tasks = sum(1 for t in self.tasks.values() if t.get("status") == "failed")
    
    # Calculate average task duration for completed tasks
    durations = []
    for task in self.tasks.values():
        if task.get("status") == "completed" and "started_at" in task and "completed_at" in task:
            try:
                started = datetime.fromisoformat(task["started_at"])
                completed = datetime.fromisoformat(task["completed_at"])
                duration = (completed - started).total_seconds()
                durations.append(duration)
            except (ValueError, TypeError):
                pass
    
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    # Return metrics
    return {
        "tasks_total": total_tasks,
        "tasks_completed": completed_tasks,
        "tasks_running": running_tasks,
        "tasks_pending": pending_tasks,
        "tasks_failed": failed_tasks,
        "avg_task_duration_sec": avg_duration,
        "tasks_per_min": (completed_tasks / (avg_duration / 60)) if avg_duration > 0 else 0,
        "semaphore_capacity": self.task_semaphore._value,
    }


# Monkey patch the TaskManager class
TaskManager.start_telemetry = start_telemetry
TaskManager.update_telemetry = update_telemetry
TaskManager.get_telemetry = get_telemetry
TaskManager.get_system_performance = get_system_performance