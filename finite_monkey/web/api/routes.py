"""
API routes for the Finite Monkey framework web interface
"""

from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ...agents import WorkflowOrchestrator
from ...db.manager import TaskManager
from ...nodes_config import nodes_config

# Create a router
router = APIRouter(prefix="/api", tags=["api"])

# Global references to components
_orchestrator = None
_task_manager = None

# Models for API requests and responses
class AuditRequest(BaseModel):
    files: List[str]
    query: str
    project_name: Optional[str] = None
    wait_for_completion: bool = False
    
class ConfigUpdateRequest(BaseModel):
    settings: Dict[str, Any]
    
class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    file: Optional[str] = None
    type: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
class TelemetryResponse(BaseModel):
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    audit_start_time: Optional[str] = None
    audit_end_time: Optional[str] = None
    active_tasks: Dict[str, Dict[str, Any]] = {}
    error: Optional[str] = None

# Dependency to get orchestrator instance
async def get_orchestrator() -> WorkflowOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        config = nodes_config()
        task_manager = await get_task_manager()
        _orchestrator = WorkflowOrchestrator(
            task_manager=task_manager,
            model_name=config.WORKFLOW_MODEL,
            base_dir=config.base_dir,
            db_url=config.ASYNC_DB_URL,
        )
    return _orchestrator

# Dependency to get task manager instance
async def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        config = nodes_config()
        _task_manager = TaskManager(db_url=config.ASYNC_DB_URL)
        await _task_manager.start()
    return _task_manager

# Routes
@router.get("/config", response_model=Dict[str, Any])
async def get_config():
    """Get the current configuration"""
    config = nodes_config()
    # Convert to dict and filter out sensitive values
    config_dict = config.model_dump()
    sensitive_keys = ["API_KEY", "PASSWORD", "SECRET", "TOKEN"]
    
    # Mask sensitive values
    for key in config_dict:
        if any(sensitive in key for sensitive in sensitive_keys):
            if config_dict[key]:  # Only mask non-empty values
                config_dict[key] = "********"
    
    return config_dict

@router.post("/config", response_model=Dict[str, Any])
async def update_config(request: ConfigUpdateRequest):
    """Update the configuration"""
    # This would update the configuration
    # For now, just return the request
    return {"message": "Configuration updated", "updated_settings": request.settings}

@router.post("/audit", response_model=Dict[str, Any])
async def start_audit(
    request: AuditRequest, 
    background_tasks: BackgroundTasks,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Start a new audit"""
    # If not waiting for completion, run in background task
    if not request.wait_for_completion:
        background_tasks.add_task(
            orchestrator.run_audit_workflow,
            solidity_paths=request.files,
            query=request.query,
            project_name=request.project_name,
            wait_for_completion=True  # We always wait in the background task
        )
        return {
            "message": "Audit started in background",
            "files": request.files,
            "project_name": request.project_name
        }
    
    # Otherwise run and wait for completion
    try:
        result = await orchestrator.run_audit_workflow(
            solidity_paths=request.files,
            query=request.query,
            project_name=request.project_name,
            wait_for_completion=True
        )
        
        # Convert report to dict
        if hasattr(result, "model_dump"):
            return result.model_dump()
        elif hasattr(result, "to_dict"):
            return result.to_dict()
        else:
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks", response_model=List[TaskStatusResponse])
async def get_tasks(
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get all tasks status"""
    # Use metrics from orchestrator
    tasks = []
    for task_id, task_data in orchestrator.metrics.get("active_tasks", {}).items():
        tasks.append(TaskStatusResponse(
            task_id=task_id,
            **task_data
        ))
    return tasks

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Get task status by ID"""
    try:
        task_status = await task_manager.get_task_status(task_id)
        
        # Enrich with any additional info from orchestrator metrics
        orchestrator = await get_orchestrator()
        task_metrics = orchestrator.metrics.get("active_tasks", {}).get(task_id, {})
        
        # Combine status and metrics
        result = {
            "task_id": task_id,
            "status": task_status.get("status", "unknown"),
        }
        
        # Add other fields if available
        for field in ["file", "type", "created_at", "started_at", "completed_at", "error"]:
            if field in task_metrics:
                result[field] = task_metrics[field]
            elif field in task_status:
                result[field] = task_status[field]
                
        return TaskStatusResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Task not found: {str(e)}")

@router.get("/telemetry", response_model=TelemetryResponse)
async def get_telemetry(
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get telemetry data"""
    return TelemetryResponse(**orchestrator.metrics)