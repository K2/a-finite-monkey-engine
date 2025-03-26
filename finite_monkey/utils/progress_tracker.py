"""
Progress tracking utilities for Finite Monkey Engine.

This module provides utilities for tracking progress of operations and
estimating completion times with structured logging.
"""

import time
from typing import Dict, Any, List, Optional
from .structured_logging import get_logger

class ProgressTracker:
    """
    Track progress of operations and estimate completion times.
    """
    
    def __init__(
        self, 
        total_stages: int, 
        estimated_prompts_per_stage: Optional[Dict[str, int]] = None,
        name: str = "pipeline"
    ):
        """
        Initialize the progress tracker.
        
        Args:
            total_stages: Total number of stages to track
            estimated_prompts_per_stage: Estimated prompts per stage for ETA calculation
            name: Name of the progress tracker for logging
        """
        self.total_stages = total_stages
        self.completed_stages = 0
        self.current_stage = ""
        self.current_stage_progress = 0.0
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.stage_times: Dict[str, float] = {}
        self.estimated_prompts_per_stage = estimated_prompts_per_stage or {}
        self.name = name
        self.logger = get_logger("progress_tracker")
        
        # Log start of tracking
        self.logger.start_task(
            task_id=self.name,
            task_name=f"Pipeline execution: {self.name}",
            total_items=total_stages
        )
    
    def start_stage(self, stage_name: str) -> None:
        """
        Mark the start of a stage.
        
        Args:
            stage_name: Name of the stage being started
        """
        self.current_stage = stage_name
        self.current_stage_progress = 0.0
        self.stage_start_time = time.time()
        
        # Log stage start
        stage_task_id = f"{self.name}.{stage_name}"
        self.logger.start_task(
            task_id=stage_task_id,
            task_name=f"Stage: {stage_name}",
            total_items=self.estimated_prompts_per_stage.get(stage_name, 100)
        )
        
        # Update overall progress
        overall_progress = (self.completed_stages / self.total_stages) * 100
        self.logger.update_progress(
            task_id=self.name,
            completed=self.completed_stages,
            total=self.total_stages
        )
    
    def update_stage_progress(self, progress: float, message: Optional[str] = None) -> None:
        """
        Update the progress within the current stage.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            message: Optional progress message
        """
        if not self.current_stage:
            return
            
        self.current_stage_progress = progress
        
        # Log stage progress
        stage_task_id = f"{self.name}.{self.current_stage}"
        estimated_total = self.estimated_prompts_per_stage.get(self.current_stage, 100)
        completed = int(progress * estimated_total)
        
        self.logger.update_progress(
            task_id=stage_task_id,
            completed=completed,
            total=estimated_total
        )
        
        # Update overall progress - current stage counts as partial completion
        overall_completed = self.completed_stages + progress / self.total_stages
        self.logger.update_progress(
            task_id=self.name,
            completed=int(overall_completed * 100),  # Scale to percentage
            total=100
        )
        
        if message:
            self.logger.info(
                f"Stage progress: {message}", 
                stage=self.current_stage, 
                progress=progress
            )
    
    def complete_stage(self, stage_name: Optional[str] = None) -> None:
        """
        Mark a stage as completed.
        
        Args:
            stage_name: Name of the completed stage, defaults to current stage
        """
        stage_to_complete = stage_name or self.current_stage
        
        if not stage_to_complete:
            return
            
        # Record completion time
        stage_time = time.time() - self.stage_start_time
        self.stage_times[stage_to_complete] = stage_time
        
        # Increment completed stages
        self.completed_stages += 1
        
        # Log stage completion
        stage_task_id = f"{self.name}.{stage_to_complete}"
        self.logger.end_task(
            task_id=stage_task_id,
            status="completed",
            duration_seconds=stage_time
        )
        
        # Update overall progress
        self.logger.update_progress(
            task_id=self.name,
            completed=self.completed_stages,
            total=self.total_stages
        )
        
        # Reset current stage if it's the one we just completed
        if stage_to_complete == self.current_stage:
            self.current_stage = ""
            self.current_stage_progress = 0.0
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """
        Get current progress statistics.
        
        Returns:
            Dictionary with progress statistics
        """
        elapsed_time = time.time() - self.start_time
        
        # Calculate estimated completion time
        eta = None
        if self.completed_stages > 0 and self.completed_stages < self.total_stages:
            time_per_stage = elapsed_time / self.completed_stages
            remaining_stages = self.total_stages - self.completed_stages
            eta = time_per_stage * remaining_stages
        
        return {
            "total_stages": self.total_stages,
            "completed_stages": self.completed_stages,
            "current_stage": self.current_stage,
            "elapsed_seconds": elapsed_time,
            "eta_seconds": eta,
            "stage_times": self.stage_times,
            "percent_complete": (self.completed_stages / self.total_stages) * 100
        }
    
    def finish(self, status: str = "completed", **result_data) -> None:
        """
        Mark the entire process as finished.
        
        Args:
            status: Final status (completed, failed, etc.)
            **result_data: Additional result data
        """
        # Log completion
        self.logger.end_task(
            task_id=self.name,
            status=status,
            stages_completed=self.completed_stages,
            stages_total=self.total_stages,
            **result_data
        )
        
        # Log final metrics
        elapsed_time = time.time() - self.start_time
        self.logger.metric(
            name="total_execution_time",
            value=elapsed_time,
            unit="seconds"
        )
        
        if self.completed_stages > 0:
            avg_stage_time = elapsed_time / self.completed_stages
            self.logger.metric(
                name="average_stage_time",
                value=avg_stage_time,
                unit="seconds"
            )
