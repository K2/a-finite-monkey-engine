"""
Utilities for monitoring and reporting LLM interactions
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

class LLMInteractionTracker:
    """
    Track and monitor LLM interactions across analyzers
    """
    
    def __init__(self):
        """Initialize the interaction tracker"""
        self.interactions = []
        self.successful_responses = 0
        self.failed_responses = 0
        self.total_tokens_used = 0
        self.response_times = []
        self.progress_tracker = None
    
    def set_progress_tracker(self, progress_tracker):
        """Set the progress tracker for ETA estimations"""
        self.progress_tracker = progress_tracker
    
    async def track_interaction(self, stage_name: str, prompt: str, llm_call_func):
        """
        Track an interaction with the LLM
        
        Args:
            stage_name: Name of the pipeline stage
            prompt: Prompt sent to the LLM
            llm_call_func: Async function that calls the LLM
            
        Returns:
            Response from the LLM
        """
        start_time = time.time()
        error = None
        response = None
        
        try:
            # Call the LLM
            response = await llm_call_func(prompt)
            self.successful_responses += 1
            
            # Estimate tokens if that info is available
            if hasattr(response, 'metadata') and 'token_usage' in response.metadata:
                self.total_tokens_used += response.metadata['token_usage'].get('total_tokens', 0)
            
        except Exception as e:
            self.failed_responses += 1
            error = str(e)
            # Re-raise the exception
            raise
            
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.response_times.append(elapsed_time)
            
            # Record the interaction
            self.interactions.append({
                "stage": stage_name,
                "timestamp": datetime.now().isoformat(),
                "prompt_length": len(prompt),
                "response_length": len(response.text) if response else 0,
                "elapsed_seconds": elapsed_time,
                "success": error is None,
                "error": error
            })
            
            # Update progress tracker if available
            if self.progress_tracker:
                self.progress_tracker.track_prompt_processed(stage_name, elapsed_time)
        
        return response
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for LLM interactions
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.response_times:
            return {
                "total_interactions": 0,
                "successful_interactions": 0,
                "failed_interactions": 0,
                "avg_response_time": 0,
                "estimated_tokens_used": 0
            }
            
        return {
            "total_interactions": len(self.interactions),
            "successful_interactions": self.successful_responses,
            "failed_interactions": self.failed_responses,
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "estimated_tokens_used": self.total_tokens_used,
            "interactions_by_stage": self._get_interactions_by_stage()
        }
    
    def _get_interactions_by_stage(self) -> Dict[str, Dict[str, Any]]:
        """Get interaction statistics grouped by stage"""
        stage_stats = {}
        
        for interaction in self.interactions:
            stage = interaction["stage"]
            if stage not in stage_stats:
                stage_stats[stage] = {
                    "count": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_time": 0
                }
            
            stats = stage_stats[stage]
            stats["count"] += 1
            if interaction["success"]:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
            stats["total_time"] += interaction["elapsed_seconds"]
        
        # Calculate averages
        for stage, stats in stage_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            
        return stage_stats
