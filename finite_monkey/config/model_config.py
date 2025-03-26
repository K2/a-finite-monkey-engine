"""
Configuration for LLM models used in the pipeline
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os

from ..nodes_config import config as node_config

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    
    # Load from nodes_config for consistency
    # Main analysis model
    analysis_model: str = node_config.DEFAULT_MODEL
    
    # Embedding model
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    
    # Validator model
    validator_model: str = node_config.VALIDATOR_MODEL
    
    # Model settings
    temperature: float = 0.2
    max_tokens: int = 4095
    
    # Timeouts
    request_timeout: int = 300  # seconds
    
    # Additional parameters by model type
    model_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: node_config.MODEL_PARAMS)
    
    def get_model_params(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters for a specific model
        
        Args:
            model_name: Name of the model to get parameters for, or None to use default
            
        Returns:
            Dictionary of model parameters
        """
        model = model_name or self.analysis_model
        
        # Start with default parameters
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Override with model-specific parameters if available
        if model in self.model_params:
            params.update(self.model_params[model])
            
        return params
