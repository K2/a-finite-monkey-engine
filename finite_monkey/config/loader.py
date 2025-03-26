"""
Configuration loader for the Finite Monkey Engine
"""

import os
import json
from typing import Dict, Any, Optional
from loguru import logger

from .model_config import ModelConfig

class ConfigLoader:
    """
    Load configuration from environment variables or config file
    """
    
    @staticmethod
    def load(config_path: Optional[str] = None) -> ModelConfig:
        """
        Load configuration
        
        Args:
            config_path: Path to configuration file, or None to use defaults
            
        Returns:
            ModelConfig instance
        """
        # Create config with defaults
        config = ModelConfig()
        
        # Try to load from environment variables
        ConfigLoader._load_from_env(config)
        
        # Try to load from file if provided
        if config_path:
            ConfigLoader._load_from_file(config, config_path)
            
        return config
    
    @staticmethod
    def _load_from_env(config: ModelConfig):
        """Load configuration from environment variables"""
        # Main models
        if os.environ.get("FM_ANALYSIS_MODEL"):
            config.analysis_model = os.environ.get("FM_ANALYSIS_MODEL")
            
        if os.environ.get("FM_EMBEDDING_MODEL"):
            config.embedding_model = os.environ.get("FM_EMBEDDING_MODEL")
            
        if os.environ.get("FM_VALIDATOR_MODEL"):
            config.validator_model = os.environ.get("FM_VALIDATOR_MODEL")
            
        # General settings
        if os.environ.get("FM_MODEL_TEMPERATURE"):
            try:
                config.temperature = float(os.environ.get("FM_MODEL_TEMPERATURE"))
            except ValueError:
                pass
                
        if os.environ.get("FM_MODEL_MAX_TOKENS"):
            try:
                config.max_tokens = int(os.environ.get("FM_MODEL_MAX_TOKENS"))
            except ValueError:
                pass
                
        if os.environ.get("FM_REQUEST_TIMEOUT"):
            try:
                config.request_timeout = int(os.environ.get("FM_REQUEST_TIMEOUT"))
            except ValueError:
                pass
    
    @staticmethod
    def _load_from_file(config: ModelConfig, config_path: str):
        """Load configuration from file"""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                
            # Main models
            if "analysis_model" in config_data:
                config.analysis_model = config_data["analysis_model"]
                
            if "embedding_model" in config_data:
                config.embedding_model = config_data["embedding_model"]
                
            if "validator_model" in config_data:
                config.validator_model = config_data["validator_model"]
                
            # General settings
            if "temperature" in config_data:
                config.temperature = float(config_data["temperature"])
                
            if "max_tokens" in config_data:
                config.max_tokens = int(config_data["max_tokens"])
                
            if "request_timeout" in config_data:
                config.request_timeout = int(config_data["request_timeout"])
                
            # Model-specific params
            if "model_params" in config_data and isinstance(config_data["model_params"], dict):
                config.model_params.update(config_data["model_params"])
                
        except Exception as e:
            logger.error(f"Error loading config from file {config_path}: {e}")
