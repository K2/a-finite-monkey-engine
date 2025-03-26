"""
Configuration module for guidance integration.
Handles loading and validating configuration for LLM providers.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "default_model": "openai:gpt-4o",
    "openai": {
        "api_key": None,
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "anthropic": {
        "api_key": None,
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "ollama": {
        "host": "localhost:11434"
    },
    "transformers": {
        "device_map": "auto"
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load guidance configuration from file or environment variables.
    
    Args:
        config_path: Optional path to config file (JSON or YAML)
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from file if provided
    if config_path:
        try:
            path = Path(config_path)
            if path.exists():
                if path.suffix.lower() in ['.json']:
                    with open(path, 'r') as f:
                        file_config = json.load(f)
                elif path.suffix.lower() in ['.yaml', '.yml']:
                    with open(path, 'r') as f:
                        file_config = yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported config file format: {path.suffix}")
                    file_config = {}
                
                # Update the default config with values from file
                _deep_update(config, file_config)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
    
    # Update with environment variables
    env_config = _get_env_config()
    _deep_update(config, env_config)
    
    return config

def _get_env_config() -> Dict[str, Any]:
    """Extract configuration from environment variables."""
    env_config = {}
    
    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        if "openai" not in env_config:
            env_config["openai"] = {}
        env_config["openai"]["api_key"] = os.environ["OPENAI_API_KEY"]
    
    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        if "anthropic" not in env_config:
            env_config["anthropic"] = {}
        env_config["anthropic"]["api_key"] = os.environ["ANTHROPIC_API_KEY"]
    
    # Ollama
    if os.environ.get("OLLAMA_HOST"):
        if "ollama" not in env_config:
            env_config["ollama"] = {}
        env_config["ollama"]["host"] = os.environ["OLLAMA_HOST"]
    
    # Default model
    if os.environ.get("GUIDANCE_DEFAULT_MODEL"):
        env_config["default_model"] = os.environ["GUIDANCE_DEFAULT_MODEL"]
    
    return env_config

def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the config file
    """
    path = Path(config_path)
    try:
        if path.suffix.lower() in ['.json']:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
        else:
            logger.error(f"Unsupported config file format: {path.suffix}")
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        raise
