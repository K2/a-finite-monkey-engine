"""
Central configuration for LLM setup
"""

from typing import Optional
import os
from loguru import logger

from llama_index.core.settings import Settings
from ..config.model_config import ModelConfig

def setup_default_llm(config: Optional[ModelConfig] = None):
    """
    Set up the default LLM in Settings for components that don't have their own adapter
    
    Args:
        config: Model configuration, or None to use default
    """
    if config is None:
        from .model_config import ModelConfig
        config = ModelConfig()
    
    # Check if an LLM is already configured
    if Settings.llm is not None:
        logger.info("Default LLM already configured, skipping setup")
        return
    
    # Try to set up the default LLM
    try:
        model_name = config.analysis_model
        if model_name.startswith("qwen") or model_name.endswith("q8_0") or ":" in model_name:
            # For Ollama models
            from llama_index.llms.ollama import Ollama
            
            # Get model parameters
            params = config.get_model_params(model_name)
            
            # Initialize the LLM
            llm = Ollama(
                model=model_name,
                **params
            )
            
            # Set it as the default LLM
            Settings.llm = llm
            logger.info(f"Configured default LLM with Ollama model: {model_name}")
        elif model_name.startswith("anthropic/"):
            # For Anthropic models
            try:
                from llama_index.llms.anthropic import Anthropic
                
                # Get model parameters
                params = config.get_model_params(model_name)
                
                # Initialize the LLM
                llm = Anthropic(
                    model=model_name,
                    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                    **params
                )
                
                # Set it as the default LLM
                Settings.llm = llm
                logger.info(f"Configured default LLM with Anthropic model: {model_name}")
            except ImportError:
                logger.warning("Anthropic package not available, falling back to OpenAI")
                _setup_fallback_llm()
        else:
            # For other models, fall back to a default
            _setup_fallback_llm()
    except Exception as e:
        logger.error(f"Failed to set up default LLM: {e}")
        _setup_fallback_llm()
    
    # Log model status
    log_model_status()

def _setup_fallback_llm():
    """Set up a fallback LLM when the primary configuration fails"""
    try:
        # Try to use Ollama with a common model
        from llama_index.llms.ollama import Ollama
        
        # Initialize a basic Ollama model
        llm = Ollama(
            model="qwen2.5-coder:7b-instruct-q8_0",
            temperature=0.2,
            max_tokens=1500
        )
        
        # Set it as the default LLM
        Settings.llm = llm
        logger.info("Configured default fallback LLM with Ollama")
    except Exception as e:
        logger.error(f"Failed to set up fallback LLM: {e}")
        logger.warning("No default LLM available - some analyzers may not work correctly")

def log_model_status():
    """Log the status of all configured models"""
    models = [
        ("Business Flow Model", node_config.BUSINESS_FLOW_MODEL),
        ("Vulnerability Scan Model", node_config.SCAN_MODEL),
        ("Cognitive Bias Model", node_config.COGNITIVE_BIAS_MODEL),
        ("Documentation Model", node_config.DOCUMENTATION_MODEL),
        ("Counterfactual Model", node_config.COUNTERFACTUAL_MODEL),
        ("Validator Model", node_config.VALIDATOR_MODEL)
    ]
    
    logger.info("------ LLM Model Configuration ------")
    for name, model in models:
        logger.info(f"{name}: {model}")
    logger.info("-------------------------------------")
