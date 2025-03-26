"""
Lazy model provider setup that integrates with llama_index Settings
"""

import os
from typing import Dict, Any, Optional
from loguru import logger

from llama_index.core.settings import Settings
from ..nodes_config import config as node_config

class ModelProvider:
    """
    Lazy model provider management that works with llama_index Settings
    """
    # Track whether models have been initialized
    _initialized_models = {}
    
    @staticmethod
    def get_analyzer_llm(analyzer_type: str):
        """
        Lazily initialize and get LLM for a specific analyzer type
        
        Args:
            analyzer_type: Type of analyzer (e.g., 'cognitive_bias', 'vulnerability')
            
        Returns:
            LLM instance configured for this analyzer
        """
        # Return cached instance if already initialized
        if analyzer_type in ModelProvider._initialized_models:
            return ModelProvider._initialized_models[analyzer_type]
        
        # Otherwise create a new instance
        try:
            llm = None
            
            # Initialize based on analyzer type
            if analyzer_type == "business_flow":
                llm = ModelProvider._init_analyzer_llm(
                    node_config.BUSINESS_FLOW_MODEL,
                    node_config.BUSINESS_FLOW_MODEL_PROVIDER,
                    node_config.BUSINESS_FLOW_MODEL_BASE_URL
                )
            elif analyzer_type == "vulnerability":
                llm = ModelProvider._init_analyzer_llm(
                    node_config.SCAN_MODEL,
                    node_config.SCAN_MODEL_PROVIDER,
                    node_config.SCAN_MODEL_BASE_URL
                )
            elif analyzer_type == "cognitive_bias":
                llm = ModelProvider._init_analyzer_llm(
                    node_config.COGNITIVE_BIAS_MODEL,
                    node_config.COGNITIVE_BIAS_MODEL_PROVIDER,
                    node_config.COGNITIVE_BIAS_MODEL_BASE_URL
                )
            elif analyzer_type == "documentation":
                llm = ModelProvider._init_analyzer_llm(
                    node_config.DOCUMENTATION_MODEL,
                    node_config.DOCUMENTATION_MODEL_PROVIDER,
                    node_config.DOCUMENTATION_MODEL_BASE_URL
                )
            elif analyzer_type == "counterfactual":
                llm = ModelProvider._init_analyzer_llm(
                    node_config.COUNTERFACTUAL_MODEL,
                    node_config.COUNTERFACTUAL_MODEL_PROVIDER,
                    node_config.COUNTERFACTUAL_MODEL_BASE_URL
                )
            elif analyzer_type == "validator":
                llm = ModelProvider._init_analyzer_llm(
                    node_config.VALIDATOR_MODEL,
                    node_config.VALIDATOR_MODEL_PROVIDER,
                    node_config.VALIDATOR_MODEL_BASE_URL
                )
            else:
                # Fall back to default LLM
                logger.warning(f"Unknown analyzer type: {analyzer_type}, using default LLM")
                llm = ModelProvider.get_default_llm()
            
            # Cache the initialized model
            if llm is not None:
                ModelProvider._initialized_models[analyzer_type] = llm
                
            return llm
                
        except Exception as e:
            logger.error(f"Failed to get LLM for analyzer {analyzer_type}: {e}")
            return ModelProvider.get_default_llm()  # Fall back to default LLM
    
    @staticmethod
    def get_default_llm():
        """Lazily initialize and get default LLM"""
        if 'default' in ModelProvider._initialized_models:
            return ModelProvider._initialized_models['default']
            
        # Check if already configured in Settings
        if Settings.llm is not None:
            ModelProvider._initialized_models['default'] = Settings.llm
            return Settings.llm
            
        # Initialize new default LLM
        try:
            # Use the default model from config
            model_name = node_config.DEFAULT_MODEL
            provider = node_config.DEFAULT_PROVIDER
            base_url = node_config.DEFAULT_BASE_URL
            model_params = node_config.MODEL_PARAMS.get(model_name, node_config.MODEL_PARAMS["default"])
            
            # Initialize based on provider
            llm = ModelProvider._init_analyzer_llm(model_name, provider, base_url)
            
            # Set as default LLM in Settings
            Settings.llm = llm
            ModelProvider._initialized_models['default'] = llm
            logger.info(f"Lazily initialized default LLM: {model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize default LLM: {e}")
            return None
    
    @staticmethod
    def _init_analyzer_llm(model_name: str, provider: str, base_url: str):
        """Initialize LLM for a specific analyzer"""
        model_params = node_config.MODEL_PARAMS.get(model_name, node_config.MODEL_PARAMS["default"])
        
        # Always use Ollama in development
        from llama_index.llms.ollama import Ollama
        
        # Handle special case for Hugging Face models
        if model_name.startswith("hf.co/"):
            model_name = model_name.replace("hf.co/", "")
        
        # Create Ollama instance
        return Ollama(
            model=model_name,
            base_url=base_url if base_url else None,  # Default URL if empty
            **model_params
        )
    
    @staticmethod
    def release_model(analyzer_type: str):
        """
        Release a model to free memory
        
        Args:
            analyzer_type: Type of analyzer to release model for
        """
        if analyzer_type in ModelProvider._initialized_models:
            logger.info(f"Releasing model for {analyzer_type}")
            del ModelProvider._initialized_models[analyzer_type]
    
    @staticmethod
    def log_model_configuration():
        """Log the current model configuration"""
        logger.info("------ LLM Model Configuration ------")
        
        # Log default LLM
        if Settings.llm is not None:
            logger.info(f"Default LLM: {Settings.llm.__class__.__name__} - {getattr(Settings.llm, 'model_name', 'unknown')}")
        else:
            logger.warning("No default LLM configured")
            
        # Log embedding model
        if Settings.embed_model is not None:
            logger.info(f"Embedding Model: {Settings.embed_model.__class__.__name__} - {getattr(Settings.embed_model, 'model_name', 'unknown')}")
        else:
            logger.warning("No embedding model configured")
            
        # Log specialized models
        if hasattr(Settings, 'llm_dict'):
            for name, llm in Settings.llm_dict.items():
                logger.info(f"Specialized LLM - {name}: {llm.__class__.__name__} - {getattr(llm, 'model_name', 'unknown')}")
                
        logger.info("-------------------------------------")
