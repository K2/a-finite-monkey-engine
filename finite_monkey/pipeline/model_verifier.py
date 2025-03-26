"""
Simplified model verification that just checks availability
"""

import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger

from llama_index.core.settings import Settings
from ..pipeline.core import Context
from ..config.model_provider import ModelProvider

class ModelVerifier:
    """
    Simple verification of model availability
    """
    
    async def verify_model(self, model_type: str) -> bool:
        """
        Simple verification - just check if model exists and can respond to a prompt
        
        Args:
            model_type: Type of model to verify
            
        Returns:
            True if model responds, False otherwise
        """
        try:
            # Get the model
            model = ModelProvider.get_analyzer_llm(model_type)
            
            # Just check if it exists
            if model is None:
                logger.warning(f"Model for {model_type} is None")
                return False
                
            # Simple test prompt
            try:
                response = await asyncio.wait_for(
                    model.acomplete("hello"), 
                    timeout=5.0
                )
                logger.info(f"Model {model_type} responded to test prompt")
                return True
            except Exception as e:
                logger.warning(f"Model {model_type} failed to respond to test prompt: {e}")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to verify model {model_type}: {e}")
            return False
