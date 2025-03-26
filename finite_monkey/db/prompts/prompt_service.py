"""
Prompt service to integrate database prompts with the prompting system

This module provides a service for retrieving dynamic prompts from the database
and rendering them with parameters.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import json
import logging
import re
from functools import lru_cache

from finite_monkey.db.prompts.manager import PromptManager
from finite_monkey.utils.prompting import get_analysis_prompt, get_validation_prompt, get_report_prompt, format_prompt

logger = logging.getLogger(__name__)

class PromptService:
    """Service for retrieving and rendering dynamic prompts"""
    
    def __init__(self):
        """Initialize the prompt service"""
        self.manager = PromptManager()
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the prompt service"""
        if self._initialized:
            return
            
        async with self._init_lock:
            if self._initialized:
                return
                
            await self.manager.initialize()
            self._initialized = True
    
    async def get_prompt(self, name: str, project_id: Optional[str] = None, 
                        **kwargs) -> str:
        """
        Get a prompt from the database
        
        Args:
            name: Prompt name
            project_id: Project ID (optional)
            **kwargs: Parameters for prompt rendering
            
        Returns:
            Rendered prompt
        """
        if not self._initialized:
            await self.initialize()
            
        # Try to get from database
        prompt_obj = await self.manager.get_prompt(name, project_id)
        
        if prompt_obj:
            # Render the prompt with parameters
            content = prompt_obj.content
            
            # Replace placeholders using a simple {var} syntax
            for key, value in kwargs.items():
                placeholder = f"{{{key}}}"
                if placeholder in content:
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, indent=2)
                    content = content.replace(placeholder, str(value))
                    
            return content
            
        # No database prompt found, use fallback from prompting module
        logger.info(f"No database prompt found for '{name}', using fallback")
        
        # Map known prompt names to fallback functions
        if name == "analysis_prompt":
            return get_analysis_prompt(**kwargs)
        elif name == "validation_prompt":
            return get_validation_prompt(**kwargs)
        elif name == "report_prompt":
            return get_report_prompt(**kwargs)
        else:
            # Try format_prompt for other prompts
            try:
                return format_prompt(name, **kwargs)
            except ValueError:
                # Construct a generic prompt as last resort
                parts = ["# " + name.replace("_", " ").title()]
                
                for key, value in kwargs.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, indent=2)
                    parts.append(f"\n## {key.replace('_', ' ').title()}\n{value}")
                    
                return "\n".join(parts)
    
    async def get_business_flow_prompt(self, function_name: str, contract_name: str,
                                      contract_code: str, project_id: Optional[str] = None) -> str:
        """
        Get a business flow analysis prompt
        
        Args:
            function_name: Function name
            contract_name: Contract name
            contract_code: Contract code
            project_id: Project ID (optional)
            
        Returns:
            Rendered prompt
        """
        if not self._initialized:
            await self.initialize()
            
        params = {
            "function_name": function_name,
            "contract_name": contract_name,
            "contract_code": contract_code
        }
        
        # Try to get a specific prompt for this function
        prompt_obj = await self.manager.get_prompt(f"business_flow_{function_name}", project_id)
        
        if prompt_obj:
            # Render the prompt with parameters
            content = prompt_obj.content
            
            # Replace placeholders
            for key, value in params.items():
                placeholder = f"{{{key}}}"
                if placeholder in content:
                    content = content.replace(placeholder, str(value))
                    
            return content
            
        # Fall back to generic business flow prompt
        return await self.get_prompt("business_flow_analysis", project_id, **params)
    
    async def get_cognitive_bias_prompt(self, bias_type: str, contract_code: str,
                                       project_id: Optional[str] = None) -> str:
        """
        Get a cognitive bias analysis prompt
        
        Args:
            bias_type: Bias type
            contract_code: Contract code
            project_id: Project ID (optional)
            
        Returns:
            Rendered prompt
        """
        if not self._initialized:
            await self.initialize()
            
        params = {
            "bias_type": bias_type,
            "contract_code": contract_code
        }
        
        # Try to get a specific prompt for this bias type
        prompt_obj = await self.manager.get_prompt(f"cognitive_bias_{bias_type}", project_id)
        
        if prompt_obj:
            # Render the prompt with parameters
            content = prompt_obj.content
            
            # Replace placeholders
            for key, value in params.items():
                placeholder = f"{{{key}}}"
                if placeholder in content:
                    content = content.replace(placeholder, str(value))
                    
            return content
            
        # Fall back to generic cognitive bias prompt
        return await self.get_prompt("cognitive_bias_analysis", project_id, **params)
    
    async def store_prompt_result(self, prompt_name: str, result: str, 
                                 project_id: Optional[str] = None) -> bool:
        """
        Store a prompt result
        
        Args:
            prompt_name: Prompt name
            result: Prompt result
            project_id: Project ID (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            await self.initialize()
            
        # Try to get the prompt
        prompt_obj = await self.manager.get_prompt(prompt_name, project_id)
        
        if prompt_obj:
            # Store the result
            prompt_obj.set_result(result)
            
            # Store in database
            async with self.manager.async_session() as session:
                session.add(prompt_obj)
                await session.commit()
                
            return True
        
        # Cache the result for prompts that don't exist in the database
        key = f"result:{prompt_name}"
        if project_id:
            key += f":{project_id}"
            
        return await self.manager.store_cache_entry(key, result)


# Global instance
prompt_service = PromptService()
