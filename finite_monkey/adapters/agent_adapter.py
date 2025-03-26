"""
Adapter for integrating agent-based analyzers into the pipeline framework.
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

from ..pipeline.core import Context
from ..llm.llama_index_adapter import LlamaIndexAdapter
from finite_monkey.nodes_config import config

class DocumentationInconsistencyAdapter:
    """Analyzes inconsistencies between documentation and implementation"""
    
    def __init__(self, llm_adapter=None):
        """Initialize the documentation inconsistency adapter"""
        if llm_adapter is None:
            try:
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                
                # Use validator model for inconsistency checks
                self.llm_adapter = LlamaIndexAdapter(
                    model_name=config.VALIDATOR_MODEL,
                    provider=config.VALIDATOR_MODEL_PROVIDER,
                    base_url=config.VALIDATOR_MODEL_BASE_URL
                )
                logger.info(f"Created documentation inconsistency adapter with model: {config.VALIDATOR_MODEL}")
            except Exception as e:
                logger.error(f"Failed to create documentation inconsistency adapter: {e}")
                self.llm_adapter = None
        else:
            self.llm_adapter = llm_adapter
    
    async def process(self, context: Context) -> Context:
        """
        Process the context to find documentation inconsistencies
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with inconsistency analysis
        """
        logger.info("Starting documentation inconsistency analysis")
        
        if not hasattr(context, 'inconsistencies'):
            context.inconsistencies = {}
            
        # Check if needed data is available
        if not hasattr(context, 'documentation_analysis'):
            logger.warning("No documentation analysis available for inconsistency check")
            context.add_error(
                stage="documentation_inconsistency_analysis",
                message="No documentation analysis available"
            )
            return context
            
        # Process each solidity file
        for file_id, file_data in context.files.items():
            if not file_data.get("is_solidity", False):
                continue
                
            # Skip files without documentation analysis
            if file_id not in context.documentation_analysis:
                continue
                
            try:
                # Find inconsistencies
                inconsistencies = await self._check_inconsistencies(
                    file_data,
                    context.documentation_analysis[file_id]
                )
                
                if inconsistencies:
                    context.inconsistencies[file_id] = inconsistencies
                    
                logger.info(f"Completed inconsistency check for {file_id}: found {len(inconsistencies)}")
                
            except Exception as e:
                logger.error(f"Error checking inconsistencies for {file_id}: {str(e)}")
                context.add_error(
                    stage="documentation_inconsistency_analysis",
                    message=f"Failed to analyze documentation inconsistencies for {file_id}",
                    exception=e
                )
        
        return context
    
    async def _check_inconsistencies(self, file_data, doc_analysis):
        """Check for inconsistencies between documentation and implementation"""
        if not self.llm_adapter:
            return []
        
        content = file_data.get("content", "")
        
        # Create prompt for inconsistency checking
        prompt = f"""
        Compare the implementation and documentation of this Solidity contract to find inconsistencies:
        
        ```solidity
        {content}
        ```
        
        Focus on:
        1. Parameter descriptions that don't match implementation
        2. Return value descriptions that don't match actual returns
        3. Function behavior descriptions that contradict the code
        4. Documented invariants that aren't enforced
        
        Respond with a JSON array with the following structure:
        [
          {{
            "function": "transfer(address to, uint256 amount)",
            "inconsistency": "Documentation states tokens are locked for 1 day, but no time check exists",
            "severity": "high",
            "recommendation": "Add time lock or update documentation"
          }}
        ]
        """
        
        try:
            # Get response from LLM
            llm = self.llm_adapter.llm
            response = await llm.acomplete(prompt)
            
            # Process response
            from ..utils.json_repair import safe_parse_json, extract_json_from_text
            
            # Extract and parse JSON
            json_str = extract_json_from_text(response.text)
            inconsistencies = safe_parse_json(json_str, [])
            
            # Ensure we have a list
            if not isinstance(inconsistencies, list):
                inconsistencies = []
            
            return inconsistencies
            
        except Exception as e:
            logger.error(f"Error checking inconsistencies: {str(e)}")
            return []
