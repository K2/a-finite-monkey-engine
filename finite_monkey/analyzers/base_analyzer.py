from matplotlib.cbook import ls_mapper


from typing import Dict, Any, Optional
from loguru import logger

from llama_index.core.settings import Settings
from ..llm.llama_index_adapter import LlamaIndexAdapter
from ..pipeline.core import Context

class BaseAnalyzer:
    """Base class for all analyzers"""
    
    def __init__(self, llm_adapter: Optional[LlamaIndexAdapter] = None):
        """
        Initialize the analyzer with an LLM adapter
        
        Args:
            llm_adapter: LlamaIndex adapter for LLM access
        """
        self.llm_adapter = llm_adapter
    
    @property
    def llm(self):
        """Get the LLM from adapter or settings"""
        if self.llm_adapter and hasattr(self.llm_adapter, 'llm'):
            return self.llm_adapter.llm
        return Settings.llm
    
    async def process(self, context: Context) -> Context:
        """
        Process the context using this analyzer
        
        Args:
            context: Context to process
            
        Returns:
            Updated context
        """
        raise NotImplementedError("Analyzers must implement process method")
