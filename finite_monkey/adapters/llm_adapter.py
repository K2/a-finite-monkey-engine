# ...existing code...
from loguru import logger
from ..nodes_config import config as Settings
class LLMAdapter:
    """Adapter for LLM interactions"""
    
    def __init__(self):
        """Initialize the LLM adapter"""
        # ...existing code...
        self.tracker = None
        
    def set_tracker(self, tracker):
        """Set the LLM interaction tracker"""
        self.tracker = tracker
    
    async def acomplete(self, prompt: str, stage_name: str = None):
        """
        Complete a prompt asynchronously with tracking
        
        Args:
            prompt: Prompt to complete
            stage_name: Optional name of the calling stage
            
        Returns:
            LLM response
        """
        # Use tracker if available
        if self.tracker and stage_name:
            llm = Settings.llm
            return await self.tracker.track_interaction(
                stage_name, 
                prompt, 
                lambda p: llm.acomplete(p)
            )
        else:
            # Fall back to regular completion
            return await Settings.llm.acomplete(prompt)
    
def _initialize_llm(self):
        """Initialize the LLM based on the parameters"""
        try:
            if self.analyzer_type and not (self.provider and self.model_name):
                # Use analyzer-specific LLM
                self._llm = ModelProvider.get_analyzer_llm(self.analyzer_type)
                logger.info(f"Lazily initialized LLM for analyzer type: {self.analyzer_type}")
            elif self.provider and self.model_name:
                # Always use Ollama in development
                from llama_index.llms.ollama import Ollama
                
                # Handle special case for Hugging Face models
                llm_model_name = self.model_name
                if llm_model_name.startswith("hf.co/"):
                    llm_model_name = llm_model_name.replace("hf.co/", "")
                
                self._llm = Ollama(
                    model=llm_model_name,
                    base_url=self.base_url if self.base_url else None,
                    **(self.model_params or {})
                )
                logger.info(f"Lazily initialized Ollama LLM with model: {llm_model_name}")
            else:
                # Fall back to default LLM
                logger.info("Lazily initializing default LLM")
                self._llm = ModelProvider.get_default_llm()
                
        except Exception as e:
            logger.error(f"Error lazily initializing LLM: {e}")
            self._llm = ModelProvider.get_default_llm()
            
        # If still no LLM, raise error
        if self._llm is None:
            raise ValueError("Failed to initialize LLM")

    # ...existing code...
