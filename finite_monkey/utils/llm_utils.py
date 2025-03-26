"""
LLM utility functions using the new Settings approach
"""

from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ..nodes_config import config

def configure_default_settings( model_name=config.WORKFLOW_MODEL, base_url="http://localhost:11434"):
    """Configure default LlamaIndex settings"""
    # Create the LLM
    llm = Ollama(
        model=model_name,
        base_url=base_url,
        temperature=0.1,
        request_timeout=config.REQUEST_TIMEOUT,
    )
    
    # Try to use HuggingFace embeddings
    try:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception as e:
        import logging
        logging.warning(f"Failed to initialize HuggingFace embedding model: {e}")
        embed_model = None
    
    # Configure global settings
    Settings.llm = llm
    if embed_model:
        Settings.embed_model = embed_model
    
    return Settings
