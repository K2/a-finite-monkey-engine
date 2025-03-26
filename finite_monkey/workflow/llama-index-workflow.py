from logging import config
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.prompts import PromptTemplate
#from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Fix: Import AsyncOllamaClient with the correct alias directly from the module
from finite_monkey.adapters.ollama import AsyncOllamaClient as Ollama
from finite_monkey.core_async_analyzer import AsyncAnalyzer
from finite_monkey.db.manager import DatabaseManager
from finite_monkey.nodes_config import nodes_config

async def main():
    # Get configuration
    config = nodes_config()
    
    # Use the base URL from config
    #api_base = config.OPENAI_API_BASE
    api_base = ""
    db_manager = DatabaseManager(db_url=config.ASYNC_DB_URL)
    
    # Use the model values directly from config (with default values set)
    workflow_model = config.WORKFLOW_MODEL
    query_model = config.QUERY_MODEL
    scan_model = config.SCAN_MODEL
    
    # Log model names to help with debugging
    print(f"Using models - Workflow: {workflow_model}, Query: {query_model}, Scan: {scan_model}")
    
    try:
        # Initialize LLM clients with validated model names
        primary_llm = Ollama(model=workflow_model)
        secondary_llm = Ollama(model=query_model)
        
        # Initialize analyzer
        analyzer = AsyncAnalyzer(
            primary_llm_client=primary_llm,
            secondary_llm_client=secondary_llm,
            db_manager=db_manager,
            primary_model_name=scan_model,
            secondary_model_name=query_model, 
        )
    except Exception as e:
        print(f"Error initializing LLM clients or analyzer: {str(e)}")
        # Fall back to initializing the analyzer without custom clients
        analyzer = AsyncAnalyzer(
            db_manager=db_manager,
            primary_model_name=config.SCAN_MODEL,  
            secondary_model_name=config.CONFIRMATION_MODEL,
        )
    
    if config.base_dir.find("github.com") > 0:
        # GitHub repo case would be handled here
        pass
    else:
        # Analyze entire project
        print(f"Analyzing project directory: {config.base_dir}")
        results = await analyzer.analyze_contract_file(
            project_path=config.base_dir,
            project_id=config.id,
            query=config.USER_QUERY
        )
        print(results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
