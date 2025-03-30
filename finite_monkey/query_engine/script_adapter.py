"""
Script adapter for query engines, enabling AI script generation and execution
based on query results and code context.
"""
import json
import os
import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from pydantic import BaseModel, Field
from loguru import logger

from finite_monkey.pipeline.core import Context
from .base_engine import QueryResult
from .flare_engine import FlareQueryEngine
from finite_monkey.pipeline.core import Pipeline

class ScriptGenerationRequest(BaseModel):
    """Request parameters for script generation"""
    query: str = Field(..., description="The query to generate a script for")
    context_snippets: List[str] = Field(default_factory=list, description="Code snippets for context")
    file_paths: List[str] = Field(default_factory=list, description="Relevant file paths")
    target_path: Optional[str] = Field(None, description="Target path for generated script")
    script_type: str = Field("analysis", description="Type of script (analysis, test, fix)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ScriptGenerationResult(BaseModel):
    """Result of script generation process"""
    script_content: str = Field(..., description="The generated script content")
    script_path: Optional[str] = Field(None, description="Path where script was saved")
    execution_command: Optional[str] = Field(None, description="Command to execute the script")
    success: bool = Field(..., description="Whether generation was successful")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryEngineScriptAdapter:
    """
    Adapter that connects query engines with AI script generation capabilities.
    
    This adapter takes query results and generates executable scripts based
    on the analysis, using the configured AI script generation tools.
    """
    
    def __init__(
        self,
        query_engine: FlareQueryEngine,
        config_path: Optional[str] = None,
        script_output_dir: Optional[str] = None
    ):
        """
        Initialize the script adapter.
        
        Args:
            query_engine: The query engine to use for script generation
            config_path: Path to the AI script configuration file
            script_output_dir: Directory where generated scripts will be saved
        """
        self.query_engine = query_engine
        self.logger = logger
        
        # Set default config path if not provided
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "genaiscript.config.json"
            )
        self.config_path = config_path
        
        # Set default output directory if not provided
        if script_output_dir is None:
            script_output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "generated_scripts"
            )
        self.script_output_dir = script_output_dir
        
        # Load configuration
        self.config = self._load_config()
        
        # Ensure output directory exists
        os.makedirs(self.script_output_dir, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the AI script configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    # Strip comments (which are not valid JSON but may be in our config)
                    content = ""
                    for line in f:
                        if '//' in line:
                            line = line.split('//')[0]
                        content += line
                    
                    if content.strip():
                        return json.loads(content)
                    return {}
            else:
                self.logger.warning(f"AI script config not found at {self.config_path}, using defaults")
                return {
                    "modelAliases": {
                        "reasoning": "openai:gpt-4o"
                    },
                    "fenceFormat": "xml"
                }
        except Exception as e:
            self.logger.error(f"Error loading AI script config: {e}")
            return {}
    
    async def generate_script(
        self,
        request: ScriptGenerationRequest,
        context: Optional[Context] = None
    ) -> ScriptGenerationResult:
        """
        Generate a script based on the given request.
        
        Args:
            request: Script generation request parameters
            context: Optional pipeline context
            
        Returns:
            ScriptGenerationResult with generated script content and metadata
        """
        try:
            # Construct a query to generate the script
            query = self._construct_script_generation_query(request)
            
            # Execute the query to get script content
            query_result = await self.query_engine.query(query, context)
            
            # Extract script content from response
            script_content = self._extract_script_content(query_result.response)
            
            # Determine script path
            script_path = self._determine_script_path(request)
            
            # Save the script if needed
            if script_path:
                os.makedirs(os.path.dirname(os.path.abspath(script_path)), exist_ok=True)
                with open(script_path, 'w') as f:
                    f.write(script_content)
                self.logger.info(f"Generated script saved to {script_path}")
            
            # Determine execution command
            execution_command = None
            if script_path and script_path.endswith('.py'):
                execution_command = f"python {script_path}"
            
            return ScriptGenerationResult(
                script_content=script_content,
                script_path=script_path,
                execution_command=execution_command,
                success=True,
                metadata={
                    "query": query,
                    "source_count": len(query_result.sources),
                    "confidence": query_result.confidence,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "script_type": request.script_type,
                    "model": self.config.get("modelAliases", {}).get("reasoning", "unknown")
                }
            )
        except Exception as e:
            self.logger.error(f"Error generating script: {e}")
            return ScriptGenerationResult(
                script_content="# Error occurred during script generation",
                success=False,
                error=str(e)
            )
    
    def _construct_script_generation_query(self, request: ScriptGenerationRequest) -> str:
        """
        Construct a query for script generation.
        
        Args:
            request: Script generation request
            
        Returns:
            Query string for the FLARE engine
        """
        # Determine purpose based on script type
        purpose_map = {
            "analysis": "analyzes the provided code for security vulnerabilities and potential issues",
            "test": "tests the functionality of the provided code to verify correctness",
            "fix": "fixes issues in the provided code while maintaining functionality",
            "generation": "implements the requested functionality based on the requirements"
        }
        purpose = purpose_map.get(request.script_type, "processes the provided code")
        
        # Format query using template
        context_text = "\n\n".join(request.context_snippets)
        file_paths_text = ", ".join(request.file_paths) if request.file_paths else "No specific files"
        
        query = f"""
Generate a complete Python script that {purpose}.

CONTEXT:
```
{context_text}
```

RELEVANT FILES: {file_paths_text}

REQUIREMENTS:
- The script should be fully functional and ready to execute
- Include proper error handling and logging
- Add comments explaining complex logic
- Make the script reusable for similar tasks
- Include a main function and proper entry point

QUERY: {request.query}

Please return the script as valid Python code without any additional text before or after.
Use proper indentation and follow PEP 8 style guidelines.
        """
        
        return query
    
    def _extract_script_content(self, response: str) -> str:
        """
        Extract script content from the query response.
        
        Args:
            response: Response from the query engine
            
        Returns:
            Extracted script content
        """
        # Look for code blocks (```python ... ```)
        import re
        code_blocks = re.findall(r'```python\s+([\s\S]+?)\s+```', response)
        
        if code_blocks:
            # Use the largest code block
            return max(code_blocks, key=len)
        
        # If no code blocks found with python tag, look for any code blocks
        code_blocks = re.findall(r'```\s+([\s\S]+?)\s+```', response)
        if code_blocks:
            return max(code_blocks, key=len)
        
        # Check for XML fence format if specified in config
        if self.config.get("fenceFormat") == "xml":
            xml_blocks = re.findall(r'<code>([\s\S]+?)</code>', response)
            if xml_blocks:
                return max(xml_blocks, key=len)
        
        # If nothing else found, just use the whole response
        return response
    
    def _determine_script_path(self, request: ScriptGenerationRequest) -> Optional[str]:
        """
        Determine the path where the script should be saved.
        
        Args:
            request: Script generation request
            
        Returns:
            Script file path or None if script shouldn't be saved
        """
        if request.target_path:
            # Use provided target path
            return request.target_path
        
        # Generate a default path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_type = request.script_type
        
        # Create a filename based on the query
        import re
        query_part = re.sub(r'[^a-zA-Z0-9]', '_', request.query[:30].lower()).strip('_')
        filename = f"{script_type}_{query_part}_{timestamp}.py"
        
        return os.path.join(self.script_output_dir, filename)


"""
LLM adapter for interacting with various LLM backends.
Provides a unified interface for making LLM calls.
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
from loguru import logger

from ..nodes_config import config

class LLMAdapter:
    """
    Adapter for LLM interactions.
    Provides a unified interface for different LLM backends.
    """
    
    def __init__(self, model: str = None, provider: str = None, base_url: str = None):
        """
        Initialize the LLM adapter.
        
        Args:
            model: Model name to use
            provider: Provider name (e.g., 'openai', 'ollama')
            base_url: Base URL for the provider's API
        """
        self.model = model or config.DEFAULT_MODEL
        self.provider = provider or config.DEFAULT_PROVIDER
        self.base_url = base_url or config.get('PROVIDER_URLS', {}).get(self.provider)
        self.logger = logger
        self._client = None
    
    async def _ensure_client(self):
        """Ensure client is initialized"""
        if self._client is None:
            if self.provider == 'ollama':
                from .ollama import AsyncOllamaClient
                self._client = AsyncOllamaClient(model=self.model, base_url=self.base_url)
            elif self.provider == 'openai':
                from .openai import AsyncOpenAIClient
                self._client = AsyncOpenAIClient(model=self.model)
            else:
                self.logger.warning(f"Unknown provider: {self.provider}, using default")
                from .default import AsyncDefaultClient
                self._client = AsyncDefaultClient(model=self.model)
    
    async def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The prompt text
            options: Additional options for the generation
            
        Returns:
            Generated response
        """
        await self._ensure_client()
        try:
            return await self._client.generate(prompt, options or {})
        except Exception as e:
            self.logger.error(f"Error in LLM generation: {e}")
            return f"Error: {str(e)}"
    
    async def llm(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Core LLM method that cognitive_bias_analyzer.py and other components expect.
        This was missing and causing the warning.
        
        Args:
            prompt: The prompt text
            options: Additional options for the generation
            
        Returns:
            Generated response
        """
        return await self.generate(prompt, options)
    
    async def structured_generate(self, prompt: str, response_format: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a structured response according to the specified format.
        
        Args:
            prompt: The prompt text
            response_format: Format specification for the response
            options: Additional options for the generation
            
        Returns:
            Structured response
        """
        await self._ensure_client()
        try:
            if hasattr(self._client, 'structured_generate'):
                return await self._client.structured_generate(prompt, response_format, options or {})
            else:
                # Fallback to regular generation and attempt to parse
                import json
                response_text = await self.generate(
                    f"{prompt}\n\nRespond with a JSON object following this format: {json.dumps(response_format)}",
                    options
                )
                try:
                    return json.loads(response_text)
                except:
                    # If JSON parsing fails, try to extract JSON from the response
                    import re
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                    if json_match:
                        try:
                            return json.loads(json_match.group(1))
                        except:
                            pass
                    return {"error": "Failed to parse structured response", "raw_response": response_text}
        except Exception as e:
            self.logger.error(f"Error in structured LLM generation: {e}")
            return {"error": str(e)}
    
    def _format_response(self, response: str, response_format: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the response based on the expected format.
        
        Args:
            response: Raw response text
            response_format: Expected format of the response
            
        Returns:
            Formatted response dictionary
        """
        formatted_response = {}
        for key, value in response_format.items():
            formatted_response[key] = response.get(value, "")
        return formatted_response


DEFAULT_QUERIES = [
    "What are the main security vulnerabilities in this contract?",
    "How could the gas efficiency be improved?",
    "Are there any business logic flaws in the contract?",
    "What are the key roles and permissions in this contract?",
    "Are there any reentrancy vulnerabilities?"
]


async def setup_pipeline(input_path: str, factory) -> Dict[str, Any]:
    """
    Set up the analysis pipeline and run document loading stages.
    
    Args:
        input_path: Path to the smart contract or project to analyze
        factory: The pipeline factory instance
        
    Returns:
        Dictionary with context and pipeline
    """
    context = Context(input_path=input_path)
    pipeline_stages = []
    
    # Add document loading stage
    load_documents_stage = await factory.load_documents(context)
    pipeline_stages.append(load_documents_stage)
    
    # Add contract extraction stage
    extract_contracts_stage = await factory.extract_contracts_from_files(context)
    pipeline_stages.append(extract_contracts_stage)
    
    # Create the pipeline
    pipeline = Pipeline(stages=pipeline_stages)
    
    # Initialize query engine in context
    context.query_engine = factory.get_query_engine()
    
    return {"context": context, "pipeline": pipeline}


async def analyze_contract(
    input_path: str,
    queries: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    generate_scripts: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Analyze a smart contract using the FLARE query engine.
    
    Args:
        input_path: Path to the smart contract or project to analyze
        queries: Optional list of specific queries to run
        output_path: Optional path to save the analysis results
        generate_scripts: Whether to generate analysis scripts
        verbose: Whether to enable verbose logging
        
    Returns:
        Analysis results dictionary
    """
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    logger.info(f"Analyzing contract/project at: {input_path}")
    
    # Initialize factory
    factory = PipelineFactory(config)
    
    # Set up pipeline
    setup_result = await setup_pipeline(input_path, factory)
    context = setup_result["context"]
    pipeline = setup_result["pipeline"]
    
    # Run pipeline
    await pipeline.run(context)
    
    # Initialize query engine adapter
    query_engine_adapter = QueryEngineScriptAdapter(context.query_engine)
    
    # Run queries
    results = {}
    queries = queries or DEFAULT_QUERIES
    for query in queries:
        request = ScriptGenerationRequest(
            query=query,
            context_snippets=context.get_snippets(),
            file_paths=context.get_file_paths(),
            script_type="analysis"
        )
        result = await query_engine_adapter.generate_script(request, context)
        results[query] = result
    
    # Save results if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Analysis results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Smart contract analyzer using FLARE query engine")
    parser.add_argument("input_path", help="Path to the smart contract or project to analyze")
    parser.add_argument("--queries", nargs="*", help="Specific queries to run")
    parser.add_argument("--output", help="Path to save the analysis results")
    parser.add_argument("--generate-scripts", action="store_true", help="Generate analysis scripts")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    asyncio.run(analyze_contract(
        input_path=args.input_path,
        queries=args.queries,
        output_path=args.output,
        generate_scripts=args.generate_scripts,
        verbose=args.verbose
    ))


if __name__ == "__main__":
    main()