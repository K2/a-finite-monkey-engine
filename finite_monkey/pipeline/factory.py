import traceback
from typing import List, Dict, Any, Optional, Callable, Awaitable
from pathlib import Path
import os
import asyncio
from loguru import logger

from llama_index.core.settings import Settings

from .core import Pipeline, Stage, Context
from ..utils.chunking import AsyncContractChunker
from ..analyzers.business_flow_extractor import BusinessFlowExtractor
from ..analyzers.vulnerability_scanner import VulnerabilityScanner
from ..analyzers.dataflow_analyzer import DataFlowAnalyzer
from ..analyzers.cognitive_bias_analyzer import CognitiveBiasAnalyzer
from ..analyzers.counterfactual_analyzer import CounterfactualAnalyzer
from ..analyzers.documentation_analyzer import DocumentationAnalyzer
from ..agents.documentation_analyzer import DocumentationAnalyzer as DocInconsistencyAnalyzer
from ..utils.flow_joiner import FlowJoiner
from ..adapters.agent_adapter import DocumentationInconsistencyAdapter
from ..utils.llm_monitor import LLMInteractionTracker
from ..adapters.validator_adapter import ValidatorAdapter
from ..adapters.llm_adapter import LLMAdapter
from ..config.llm_setup import setup_default_llm
from ..nodes_config import config  # This is the correct import

from ..query_engine.base_engine import BaseQueryEngine, QueryResult
from ..query_engine.flare_engine import FlareQueryEngine
from ..query_engine.existing_engine import ExistingQueryEngine
from ..query_engine.script_adapter import QueryEngineScriptAdapter, ScriptGenerationRequest


class PipelineFactory:
    """
    Factory for creating document processing pipelines
    
    This class encapsulates the creation of different analysis pipelines with
    configurable stages for processing smart contracts and related documents.
    It manages LLM settings, adapter configuration, and stage construction.
    """
    
    def __init__(self, config_module=None):
        """
        Initialize the pipeline factory.
        
        Args:
            config_module: Optional configuration module
        """
        self.config = config_module or config
        
        # Initialize common components
        self.llm_adapter = None
        self.llm_provider = getattr(self.config, "LLM_PROVIDER", "")
        self.llm_model = getattr(self.config, "LLM_MODEL", "")
        
        # Initialize vector store directory
        self.engine_dir = getattr(self.config, "VECTOR_STORE_DIR", "./vector_store")
        
        # Initialize LLM adapter
        self._initialize_llm_adapter()
        
        logger.info(f"Pipeline factory initialized using {self.llm_provider} with model {self.llm_model}")
        
        self.validator_adapter = None
        self.llm_tracker = LLMInteractionTracker()
        
        # Set up default LLM in Settings
        setup_default_llm(self.config)
        
        # Initialize validator adapter
        try:
            self.validator_adapter = ValidatorAdapter()
            logger.info("Validator adapter initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize validator adapter: {e}")
        
        # Initialize and cache query engine components
        self._vector_index = None  # Will be populated when needed
        self._query_engine = None  # Initialize as None and create on demand
        self._script_adapter = None  # Initialize as None and create on demand
    
    def _initialize_llm_adapter(self):
        """
        Initialize the LLM adapter based on configuration.
        Sets up the appropriate LLM client based on the provider and model.
        """
        try:
            provider = self.llm_provider.lower()
            logger.info(f"Initializing LLM adapter for provider: {provider}")
            
            if provider == "openai":
                self._initialize_openai_adapter()
            elif provider == "anthropic":
                self._initialize_anthropic_adapter()
            elif provider == "llama_index":
                self._initialize_llama_index_adapter()
            elif provider == "ollama":
                self._initialize_ollama_adapter()
            elif provider == "mock" or provider == "test":
                self._initialize_mock_adapter()
            else:
                logger.warning(f"Unknown LLM provider: {provider}. Using mock adapter.")
                self._initialize_mock_adapter()
                
        except Exception as e:
            logger.error(f"Error initializing LLM adapter: {e}")
            logger.info("Falling back to mock adapter")
            self._initialize_mock_adapter()

    def _initialize_openai_adapter(self):
        """Initialize OpenAI adapter."""
        try:
            from ..llm.openai_adapter import OpenAIAdapter
            
            # Get API key from config or environment
            api_key = getattr(self.config, "OPENAI_API_KEY", None)
            model = self.llm_model or getattr(self.config, "OPENAI_MODEL", "gpt-4")
            temperature = getattr(self.config, "TEMPERATURE", 0.0)
            
            self.llm_adapter = OpenAIAdapter(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            logger.info(f"Initialized OpenAI adapter with model: {model}")
            
        except ImportError:
            logger.error("OpenAI package not available. Install with 'pip install openai'")
            raise

    def _initialize_anthropic_adapter(self):
        """Initialize Anthropic adapter."""
        try:
            from ..llm.anthropic_adapter import AnthropicAdapter
            
            # Get API key from config or environment
            api_key = getattr(self.config, "ANTHROPIC_API_KEY", None)
            model = self.llm_model or getattr(self.config, "ANTHROPIC_MODEL", "claude-2")
            temperature = getattr(self.config, "TEMPERATURE", 0.0)
            
            self.llm_adapter = AnthropicAdapter(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            logger.info(f"Initialized Anthropic adapter with model: {model}")
            
        except ImportError:
            logger.error("Anthropic package not available. Install with 'pip install anthropic'")
            raise

    def _initialize_llama_index_adapter(self):
        """Initialize LlamaIndex adapter."""
        try:
            from ..llm.llama_index_adapter import LlamaIndexAdapter
            
            # Get LlamaIndex settings
            model = self.llm_model or getattr(self.config, "LLAMA_INDEX_MODEL", "gpt-4")
            temperature = getattr(self.config, "TEMPERATURE", 0.0)
            
            self.llm_adapter = LlamaIndexAdapter(
                model=model,
                temperature=temperature
            )
            logger.info(f"Initialized LlamaIndex adapter with model: {model}")
            
        except ImportError:
            logger.error("LlamaIndex package not available. Install with 'pip install llama-index'")
            raise

    def _initialize_ollama_adapter(self):
        """Initialize Ollama adapter."""
        try:
            from ..llm.ollama_adapter import OllamaAdapter
            
            # Get Ollama settings
            host = getattr(self.config, "OLLAMA_HOST", "http://localhost:11434")
            model = self.llm_model or getattr(self.config, "OLLAMA_MODEL", "llama2")
            temperature = getattr(self.config, "TEMPERATURE", 0.0)
            
            self.llm_adapter = OllamaAdapter(
                host=host,
                model=model,
                temperature=temperature
            )
            logger.info(f"Initialized Ollama adapter with model: {model}")
            
        except ImportError:
            logger.error("Ollama package not available.")
            raise

    def _initialize_mock_adapter(self):
        """Initialize mock adapter for testing."""
        try:
            from ..llm.mock_adapter import MockAdapter
            
            self.llm_adapter = MockAdapter()
            logger.info("Initialized mock LLM adapter")
            
        except ImportError:
            logger.error("Mock adapter not available.")
            # Create a minimal mock adapter inline
            class MinimalMockAdapter:
                async def acomplete(self, prompt):
                    return {"result": "This is a mock response"}
                    
                async def achat(self, messages):
                    return {"message": {"content": "This is a mock chat response"}}
            
            self.llm_adapter = MinimalMockAdapter()
            logger.info("Initialized minimal mock LLM adapter")

    async def verify_models(self) -> Dict[str, bool]:
        """
        Stub for model verification - previously used ModelVerifier
        
        This method is kept as a stub to maintain API compatibility.
        ModelVerifier has been removed due to causing issues.
        
        Returns:
            An empty dictionary as we're no longer doing model verification
        """
        logger.info("Model verification disabled")
        return {}
    
    def create_standard_pipeline(
        self, 
        documents: List[Any],
        output_path: str,
        config: Dict[str, Any] = None
    ) -> Pipeline:
        """
        Create a standard analysis pipeline for smart contracts
        
        Assembles a complete pipeline with all standard analysis stages including:
        - Document loading
        - Contract chunking
        - Function extraction
        - Business flow analysis
        - Data flow analysis
        - Vulnerability scanning
        - Cognitive bias analysis
        - Documentation analysis
        - Report generation
        
        Args:
            documents: Pre-loaded documents to process
            output_path: Path to save the output
            config: Optional configuration overrides
            
        Returns:
            Configured Pipeline ready for execution
        """
        if config is None:
            config = {}
            
        # Extract configuration
        chunk_size = config.get('chunk_size', 1000)
        overlap = config.get('overlap', 100)
        
        # Create document loading function
        load_documents_func = self._load_documents_stage(documents)
        
        # Create chunking function
        chunk_documents_func = self._chunk_documents_stage(chunk_size, overlap)
        
        # Create function extraction function
        extract_functions_func = self._extract_functions_stage()
        
        # Create business flow extraction function
        extract_business_flows_func = self._extract_business_flows_stage()
        
        # Create data flow analysis function
        analyze_data_flows_func = self._analyze_data_flows_stage()
        
        # Create vulnerability scanning function
        scan_vulnerabilities_func = self._scan_vulnerabilities_stage()
        
        # Create cognitive bias analysis function
        analyze_cognitive_bias_func = self._analyze_cognitive_bias_stage()
        
        # Create counterfactual analysis function 
        analyze_counterfactuals_func = self._analyze_counterfactual_stage()
        
        # Create documentation analysis function
        analyze_documentation_func = self._analyze_documentation_stage()
        
        # Create documentation inconsistency analysis function
        analyze_doc_inconsistency_func = self._analyze_documentation_inconsistency_stage()
        
        # Create report generation function
        generate_report_func = self._generate_report_stage(output_path)
        
        # Add the LLM stats function
        def add_llm_stats(context: Context) -> Context:
            """Add LLM usage statistics to the context"""
            context.llm_stats = self.llm_tracker.get_summary()
            logger.info(f"Added LLM usage stats: {len(context.llm_stats)} entries")
            return context
        
        # Create validation stage
        validation_stage = ValidationStage(validator_adapter=self.validator_adapter)
        
        # Create pipeline without model verification
        pipeline = Pipeline(
            stages=[
                # Model verification removed as it was causing issues
                load_documents_func,
                chunk_documents_func,
                extract_functions_func,
                extract_business_flows_func,
                analyze_data_flows_func,
                scan_vulnerabilities_func,
                analyze_cognitive_bias_func,
                analyze_documentation_func,
                analyze_doc_inconsistency_func,
                analyze_counterfactuals_func,
                validation_stage,
                add_llm_stats,
                generate_report_func
            ]
        )
        
        # Connect progress tracker to LLM monitor if available
        if hasattr(pipeline, 'get_progress_tracker'):
            progress_tracker = pipeline.get_progress_tracker()
            self.llm_tracker.set_progress_tracker(progress_tracker)
        
        return pipeline
    
    def _load_documents_stage(self, documents: List[Any]) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for loading documents into context"""
        
        async def load_documents(context: Context) -> Context:
            """Load provided documents into the context"""
            logger.info(f"Loading {len(documents)} documents into context")
            
            # Process documents in batches to avoid memory issues with large datasets
            batch_size = 50  # Adjust based on expected document size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                # Process batch concurrently
                tasks = [self._process_document(doc, context) for doc in batch]
                await asyncio.gather(*tasks)
                
                # Small delay between batches to prevent resource exhaustion
                if i + batch_size < len(documents):
                    await asyncio.sleep(0.1)
            
            logger.info(f"Loaded {len(context.files)} Solidity files into context")
            return context
        
        return load_documents

    async def _process_document(self, doc: Any, context: Context):
        """Process a single document asynchronously"""
        file_path = doc.metadata.get("file_path", "unknown")
        file_name = doc.metadata.get("file_name", "unknown")
        file_ext = doc.metadata.get("file_type", ".sol")
        
        # Only process Solidity files
        if file_ext.lower() != ".sol":
            logger.debug(f"Skipping non-Solidity file: {file_path}")
            return
        
        # Add file to context (this is a quick in-memory operation)
        context.files[file_path] = {
            "path": file_path,
            "name": file_name,
            "content": doc.text,
            "is_solidity": True
        }
        
        logger.debug(f"Added file to context: {file_path}")
    
    def _chunk_documents_stage(self, chunk_size: int, overlap: int) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for chunking documents"""
        
        async def chunk_documents(context: Context) -> Context:
            """Chunk Solidity contracts into processable segments"""
            logger.info("Chunking Solidity contracts asynchronously")
            
            # Create async chunker
            chunker = AsyncContractChunker(
                max_chunk_size=chunk_size,
                overlap_size=overlap,
                chunk_by_contract=True,
                chunk_by_function=True,
                include_call_graph=True
            )
            
            # Ensure chunks dictionary exists
            if not hasattr(context, 'chunks'):
                context.chunks = {}
            
            # Process files concurrently with semaphore
            semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
            tasks = []
            
            for file_id, file_data in list(context.files.items()):
                if not file_data.get("is_solidity", False):
                    continue
                
                # Create task to process this file
                task = self._process_file_chunks(chunker, semaphore, context, file_id, file_data)
                tasks.append(task)
            
            # Wait for all chunking tasks to complete
            if tasks:
                await asyncio.gather(*tasks)
            
            logger.info(f"Created {len(context.chunks)} chunks from {len(context.files)} files")
            return context
        
        return chunk_documents

    async def _process_file_chunks(self, chunker: AsyncContractChunker, semaphore: asyncio.Semaphore, context: Context, file_id: str, file_data: Dict[str, Any]):
        """Process chunks for a single file asynchronously with semaphore"""
        async with semaphore:
            try:
                # Chunk the file
                chunks = await chunker.chunk_code(
                    code=file_data["content"],
                    name=file_data["name"],
                    file_path=file_data["path"]
                )
                
                # Add chunks to file data
                file_data["chunks"] = chunks
                
                # Also add to global chunks dictionary
                for chunk in chunks:
                    # Use chunk_id consistently instead of id
                    chunk_id = chunk.get("chunk_id", f"{file_id}:chunk:{len(context.chunks)}")
                    context.chunks[chunk_id] = chunk
                
                logger.debug(f"Added {len(chunks)} chunks for file: {file_id}")
            
            except Exception as e:
                logger.error(f"Error chunking file {file_id}: {str(e)}")
                context.add_error(
                    stage="contract_chunking",
                    message=f"Failed to chunk file: {file_id}",
                    exception=e
                )
    
    def _extract_functions_stage(self) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for extracting functions"""
        
        async def extract_functions(context: Context) -> Context:
            """Extract function definitions from contract chunks"""
            logger.info("Extracting functions from contracts")
            
            # Initialize functions dictionary if not exists
            if not hasattr(context, 'functions'):
                context.functions = {}
            
            # Process each file with chunks
            for file_id, file_data in list(context.files.items()):
                if not file_data.get("is_solidity", False) or "chunks" not in file_data:
                    continue
                
                # Initialize functions list for this file
                file_data["functions"] = []
                
                # Extract functions from chunks
                for chunk in file_data["chunks"]:
                    if chunk["chunk_type"] == "function":
                        # This chunk contains a function, add it to functions
                        function_data = {
                            "id": chunk.get("id", f"{file_id}:function:{len(file_data['functions'])}"),
                            "name": chunk.get("function_name", "unknown"),
                            "start_line": chunk.get("start_line", 0),
                            "end_line": chunk.get("end_line", 0),
                            "contract_name": chunk.get("contract_name", "unknown"),
                            "full_text": chunk.get("content", ""),
                            "visibility": chunk.get("visibility", "unknown"),
                        }
                        
                        # Add to file's functions
                        file_data["functions"].append(function_data)
                        
                        # Add to global functions dictionary
                        context.functions[function_data["id"]] = function_data
            
            logger.info(f"Extracted {len(context.functions)} functions")
            return context
        
        return extract_functions
    
    def _extract_business_flows_stage(self) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for extracting business flows"""
        
        async def extract_business_flows(context: Context) -> Context:
            """Extract business flows from functions"""
            logger.info("Extracting business flows from functions")
            
            # Create business flow extractor with proper LLM initialization
            try:
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                
                # Use the adapter directly with proper configuration
                llm_adapter = LlamaIndexAdapter(
                    provider=self.config.BUSINESS_FLOW_MODEL_PROVIDER,  # Using self.config from nodes_config
                    model_name=self.config.BUSINESS_FLOW_MODEL,         # Using self.config from nodes_config
                    base_url=self.config.BUSINESS_FLOW_MODEL_BASE_URL,  # Using self.config from nodes_config
                )
                
                # Create flow extractor with the LLM adapter
                flow_extractor = BusinessFlowExtractor(llm_adapter=llm_adapter)
                logger.info(f"Created BusinessFlowExtractor with model: {self.config.BUSINESS_FLOW_MODEL}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize LLM adapter: {e}, using pattern-based extraction")
                flow_extractor = BusinessFlowExtractor()
            
            # Process the context to extract flows
            try:
                context = await flow_extractor.process(context)
                logger.info("Business flow extraction completed successfully")
            except Exception as e:
                logger.error(f"Error in business flow extraction: {str(e)}")
                context.add_error(
                    stage="business_flow_extraction",
                    message="Failed to extract business flows",
                    exception=e
                )
            
            return context
        
        return extract_business_flows
    
    def _analyze_data_flows_stage(self) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for analyzing data flows"""
        
        async def analyze_data_flows(context: Context) -> Context:
            """Analyze data flows between contract components"""
            logger.info("Analyzing data flows in contracts")
            
            # Create data flow analyzer
            data_flow_analyzer = DataFlowAnalyzer()
            
            # Process the context to analyze data flows
            try:
                context = await data_flow_analyzer.process(context)
                logger.info("Data flow analysis completed successfully")
            except Exception as e:
                logger.error(f"Error in data flow analysis: {str(e)}")
                context.add_error(
                    stage="data_flow_analysis",
                    message="Failed to analyze data flows",
                    exception=e
                )
            
            return context
        
        return analyze_data_flows
    
    def _scan_vulnerabilities_stage(self) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for vulnerability scanning"""
        
        async def scan_vulnerabilities(context: Context) -> Context:
            """Scan contracts for common vulnerabilities"""
            logger.info("Scanning for vulnerabilities")
                
            try:
                # Create adapter with scan-specific model configuration
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                
                llm_adapter = LlamaIndexAdapter(
                    provider=self.config.SCAN_MODEL_PROVIDER,  # Using self.config from nodes_config
                    model_name=self.config.SCAN_MODEL,         # Using self.config from nodes_config
                    base_url=self.config.SCAN_MODEL_BASE_URL,  # Using self.config from nodes_config
                    model_params=self.config.MODEL_PARAMS.get(self.config.SCAN_MODEL, self.config.MODEL_PARAMS["default"])
                )
                
                # Create vulnerability scanner
                scanner = VulnerabilityScanner(llm_adapter=llm_adapter)
                
                # Process the context
                context = await scanner.process(context)
                logger.info("Vulnerability scanning completed successfully")
                
            except Exception as e:
                logger.error(f"Error in vulnerability scanning: {str(e)}")
                context.add_error(
                    stage="vulnerability_scanning",
                    message="Failed to scan for vulnerabilities",
                    exception=e
                )
            
            return context
        
        return scan_vulnerabilities
    
    def _analyze_cognitive_bias_stage(self) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for analyzing cognitive biases"""
        
        async def analyze_cognitive_bias(context: Context) -> Context:
            """Analyze contract code for cognitive biases"""
            logger.info("Analyzing cognitive biases")
            
            # Create cognitive bias analyzer
            bias_analyzer = CognitiveBiasAnalyzer()
            
            # Process the context
            try:
                context = await bias_analyzer.process(context)
                logger.info("Cognitive bias analysis completed successfully")
            except Exception as e:
                logger.error(f"Error in cognitive bias analysis: {str(e)}")
                context.add_error(
                    stage="cognitive_bias_analysis",
                    message="Failed to analyze cognitive biases",
                    exception=e
                )
            
            return context
        
        return analyze_cognitive_bias
    
    def _analyze_counterfactual_stage(self) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for counterfactual analysis"""
        
        async def analyze_counterfactuals(context: Context) -> Context:
            """Analyze contract code using counterfactual scenarios"""
            logger.info("Performing counterfactual analysis")
            
            # Create counterfactual analyzer
            counterfactual_analyzer = CounterfactualAnalyzer()
            
            # Process the context
            try:
                context = await counterfactual_analyzer.process(context)
                logger.info("Counterfactual analysis completed successfully")
            except Exception as e:
                logger.error(f"Error in counterfactual analysis: {str(e)}")
                context.add_error(
                    stage="counterfactual_analysis",
                    message="Failed to analyze counterfactuals",
                    exception=e
                )
            
            return context
        
        return analyze_counterfactuals
    
    def _analyze_documentation_stage(self) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for documentation analysis"""
        
        async def analyze_documentation(context: Context) -> Context:
            """Analyze documentation quality and completeness"""
            logger.info("Analyzing documentation")
            
            # Create documentation analyzer
            doc_analyzer = DocumentationAnalyzer()
            
            # Process the context
            try:
                context = await doc_analyzer.process(context)
                logger.info("Documentation analysis completed successfully")
            except Exception as e:
                logger.error(f"Error in documentation analysis: {str(e)}")
                context.add_error(
                    stage="documentation_analysis",
                    message="Failed to analyze documentation",
                    exception=e
                )
            
            return context
        
        return analyze_documentation
    
    def _analyze_documentation_inconsistency_stage(self) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for documentation inconsistency analysis"""
        
        async def analyze_documentation_inconsistency(context: Context) -> Context:
            """Analyze documentation for inconsistencies with implementation"""
            logger.info("Analyzing documentation inconsistencies")
            
            # Create documentation inconsistency analyzer
            doc_inconsistency_analyzer = DocInconsistencyAnalyzer()
            
            # Process the context
            try:
                context = await doc_inconsistency_analyzer.process(context)
                logger.info("Documentation inconsistency analysis completed successfully")
            except Exception as e:
                logger.error(f"Error in documentation inconsistency analysis: {str(e)}")
                context.add_error(
                    stage="documentation_inconsistency_analysis",
                    message="Failed to analyze documentation inconsistencies",
                    exception=e
                )
            
            return context
        
        return analyze_documentation_inconsistency
    
    def _generate_report_stage(self, output_path: str) -> Callable[[Context], Awaitable[Context]]:
        """Create a stage function for generating the final report"""
        
        async def generate_report(context: Context) -> Context:
            """Generate comprehensive analysis report"""
            logger.info("Generating final report")
            
            try:
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Compile all analysis results
                report_data = {
                    "project": context.project if hasattr(context, "project") else {},
                    "files": len(context.files),
                    "functions": len(context.functions) if hasattr(context, "functions") else 0,
                    "vulnerabilities": context.findings if hasattr(context, "findings") else [],
                    "cognitive_biases": context.biases if hasattr(context, "biases") else [],
                    "documentation_issues": context.documentation_issues if hasattr(context, "documentation_issues") else [],
                    "counterfactuals": context.counterfactuals if hasattr(context, "counterfactuals") else [],
                    "flow_analysis": context.flow_analysis if hasattr(context, "flow_analysis") else {},
                    "data_flow_analysis": context.data_flow_analysis if hasattr(context, "data_flow_analysis") else {},
                    "errors": context.errors if hasattr(context, "errors") else [],
                    "llm_stats": context.llm_stats if hasattr(context, "llm_stats") else {},
                    # Add attack surface data to the report
                    "attack_surfaces": context.attack_surfaces if hasattr(context, "attack_surfaces") else {},
                    "attack_surface_summary": context.attack_surface_summary if hasattr(context, "attack_surface_summary") else {}
                }
                
                # Write the report to disk
                import json
                with open(output_path, 'w') as f:
                    json.dump(report_data, f, indent=2)
                
                logger.info(f"Report generated successfully: {output_path}")
                
                # Store report path in context
                context.report_path = output_path
                
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                context.add_error(
                    stage="report_generation",
                    message="Failed to generate report",
                    exception=e
                )
            
            return context
        
        return generate_report

    def _initialize_existing_query_engine(self):
        """Initialize a query engine using existing data"""
        from llama_index.core import load_index_from_storage, StorageContext
        from llama_index.core.schema import Document
        from llama_index.core import VectorStoreIndex
        from llama_index.core import Settings
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI
        import os
        
        # Get the vector store and query engine
        try:
            # Check if index exists
            if os.path.exists(self.engine_dir):
                # Load from storage
                storage_context = StorageContext.from_defaults(persist_dir=self.engine_dir)
                index = load_index_from_storage(storage_context)
                logger.info(f"Loaded existing vector index from {self.engine_dir}")
            else:
                # Create settings with modern API
                embed_model = OpenAIEmbedding()
                llm = OpenAI(model=config.QUERY_MODEL, temperature=config.TEMPERATURE)
                
                # Use the Settings API instead of ServiceContext
                Settings.embed_model = embed_model
                Settings.llm = llm
                
                # Create an empty index if none exists
                logger.warning("No vector index available. Creating empty index.")
                index = VectorStoreIndex([Document(text="Placeholder document", id="placeholder")])
                
                # Persist the index
                index.storage_context.persist(persist_dir=self.engine_dir)
                
            return index.as_query_engine(similarity_top_k=3)
        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            raise

    def get_query_engine(self) -> FlareQueryEngine:
        """
        Get the FLARE query engine instance, initializing it if needed.
        
        Returns:
            FlareQueryEngine: The query engine instance
        """
        if self._query_engine is None:
            # Get model from config if available
            model_name = getattr(self.config, "DEFAULT_REASONING_MODEL", "openai:gpt-4o")
            
            # Initialize the underlying engine
            underlying_engine = self._initialize_existing_query_engine()
            
            # Create the FLARE engine
            self._query_engine = FlareQueryEngine(
                underlying_engine=underlying_engine,
                max_iterations=10,
                verbose=True,
                config={"model": model_name}
            )
            
            # Initialize immediately
            asyncio.create_task(self._query_engine.initialize())
        
        return self._query_engine

    def get_script_adapter(self) -> QueryEngineScriptAdapter:
        """
        Get the script adapter instance, initializing it if needed.
        
        Returns:
            QueryEngineScriptAdapter: The script adapter instance
        """
        if self._script_adapter is None:
            self._script_adapter = QueryEngineScriptAdapter(
                query_engine=self.get_query_engine(),
                script_output_dir=os.path.join(os.getcwd(), "generated_scripts")
            )
        return self._script_adapter

    async def create_document_processing_stage(self, context: Context) -> Stage:
        """
        Create a combined document processing stage that handles loading,
        contract extraction, and function parsing in a single stage.
        
        This stage implements the complete flow from files → contracts → functions,
        while maintaining proper state transitions and metadata updates.
        
        Args:
            context: Pipeline context
            
        Returns:
            Combined document processing stage
        """
        @stage(name="document_processing", description="Process documents: load files, extract contracts, parse functions")
        async def document_processing_stage(ctx: Context, *args, **kwargs) -> Context:
            """Process documents through the complete file → contract → function pipeline"""
            logger.info("Starting document processing stage")
            
            # Step 1: Load files from input path
            if hasattr(ctx, 'input_path') and ctx.input_path:
                input_path = ctx.input_path
                logger.info(f"Loading files from: {input_path}")
                
                # Use the document loader component
                document_loader = await self.get_document_loader()
                ctx = await document_loader(ctx)
                ctx.set_files_loaded()
                logger.info(f"Loaded {len(ctx.files)} files")
            else:
                logger.warning("No input path specified, skipping file loading")
            
            # Step 2: Extract contracts from files
            if ctx.files:
                logger.info("Extracting contracts from files")
                contract_extractor = await self.get_contract_extractor()
                ctx = await contract_extractor(ctx)
                ctx.set_contracts_extracted()
                logger.info(f"Extracted {len(ctx.contracts)} contracts")
            else:
                logger.warning("No files to extract contracts from")
            
            # Step 3: Parse functions from contracts
            if ctx.contracts:
                logger.info("Parsing functions from contracts")
                function_parser = await self.get_function_parser()
                ctx = await function_parser(ctx)
                ctx.set_functions_extracted()
                logger.info(f"Parsed {len(ctx.functions)} functions")
            else:
                logger.warning("No contracts to parse functions from")
            
            return ctx
        
        return document_processing_stage

    async def create_business_flow_extractor(self, context: Context) -> Stage:
        """
        Create a business flow extraction stage.
        
        This stage analyzes smart contracts to identify and extract logical business flows,
        state transitions, and interaction patterns.
        
        Args:
            context: Pipeline context with contracts loaded
            
        Returns:
            Configured business flow extraction stage
        """
        logger.info("Creating business flow extractor stage")
        
        # Initialize the business flow extractor with appropriate settings
        flow_extractor = BusinessFlowExtractor(
            flow_types=[
                "token_transfer",
                "access_control", 
                "state_transition",
                "external_call",
                "fund_management",
                "Source identification (user inputs, transaction data)",
                "Sink identification (external calls, state changes)",
                "Taint propagation tracking",
                "Control variable analysis",
                "Business context integration",
                "Source-to-sink paths with exploitation potential",
                "Affected business flows",
                "Exploitability scores",
                "Optimism bias detection",
                "Anchoring bias detection",
                "Confirmation bias detection",
                "Authority bias detection",
                "Status quo bias detection",
                "extreme value analysis",
                "State divergence analysis",
                "External call failure scenarios",
                "Transaction ordering attack scenarios",
                "Permission change vulnerability detection"
            ]       
        )  # Configure with LLM if available for enhanced extraction
        if hasattr(self, 'llm_adapter') and self.llm_adapter:
            flow_extractor.set_llm_adapter(self.llm_adapter)
            logger.info("Business flow extractor configured with LLM adapter")
        
        async def extract_business_flows_stage(ctx: Context) -> Context:
            """Extract business flows from smart contracts"""
            logger.info("Extracting business flows from contracts")
            
            try:
                # Ensure business_flows collection exists
                if not hasattr(ctx, 'business_flows'):
                    ctx.business_flows = {}
                
                # Extract business flows from contracts
                result_ctx = await flow_extractor.process(ctx)
                
                # Log summary of extraction
                flow_count = sum(len(flows) for flows in ctx.business_flows.values()) if hasattr(ctx, 'business_flows') else 0
                logger.info(f"Extracted {flow_count} business flows from contracts")
                
                return result_ctx
                
            except Exception as e:
                logger.error(f"Error in business flow extraction: {str(e)}")
                logger.error(traceback.format_exc())
                ctx.add_error(
                    stage="business_flow_extraction",
                    message="Failed to extract business flows",
                    exception=e
                )
                return ctx
        
        return extract_business_flows_stage

    async def create_vulnerability_scanner(self, context: Context) -> Stage:
        """
        Create a vulnerability scanning stage for smart contracts.
        
        This stage analyzes smart contracts to identify potential security vulnerabilities,
        using both pattern-based detection and LLM-enhanced semantic analysis.
        
        Args:
            context: Pipeline context with contracts loaded
            
        Returns:
            Configured vulnerability scanning stage
        """
        logger.info("Creating vulnerability scanner stage")
        
        # Initialize the vulnerability scanner
        try:
            # Initialize with appropriate LLM adapter if possible
            if hasattr(self, 'llm_adapter') and self.llm_adapter:
                scanner = VulnerabilityScanner(llm_adapter=self.llm_adapter)
                logger.info("Vulnerability scanner configured with LLM adapter")
            else:
                # Fall back to pattern-based scanning without LLM
                scanner = VulnerabilityScanner()
                logger.info("Vulnerability scanner initialized in pattern-only mode")
                
            async def scan_vulnerabilities_stage(ctx: Context) -> Context:
                """Scan contracts for security vulnerabilities"""
                logger.info("Scanning contracts for vulnerabilities")
                
                try:
                    # Ensure vulnerabilities collection exists
                    if not hasattr(ctx, 'vulnerabilities'):
                        ctx.vulnerabilities = {}
                    
                    # Scan contracts for vulnerabilities
                    result_ctx = await scanner.process(ctx)
                    
                    # Log summary of scan
                    vuln_count = sum(len(vulns) for vulns in ctx.vulnerabilities.values()) if hasattr(ctx, 'vulnerabilities') else 0
                    logger.info(f"Identified {vuln_count} potential vulnerabilities in contracts")
                    
                    return result_ctx
                    
                except Exception as e:
                    logger.error(f"Error in vulnerability scanning: {str(e)}")
                    logger.error(traceback.format_exc())
                    ctx.add_error(
                        stage="vulnerability_scanning",
                        message="Failed to scan for vulnerabilities",
                        exception=e
                    )
                    return ctx
            
            return scan_vulnerabilities_stage
            
        except Exception as e:
            logger.error(f"Failed to create vulnerability scanner: {e}")
            
            # Return a fallback stage that does nothing
            async def fallback_stage(ctx: Context) -> Context:
                logger.warning("Using fallback vulnerability scanning stage (no-op)")
                return ctx
                
            return fallback_stage

    async def create_dataflow_analyzer(self, context: Context) -> Stage:
        """
        Create a data flow analysis stage for smart contracts.
        
        This stage analyzes how data flows through contracts, identifying sources,
        sinks, and potential data-related vulnerabilities.
        
        Args:
            context: Pipeline context with contracts loaded
            
        Returns:
            Configured data flow analysis stage
        """
        logger.info("Creating data flow analyzer stage")
        
        # Initialize the data flow analyzer
        try:
            # Initialize with appropriate LLM adapter if possible
            if hasattr(self, 'llm_adapter') and self.llm_adapter:
                analyzer = DataFlowAnalyzer(llm_adapter=self.llm_adapter)
                logger.info("Data flow analyzer configured with LLM adapter")
            else:
                # Fall back to pattern-based analysis without LLM
                analyzer = DataFlowAnalyzer()
                logger.info("Data flow analyzer initialized in pattern-only mode")
                
            async def analyze_data_flows_stage(ctx: Context) -> Context:
                """Analyze data flows in smart contracts"""
                logger.info("Analyzing data flows in contracts")
                
                try:
                    # Ensure dataflows collection exists
                    if not hasattr(ctx, 'dataflows'):
                        ctx.dataflows = {}
                    
                    # Analyze data flows in contracts
                    result_ctx = await analyzer.process(ctx)
                    
                    # Log summary of analysis
                    flow_count = sum(len(flows) for flows in ctx.dataflows.values()) if hasattr(ctx, 'dataflows') else 0
                    logger.info(f"Identified {flow_count} data flows in contracts")
                    
                    return result_ctx
                    
                except Exception as e:
                    logger.error(f"Error in data flow analysis: {str(e)}")
                    logger.error(traceback.format_exc())
                    ctx.add_error(
                        stage="data_flow_analysis",
                        message="Failed to analyze data flows",
                        exception=e
                    )
                    return ctx
            
            return analyze_data_flows_stage
            
        except Exception as e:
            logger.error(f"Failed to create data flow analyzer: {e}")
            
            # Return a fallback stage that does nothing
            async def fallback_stage(ctx: Context) -> Context:
                logger.warning("Using fallback data flow analysis stage (no-op)")
                return ctx
                
            return fallback_stage

    async def create_cognitive_bias_analyzer(self, context: Context) -> Stage:
        """
        Create a cognitive bias analysis stage for smart contracts.
        
        This stage identifies potential cognitive biases in contract design and implementation
        that could lead to security issues or unintended behaviors.
        
        Args:
            context: Pipeline context with contracts loaded
            
        Returns:
            Configured cognitive bias analysis stage
        """
        logger.info("Creating cognitive bias analyzer stage")
        
        # Initialize the cognitive bias analyzer
        try:
            # Initialize with appropriate LLM adapter if possible
            if hasattr(self, 'llm_adapter') and self.llm_adapter:
                analyzer = CognitiveBiasAnalyzer(llm_adapter=self.llm_adapter)
                logger.info("Cognitive bias analyzer configured with LLM adapter")
            else:
                # Fall back to basic analysis without LLM
                analyzer = CognitiveBiasAnalyzer()
                logger.info("Cognitive bias analyzer initialized in basic mode")
                
            async def analyze_cognitive_biases_stage(ctx: Context) -> Context:
                """Analyze cognitive biases in smart contracts"""
                logger.info("Analyzing cognitive biases in contracts")
                
                try:
                    # Ensure cognitive_biases collection exists
                    if not hasattr(ctx, 'cognitive_biases'):
                        ctx.cognitive_biases = {}
                    
                    # Analyze cognitive biases in contracts
                    result_ctx = await analyzer.process(ctx)
                    
                    # Log summary of analysis
                    bias_count = sum(len(biases) for biases in ctx.cognitive_biases.values()) if hasattr(ctx, 'cognitive_biases') else 0
                    logger.info(f"Identified {bias_count} potential cognitive biases in contracts")
                    
                    return result_ctx
                    
                except Exception as e:
                    logger.error(f"Error in cognitive bias analysis: {str(e)}")
                    logger.error(traceback.format_exc())
                    ctx.add_error(
                        stage="cognitive_bias_analysis",
                        message="Failed to analyze cognitive biases",
                        exception=e
                    )
                    return ctx
            
            return analyze_cognitive_biases_stage
            
        except Exception as e:
            logger.error(f"Failed to create cognitive bias analyzer: {e}")
            
            # Return a fallback stage that does nothing
            async def fallback_stage(ctx: Context) -> Context:
                logger.warning("Using fallback cognitive bias analysis stage (no-op)")
                return ctx
                
            return fallback_stage

    async def create_documentation_analyzer(self, context: Context) -> Stage:
        """
        Create a documentation analysis stage for smart contracts.
        
        This stage analyzes documentation consistency with implementation,
        identifying areas where code behavior differs from what is documented
        in comments, NatSpec, or other documentation forms.
        
        Args:
            context: Pipeline context with contracts loaded
            
        Returns:
            Configured documentation analysis stage
        """
        logger.info("Creating documentation analyzer stage")
        
        # Initialize the documentation analyzer
        try:
            # Initialize with appropriate LLM adapter if possible
            if hasattr(self, 'llm_adapter') and self.llm_adapter:
                analyzer = DocumentationAnalyzer(llm_adapter=self.llm_adapter)
                logger.info("Documentation analyzer configured with LLM adapter")
            else:
                # Fall back to basic analysis without LLM
                analyzer = DocumentationAnalyzer()
                logger.info("Documentation analyzer initialized in basic mode")
                
            async def analyze_documentation_stage(ctx: Context) -> Context:
                """Analyze consistency between code and documentation"""
                logger.info("Analyzing documentation consistency in contracts")
                
                try:
                    # Ensure documentation_issues collection exists
                    if not hasattr(ctx, 'documentation_issues'):
                        ctx.documentation_issues = {}
                    
                    # Analyze documentation in contracts
                    result_ctx = await analyzer.process(ctx)
                    
                    # Log summary of analysis
                    issue_count = sum(len(issues) for issues in ctx.documentation_issues.values()) if hasattr(ctx, 'documentation_issues') else 0
                    logger.info(f"Identified {issue_count} documentation inconsistencies in contracts")
                    
                    return result_ctx
                    
                except Exception as e:
                    logger.error(f"Error in documentation analysis: {str(e)}")
                    logger.error(traceback.format_exc())
                    ctx.add_error(
                        stage="documentation_analysis",
                        message="Failed to analyze documentation consistency",
                        exception=e
                    )
                    return ctx
            
            return analyze_documentation_stage
            
        except Exception as e:
            logger.error(f"Failed to create documentation analyzer: {e}")
            
            # Return a fallback stage that does nothing
            async def fallback_stage(ctx: Context) -> Context:
                logger.warning("Using fallback documentation analysis stage (no-op)")
                return ctx
                
            return fallback_stage

    async def create_vector_match_analyzer(self, context: Context) -> Stage:
        """
        Create a vector match analysis stage for smart contracts.
        
        This stage uses vector embedding similarity with adaptive decay functions 
        to identify semantically similar code patterns across contracts and match them 
        against known vulnerability patterns from external sources.
        
        Args:
            context: Pipeline context with contracts loaded
            
        Returns:
            Configured vector match analysis stage
        """
        logger.info("Creating vector match analyzer stage")
        
        # Use cached analyzer if available in the factory instance
        analyzer = getattr(self, 'vector_match_analyzer', None)
        
        # Initialize the vector match analyzer if not already cached
        try:
            if analyzer is None:
                # Prepare database connection for vector storage
                # Uses pgvector with PostgreSQL, database name is postgres_vector
                db_url = self.config.ASYNC_DB_URL
                if db_url:
                    # Replace the database name in the connection string
                    vector_db_url = db_url.replace("/postgres", "/postgres_vector")
                else:
                    logger.warning("No ASYNC_DB_URL configured, vector analyzer will use in-memory storage")
                    vector_db_url = None
                    
                # Initialize with appropriate LLM adapter for enhanced analysis
                if hasattr(self, 'llm_adapter') and self.llm_adapter:
                    analyzer = VectorMatchAnalyzer(
                        llm_adapter=self.llm_adapter,
                        similarity_threshold=0.75,
                        use_decay_functions=True,
                        match_sources=["github_issues", "known_vulnerabilities"],
                        db_connection=vector_db_url,
                        vector_table="issues",
                        vector_column="embedding"
                    )
                    self.vector_match_analyzer = analyzer  # Cache for reuse
                    logger.info("Vector match analyzer configured with LLM adapter, decay functions, and pgvector backend")
                else:
                    # Fall back to basic analysis without LLM
                    analyzer = VectorMatchAnalyzer(
                        similarity_threshold=0.8,  # Higher threshold when no LLM verification
                        use_decay_functions=True,
                        db_connection=vector_db_url,
                        vector_table="issues",
                        vector_column="embedding"
                    )
                    self.vector_match_analyzer = analyzer  # Cache for reuse
                    logger.info("Vector match analyzer initialized in basic mode with pgvector backend")
                
            async def analyze_vector_matches_stage(ctx: Context) -> Context:
                """Analyze vector similarity matches in smart contracts"""
                logger.info("Analyzing vector matches in contracts")
                
                try:
                    # Ensure vector_matches collection exists
                    if not hasattr(ctx, 'vector_matches'):
                        ctx.vector_matches = {}
                    
                    # Analyze vector matches in contracts
                    result_ctx = await analyzer.process(ctx)
                    
                    # Log summary of analysis
                    match_count = sum(len(matches) for matches in ctx.vector_matches.values()) if hasattr(ctx, 'vector_matches') else 0
                    logger.info(f"Identified {match_count} vector matches in contracts")
                    
                    return result_ctx
                    
                except Exception as e:
                    logger.error(f"Error in vector match analysis: {str(e)}")
                    logger.error(traceback.format_exc())
                    ctx.add_error(
                        stage="vector_match_analysis",
                        message="Failed to analyze vector matches",
                        exception=e
                    )
                    return ctx
            
            return analyze_vector_matches_stage
            
        except Exception as e:
            logger.error(f"Failed to create vector match analyzer: {e}")
            
            # Return a fallback stage that does nothing
            async def fallback_stage(ctx: Context) -> Context:
                logger.warning("Using fallback vector match analysis stage (no-op)")
                return ctx
                
            return fallback_stage

    async def create_threat_detector(self, context: Context) -> Stage:
        """
        Create a threat detection stage for smart contracts.
        
        This stage builds on vector match analysis to identify potential security
        threats by generating targeted LLM prompts that probe each function for 
        specific vulnerability characteristics identified in similar code patterns.
        
        Args:
            context: Pipeline context with contracts and vector matches
            
        Returns:
            Configured threat detection stage
        """
        logger.info("Creating threat detector stage")
        
        # Use cached detector if available in the factory instance
        detector = getattr(self, 'threat_detector', None)
        
        # Initialize the threat detector if not already cached
        try:
            if detector is None:
                # Initialize with appropriate LLM adapter for enhanced detection
                if hasattr(self, 'llm_adapter') and self.llm_adapter:
                    detector = ThreatDetector(
                        llm_adapter=self.llm_adapter,
                        generate_mitigation=True,
                        analyze_exploit_paths=True,
                        threat_categories=[
                            "arithmetic_flaws", 
                            "access_control_issues",
                            "external_calls_exposure",
                            "state_manipulation",
                            "gas_optimization_issues"
                        ]
                    )
                    self.threat_detector = detector  # Cache for reuse
                    logger.info("Threat detector configured with LLM adapter and exploit path analysis")
                else:
                    # Fall back to basic detection without LLM
                    detector = ThreatDetector(
                        generate_mitigation=False,  # Can't generate mitigations without LLM
                        analyze_exploit_paths=False
                    )
                    self.threat_detector = detector  # Cache for reuse
                    logger.info("Threat detector initialized in basic mode")
                
            async def detect_threats_stage(ctx: Context) -> Context:
                """Detect threats in smart contracts based on vector matches and known patterns"""
                logger.info("Detecting threats in contracts")
                
                try:
                    # Ensure threats collection exists
                    if not hasattr(ctx, 'threats'):
                        ctx.threats = {}
                    
                    # Check if vector matches are available for enhanced detection
                    has_vector_matches = hasattr(ctx, 'vector_matches') and ctx.vector_matches
                    if has_vector_matches:
                        logger.info("Using vector matches for enhanced threat detection")
                    else:
                        logger.warning("No vector matches available, threat detection may be limited")
                    
                    # Detect threats in contracts
                    result_ctx = await detector.process(ctx)
                    
                    # Log summary of detection
                    threat_count = sum(len(threats) for threats in ctx.threats.values()) if hasattr(ctx, 'threats') else 0
                    logger.info(f"Identified {threat_count} potential threats in contracts")
                    
                    return result_ctx
                    
                except Exception as e:
                    logger.error(f"Error in threat detection: {str(e)}")
                    logger.error(traceback.format_exc())
                    ctx.add_error(
                        stage="threat_detection",
                        message="Failed to detect threats",
                        exception=e
                    )
                    return ctx
            
            return detect_threats_stage
            
        except Exception as e:
            logger.error(f"Failed to create threat detector: {e}")
            
            # Return a fallback stage that does nothing
            async def fallback_stage(ctx: Context) -> Context:
                logger.warning("Using fallback threat detection stage (no-op)")
                return ctx
                
            return fallback_stage


class ValidationStage:
    """Stage for validating analysis results"""
    
    def __init__(self, validator_adapter: Optional[ValidatorAdapter] = None):
        """Initialize with optional validator adapter"""
        self.validator_adapter = validator_adapter
    
    async def __call__(self, context: Context) -> Context:
        """Execute validation on the context"""
        if not self.validator_adapter:
            logger.warning("No validator adapter available, skipping validation")
            return context
        
        logger.info("Running validation stage")
        
        try:
            # Validate results using the validator adapter
            validation_results = await self.validator_adapter.validate_results(context)
            
            # Add validation results to context
            context.validation_results = validation_results
            
            # Log validation summary
            valid_count = sum(1 for r in validation_results if r.get("valid", False))
            logger.info(f"Validation completed: {valid_count}/{len(validation_results)} checks passed")
            
        except Exception as e:
            logger.error(f"Error in validation stage: {str(e)}")
            context.add_error(
                stage="validation",
                message="Failed to validate results",
                exception=e
            )
        
        return context