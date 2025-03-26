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


class PipelineFactory:
    """
    Factory for creating document processing pipelines
    
    This class encapsulates the creation of different analysis pipelines with
    configurable stages for processing smart contracts and related documents.
    It manages LLM settings, adapter configuration, and stage construction.
    """
    
    def __init__(self):
        """
        Initialize the pipeline factory
        
        Sets up default LLM configuration, initializes adapters, and prepares
        the system for creating document processing pipelines.
        """
        self.config = config  # Using the imported config from nodes_config
        self.llm_adapter = None
        self.validator_adapter = None
        self.llm_tracker = LLMInteractionTracker()
        
        # Set up default LLM in Settings
        setup_default_llm(self.config)
        
        # Initialize LLM adapter
        try:
            self.llm_adapter = LLMAdapter()
            # Set the tracker in the adapter
            self.llm_adapter.set_tracker(self.llm_tracker)
            logger.info("LLM adapter initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM adapter: {e}")
        
        # Initialize validator adapter
        try:
            self.validator_adapter = ValidatorAdapter()
            logger.info("Validator adapter initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize validator adapter: {e}")
    
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