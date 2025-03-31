"""
Factory pattern implementation for creating document processing pipelines.
"""
from typing import List, Dict, Any, Optional, Callable, Awaitable, Union
from pathlib import Path
import os
import asyncio
from box import Box
from loguru import logger
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, get_response_synthesizer, Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

from finite_monkey.adapters.ollama import AsyncOllamaClient
# Import pipeline components
from .core import Context, Pipeline, Stage
from finite_monkey.pipeline.validation_stage import ValidationStage
from ..utils.chunking import AsyncContractChunker
from ..analyzers.business_flow_extractor import BusinessFlowExtractor
from ..analyzers.vulnerability_scanner import VulnerabilityScanner
from ..analyzers.data_flow_analyzer import DataFlowAnalyzer
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
from ..nodes_config import config
from ..analyzers.vector_match_analyzer import VectorMatchAnalyzer
from ..analyzers.threat_detector import ThreatDetector
from .reporting import patch_report_generator
from .extractors import extract_contracts_from_files
from llama_index.core import PromptTemplate
from llama_index.core.query_engine.sub_question_query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
from ..models.pipeline_data import FileData, ChunkData, FunctionData, DataFlowData, VectorMatchData, ThreatData, ProjectData
from ..utils.async_parser import AsyncSolidityParser
from ..llama_index.processor import AsyncIndexProcessor
from finite_monkey.query_engine import FLAREInstructQueryEngine  # new import
from finite_monkey.query_engine import ExistingQueryEngine         # new import for underlying engine

def require_attributes(context, attrs):
    missing = [attr for attr in attrs if not hasattr(context, attr) or not getattr(context, attr)]
    if missing:
        logger.warning(f"Missing required context attributes: {', '.join(missing)}")
        return False
    return True

class PipelineFactory:
    """
    Factory for creating document processing pipelines.
    """
    
    def __init__(self, config):
        """
        Initialize the factory with the given configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self._llm_adapter = None  # Initialize LLM adapter as None
        self.validator_adapter = None  # Initialize validator adapter as None
        self.llm_tracker = LLMInteractionTracker()
        self.logger = logger
        # Set up default LLM in Settings
        setup_default_llm(self.config)
        # Cache analyzer instances for performance and consistency
        self.threat_detector = ThreatDetector()
        self.counterfactual_analyzer = CounterfactualAnalyzer()
        self.cognitive_bias_analyzer = CognitiveBiasAnalyzer(AsyncOllamaClient(model=config.DEFAULT_MODEL))
        self.documentation_analyzer = DocumentationAnalyzer()
        self.doc_inconsistency_adapter = DocumentationInconsistencyAdapter(self.documentation_analyzer)
        # Initialize and cache the query engine instance for use throughout the pipeline
        self.query_engine = FLAREInstructQueryEngine(
            query_engine=self._initialize_existing_query_engine(),
            max_iterations=10,
            verbose=True
        )
    
    def _initialize_existing_query_engine(self):
        """
        Initialize the existing underlying query engine.
        This function can be adjusted based on how the existing query engine is configured.
        """
        return ExistingQueryEngine()  # For demonstration; replace with actual initialization if needed
    
    def get_query_engine(self) -> FLAREInstructQueryEngine:
        """
        Return the cached FLAREInstructQueryEngine instance.
        """
        return self.query_engine
    
    def _get_llm_adapter(self) -> LLMAdapter:
        """
        Get the LLM adapter, initializing it if it's not already initialized.
        
        Returns:
            The LLM adapter.
        """
        if self._llm_adapter is None:
            try:
                self._llm_adapter = LLMAdapter()
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM adapter: {e}")
                raise
        return self._llm_adapter
        
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
        
    async def load_project(self, context: Context) -> Context:
        """
        Load project using AsyncProjectLoader
        
        Args:
            context: Pipeline context
        
        Returns:
            Updated context with loaded project
        """
        logger.info("Loading project using AsyncProjectLoader")
        if not hasattr(context, 'input_path') or not context.input_path:
            logger.warning("No input path specified")
            return context
            
        try:
            # Implementation would go here
            pass 
        except Exception as e:
            logger.error(f"Error loading project: {e}")
            if hasattr(context, 'add_error'):
                context.add_error(
                    stage="project_loading",
                    message=f"Failed to load project: {str(e)}",
                    exception=e
                )
        return context
        
    async def load_documents(self, context: Context) -> Context:
        """
        Load documents from the specified source directory into the context.
        
        Args:
            context: Pipeline context
        
        Returns:
            Updated context with loaded documents.
        """
        logger.info("Loading documents")
        if not hasattr(context, 'input_path') or not context.input_path:
            logger.warning("No input path specified")
            return context
            
        try:
            input_path = context.input_path
            logger.info(f"Loading documents from {input_path}")
            # Parse the project using AsyncSolidityParser
            project_data_list = await AsyncSolidityParser.parse_project(input_path)
            # Populate the context with the parsed data
            context.project_data = project_data_list
            # Consolidate files, contracts, and functions into separate lists
            context.files = []
            context.contracts = []
            context.functions = []
            for project_data in project_data_list:
                context.files.append(project_data.file)
                context.contracts.extend(project_data.contracts)
                context.functions.extend(project_data.functions)
            logger.info(f"Loaded {len(context.files)} files, {len(context.contracts)} contracts, {len(context.functions)} functions")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            if hasattr(context, 'add_error'):
                context.add_error(
                    stage="document_loading",
                    message=f"Failed to load documents: {str(e)}",
                    exception=e
                )
        return context
        
    async def _load_file(self, file_path: str, context: Context) -> None:
        """Load a single file into the context"""
        try:
            # Generate a file ID
            file_id = os.path.basename(file_path)
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Determine if it's a Solidity file
            is_solidity = file_path.endswith('.sol')
            # Add to context
            context.files[file_id] = {
                "id": file_id,
                "path": file_path,
                "content": content,
                "is_solidity": is_solidity
            }
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
            
    async def chunk_documents(self, context: Context) -> Context:
        """
        Chunk documents into manageable pieces
        
        Args:
            context: Pipeline context
        
        Returns:
            Updated context with chunked documents
        """
        logger.info("Chunking documents")
        if not hasattr(context, 'files') or not context.files:
            logger.warning("No files to chunk")
            return context
            
        if not hasattr(context, 'chunks'):
            context.chunks = {}
            
        try:
            chunk_size = getattr(context, 'chunk_size', 1000)
            chunk_overlap = getattr(context, 'chunk_overlap', 200)
            chunker = AsyncContractChunker(chunk_size, chunk_overlap)
            
            for file_id, file_data in context.files.items():
                try:
                    if "meta" in file_data and "file_name" in file_data["meta"] and "file_path" in file_data["meta"]:
                        content = file_data["content"]
                        if len(content) > 51 and content.startswith("Text: "):
                            content = content[51:]
                        # Get chunks
                        chunks = await chunker.chunk_code(
                            content,
                            file_data["meta"]["file_name"], 
                            file_data["meta"]["file_path"], 
                        )
                        for chunk in chunks:
                            chunk_id = chunk.get("chunk_id", f"{file_id}_{len(context.chunks)}")
                            context.chunks[chunk_id] = Box({
                                "id": chunk_id,
                                "file_id": file_data["meta"]["file_path"],
                                "content": chunk.get("content", ""),
                                "start_char": chunk.get("start_char", 0),
                                "end_char": chunk.get("end_char", 0),
                                "chunk_type": chunk.get("chunk_type", "unknown"),
                                "index": chunk_id,
                            })
                    else:
                        is_solidity = file_data.get("is_solidity", file_id.endswith('.sol'))
                        if is_solidity:
                            content = file_data.get("content", "")
                            if len(content) > 51 and content.startswith("Text: "):
                                content = content[51:]
                            # Get chunks
                            file_chunks = await chunker.chunk_code(
                                content,
                                os.path.basename(file_id), 
                                file_id
                            )
                            for chunk in file_chunks:
                                chunk_id = chunk.get("chunk_id", f"{file_id}_{len(context.chunks)}")
                                context.chunks[chunk_id] = Box({
                                    "id": chunk_id,
                                    "file_id": file_id,
                                    "content": chunk.get("content", ""),
                                    "start_char": chunk.get("start_char", 0),
                                    "end_char": chunk.get("end_char", 0),
                                    "chunk_type": chunk.get("chunk_type", "unknown"),
                                    "index": chunk_id,
                                })
                        else:
                            logger.debug(f"Skipping non-Solidity file: {file_id}")
                except Exception as ex:
                    logger.error(f"Error chunking file {file_id}: {str(ex)}")
                    if hasattr(context, 'add_error'):
                        context.add_error(
                            stage="document_chunking",
                            message=f"Failed to chunk file: {file_id}",
                            exception=ex
                        )
                        
            # Convert chunks to ChunkData objects
            chunk_data_list = []
            for chunk_id, chunk_data in context.chunks.items():
                chunk_data_list.append(ChunkData(**chunk_data))
            context.chunks = chunk_data_list
            logger.info(f"Created {len(context.chunks)} chunks from {len(context.files)} files")
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            if hasattr(context, 'add_error'):
                context.add_error(
                    stage="document_chunking",
                    message=f"Failed to chunk documents: {str(e)}",
                    exception=e
                )
        return context
        
    async def extract_functions(self, context: Context) -> Context:
        """
        Extract functions from the loaded documents.
        
        Args:
            context (Context): The pipeline context.
        
        Returns:
            Context: The updated pipeline context.
        """
        logger.info("Extracting functions")
        if not hasattr(context, 'project_data') or not context.project_data:
            logger.warning("No project data found. Skipping function extraction.")
            return context
            
        try:
            # Consolidate functions into a single list
            all_functions = []
            for project_data in context.project_data:
                all_functions.extend(project_data.functions)
            # Update the context with the consolidated list of functions
            context.functions = all_functions
            logger.info(f"Extracted {len(context.functions)} functions")
            return context
        except Exception as e:
            logger.error(f"Error extracting functions: {e}")
            if hasattr(context, 'add_error'):
                context.add_error(
                    stage="function_extraction",
                    message=f"Failed to extract functions: {str(e)}",
                    exception=e
                )
            return context

    async def extract_contracts_from_files(self, context: Context) -> Context:
        """
        Extract contracts from files using the extractor utility
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with extracted contracts
        """
        logger.info("Extracting contracts from files")
        if not hasattr(context, 'files') or not context.files:
            logger.warning("No files to extract contracts from")
            return context
            
        try:
            # Use the extract_contracts_from_files function from extractors module
            await extract_contracts_from_files(context)
            logger.info(f"Extracted contracts: {len(context.contracts) if hasattr(context, 'contracts') else 0}")
        except Exception as e:
            logger.error(f"Error extracting contracts: {e}")
            if hasattr(context, 'add_error'):
                context.add_error(
                    stage="contract_extraction",
                    message=f"Failed to extract contracts: {str(e)}",
                    exception=e
                )
        return context
    
    async def extract_business_flows(self, context: Context) -> Context:
        """
        Extract business flows from contracts and functions
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with business flows
        """
        logger.info("Extracting business flows")
        if not hasattr(context, 'contracts') or not context.contracts:
            logger.warning("No contracts to extract business flows from")
            return context
            
        if not hasattr(context, 'functions') or not context.functions:
            logger.warning("No functions to extract business flows from")
            return context
        
        try:
            # Create the business flow extractor
            extractor = BusinessFlowExtractor(llm_adapter=AsyncOllamaClient(config.DEFAULT_MODEL))
            
            # Extract flows
            if not hasattr(context, 'business_flows'):
                context.business_flows = []
                
            # Pass the contracts and functions to the extractor
            business_flows = await extractor.extract_business_flows(context.contracts, context.functions)
            context.business_flows = business_flows
            
            logger.info(f"Extracted {len(context.business_flows)} business flows")
        except Exception as e:
            logger.error(f"Error extracting business flows: {e}")
            if hasattr(context, 'add_error'):
                context.add_error(
                    stage="business_flow_extraction",
                    message=f"Failed to extract business flows: {str(e)}",
                    exception=e
                )
        return context
    
    async def extract_data_flows(self, context: Context) -> Context:
        """
        Extract data flows from contracts and functions
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with data flows
        """
        logger.info("Extracting data flows")
        if not hasattr(context, 'contracts') or not context.contracts:
            logger.warning("No contracts to extract data flows from")
            return context
            
        if not hasattr(context, 'functions') or not context.functions:
            logger.warning("No functions to extract data flows from")
            return context
            
        try:
            # Create the data flow analyzer
            analyzer = DataFlowAnalyzer()
            
            # Extract data flows
            if not hasattr(context, 'data_flows'):
                context.data_flows = []
                
            # Pass the contracts and functions to the analyzer
            data_flows = await analyzer.analyze_data_flows(context.contracts, context.functions)
            context.data_flows = data_flows
            
            logger.info(f"Extracted {len(context.data_flows)} data flows")
        except Exception as e:
            logger.error(f"Error extracting data flows: {e}")
            if hasattr(context, 'add_error'):
                context.add_error(
                    stage="data_flow_analysis",
                    message=f"Failed to extract data flows: {str(e)}",
                    exception=e
                )
        return context
    
    async def scan_vulnerabilities(self, context: Context) -> Context:
        """
        Scan for vulnerabilities in contracts
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with vulnerabilities
        """
        logger.info("Scanning for vulnerabilities")
        if not hasattr(context, 'contracts') or not context.contracts:
            logger.warning("No contracts to scan for vulnerabilities")
            return context
            
        try:
            # Create the vulnerability scanner
            scanner = VulnerabilityScanner()
            
            # Scan for vulnerabilities
            if not hasattr(context, 'vulnerabilities'):
                context.vulnerabilities = []
                
            # Scan the contracts for vulnerabilities
            vulnerabilities = await scanner.scan_contracts(context.contracts)
            context.vulnerabilities = vulnerabilities
            
            logger.info(f"Found {len(context.vulnerabilities)} vulnerabilities")
        except Exception as e:
            logger.error(f"Error scanning for vulnerabilities: {e}")
            if hasattr(context, 'add_error'):
                context.add_error(
                    stage="vulnerability_scanning",
                    message=f"Failed to scan for vulnerabilities: {str(e)}",
                    exception=e
                )
        return context
    
    def threat_detection_stage(self) -> Stage:
        async def detect_threats(context: Context) -> Context:
            logger.info("Detecting threats based on business flows")
            if not require_attributes(context, ['contracts', 'business_flows']):
                return context
            try:
                if not hasattr(context, 'threats'):
                    context.threats = []
                context.threats = await self.threat_detector.detect_threats(context.contracts, context.business_flows)
                logger.info(f"Detected {len(context.threats)} potential threats")
            except Exception as e:
                logger.error(f"Error in threat detection: {str(e)}")
                if hasattr(context, 'add_error'):
                    context.add_error(
                        stage="threat_detection",
                        message=f"Failed to detect threats: {str(e)}",
                        exception=e
                    )
            return context
        return Stage(detect_threats, name="threat_detection")

    def counterfactual_stage(self) -> Stage:
        async def analyze_counterfactuals(context: Context) -> Context:
            logger.info("Generating counterfactual scenarios")
            if not require_attributes(context, ['contracts', 'business_flows', 'functions']):
                return context
            try:
                if not hasattr(context, 'counterfactuals'):
                    context.counterfactuals = []
                if not hasattr(context, 'analysis_results'):
                    context.analysis_results = {}
                counterfactuals = await self.counterfactual_analyzer.generate_counterfactuals(
                    context.contracts, context.functions, context.business_flows)
                context.counterfactuals = counterfactuals
                context.analysis_results['counterfactual_analysis'] = {
                    'count': len(counterfactuals),
                    'summary': self.counterfactual_analyzer.get_summary()
                }
                logger.info(f"Generated {len(context.counterfactuals)} counterfactual scenarios")
            except Exception as e:
                logger.error(f"Error in counterfactual analysis: {str(e)}")
                if hasattr(context, 'add_error'):
                    context.add_error(
                        stage="counterfactual_analysis",
                        message=f"Failed to generate counterfactuals: {str(e)}",
                        exception=e
                    )
            return context
        return Stage(analyze_counterfactuals, name="counterfactual_analysis")

    def cognitive_bias_stage(self) -> Stage:
        async def analyze_cognitive_biases(context: Context) -> Context:
            logger.info("Analyzing for cognitive biases")
            if not require_attributes(context, ['contracts']):
                return context
            try:
                if not hasattr(context, 'analysis_results'):
                    context.analysis_results = {}
                # Assuming a single contract analysis per stage; adjust if multiple
                bias_result = await self.cognitive_bias_analyzer.analyze_cognitive_biases(
                    contract_code="...contract code...",  # Ideally use context.selected_contract or similar
                    contract_name="...contract name..."
                )
                context.analysis_results['cognitive_bias'] = bias_result.bias_findings
                logger.info("Completed cognitive bias analysis")
            except Exception as e:
                logger.error(f"Error in cognitive bias analysis: {str(e)}")
                if hasattr(context, 'add_error'):
                    context.add_error(
                        stage="cognitive_bias_analysis",
                        message=f"Failed to analyze cognitive biases: {str(e)}",
                        exception=e
                    )
            return context
        return Stage(analyze_cognitive_biases, name="cognitive_bias_analysis")

    def documentation_analysis_stage(self) -> Stage:
        async def analyze_documentation(context: Context) -> Context:
            logger.info("Analyzing documentation quality")
            if not require_attributes(context, ['contracts', 'functions']):
                return context
            try:
                if not hasattr(context, 'analysis_results'):
                    context.analysis_results = {}
                doc_results = await self.documentation_analyzer.analyze_documentation(context.contracts, context.functions)
                context.analysis_results['documentation'] = doc_results
                logger.info("Completed documentation analysis")
            except Exception as e:
                logger.error(f"Error in documentation analysis: {str(e)}")
                if hasattr(context, 'add_error'):
                    context.add_error(
                        stage="documentation_analysis",
                        message=f"Failed to analyze documentation: {str(e)}",
                        exception=e
                    )
            return context
        return Stage(analyze_documentation, name="documentation_analysis")

    def analyze_documentation_inconsistency_stage(self) -> Stage:
        async def analyze_documentation_inconsistencies(context: Context) -> Context:
            logger.info("Analyzing documentation inconsistencies")
            if not require_attributes(context, ['contracts']):
                return context
            try:
                if not hasattr(context, 'analysis_results'):
                    context.analysis_results = {}
                if 'documentation_inconsistencies' not in context.analysis_results:
                    context.analysis_results['documentation_inconsistencies'] = []
                inconsistencies = await self.doc_inconsistency_adapter.analyze(context)
                context.analysis_results['documentation_inconsistencies'] = inconsistencies
                logger.info(f"Found {len(inconsistencies)} documentation inconsistencies")
            except Exception as e:
                logger.error(f"Error in documentation inconsistency analysis: {str(e)}")
                if hasattr(context, 'add_error'):
                    context.add_error(
                        stage="documentation_inconsistency_analysis",
                        message=f"Failed to analyze documentation inconsistencies: {str(e)}",
                        exception=e
                    )
            return context
        return Stage(analyze_documentation_inconsistencies, name="documentation_inconsistency_analysis")
    
    def _vector_prompt_stage(self) -> Stage:
        """
        Create a stage for generating prompts from vector database matches.
        
        This stage takes vector matching results and generates prompts
        that can be used with an LLM to gain insights about similar patterns.
        
        Returns:
            A configured Stage for vector match prompting
        """
        async def generate_vector_prompts(context: Context) -> Context:
            """Generate prompts from vector database matches"""
            logger.info("Generating prompts from vector database matches")
            if not hasattr(context, 'vector_matches') or not context.vector_matches:
                logger.info("No vector matches found for prompt generation")
                return context
            
            # Create a place to store prompts and responses
            if not hasattr(context, 'vector_prompts'):
                context.vector_prompts = {}
            if getattr(context, 'execute_vector_prompts', True) and not hasattr(context, 'vector_prompt_results'):
                context.vector_prompt_results = {}
            
            try:
                # Process each contract's vector matches
                for contract_id, matches in context.vector_matches.items():
                    contract = context.contracts.get(contract_id)
                    if not contract:
                        logger.warning(f"Contract {contract_id} not found for vector prompt generation")
                        continue
                    
                    # Get contract name for logging
                    contract_name = contract.get('name', contract_id) if isinstance(contract, dict) else getattr(contract, 'name', contract_id)
                    
                    # Create prompts for this contract
                    contract_prompts = []
                    for i, match in enumerate(matches[:5]):  # Limit to 5 top matches
                        match_score = match.get('score', 0)
                        if match_score < 0.5:  # Skip low relevance matches
                            continue
                        
                        prompt = {
                            "id": f"{contract_id}_prompt_{i}",
                            "contract_id": contract_id,
                            "contract_name": contract_name,
                            "match_id": match.get('id', f"match_{i}"),
                            "match_score": match_score,
                            "prompt_text": f"Analyze this code pattern in {contract_name} that matches with {match.get('pattern', 'unknown pattern')} with {match_score:.2f} relevance score."
                        }
                        contract_prompts.append(prompt)
                    
                    # Store prompts for this contract
                    if contract_prompts:
                        context.vector_prompts[contract_id] = contract_prompts
                
                logger.info(f"Generated vector prompts for {len(context.vector_prompts)} contracts")
            except Exception as e:
                logger.error(f"Error in vector prompt generation: {str(e)}")
                if hasattr(context, 'add_error'):
                    context.add_error(
                        stage="vector_prompt_generation",
                        message=f"Failed to generate vector prompts: {str(e)}",
                        exception=e
                    )
                    
            return context
        
        return Stage(generate_vector_prompts, name="vector_prompt_generation")
    
    def _get_process_function(self, stage: Stage) -> Callable[[Context], Awaitable[Context]]:
        """
        Extract the process function from a Stage object.
        
        Args:
            stage: A Stage object
            
        Returns:
            The callable process function
        """
        # Try different attribute names used in Stage objects
        if hasattr(stage, 'process'):
            return stage.process
        elif hasattr(stage, '_process_func'):
            return stage._process_func
        elif hasattr(stage, '__call__'):
            return stage.__call__
        elif callable(stage):
            return stage
        else:
            raise AttributeError(f"Cannot find process function in Stage object: {stage}")
    
    def _analyze_vector_prompts_stage(self) -> Callable[[Context], Awaitable[Context]]:
        """
        Create a stage for generating and analyzing prompts from vector matches.
        
        Returns:
            Callable stage function
        """
        vector_prompt_stage = self._vector_prompt_stage()
        if vector_prompt_stage is None:
            raise ValueError("Vector prompt stage is None - implementation error in _vector_prompt_stage")
        
        return self._get_process_function(vector_prompt_stage)