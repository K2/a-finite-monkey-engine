"""
Pipeline Orchestrator for Finite Monkey Engine
Implements dual-ring architecture with:
- Outer ring: Manager agents for coordination, evaluation, and reporting
- Inner ring: Detail-oriented agents leveraging LlamaIndex for specific tasks
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Callable
from loguru import logger
from enum import Enum
from dataclasses import dataclass, field
import time
from pathlib import Path

from .core import Context, Pipeline, Stage
from .sources import FileSource, DirectorySource, GithubSource
# Updated import to include BusinessFlowExtractor
from .transformers import AgentState, AsyncContractChunker, FunctionExtractor, CallGraphBuilder, ASTAnalyzer, BusinessFlowExtractor, agent_workflow

# Add imports for functional utilities
from ..utils.functional import async_pipe, amap, AsyncPipeline
from .functional_pipeline import FunctionalPipeline
from .effect_orchestrator import EffectOrchestrator, EffectResult, WorkflowEvent

# Flow state for tracking pipeline progress
class FlowState(str, Enum):
    """States for tracking pipeline workflow"""
    PREPARING = "preparing"
    LOADING = "loading_data"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    VALIDATING = "validating"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowEvent:
    """Event for tracking state transitions in workflow"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = "state_change"
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    agent_name: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@agent_workflow
class PipelineManager:
    """
    Pipeline manager for orchestrating the full analysis workflow
    
    This is an outer ring manager agent that coordinates the pipeline stages,
    tracks state, and ensures the workflow progresses correctly.
    """
    
    def __init__(self, name: str = "PipelineManager"):
        """Initialize the pipeline manager"""
        self.name = name
        self.state = AgentState.IDLE
        self.flow_state = FlowState.PREPARING
        self.events: List[WorkflowEvent] = []
        self.start_time = time.time()
        
    def record_event(self, event_type: str, description: str, **metadata) -> WorkflowEvent:
        """Record a workflow event"""
        event = WorkflowEvent(
            event_type=event_type,
            description=description,
            metadata=metadata
        )
        self.events.append(event)
        
        # Log the event
        logger.info(f"Event: {event_type} - {description}")
        
        return event
        
    def set_flow_state(self, new_state: FlowState) -> None:
        """Update the flow state with event tracking"""
        old_state = self.flow_state
        self.flow_state = new_state
        
        # Record the state transition
        self.record_event(
            event_type="state_change",
            description=f"Workflow state change: {old_state} → {new_state}",
            from_state=old_state,
            to_state=new_state
        )
        
    async def create_pipeline(
        self,
        pipeline_name: str = "smart_contract_analysis",
        include_ast_analysis: bool = True,
        include_call_graph: bool = True,
        include_business_flow: bool = True
    ) -> Pipeline:
        """
        Create a pipeline with the specified stages
        
        Args:
            pipeline_name: Name for the pipeline
            include_ast_analysis: Whether to include AST analysis
            include_call_graph: Whether to include call graph building
            include_business_flow: Whether to include business flow extraction
            
        Returns:
            Configured Pipeline instance
        """
        # Create the pipeline
        pipeline = Pipeline(name=pipeline_name)
        
        # Extract functions from contracts
        chunker = AsyncContractChunker()
        function_extractor = FunctionExtractor()
        
        # Add base stages
        pipeline.add_stage(Stage(
            name="contract_chunking",
            func=chunker.process,
            description="Split contracts into manageable chunks",
            retry_count=1
        ))
        
        pipeline.add_stage(Stage(
            name="function_extraction",
            func=function_extractor.process,
            description="Extract functions from contracts",
            retry_count=1
        ))
        
        # Add business flow extraction if enabled
        if include_business_flow:
            business_flow_extractor = BusinessFlowExtractor()
            pipeline.add_stage(Stage(
                name="business_flow_extraction",
                func=business_flow_extractor.process,
                description="Extract business flows from functions",
                retry_count=1,
                required=False
            ))
        
        # Add optional stages
        if include_call_graph:
            call_graph_builder = CallGraphBuilder()
            pipeline.add_stage(Stage(
                name="call_graph_building",
                func=call_graph_builder.process,
                description="Build function call graph",
                retry_count=1,
                required=False
            ))
        
        if include_ast_analysis:
            ast_analyzer = ASTAnalyzer()
            pipeline.add_stage(Stage(
                name="ast_analysis",
                func=ast_analyzer.process,
                description="Analyze contract AST for patterns",
                retry_count=1,
                required=False
            ))
        
        return pipeline

    async def create_functional_pipeline(
        self,
        pipeline_name: str = "smart_contract_analysis",
        include_ast_analysis: bool = True,
        include_call_graph: bool = True,
        include_business_flow: bool = True
    ) -> FunctionalPipeline[Context]:
        """
        Create a functional pipeline with the specified stages
        
        Args:
            pipeline_name: Name for the pipeline
            include_ast_analysis: Whether to include AST analysis
            include_call_graph: Whether to include call graph building
            include_business_flow: Whether to include business flow extraction
            
        Returns:
            Configured FunctionalPipeline instance
        """
        # Create the pipeline
        pipeline = FunctionalPipeline[Context](pipeline_name)
        
        # Create component instances
        chunker = AsyncContractChunker()
        function_extractor = FunctionExtractor()
        
        # Add base stages with functional composition
        pipeline.add_stage("contract_chunking", chunker.process)
        pipeline.add_stage("function_extraction", function_extractor.process)
        
        # Add business flow extraction if enabled
        if include_business_flow:
            business_flow_extractor = BusinessFlowExtractor()
            pipeline.add_stage("business_flow_extraction", business_flow_extractor.process)
        
        # Add optional stages
        if include_call_graph:
            call_graph_builder = CallGraphBuilder()
            pipeline.add_stage("call_graph_building", call_graph_builder.process)
        
        if include_ast_analysis:
            ast_analyzer = ASTAnalyzer()
            pipeline.add_stage("ast_analysis", ast_analyzer.process)
        
        return pipeline
    
    async def run_pipeline_with_effects(
        self,
        source_path: Union[str, Path],
        is_directory: bool = False,
        is_github_repo: bool = False,
        file_pattern: str = "**/*.sol",
        config: Optional[Dict[str, Any]] = None
    ) -> EffectResult[Context]:
        """
        Run a pipeline with effect handling for improved functional composition
        
        Args:
            source_path: Path to the source (file, directory, or repository)
            is_directory: Whether the source is a directory
            is_github_repo: Whether the source is a GitHub repository
            file_pattern: Pattern for matching files
            config: Optional configuration
            
        Returns:
            EffectResult containing the pipeline context and events
        """
        # Create an orchestrator for effect handling
        orchestrator = EffectOrchestrator(name=f"{self.name}_orchestrator")
        orchestrator.record_event(orchestrator.create_event(
            event_type="pipeline_start",
            description=f"Starting pipeline for {source_path}"
        ))
        
        # Create initial context
        context = Context(config=config or {})
        
        # Select source and load data
        try:
            source = await self._get_source(source_path, is_directory, is_github_repo, file_pattern)
            orchestrator.transition_state(FlowState.LOADING)
            
            # Load data with effect tracking
            context = await source.load_into_context(context)
            orchestrator.record_event(orchestrator.create_event(
                event_type="data_loaded",
                description=f"Loaded {len(context.files)} files from source"
            ))
            
            # Create and run functional pipeline
            pipeline = await self.create_functional_pipeline()
            orchestrator.transition_state(FlowState.PROCESSING)
            
            # Run the pipeline with effect handling
            result = await orchestrator.run_with_effects(pipeline, context)
            
            # Transition to reporting state
            orchestrator.transition_state(FlowState.REPORTING)
            
            return result
            
        except Exception as e:
            # Handle errors functionally
            error_event = orchestrator.create_event(
                event_type="error",
                description=f"Pipeline execution failed: {str(e)}",
                error=str(e)
            )
            orchestrator.record_event(error_event)
            orchestrator.transition_state(FlowState.FAILED)
            
            # Return error result
            return EffectResult(
                value=context,
                events=orchestrator.events,
                metadata={"error": str(e)}
            )
    
    async def _get_source(
        self, 
        source_path: Union[str, Path],
        is_directory: bool,
        is_github_repo: bool,
        file_pattern: str
    ) -> Union[FileSource, DirectorySource, GithubSource]:
        """Get the appropriate source handler using pattern matching"""
        source_path = Path(source_path) if isinstance(source_path, str) else source_path
        
        if is_github_repo:
            return GithubSource(str(source_path), file_pattern=file_pattern)
        elif is_directory or source_path.is_dir():
            return DirectorySource(source_path, file_pattern=file_pattern)
        else:
            return FileSource(source_path)
        
    async def run_pipeline(
        self,
        source_path: Union[str, Path],
        is_directory: bool = False,
        is_github_repo: bool = False,
        file_pattern: str = "**/*.sol",
        config: Optional[Dict[str, Any]] = None
    ) -> Context:
        """
        Run the pipeline with the specified source
        
        Args:
            source_path: Path to file, directory, or GitHub repo
            is_directory: Whether source_path is a directory
            is_github_repo: Whether source_path is a GitHub repo URL
            file_pattern: Pattern for matching files (if directory)
            config: Optional configuration parameters
            
        Returns:
            Pipeline context with results
        """
        self.state = AgentState.INITIALIZING
        self.set_flow_state(FlowState.PREPARING)
        
        # Create context
        context = Context()
        if config:
            context.config.update(config)
        
        # Set workflow tracking info
        context.state["workflow_id"] = str(uuid.uuid4())
        context.state["workflow_start"] = time.time()
        
        # Create pipeline
        pipeline = await self.create_pipeline()
        
        try:
            # Load data
            self.state = AgentState.EXECUTING
            self.set_flow_state(FlowState.LOADING)
            
            if is_github_repo:
                github_source = GithubSource()
                context = await github_source.process(
                    context=context,
                    repo_url=source_path,
                    pattern=file_pattern
                )
            elif is_directory:
                directory_source = DirectorySource()
                context = await directory_source.process(
                    context=context,
                    directory_path=source_path,
                    pattern=file_pattern
                )
            else:
                file_source = FileSource()
                context = await file_source.process(
                    context=context,
                    file_path=source_path
                )
            
            # Run pipeline
            self.set_flow_state(FlowState.PROCESSING)
            result, context = await pipeline.run(context=context)
            
            # Handle success
            self.set_flow_state(FlowState.COMPLETED)
            self.state = AgentState.COMPLETED
            
            # Record completion metrics
            context.state["workflow_end"] = time.time()
            context.state["workflow_duration"] = context.state["workflow_end"] - context.state["workflow_start"]
            
            return context
            
        except Exception as e:
            # Handle failure
            logger.exception(f"Pipeline failed: {str(e)}")
            self.set_flow_state(FlowState.FAILED)
            self.state = AgentState.FAILED
            
            # Record failure
            context.add_error(
                stage="pipeline_manager",
                message=f"Pipeline failed: {str(e)}",
                exception=e
            )
            
            # Still return context for partial results
            return context


@agent_workflow(log_level="DEBUG")
class LlamaIndexAgent:
    """
    Inner loop agent leveraging LlamaIndex for detailed analysis
    
    This agent specializes in contextual understanding of smart contracts
    and provides vector-based semantic search capabilities.
    """
    
    def __init__(self, name: str = "LlamaIndexAgent"):
        """Initialize the LlamaIndex agent"""
        self.name = name
        self.state = AgentState.IDLE
        self.index = None
        self.project_id = None
        
    async def initialize(self, project_id: str) -> None:
        """Initialize the LlamaIndex engine"""
        self.state = AgentState.INITIALIZING
        
        try:
            # Import LlamaIndex only when needed
            try:
                from ...llama_index import AsyncIndexProcessor
                self.index = AsyncIndexProcessor(project_id=project_id)
                self.project_id = project_id
                
                logger.info(f"LlamaIndex agent initialized for project: {project_id}")
                self.state = AgentState.IDLE
                
            except ImportError:
                logger.error("LlamaIndex not available, agent initialization failed")
                self.state = AgentState.FAILED
                raise ImportError("LlamaIndex not available")
                
        except Exception as e:
            logger.exception(f"Failed to initialize LlamaIndex agent: {str(e)}")
            self.state = AgentState.FAILED
            raise
    
    async def process_context(self, context: Context) -> Context:
        """
        Process the pipeline context with LlamaIndex
        
        Args:
            context: Pipeline context with files and chunks
            
        Returns:
            Updated context with LlamaIndex results
        """
        self.state = AgentState.INITIALIZING
        
        # Get project ID from context or generate one
        project_id = context.state.get("project_id", f"project_{context.pipeline_id}")
        
        # Initialize if needed
        if self.index is None or self.project_id != project_id:
            await self.initialize(project_id)
            
        self.state = AgentState.EXECUTING
        
        # Prepare documents for indexing
        documents = []
        
        # Add files
        for file_id, file_data in context.files.items():
            if not file_data.get("is_solidity", False):
                continue
                
            documents.append({
                "id": file_id,
                "text": file_data["content"],
                "metadata": {
                    "path": file_data["path"],
                    "name": file_data["name"],
                    "type": "file"
                }
            })
            
            # Add chunks if available
            if "chunks" in file_data:
                for chunk in file_data["chunks"]:
                    documents.append({
                        "id": chunk["id"],
                        "text": chunk["content"],
                        "metadata": {
                            "path": file_data["path"],
                            "name": file_data["name"],
                            "chunk_type": chunk["chunk_type"],
                            "start_line": chunk.get("start_line"),
                            "end_line": chunk.get("end_line"),
                            "type": "chunk"
                        }
                    })
                    
            # Add functions if available
            if "functions" in file_data:
                for function in file_data["functions"]:
                    documents.append({
                        "id": function.get("id", f"{file_id}:{function['name']}"),
                        "text": function.get("full_text", ""),
                        "metadata": {
                            "path": file_data["path"],
                            "name": function["name"],
                            "visibility": function.get("visibility", ""),
                            "start_line": function.get("start_line", 0),
                            "end_line": function.get("end_line", 0),
                            "type": "function"
                        }
                    })
                    
            # Add business flows if available
            if "business_flows" in file_data:
                for flow in file_data["business_flows"]:
                    documents.append({
                        "id": flow.get("id", f"{file_id}:flow:{flow['name']}"),
                        "text": flow.get("description", "") + "\n" + flow.get("flow_text", ""),
                        "metadata": {
                            "path": file_data["path"],
                            "name": flow["name"],
                            "flow_type": flow.get("flow_type", ""),
                            "start_function": flow.get("start_function", ""),
                            "type": "business_flow"
                        }
                    })
        
        # Index documents
        logger.info(f"Indexing {len(documents)} documents with LlamaIndex")
        await self.index.add_documents(documents)
        
        # Store index info in context
        context.state["llama_index_info"] = {
            "project_id": project_id,
            "document_count": len(documents),
            "indexed_at": time.time()
        }
        
        self.state = AgentState.COMPLETED
        return context
        
    async def search(
        self, 
        context: Context,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search the indexed documents
        
        Args:
            context: Pipeline context
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Search results
        """
        self.state = AgentState.EXECUTING
        
        if self.index is None:
            project_id = context.state.get("project_id", f"project_{context.pipeline_id}")
            await self.initialize(project_id)
        
        # Perform search
        logger.info(f"Searching with query: '{query}'")
        results = await self.index.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # Format results
        formatted_results = {
            "query": query,
            "result_count": len(results),
            "results": results
        }
        
        # Store in context
        context.state["last_search_query"] = query
        context.state["last_search_results"] = formatted_results
        
        self.state = AgentState.COMPLETED
        return formatted_results


@agent_workflow
class WorkflowOrchestrator:
    """
    Top-level orchestrator managing the dual-ring architecture
    
    This orchestrator coordinates:
    1. Outer ring: Manager agents (PipelineManager)
    2. Inner ring: Detail agents (LlamaIndex, LLM processors)
    """
    
    def __init__(self, name: str = "WorkflowOrchestrator"):
        """Initialize the workflow orchestrator"""
        self.name = name
        self.state = AgentState.IDLE
        
        # Initialize agents
        self.pipeline_manager = PipelineManager()
        self.llama_index_agent = LlamaIndexAgent()
        
        # Telemetry
        self.workflow_events = []
        
    def track_workflow(self, context: Context, event_type: str, description: str, **metadata) -> None:
        """Track workflow events"""
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "event_type": event_type,
            "description": description,
            "metadata": metadata,
            "pipeline_id": context.pipeline_id,
        }
        
        self.workflow_events.append(event)
        
        # Log the event
        logger.info(f"Workflow: {event_type} - {description}")
        
        # Store in context
        if "workflow_events" not in context.state:
            context.state["workflow_events"] = []
        context.state["workflow_events"].append(event)
        
    async def run_analysis(
        self,
        source_path: Union[str, Path],
        query: Optional[str] = None,
        is_directory: bool = False,
        is_github_repo: bool = False,
        file_pattern: str = "**/*.sol",
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> Tuple[Context, Dict[str, Any]]:
        """
        Run the complete analysis workflow
        
        Args:
            source_path: Path to file, directory, or GitHub repo
            query: Optional analysis query
            is_directory: Whether source_path is a directory
            is_github_repo: Whether source_path is a GitHub repo URL
            file_pattern: Pattern for matching files (if directory)
            config: Optional configuration parameters
            callbacks: Optional callbacks for workflow events
            
        Returns:
            Tuple of (context, results)
        """
        self.state = AgentState.INITIALIZING
        
        # Initialize config if needed
        if config is None:
            config = {}
        
        # Set up defaults
        config.setdefault("project_id", str(uuid.uuid4()))
        config.setdefault("include_ast_analysis", True)
        config.setdefault("include_call_graph", True)
        config.setdefault("include_business_flow", True)
        
        # Initialize the workflow
        self.track_workflow(
            context=Context(pipeline_id=config["project_id"]),
            event_type="workflow_start",
            description=f"Starting analysis workflow for {source_path}",
            source_path=str(source_path),
            is_directory=is_directory,
            is_github_repo=is_github_repo
        )
        
        try:
            # Phase 1: Data Loading and Processing (Outer Ring)
            self.state = AgentState.EXECUTING
            
            # Run pipeline
            context = await self.pipeline_manager.run_pipeline(
                source_path=source_path,
                is_directory=is_directory,
                is_github_repo=is_github_repo,
                file_pattern=file_pattern,
                config=config
            )
            
            # Track phase completion
            self.track_workflow(
                context=context,
                event_type="phase_complete",
                description="Completed data loading and processing phase",
                file_count=len(context.files),
                chunk_count=len(context.chunks),
                function_count=len(context.functions)
            )
            
            # Phase 2: LlamaIndex Processing (Inner Ring)
            self.state = AgentState.ANALYZING
            
            # Process with LlamaIndex
            context = await self.llama_index_agent.process_context(context)
            
            # Run query if provided
            results = {}
            if query:
                results = await self.llama_index_agent.search(
                    context=context,
                    query=query
                )
                
                # Extract findings from results
                self._extract_findings_from_results(context, results)
                
            # Phase 3: Finalization and Reporting
            self.state = AgentState.REPORTING
            
            # Execute callbacks if provided
            if callbacks:
                for event_name, callback_fn in callbacks.items():
                    if callable(callback_fn):
                        await self._execute_callback(callback_fn, context, event_name)
            
            # Generate summary
            summary = self._generate_summary(context)
            results["summary"] = summary
            
            # Track workflow completion
            self.track_workflow(
                context=context,
                event_type="workflow_complete",
                description="Analysis workflow completed successfully",
                finding_count=len(context.findings),
                duration=time.time() - context.state.get("workflow_start", time.time())
            )
            
            self.state = AgentState.COMPLETED
            return context, results
            
        except Exception as e:
            logger.exception(f"Analysis workflow failed: {str(e)}")
            
            # Create minimal context if we don't have one yet
            if not hasattr(self, 'context'):
                context = Context(pipeline_id=config["project_id"])
                
            # Track failure
            self.track_workflow(
                context=context,
                event_type="workflow_failed",
                description=f"Analysis workflow failed: {str(e)}",
                error=str(e),
                error_type=type(e).__name__
            )
            
            self.state = AgentState.FAILED
            raise
            
    async def _execute_callback(self, callback: Callable, context: Context, event_name: str) -> None:
        """Execute a callback function, handling both sync and async callbacks"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(context, event_name)
            else:
                callback(context, event_name)
        except Exception as e:
            logger.error(f"Error in callback {event_name}: {str(e)}")
            
    def _extract_findings_from_results(self, context: Context, results: Dict[str, Any]) -> None:
        """Extract potential findings from search results"""
        search_results = results.get("results", [])
        
        for result in search_results:
            # Check if the result suggests a vulnerability
            text = result.lower() if isinstance(result, str) else str(result).lower()
            
            if any(term in text for term in [
                "vulnerability", "exploit", "attack", "bug", "issue", 
                "reentrancy", "overflow", "underflow", "unsafe"
            ]):
                finding = {
                    "title": "Potential Issue Detected",
                    "description": result if isinstance(result, str) else str(result),
                    "severity": "Medium",  # Default
                    "source": "LlamaIndex"
                }
                
                # Try to determine severity
                if any(term in text for term in ["critical", "severe", "high"]):
                    finding["severity"] = "High"
                elif any(term in text for term in ["low", "minor", "info"]):
                    finding["severity"] = "Low"
                
                context.add_finding(finding)
                
    def _generate_summary(self, context: Context) -> Dict[str, Any]:
        """Generate a summary of the analysis"""
        # Count business flows if they exist
        business_flow_count = 0
        for file_data in context.files.values():
            if "business_flows" in file_data:
                business_flow_count += len(file_data["business_flows"])
        
        return {
            "files_analyzed": len(context.files),
            "functions_found": len(context.functions),
            "chunks_generated": len(context.chunks),
            "business_flows_identified": business_flow_count,
            "findings_count": len(context.findings),
            "finding_severity": {
                "High": len([f for f in context.findings if f.get("severity") == "High"]),
                "Medium": len([f for f in context.findings if f.get("severity") == "Medium"]),
                "Low": len([f for f in context.findings if f.get("severity") == "Low"])
            },
            "processing_time": time.time() - context.state.get("workflow_start", time.time()),
            "top_issues": [f["title"] for f in context.findings[:5]] if context.findings else []
        }