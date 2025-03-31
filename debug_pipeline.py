#!/usr/bin/env python3
"""
Debug entrypoint for the Finite Monkey Engine.

This script creates a complete pipeline with all stages and processes
sample contracts from the examples/src directory, generating a detailed
markdown report for analysis and debugging.
"""
import os
import sys
import asyncio
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
import aiofiles
import aiofiles.os as aio_os

from finite_monkey.utils.async_doc_loader import AsyncDocumentLoader
from finite_monkey.utils.async_parser import AsyncSolidityParser

# Add detailed logging for debugging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
)
logger.add("logs/debug_pipeline_{time}.log", rotation="100 MB", level="DEBUG")

# Import all required components
try:
    from finite_monkey.pipeline.factory import PipelineFactory
    from finite_monkey.pipeline.core import Pipeline, Stage, Context
    from finite_monkey.utils.chunking import AsyncContractChunker
    from finite_monkey.analyzers.business_flow_extractor import BusinessFlowExtractor
    from finite_monkey.analyzers.vulnerability_scanner import VulnerabilityScanner
    from finite_monkey.analyzers.dataflow_analyzer import DataFlowAnalyzer
    from finite_monkey.analyzers.cognitive_bias_analyzer import CognitiveBiasAnalyzer
    from finite_monkey.analyzers.counterfactual_analyzer import CounterfactualAnalyzer
    from finite_monkey.analyzers.documentation_analyzer import DocumentationAnalyzer
    from finite_monkey.utils.flow_joiner import FlowJoiner
    from finite_monkey.adapters.agent_adapter import DocumentationInconsistencyAdapter
    from finite_monkey.utils.llm_monitor import LLMInteractionTracker
    from finite_monkey.adapters.validator_adapter import ValidatorAdapter
    from finite_monkey.adapters.llm_adapter import LLMAdapter
    from finite_monkey.nodes_config import config
    from finite_monkey.query_engine.flare_engine import FlareQueryEngine
except ImportError as e:
    logger.error(f"Failed to import required components: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Constants for file paths
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples", "src")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "debug_output")
REPORT_PATH = os.path.join(OUTPUT_DIR, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
ANALYSIS_PATH = os.path.join(os.path.dirname(__file__), "analyzed.md")

# Custom queries for testing FLARE engine
DEBUG_QUERIES = [
    "What are the main contracts in the codebase and what do they do?",
    "What potential security vulnerabilities exist in these contracts?",
    "Are there any gas optimization opportunities?",
    "Identify the key functions and their purposes",
    "Are there any inconsistencies between code and documentation?"
]

class DebugStage(Stage):
    """Debug stage that wraps another stage and logs detailed execution info"""
    
    def __init__(self, stage: Stage, name: Optional[str] = None):
        """Initialize with wrapped stage"""
        self.stage = stage
        self.name = name or f"Debug({stage.__class__.__name__})"
        
    async def __call__(self, context: Context) -> Context:
        """Execute stage with detailed logging"""
        logger.debug(f"Starting stage: {self.name}")
        logger.debug(f"Context before: {self._summarize_context(context)}")
        
        try:
            start_time = datetime.now()
            result = await self.stage(context)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.debug(f"Completed stage: {self.name} in {duration:.2f} seconds")
            logger.debug(f"Context after: {self._summarize_context(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in stage {self.name}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _summarize_context(self, context: Context) -> Dict[str, Any]:
        """Create a summarized version of context for logging"""
        summary = {}
        for key, value in context.__dict__.items():
            if key == "state":
                # Summarize the state dictionary
                state_summary = {}
                for state_key, state_value in context.state.items():
                    if isinstance(state_value, list):
                        state_summary[state_key] = f"List[{len(state_value)} items]"
                    elif isinstance(state_value, dict):
                        state_summary[state_key] = f"Dict[{len(state_value)} items]"
                    else:
                        state_summary[state_key] = str(type(state_value))
                summary[key] = state_summary
            elif isinstance(value, list):
                summary[key] = f"List[{len(value)} items]"
            elif isinstance(value, dict):
                summary[key] = f"Dict[{len(value)} items]"
            else:
                summary[key] = str(type(value))
        return summary


async def generate_markdown_report(context: Context, queries_results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive markdown report from the analysis results.
    
    Args:
        context: The context containing analysis results
        queries_results: Results from FLARE query engine
        
    Returns:
        Markdown report content
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start with the report header
    report = [
        "# Smart Contract Analysis Report",
        f"\nGenerated: {now}",
        "\n## Overview\n",
    ]
    
    # Add contract information
    if hasattr(context, 'contracts') and context.contracts:
        report.append(f"Analyzed {len(context.contracts)} contract(s)\n")
        for i, contract in enumerate(context.contracts):
            name = getattr(contract, 'name', f"Contract_{i}")
            path = getattr(contract, 'file_path', "Unknown")
            report.append(f"### {name}")
            report.append(f"- File: `{path}`")
            
            # Add contract size info if available
            code = getattr(contract, 'code', getattr(contract, 'content', ''))
            if code:
                lines = len(code.split('\n'))
                report.append(f"- Lines of code: {lines}")
            
            report.append("")
    else:
        report.append("No contracts were found or loaded.\n")
    
    # Add vulnerability scan results
    if hasattr(context, 'vulnerabilities') and context.vulnerabilities:
        report.append("## Security Vulnerabilities\n")
        vulns = context.vulnerabilities
        if isinstance(vulns, dict):
            for contract_name, contract_vulns in vulns.items():
                report.append(f"### {contract_name}\n")
                if contract_vulns:
                    for vuln in contract_vulns:
                        vuln_name = vuln.get('name', 'Unknown vulnerability')
                        severity = vuln.get('severity', 'Unknown')
                        description = vuln.get('description', 'No description available')
                        location = vuln.get('location', 'Unknown location')
                        
                        report.append(f"#### {vuln_name} ({severity})")
                        report.append(f"- **Description**: {description}")
                        report.append(f"- **Location**: {location}")
                        report.append("")
                else:
                    report.append("No vulnerabilities detected.\n")
        else:
            report.append("Vulnerability data is available but in an unexpected format.\n")
    
    # Add business flow analysis
    if hasattr(context, 'business_flows') and context.business_flows:
        report.append("## Business Flow Analysis\n")
        flows = context.business_flows
        if isinstance(flows, dict):
            for contract_name, contract_flows in flows.items():
                report.append(f"### {contract_name}\n")
                if contract_flows:
                    for flow in contract_flows:
                        # Handle both dictionary and BusinessFlow object instances
                        if isinstance(flow, dict):
                            flow_name = flow.get('name', 'Unnamed flow')
                            flow_desc = flow.get('description', 'No description')
                            flow_type = flow.get('flow_type', 'Unknown type')
                            flow_steps = flow.get('steps', [])
                            flow_functions = flow.get('functions', [])
                            flow_actors = flow.get('actors', [])
                        else:
                            # BusinessFlow object - use attribute access instead of dict access
                            flow_name = getattr(flow, 'name', 'Unnamed flow')
                            flow_desc = getattr(flow, 'description', 'No description')
                            flow_type = getattr(flow, 'flow_type', 'Unknown type')
                            flow_steps = getattr(flow, 'steps', [])
                            flow_functions = getattr(flow, 'functions', [])
                            flow_actors = getattr(flow, 'actors', [])
                        
                        report.append(f"#### {flow_name}\n")
                        report.append(f"**Type**: {flow_type}\n")
                        report.append(f"**Description**: {flow_desc}\n")
                        
                        if flow_steps:
                            report.append("**Steps**:\n")
                            for step in flow_steps:
                                report.append(f"- {step}")
                            report.append("")
                        
                        if flow_functions:
                            report.append("**Functions**:\n")
                            for func in flow_functions:
                                report.append(f"- `{func}`")
                            report.append("")
                            
                        if flow_actors:
                            report.append("**Actors**:\n")
                            for actor in flow_actors:
                                report.append(f"- {actor}")
                            report.append("")
                else:
                    report.append("No business flows identified.\n")
        else:
            report.append("Business flow data is available but in an unexpected format.\n")
    
    # Add data flow analysis
    if hasattr(context, 'dataflows') and context.dataflows:
        report.append("## Data Flow Analysis\n")
        dataflows = context.dataflows
        if isinstance(dataflows, dict):
            for contract_name, contract_dataflows in dataflows.items():
                report.append(f"### {contract_name}\n")
                if contract_dataflows:
                    for dataflow in contract_dataflows:
                        source = dataflow.get('source', 'Unknown')
                        target = dataflow.get('target', 'Unknown')
                        data_type = dataflow.get('type', 'Unknown')
                        impact = dataflow.get('impact', 'Unknown')
                        
                        report.append(f"- **{source}** → **{target}**")
                        report.append(f"  - Type: {data_type}")
                        report.append(f"  - Impact: {impact}")
                        report.append("")
                else:
                    report.append("No data flows identified.\n")
        else:
            report.append("Data flow information is available but in an unexpected format.\n")
    
    # Add cognitive bias analysis
    if hasattr(context, 'cognitive_biases') and context.cognitive_biases:
        report.append("## Cognitive Bias Analysis\n")
        biases = context.cognitive_biases
        if isinstance(biases, dict):
            for contract_name, contract_biases in biases.items():
                report.append(f"### {contract_name}\n")
                if contract_biases:
                    for bias in contract_biases:
                        bias_type = bias.get('type', 'Unknown bias')
                        description = bias.get('description', 'No description')
                        impact = bias.get('impact', 'Unknown impact')
                        
                        report.append(f"#### {bias_type}")
                        report.append(f"- **Description**: {description}")
                        report.append(f"- **Impact**: {impact}")
                        report.append("")
                else:
                    report.append("No cognitive biases detected.\n")
        else:
            report.append("Cognitive bias data is available but in an unexpected format.\n")
    
    # Add FLARE query results
    if queries_results:
        report.append("## FLARE Query Analysis\n")
        for query_id, result in queries_results.items():
            if query_id.startswith("query_"):
                query = result.get('query', 'Unknown query')
                response = result.get('response', 'No response')
                confidence = result.get('confidence', 0.0)
                
                report.append(f"### Query: {query}")
                report.append(f"**Confidence**: {confidence:.2f}\n")
                report.append(f"{response}\n")
    
    # Add recommendations section
    report.append("## Recommendations\n")
    report.append("Based on the analysis, consider the following recommendations:\n")
    
    # Try to extract recommendations from various analysis sections
    recommendations = []
    
    # From vulnerabilities
    if hasattr(context, 'vulnerabilities') and context.vulnerabilities:
        for contract_name, contract_vulns in context.vulnerabilities.items():
            if isinstance(contract_vulns, list):
                for vuln in contract_vulns:
                    if vuln.get('recommendation'):
                        recommendations.append(
                            f"- Fix {vuln.get('name', 'vulnerability')} in {contract_name}: {vuln.get('recommendation')}"
                        )
    
    # From FLARE queries
    if queries_results:
        for query_id, result in queries_results.items():
            if query_id.startswith("query_") and "recommendation" in result.get('response', '').lower():
                # Extract recommendations from the response (simplified approach)
                response_lines = result.get('response', '').split('\n')
                for line in response_lines:
                    if "recommend" in line.lower():
                        recommendations.append(f"- {line.strip()}")
    
    # Add recommendations to report
    if recommendations:
        for rec in recommendations:
            report.append(rec)
    else:
        report.append("- No specific recommendations available from the analysis")
    
    # Add debugging notes
    report.append("\n## Debug Information\n")
    report.append("This report was generated as part of the debugging process.")
    report.append(f"- Pipeline executed at: {now}")
    report.append(f"- Input directory: {EXAMPLES_DIR}")
    report.append(f"- Output directory: {OUTPUT_DIR}")
    
    return "\n".join(report)


async def find_files_by_extension(directory: str, extension: str = '.sol') -> List[Dict[str, Any]]:
    """
    Asynchronously enumerate all files with the specified extension in the given directory.
    
    Args:
        directory: The directory path to search
        extension: File extension to filter by (default: '.sol')
        
    Returns:
        List of dictionaries containing file information
    """
    logger.info(f"Finding files with extension {extension} in {directory}")
    found_files = []
    directory_path = Path(directory)
    
    # Get list of all potential files (synchronously to avoid excessive async overhead for simple listing)
    potential_files = list(directory_path.rglob(f"*{extension}"))
    logger.debug(f"Found {len(potential_files)} potential {extension} files")
    

    # Call graph is a bit wonky at the moment, were going to lean on TreeSitter a bit more due to this 
    chunker = AsyncContractChunker(include_call_graph=False)
    if chunker.include_call_graph:
        await chunker.initialize_call_graph(directory)

    # Process files concurrently
    async def process_file(file_path):
        try:
            # Get file stats
            stats = await aio_os.stat(file_path)
            
            # Read file content asynchronously
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = await f.read()
            
                chunks = await chunker.chunk_code(content, file_path.name, file_path)


            # Create file info dictionary
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "content": content,


                "size": stats.st_size,
                "modified_time": stats.st_mtime,
                "is_solidity": extension.lower() == '.sol'
            }
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "error": str(e),
                "is_solidity": extension.lower() == '.sol'
            }
    
    # Create and gather tasks
    tasks = [process_file(file_path) for file_path in potential_files]
    results = await asyncio.gather(*tasks)
    
    # Filter out any None results
    found_files = [result for result in results if result]
    logger.info(f"Successfully processed {len(found_files)} {extension} files")
    
    return found_files


async def create_combined_document_processing_stage(factory: PipelineFactory) -> Stage:
    """
    Create a combined stage that handles document loading, contract extraction, 
    and code chunking in a single pass to improve cache efficiency.
    
    Args:
        factory: The PipelineFactory instance
        
    Returns:
        A stage that performs all document processing operations
    """
    logger.info("Creating combined document processing stage")
    
    # Get the chunker instance ready for use inside the combined stage
    chunker = AsyncContractChunker(
        max_chunk_size=500,
        overlap_size=100,
        preserve_imports=True,
        chunk_by_contract=True,
        chunk_by_function=True,
        include_call_graph=False
    )
    
    async def combined_document_processor(context: Context) -> Context:
        """Process documents from loading through chunking in a single stage"""
        # Step 1: Asynchronously load and parse files
        logger.info("Loading documents from input path")
        
        if not hasattr(context, 'input_path') or not context.input_path:
            logger.warning("No input path specified")
            return context
        
        # Use async file finder instead of the previous document loading approach
        solidity_files = await find_files_by_extension(context.input_path, '.sol')
        logger.info(f"Found {len(solidity_files)} Solidity files")
        
        # Initialize containers if they don't exist
        if not hasattr(context, 'files'):
            context.files = {}
        #if not hasattr(context, 'contracts'):
        context.contracts = []
        #if not hasattr(context, 'functions'):
        context.functions = []
        if not hasattr(context, 'chunks'):
            context.chunks = {}
        
        # Store files in context
        for file_info in solidity_files:
            file_id = file_info["file_path"]
            context.files[file_id] = file_info
        
        # Step 2: Extract contracts from files and chunk them
        logger.info("Extracting and chunking contracts from files")
        
        # Process each file to extract contracts using the chunker
        for file_id, file_info in context.files.items():
            try:
                # Get file content and metadata
                content = file_info.get("content", "")
                file_name = file_info.get("file_name", os.path.basename(file_id))
                
                # Skip files with errors
                if "error" in file_info:
                    logger.warning(f"Skipping file with error: {file_id}, Error: {file_info['error']}")
                    continue
                
                # Process file with chunker to get contract information - always returns a dictionary now
                chunk_result = await chunker.chunk_code(content, file_name, file_id)
                
                # Store chunk result with the file
                file_info["chunk_data"] = chunk_result
                
                # Add the file-level chunk to global chunks
                file_chunk_id = chunk_result.get("chunk_id", f"{file_id}:file")
                context.chunks[file_chunk_id] = chunk_result
                
                # Process all contracts from the chunk result
                if "contracts" in chunk_result and isinstance(chunk_result["contracts"], list):
                    for contract in chunk_result["contracts"]:
                        # Add file path information to contract if not already present
                        if not contract.get("file_path"):
                            contract["file_path"] = file_id
                        if not contract.get("file_name"):
                            contract["file_name"] = file_name
                        
                        # Add to global contracts list
                        context.contracts.append(contract)
                        
                        # Add contract chunk to global chunks
                        contract_chunk_id = contract.get("chunk_id", f"{file_id}:{contract.get('name', 'unknown')}")
                        context.chunks[contract_chunk_id] = contract
                        
                        # Process functions if they exist
                        if "functions" in contract and isinstance(contract["functions"], list):
                            for func in contract["functions"]:
                                # Add function to functions dictionary
                                func_id = func.get("full_name", f"{contract.get('name', 'unknown')}_{func.get('name', 'unknown')}")
                                context.functions.append(func)
                                
                                # Add function chunk to global chunks
                                func_chunk_id = func.get("chunk_id", f"{file_id}:{func_id}")
                                context.chunks[func_chunk_id] = func
                
                # Log extraction details
                contract_count = len([c for c in context.contracts if c.get("file_path") == file_id])
                function_count = len([f for f in context.functions if f.get("source_file") == file_id])
                
                logger.debug(f"Processed file {file_name}: extracted {contract_count} contracts and {function_count} functions")
                
            except Exception as e:
                logger.error(f"Error processing file {file_id}: {str(e)}")
                logger.error(traceback.format_exc())
                context.add_error(
                    stage="combined_document_processing",
                    message=f"Failed to process file: {file_id}",
                    exception=e
                )
        
        # Final counts
        logger.info(f"Total extracted: {len(context.contracts)} contracts, {len(context.functions)} functions, {len(context.chunks)} chunks")
        
        return context
    
    return combined_document_processor


async def setup_debug_pipeline(factory: PipelineFactory, input_path: str) -> tuple:
    """
    Set up a debug pipeline with all necessary stages using the pipeline factory.
    
    Args:
        factory: The PipelineFactory instance
        input_path: Path to input files for analysis
        
    Returns:
        Tuple of (context, pipeline) ready for execution
    """
    logger.info(f"Setting up debug pipeline for input: {input_path}")
    
    # Create the initial context
    context = Context(input_path=input_path)
    
    # Create combined document processing stage (load, extract, chunk)
    combined_document_stage = await create_combined_document_processing_stage(factory)
    
    # Get business flow extraction stage
    business_flow_stage = await factory.create_business_flow_extractor(context)
    
    # Get vulnerability scanning stage
    vulnerability_stage = await factory.create_vulnerability_scanner(context)
    
    # Get data flow analysis stage
    dataflow_stage = await factory.create_dataflow_analyzer(context)
    
    # Get cognitive bias analysis stage
    cognitive_bias_stage = await factory.create_cognitive_bias_analyzer(context)
    
    # Get documentation analysis stage
    documentation_stage = await factory.create_documentation_analyzer(context)
    
    # Create flow joiner stage
    flow_joiner_stage = FlowJoiner()
    
    # Wrap all stages with debug wrapper
    pipeline_stages = [
        DebugStage(combined_document_stage, "Combined Document Processing"),
        DebugStage(business_flow_stage, "Business Flow Analysis"),
        DebugStage(vulnerability_stage, "Vulnerability Scanning"),
        DebugStage(dataflow_stage, "Data Flow Analysis"),
        DebugStage(cognitive_bias_stage, "Cognitive Bias Analysis"),
        DebugStage(documentation_stage, "Documentation Analysis"),
        DebugStage(flow_joiner_stage, "Result Aggregation")
    ]
    
    # Create the pipeline
    pipeline = Pipeline(stages=pipeline_stages)
    
    logger.info(f"Created debug pipeline with {len(pipeline_stages)} stages")
    return context, pipeline


async def main():
    """Main function to run the complete pipeline for debugging"""
    logger.info("Starting debug pipeline")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Configure input path
    if not os.path.exists(EXAMPLES_DIR):
        logger.error(f"Examples directory not found: {EXAMPLES_DIR}")
        return
    
    # Initialize factory and create pipeline stages
    try:
        logger.info("Initializing pipeline factory")
        factory = PipelineFactory()
        
        # Set up the debug pipeline directly using factory
        context, pipeline = await setup_debug_pipeline(factory, EXAMPLES_DIR)
        
        # Run the pipeline directly
        logger.info("Executing pipeline")
        try:
            # Execute the pipeline using the run method instead of calling it directly
            context = await pipeline.run(context)  # Changed from await pipeline(context)
            logger.info("Pipeline execution completed successfully")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            logger.error(traceback.format_exc())
            # Continue to query engine to test as much as possible
        
        # Get query engine
        logger.info("Initializing FLARE query engine")
        try:
            # Initialize query engine
            query_engine = factory.get_query_engine()
            context.query_engine = query_engine
            
            # Execute queries for testing
            logger.info("Executing test queries using FLARE engine")
            queries_results = {}
            
            for i, query in enumerate(DEBUG_QUERIES):
                logger.info(f"Executing query: {query}")
                query_id = f"query_{i}"
                try:
                    result = await query_engine.query(query, context)
                    queries_results[query_id] = {
                        "query": query,
                        "response": result.response,
                        "confidence": result.confidence,
                        "sources_count": len(result.sources)
                    }
                    logger.info(f"Query completed: {query_id}")
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    queries_results[query_id] = {
                        "query": query,
                        "response": f"Error: {str(e)}",
                        "confidence": 0.0,
                        "sources_count": 0
                    }
        except Exception as e:
            logger.error(f"Query engine initialization failed: {e}")
            logger.error(traceback.format_exc())
            queries_results = {}
        
        # Generate markdown report
        logger.info("Generating markdown report")
        report_content = await generate_markdown_report(context, queries_results)
        
        # Save report
        with open(REPORT_PATH, 'w') as f:
            f.write(report_content)
        
        # Also save to analyzed.md for easier access
        with open(ANALYSIS_PATH, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {REPORT_PATH}")
        logger.info(f"Analysis saved to: {ANALYSIS_PATH}")
        logger.info("Debug pipeline completed successfully")
        
        print(f"\nDebug pipeline execution completed.")
        print(f"Report generated at: {REPORT_PATH}")
        print(f"Analysis summary saved to: {ANALYSIS_PATH}")
        print("Check the log file for detailed execution information.")
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())