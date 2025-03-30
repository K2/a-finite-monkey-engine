#!/usr/bin/env python3
"""
Smart contract analyzer entry point script using the FLARE query engine.

This script provides a command-line interface for analyzing smart contracts
using the Finite Monkey Engine with advanced reasoning capabilities.
"""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from finite_monkey.pipeline.factory import PipelineFactory
from finite_monkey.pipeline.core import Context, Pipeline, Stage
from finite_monkey.nodes_config import config
from finite_monkey.query_engine.flare_engine import FlareQueryEngine
from finite_monkey.query_engine.script_adapter import QueryEngineScriptAdapter, ScriptGenerationRequest

DEFAULT_QUERIES = [
    "What are the main security vulnerabilities in this contract?",
    "How could the gas efficiency be improved?",
    "Are there any business logic flaws in the contract?",
    "What are the key roles and permissions in this contract?",
    "Are there any reentrancy vulnerabilities?"
]

async def setup_pipeline(input_path: str, factory: PipelineFactory) -> Dict[str, Any]:
    """
    Set up the analysis pipeline and run document loading stages.
    
    Args:
        input_path: Path to the smart contract or project to analyze
        factory: The pipeline factory instance
        
    Returns:
        Dictionary with context and pipeline
    """
    # Create context with input path
    context = Context(input_path=input_path)
    
    # Define pipeline stages
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
    
    # Set up pipeline and get context
    setup_result = await setup_pipeline(input_path, factory)
    context = setup_result["context"]
    pipeline = setup_result["pipeline"]
    
    # Run the pipeline to load documents and contracts
    logger.info("Running pipeline to load documents and contracts")
    for stage in pipeline.stages:
        stage_name = getattr(stage, "name", str(stage))
        logger.info(f"Executing stage: {stage_name}")
        context = await stage(context)
    
    # Check if contracts were loaded
    if not hasattr(context, 'contracts') or not context.contracts:
        logger.error("No contracts found in the specified path")
        return {"error": "No contracts found"}
    
    logger.info(f"Found {len(context.contracts)} contracts")
    
    # Use default queries if none provided
    if not queries:
        queries = DEFAULT_QUERIES
    
    # Store queries in context
    context.queries = {f"query_{i}": query for i, query in enumerate(queries)}
    
    # Execute queries
    results = {}
    try:
        for query_id, query_text in context.queries.items():
            logger.info(f"Executing query: {query_text}")
            result = await context.query_engine.query(query_text, context)
            results[query_id] = {
                "query": query_text,
                "response": result.response,
                "confidence": result.confidence,
                "sources_count": len(result.sources)
            }
            logger.success(f"Query completed: {query_id}")
    except Exception as e:
        logger.error(f"Error executing queries: {e}")
        results["error"] = str(e)
    
    # Generate scripts if requested
    if generate_scripts:
        try:
            logger.info("Generating analysis scripts")
            script_adapter = QueryEngineScriptAdapter(
                query_engine=context.query_engine,
                script_output_dir=os.path.join(os.getcwd(), "generated_scripts")
            )
            
            # Add script adapter to context
            context.script_adapter = script_adapter
            
            # Store script results
            results["generated_scripts"] = {}
            
            # Generate a script for each contract
            for i, contract in enumerate(context.contracts):
                # Extract contract information
                contract_name = getattr(contract, 'name', f"Contract_{i}")
                contract_code = getattr(contract, 'code', getattr(contract, 'content', ''))
                contract_path = getattr(contract, 'file_path', getattr(contract, 'path', input_path))
                
                # Create script generation request
                request = ScriptGenerationRequest(
                    query=f"Create a comprehensive security analysis script for the {contract_name} contract",
                    context_snippets=[contract_code],
                    file_paths=[contract_path],
                    script_type="analysis"
                )
                
                # Generate the script
                script_result = await script_adapter.generate_script(request, context)
                
                # Store the result
                results["generated_scripts"][contract_name] = {
                    "success": script_result.success,
                    "script_path": script_result.script_path,
                    "execution_command": script_result.execution_command
                }
                
                logger.success(f"Generated script for {contract_name}")
        except Exception as e:
            logger.error(f"Error generating scripts: {e}")
            results["script_error"] = str(e)
    
    # Calculate summary metrics
    results["summary"] = {
        "contracts_analyzed": len(context.contracts),
        "queries_executed": len(results) - (1 if "error" in results else 0) - (1 if "generated_scripts" in results else 0) - (1 if "summary" in results else 0),
        "scripts_generated": len(results.get("generated_scripts", {})),
        "average_confidence": sum(r.get("confidence", 0) for r in results.values() if isinstance(r, dict) and "confidence" in r) / max(1, len([r for r in results.values() if isinstance(r, dict) and "confidence" in r]))
    }
    
    # Save results if output path provided
    if output_path:
        try:
            logger.info(f"Saving analysis results to {output_path}")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.success(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            results["save_error"] = str(e)
    
    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Smart Contract Analyzer - Analyze smart contracts using the FLARE query engine"
    )
    parser.add_argument(
        "input_path",
        help="Path to the smart contract or project to analyze"
    )
    parser.add_argument(
        "-q", "--query",
        help="Custom query to execute (can be specified multiple times)",
        action="append"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save analysis results",
        default=None
    )
    parser.add_argument(
        "-s", "--generate-scripts",
        help="Generate analysis scripts",
        action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Enable verbose logging",
        action="store_true"
    )
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    logger.info("Starting Smart Contract Analyzer")
    
    # Run the analysis
    results = await analyze_contract(
        input_path=args.input_path,
        queries=args.query,
        output_path=args.output,
        generate_scripts=args.generate_scripts,
        verbose=args.verbose
    )
    
    # Print a summary
    print("\n=== ANALYSIS SUMMARY ===")
    if "summary" in results:
        summary = results["summary"]
        print(f"Contracts analyzed: {summary['contracts_analyzed']}")
        print(f"Queries executed: {summary['queries_executed']}")
        print(f"Average confidence: {summary['average_confidence']:.2f}")
        if summary.get('scripts_generated', 0) > 0:
            print(f"Scripts generated: {summary['scripts_generated']}")
    
    print("\n=== QUERY RESULTS ===")
    for query_id, result in results.items():
        if query_id.startswith("query_"):
            print(f"\nQuery: {result['query']}")
            print(f"Confidence: {result['confidence']}")
            # Print first 200 chars of response with ellipsis if needed
            response_preview = result['response'][:200]
            if len(result['response']) > 200:
                response_preview += "..."
            print(f"Response: {response_preview}")
    
    if "generated_scripts" in results:
        print("\n=== GENERATED SCRIPTS ===")
        for contract_name, script_result in results["generated_scripts"].items():
            if script_result["success"]:
                print(f"- {contract_name}: {script_result['script_path']}")
                print(f"  Execute with: {script_result['execution_command']}")
    
    if "error" in results:
        print(f"\nError: {results['error']}")
    
    logger.info("Analysis complete")