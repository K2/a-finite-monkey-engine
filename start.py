nb import aiofiles
import yappi
import os
from configparser import ConfigParser, SectionProxy
from typing import Any, Type, List, Dict
from pathlib import Path
import asyncio
import tempfile
import shutil
import ghapi
import argparse
import glob
import sys
import subprocess
from loguru import logger
from ollama import embed

# Import configuration from nodes_config
from finite_monkey.nodes_config import config

from llama_index.llms.ollama import Ollama
from finite_monkey.test_chunking import test_chunking, test_chunking_in_context
from finite_monkey.utils.chunking import AsyncContractChunker
from finite_monkey.utils.project_loader import AsyncProjectLoader
import json
from datetime import datetime

# Set up logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Current llama_index namespace structure - updated to the latest version
from llama_index.core.settings import Settings
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from finite_monkey import nodes_config
from finite_monkey.pipeline.factory import PipelineFactory
from finite_monkey.pipeline.core import Context
from finite_monkey.utils.async_doc_loader import AsyncDocumentLoader

# Import additional analyzers for pipeline registration
from finite_monkey.analyzers.dataflow_analyzer import DataFlowAnalyzer
from finite_monkey.analyzers.cognitive_bias_analyzer import CognitiveBiasAnalyzer  
from finite_monkey.analyzers.counterfactual_analyzer import CounterfactualAnalyzer
from finite_monkey.analyzers.documentation_analyzer import DocumentationAnalyzer
from finite_monkey.adapters.agent_adapter import DocumentationInconsistencyAdapter

from finite_monkey.config.loader import ConfigLoader

def setup_llama_service():
    """Initialize LlamaIndex settings with appropriate LLM and embedding models"""
    try:
        # Configure Ollama LLM using configuration instead of hardcoded values
        Settings.llm = Ollama(
            model=config.WORKFLOW_MODEL,  # Use configured model instead of hardcoded value
            temperature=0.1,
            request_timeout=config.REQUEST_TIMEOUT,  # Use configured timeout
        )
        
        # Configure embedding model
        try:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=config.EMBEDDING_MODEL  # Use configured embedding model
            )
        except Exception as embed_error:
            logger.warning(f"Could not initialize embedding model: {embed_error}")
        
        logger.info(f"Successfully configured LlamaIndex settings with model {config.WORKFLOW_MODEL}")
        
    except Exception as e:
        logger.error(f"Error configuring LlamaIndex settings: {str(e)}")
        # No need to return anything, Settings is globally configured

# Custom directory reader to replace SimpleDirectoryReader
async def read_files_from_directory(directory_path, pattern="**/*.sol"):
    """Read files from a directory asynchronously"""
    logger.info(f"Reading files from directory: {directory_path} with pattern {pattern}")
    
    # Convert to Path object
    directory = Path(directory_path)
    if not await asyncio.to_thread(directory.exists):
        raise ValueError(f"Directory {directory_path} does not exist")
    
    documents = []
    
    # Get matching files using thread pool
    matching_files = await asyncio.to_thread(
        lambda: list(directory.glob(pattern))
    )
    
    # Process files concurrently with semaphore to limit open files
    semaphore = asyncio.Semaphore(20)  # Limit to 20 concurrent file operations
    
    async def process_file(file_path):
        async with semaphore:
            if not await asyncio.to_thread(file_path.is_file):
                return None
                
            try:
                # Read file content asynchronously
                async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = await f.read()
                
                # Create document with metadata
                return Document(
                    text=content,
                    metadata={
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix,
                        "is_solidity": file_path.suffix.lower() == ".sol"
                    }
                )
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                return None
    
    # Process files in batches to limit memory usage
    batch_size = 50  # Process 50 files at a time
    results = []
    
    for i in range(0, len(matching_files), batch_size):
        batch = matching_files[i:i+batch_size]
        batch_results = await asyncio.gather(*[process_file(f) for f in batch])
        results.extend([doc for doc in batch_results if doc is not None])
        
        # Small delay between batches
        if i + batch_size < len(matching_files):
            await asyncio.sleep(0.1)
    
    # Count Solidity files
    solidity_count = sum(1 for doc in results if doc.metadata.get("is_solidity"))
    
    logger.info(f"Loaded {len(results)} documents from {directory_path} ({solidity_count} Solidity files)")
    return results

async def load_files_from_directory(directory_path):
    """Load files from a local directory asynchronously"""
    return await read_files_from_directory(directory_path)

async def load_files_from_github(repo_url, branch="main", subdirectory=None):
    """Clone a GitHub repository and load its files asynchronously"""
    logger.info(f"Loading files from GitHub repo: {repo_url} (branch: {branch})")
    temp_dir = tempfile.mkdtemp()
    try:
        # Clone the repository to a temporary directory
        logger.info(f"Cloning repository to {temp_dir}...")
        
        # Run git clone in subprocess asynchronously
        process = await asyncio.create_subprocess_exec(
            "git", "clone", "--branch", branch, "--single-branch", "--depth=1", 
            repo_url, temp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for process to complete
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise ValueError(f"Failed to clone repository: {error_msg}")
        
        # Determine the directory to read from
        read_dir = temp_dir
        if subdirectory:
            read_dir = os.path.join(temp_dir, subdirectory)
            if not await asyncio.to_thread(os.path.exists, read_dir):
                raise ValueError(f"Subdirectory '{subdirectory}' not found in repository")
        
        # Load documents from the directory
        documents = await load_files_from_directory(read_dir)
        return documents
    except Exception as e:
        logger.error(f"Error loading files from GitHub: {str(e)}")
        raise
    finally:
        # Clean up the temporary directory
        logger.info(f"Cleaning up temporary directory {temp_dir}")
        await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)

from finite_monkey.config.model_provider import ModelProvider

async def run_pipeline(
    input_path: str,
    output_file: str,
    is_github: bool = False,
    branch: str = "main",
    subdirectory: str = "/src",
    config=None
):
    """
    Run analysis pipeline on input files
    
    Args:
        input_path: Path to input directory or GitHub repository URL
        output_file: Path to output file
        is_github: Whether the input is a GitHub repository
        branch: Branch to use for GitHub repositories
        subdirectory: Subdirectory to focus on in GitHub repositories
        config: Optional configuration
        
    Returns:
        Tuple of (analysis result, document count)
    """
    logger.info(f"Loading documents from {'GitHub repository' if is_github else 'local directory'}: {input_path}")
    
    # Create document loader
    loader = AsyncDocumentLoader(max_workers=5)
    
    # Load documents
    documents = []
    try:
        if is_github:
            logger.info(f"Loading from GitHub: {input_path}, branch: {branch}, subdirectory: {subdirectory}")
            documents = await loader.load_from_github(
                input_path,
                branch=branch,
                subdirectory=subdirectory
            )
        else:
            logger.info(f"Loading from directory: {input_path}")
            documents = await loader.load_from_directory(input_path)
        
        doc_count = len(documents)
        logger.info(f"Loaded {doc_count} documents")
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        return None, 0
    
    # Create initial context
    context = Context()
    
    # Create the pipeline factory
    factory = PipelineFactory()
    
    # Create and run pipeline with more detailed logging
    logger.info("Creating analysis pipeline...")
    try:
        pipeline = factory.create_standard_pipeline(
            documents=documents,
            output_path=output_file,
            config={
                'chunk_size': 1000,
                'overlap': 100
            }
        )
        
        logger.info(f"Running pipeline with {len(pipeline.stages) if hasattr(pipeline, 'stages') else 'unknown'} stages")
        result = await pipeline.run(context)
        
        # Log detailed results summary
        logger.info("Pipeline execution completed")
        if hasattr(result, 'completed_stages'):
            logger.info(f"Completed stages: {', '.join(result.completed_stages)}")
        
        # Check for errors
        if hasattr(result, 'errors') and result.errors:
            logger.warning(f"Pipeline completed with {len(result.errors)} errors")
        
        # Save results to output file
        try:
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save detailed results for debugging
            debug_output = f"{os.path.splitext(output_file)[0]}_debug.json"
            with open(debug_output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Saved debug output to {debug_output}")
        except Exception as e:
            logger.error(f"Error saving debug output: {str(e)}")
        
        return result, doc_count
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, doc_count

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Finite Monkey Engine - Smart Contract Analysis Tool')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Main pipeline parser
    pipeline_parser = subparsers.add_parser('analyze', help='Run the analysis pipeline')
    pipeline_parser.add_argument('--input', type=str, required=True,
                        help='Input directory path or GitHub repository URL')
    pipeline_parser.add_argument('--output', type=str, default="output/analysis.html",
                        help='Output file path (default: output/analysis.html)')
    pipeline_parser.add_argument('--github', action='store_true',
                        help='Specify if the input is a GitHub repository URL')
    pipeline_parser.add_argument('--branch', type=str, default="main",
                        help='Branch to use for GitHub repositories (default: main)')
    pipeline_parser.add_argument('--subdirectory', type=str, default='/src',
                        help='Subdirectory within the GitHub repository to process')
    
    # Test chunk parser
    chunk_parser = subparsers.add_parser('test-chunk', help='Test chunking functionality on a single file')
    chunk_parser.add_argument('file', type=str, help='File to chunk')
    
    # Test chunk context parser
    chunk_context_parser = subparsers.add_parser('test-chunk-context', help='Test chunking with context integration')
    chunk_context_parser.add_argument('file', type=str, help='File to chunk')
    
    # Test chunking parser
    chunking_parser = subparsers.add_parser('test-chunking', help='Test the chunking functionality on a Solidity file')
    chunking_parser.add_argument('file', type=str, help='File to chunk')
    chunking_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    # Chunk directory parser
    chunk_dir_parser = subparsers.add_parser('chunk-directory', help='Chunk all Solidity files in a directory')
    chunk_dir_parser.add_argument('directory', type=str, help='Directory containing files to chunk')
    chunk_dir_parser.add_argument('--output', '-o', type=str, default="./output", help='Output directory for results')
    chunk_dir_parser.add_argument('--max-concurrent', '-m', type=int, default=5, help='Maximum concurrent files')
    chunk_dir_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    # Process project parser
    process_parser = subparsers.add_parser('process-project', help='Process a project directory with loading and chunking')
    process_parser.add_argument('directory', type=str, help='Directory to process')
    process_parser.add_argument('--output', '-o', type=str, default="./output", help='Output directory for results')
    process_parser.add_argument('--max-concurrent', '-m', type=int, default=5, help='Maximum concurrent files')
    process_parser.add_argument('--chunk-size', '-c', type=int, default=8000, help='Maximum chunk size')
    process_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    # If no arguments are provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()

# Replace typer commands with regular functions
def cmd_test_chunk(file: str):
    """Test chunking functionality on a single file"""
    try:
        result = asyncio.run(test_chunking(file))
        print(f"Chunking test completed successfully with {len(result)} chunks")
        return 0
    except Exception as e:
        print(f"Chunking test failed: {e}", file=sys.stderr)
        return 1

def cmd_test_chunk_context(file: str):
    """Test chunking with context integration on a single file"""
    try:
        result = asyncio.run(test_chunking_in_context(file))
        print(f"Context integration test completed successfully")
        print(f"Context contains {len(result.chunks)} chunks")
        return 0
    except Exception as e:
        print(f"Context integration test failed: {e}", file=sys.stderr)
        return 1

def cmd_test_chunking(file: str, verbose: bool = False):
    """Test the chunking functionality on a Solidity file"""
    # Set up logging
    if verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    
    # Validate file path
    file_path = Path(file)
    if not file_path.exists():
        print(f"Error: File {file} does not exist", file=sys.stderr)
        return 1
    
    if not file_path.is_file():
        print(f"Error: {file} is not a file", file=sys.stderr)
        return 1
        
    if not str(file_path).endswith(".sol"):
        print(f"Warning: {file} does not appear to be a Solidity file", file=sys.stderr)
    
    # Run chunking
    print(f"Chunking file: {file}")
    try:
        async def run_test():
            # Create chunker
            chunker = AsyncContractChunker(
                max_chunk_size=8000,
                include_call_graph=False  # Skip call graph for simple test
            )
            
            # Chunk the file
            chunks = await chunker.chunk_file(str(file_path))
            
            # Display results
            print(f"Successfully chunked into {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks):
                chunk_type = chunk.get("chunk_type", "unknown")
                content_len = len(chunk.get("content", ""))
                print(f"  {i+1}. {chunk_type} - {content_len} chars")
            
            return chunks
        
        chunks = asyncio.run(run_test())
        print(f"Chunking complete with {len(chunks)} chunks")
        return 0
        
    except Exception as e:
        print(f"Error during chunking: {e}", file=sys.stderr)
        return 1

def cmd_chunk_directory(directory: str, output: str = "./output", max_concurrent: int = 5, verbose: bool = False):
    """Chunk all Solidity files in a directory structure"""
    # Set up logging
    if verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    else:
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")
    
    # Validate directory path
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        return 1
    
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run chunking
    print(f"Chunking directory: {directory}")
    
    async def run_directory_chunking():
        try:
            # Create chunker with appropriate settings
            chunker = AsyncContractChunker(
                max_chunk_size=8000,
                include_call_graph=True
            )
            
            # Process entire directory
            results = await chunker.chunk_project(str(dir_path))
            
            # Write results to output files
            if results:
                file_count = len(results)
                chunk_count = sum(len(chunks) for chunks in results.values())
                
                # Create summary file
                summary = {
                    "processed_at": datetime.now().isoformat(),
                    "directory": str(dir_path),
                    "file_count": file_count,
                    "chunk_count": chunk_count,
                    "files": {os.path.relpath(fp, str(dir_path)): len(chunks) 
                              for fp, chunks in results.items()}
                }
                
                with open(output_path / "chunking_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                
                # Create detailed results file - just first 100 chars of content to keep it manageable
                detailed = {os.path.relpath(fp, str(dir_path)): 
                           [{**c, "content": c["content"][:100] + "..."} for c in chunks]
                           for fp, chunks in results.items()}
                
                with open(output_path / "chunking_details.json", "w") as f:
                    json.dump(detailed, f, indent=2)
                
                return file_count, chunk_count
            else:
                logger.warning("No files were processed")
                return 0, 0
        except Exception as e:
            logger.error(f"Directory chunking failed: {e}")
            raise
    
    try:
        # Run the chunking asynchronously
        file_count, chunk_count = asyncio.run(run_directory_chunking())
        
        if file_count > 0:
            print(f"Successfully chunked {file_count} files into {chunk_count} chunks")
            print(f"Results saved to {output_path}")
        else:
            print("No Solidity files were processed")
        return 0
    except Exception as e:
        print(f"Error during directory chunking: {e}", file=sys.stderr)
        return 1

def cmd_process_project(directory: str, output: str = "./output", max_concurrent: int = 5, 
                    chunk_size: int = 8000, verbose: bool = False):
    """Process a project directory with loading and chunking"""
    # Set up logging
    if verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    else:
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")
    
    # Validate directory path
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory {directory} does not exist", file=sys.stderr)
        return 1
    
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run processing
    print(f"Processing project: {directory}")
    start_time = datetime.now()
    
    try:
        # Create project loader
        loader = AsyncProjectLoader(max_concurrency=max_concurrent)
        
        # Run loading and chunking
        context = asyncio.run(loader.load_and_chunk_project(
            project_path=str(dir_path),
            chunk_size=chunk_size
        ))
        
        # Print results
        file_count = len(context.files)
        chunk_count = len(context.chunks)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Successfully processed {file_count} files into {chunk_count} chunks")
        print(f"Processing took {duration:.2f} seconds")
        
        # Save context summary to output file
        summary = context.to_dict()
        with open(output_path / "project_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"Results saved to {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error during project processing: {e}", file=sys.stderr)
        return 1

def run_analyze(args):
    """Run the analysis pipeline based on args"""
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_path = args.config if hasattr(args, "config") else None
    config = ConfigLoader.load(config_path)

    # Create pipeline factory with configuration
    factory = PipelineFactory()

    # Run the pipeline
    logger.info(f"Starting analysis pipeline...")
    result, doc_count = asyncio.run(run_pipeline(
        args.input,
        args.output,
        is_github=args.github,
        branch=args.branch,
        subdirectory=args.subdirectory,
        config=config  # Pass the config to run_pipeline
    ))

    # Print results summary
    logger.info(f"Analysis completed. Processed {doc_count} documents.")
    logger.info(f"Results saved to: {args.output}")
    if hasattr(result, 'metrics'):
        logger.info("\nMetrics:")
        for key, value in result.metrics.items():
            logger.info(f"  {key}: {value}")
    return 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Finite Monkey Engine")
    subparsers = parser.add_subparsers(dest="command")
    
    # Main pipeline parser
    analyze_parser = subparsers.add_parser("analyze", help="Run the analysis pipeline")
    analyze_parser.add_argument("--input", type=str, required=True, help="Input directory path or GitHub repository URL")
    analyze_parser.add_argument("--output", type=str, default="output/analysis.html", help="Output file path")
    analyze_parser.add_argument("--github", action="store_true", help="Specify if the input is a GitHub repository URL")
    analyze_parser.add_argument("--branch", type=str, default="main", help="Branch to use for GitHub repositories")
    analyze_parser.add_argument("--subdirectory", type=str, default="/src", help="Subdirectory within the GitHub repository to process")
    
    # Test chunk parser
    test_parser = subparsers.add_parser("test-chunk", help="Test chunking functionality on a single file")
    test_parser.add_argument("file", type=str, help="File to chunk")
    
    # Test chunk context parser
    chunk_parser = subparsers.add_parser("test-chunk-context", help="Test chunking with context integration")
    chunk_parser.add_argument("file", type=str, help="File to chunk")
    
    # Test chunking parser
    chunking_parser = subparsers.add_parser("test-chunking", help="Test the chunking functionality on a Solidity file")
    chunking_parser.add_argument("file", type=str, help="File to chunk")
    chunking_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Chunk directory parser
    chunk_dir_parser = subparsers.add_parser("chunk-directory", help="Chunk all Solidity files in a directory")
    chunk_dir_parser.add_argument("directory", type=str, help="Directory containing files to chunk")
    chunk_dir_parser.add_argument("--output", "-o", type=str, default="./output", help="Output directory for results")
    chunk_dir_parser.add_argument("--max-concurrent", "-m", type=int, default=5, help="Maximum concurrent files")
    chunk_dir_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Process project parser
    process_parser = subparsers.add_parser("process-project", help="Process a project directory with loading and chunking")
    process_parser.add_argument("directory", type=str, help="Directory to process")
    process_parser.add_argument("--output", "-o", type=str, default="./output", help="Output directory for results")
    process_parser.add_argument("--max-concurrent", "-m", type=int, default=5, help="Maximum concurrent files")
    process_parser.add_argument("--chunk-size", "-c", type=int, default=8000, help="Maximum chunk size")
    process_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Add config option to all parsers
    for subparser in [analyze_parser, test_parser, chunk_parser, process_parser]:
        subparser.add_argument(
            "--config",
            help="Path to configuration file"
        )
    
    # If no arguments are provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Determine which command to run
    exit_code = 0
    if args.command == 'analyze':
        exit_code = run_analyze(args)
    elif args.command == 'test-chunk':
        exit_code = cmd_test_chunk(args.file)
    elif args.command == 'test-chunk-context':
        exit_code = cmd_test_chunk_context(args.file)
    elif args.command == 'test-chunking':
        exit_code = cmd_test_chunking(args.file, args.verbose)
    elif args.command == 'chunk-directory':
        exit_code = cmd_chunk_directory(args.directory, args.output, args.max_concurrent, args.verbose)
    elif args.command == 'process-project':
        exit_code = cmd_process_project(args.directory, args.output, args.max_concurrent, 
                                      args.chunk_size, args.verbose)
    
    # Exit with the appropriate code
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
