#!/usr/bin/env python3
"""
Chunking Demonstration Script with CallGraph integration

This script demonstrates the contract chunking functionality
of the Finite Monkey Engine, including the CallGraph integration
for hierarchical file->contract->function structure analysis.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple

# Add the parent directory to the path so we can import the module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from finite_monkey.utils.chunking import (
    ContractChunker, AsyncContractChunker, CallGraph,
    chunk_solidity_file, chunk_solidity_code, async_chunk_solidity_file
)

def print_chunk_info(chunk, detailed=False, indent=0):
    """Print information about a chunk"""
    chunk_id = chunk.get('chunk_id', 'Unknown')
    chunk_type = chunk.get('chunk_type', 'Unknown')
    
    print(f"{'  ' * indent}Chunk ID: {chunk_id}")
    print(f"{'  ' * indent}Type: {chunk_type}")
    
    if chunk_type == 'contract':
        contract_name = chunk.get('contract_name', 'Unknown')
        print(f"{'  ' * indent}Contract: {contract_name}")
        
        if 'contract_functions' in chunk:
            functions = chunk['contract_functions']
            print(f"{'  ' * indent}Functions in contract ({len(functions)}):")
            for function in functions[:5]:  # Show first 5 functions
                print(f"{'  ' * (indent+1)}- {function}")
            if len(functions) > 5:
                print(f"{'  ' * (indent+1)}... and {len(functions) - 5} more functions")
    
    elif chunk_type == 'function':
        contract_name = chunk.get('contract_name', 'Unknown')
        function_name = chunk.get('function_name', 'Unknown')
        print(f"{'  ' * indent}Function: {contract_name}.{function_name}")
        
        # Show function calls
        if 'function_calls' in chunk:
            calls = chunk['function_calls']
            print(f"{'  ' * indent}Calls to other functions ({len(calls)}):")
            for call in calls[:3]:  # Show first 3 calls
                print(f"{'  ' * (indent+1)}- {call['contract']}.{call['function']}")
            if len(calls) > 3:
                print(f"{'  ' * (indent+1)}... and {len(calls) - 3} more calls")
        
        # Show function callers
        if 'called_by' in chunk:
            callers = chunk['called_by']
            print(f"{'  ' * indent}Called by ({len(callers)}):")
            for caller in callers[:3]:  # Show first 3 callers
                print(f"{'  ' * (indent+1)}- {caller['contract']}.{caller['function']}")
            if len(callers) > 3:
                print(f"{'  ' * (indent+1)}... and {len(callers) - 3} more callers")
    
    print(f"{'  ' * indent}Content Length: {len(chunk['content'])} chars")
    
    if "imports" in chunk and chunk["imports"]:
        print(f"{'  ' * indent}Imports: {len(chunk['imports'])}")
        for imp in chunk["imports"][:3]:  # Show first 3 imports
            print(f"{'  ' * (indent+1)}- {imp}")
        if len(chunk["imports"]) > 3:
            print(f"{'  ' * (indent+1)}... and {len(chunk['imports']) - 3} more imports")
    
    # Show content snippet for detailed view
    if detailed:
        content = chunk.get('content', '')
        if len(content) > 200:
            snippet = content[:200] + '...'
        else:
            snippet = content
        print(f"{'  ' * indent}Content snippet:")
        print(f"{'  ' * (indent+1)}{snippet.replace(chr(10), chr(10) + '  ' * (indent+1))}")
    
    print()

def analyze_file(file_path, max_size=8000, detailed=False, include_call_graph=True):
    """Analyze a single Solidity file"""
    print(f"Analyzing file: {file_path}")
    
    # Initialize the chunker with call graph support
    chunker = ContractChunker(
        max_chunk_size=max_size,
        chunk_by_contract=True,
        chunk_by_function=True,
        include_call_graph=include_call_graph,
    )
    
    # Initialize the call graph for the project if needed
    if include_call_graph:
        project_path = os.path.dirname(file_path)
        chunker.initialize_call_graph(project_path)
    
    # Chunk the file
    chunks = chunker.chunk_file(file_path)
    
    # Display information about each chunk
    print(f"\nChunked into {len(chunks)} segments:")
    for i, chunk in enumerate(chunks):
        print(f"CHUNK {i+1}/{len(chunks)}")
        print_chunk_info(chunk, detailed)
    
    # Show how the file relationships are structured
    print("\n--- File Structure Information ---")
    contract_counts = {}
    function_counts = {}
    
    for chunk in chunks:
        if chunk['chunk_type'] == 'contract':
            contract_name = chunk.get('contract_name', 'Unknown')
            if contract_name in contract_counts:
                contract_counts[contract_name] += 1
            else:
                contract_counts[contract_name] = 1
        
        elif chunk['chunk_type'] == 'function':
            contract_name = chunk.get('contract_name', 'Unknown')
            if contract_name in function_counts:
                function_counts[contract_name] += 1
            else:
                function_counts[contract_name] = 1
    
    print(f"Contracts found: {len(contract_counts)}")
    for contract, count in contract_counts.items():
        funcs = function_counts.get(contract, 0)
        print(f"  - {contract}: {funcs} functions")
    
    return chunks

async def analyze_file_async(file_path, max_size=8000, detailed=False):
    """Analyze a Solidity file with AsyncContractChunker"""
    print(f"Analyzing file asynchronously: {file_path}")
    
    # Initialize the async chunker with call graph support
    chunker = AsyncContractChunker(
        max_chunk_size=max_size,
        chunk_by_contract=True,
        chunk_by_function=True,
        include_call_graph=True,
    )
    
    # Initialize the call graph for the project
    project_path = os.path.dirname(file_path)
    await chunker.initialize_call_graph(project_path)
    
    # Chunk the file
    chunks = await chunker.chunk_file(file_path)
    
    # Display information about each chunk
    print(f"\nChunked into {len(chunks)} segments:")
    for i, chunk in enumerate(chunks):
        print(f"CHUNK {i+1}/{len(chunks)}")
        print_chunk_info(chunk, detailed)
    
    return chunks

def analyze_project(project_path, max_size=8000, detailed=False, limit=None, include_call_graph=True):
    """
    Analyze all Solidity files in a project directory
    
    Args:
        project_path: Path to the project directory
        max_size: Maximum chunk size
        detailed: Whether to show detailed information
        limit: Optional limit on number of files to process
        include_call_graph: Whether to include call graph information
    """
    print(f"Analyzing project directory: {project_path}")
    
    # Find all Solidity files
    solidity_files = []
    for root, _, files in os.walk(project_path):
        for file in sorted(files):
            if file.endswith(".sol") and not file.endswith(".t.sol"):
                solidity_files.append(os.path.join(root, file))
    
    # Apply limit if specified
    if limit and limit > 0:
        print(f"Limiting analysis to {limit} of {len(solidity_files)} files")
        solidity_files = solidity_files[:limit]
    else:
        print(f"Analyzing all {len(solidity_files)} Solidity files")
    
    # Initialize the chunker with call graph support
    chunker = ContractChunker(
        max_chunk_size=max_size,
        chunk_by_contract=True,
        chunk_by_function=True,
        include_call_graph=include_call_graph,
    )
    
    # Initialize the call graph for the project if needed
    if include_call_graph:
        start_time = time.time()
        print("Initializing call graph for the project...")
        chunker.initialize_call_graph(project_path)
        print(f"Call graph initialized in {time.time() - start_time:.2f} seconds")
    
    # Process each file
    chunked_files = {}
    file_summaries = []
    total_contracts = 0
    total_functions = 0
    
    for i, file_path in enumerate(solidity_files):
        print(f"\nProcessing file {i+1}/{len(solidity_files)}: {os.path.basename(file_path)}")
        
        try:
            # Chunk the file
            chunks = chunker.chunk_file(file_path)
            chunked_files[file_path] = chunks
            
            # Count contracts and functions
            contracts = set()
            functions = 0
            
            for chunk in chunks:
                if chunk['chunk_type'] == 'contract':
                    contracts.add(chunk.get('contract_name', 'Unknown'))
                
                elif chunk['chunk_type'] == 'function':
                    functions += 1
            
            total_contracts += len(contracts)
            total_functions += functions
            
            # Create a file summary
            file_name = os.path.basename(file_path)
            file_summaries.append({
                'file': file_name,
                'contracts': len(contracts),
                'functions': functions,
                'chunks': len(chunks)
            })
            
            # Print a brief summary for this file
            print(f"  - Contracts: {len(contracts)}")
            print(f"  - Functions: {functions}")
            print(f"  - Chunks: {len(chunks)}")
            
            # Print detailed information if requested
            if detailed:
                for j, chunk in enumerate(chunks):
                    print(f"\n  CHUNK {j+1}/{len(chunks)}")
                    print_chunk_info(chunk, detailed=False, indent=2)
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Print the overall statistics
    print("\n" + "=" * 70)
    print("Project Analysis Summary")
    print("=" * 70)
    print(f"Total files processed: {len(chunked_files)}")
    print(f"Total contracts found: {total_contracts}")
    print(f"Total functions found: {total_functions}")
    
    # Print file summaries
    print("\nFile summaries:")
    for summary in file_summaries:
        print(f"  - {summary['file']}: {summary['contracts']} contracts, {summary['functions']} functions, {summary['chunks']} chunks")
    
    return chunked_files

async def analyze_project_async(project_path, max_size=8000, detailed=False, limit=None):
    """
    Analyze all Solidity files in a project directory asynchronously
    
    Args:
        project_path: Path to the project directory
        max_size: Maximum chunk size
        detailed: Whether to show detailed information
        limit: Optional limit on number of files to process
    """
    print(f"Analyzing project directory asynchronously: {project_path}")
    
    # Find all Solidity files
    solidity_files = []
    for root, _, files in os.walk(project_path):
        for file in sorted(files):
            if file.endswith(".sol") and not file.endswith(".t.sol"):
                solidity_files.append(os.path.join(root, file))
    
    # Apply limit if specified
    if limit and limit > 0:
        print(f"Limiting analysis to {limit} of {len(solidity_files)} files")
        solidity_files = solidity_files[:limit]
    else:
        print(f"Analyzing all {len(solidity_files)} Solidity files")
    
    # Initialize the chunker with call graph support
    chunker = AsyncContractChunker(
        max_chunk_size=max_size,
        chunk_by_contract=True,
        chunk_by_function=True,
        include_call_graph=True,
    )
    
    # Initialize the call graph for the project
    start_time = time.time()
    print("Initializing call graph for the project...")
    await chunker.initialize_call_graph(project_path)
    print(f"Call graph initialized in {time.time() - start_time:.2f} seconds")
    
    # Process all files concurrently
    async def process_file(file_path):
        try:
            print(f"Processing: {os.path.basename(file_path)}")
            return file_path, await chunker.chunk_file(file_path)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return file_path, []
    
    # Create tasks for all files
    tasks = [process_file(file_path) for file_path in solidity_files]
    
    # Run all tasks and collect results
    results = await asyncio.gather(*tasks)
    
    # Create dictionary of file -> chunks
    chunked_files = {file_path: chunks for file_path, chunks in results if chunks}
    
    # Count totals and create summaries
    total_contracts = 0
    total_functions = 0
    file_summaries = []
    
    for file_path, chunks in chunked_files.items():
        # Count contracts and functions
        contracts = set()
        functions = 0
        
        for chunk in chunks:
            if chunk['chunk_type'] == 'contract':
                contracts.add(chunk.get('contract_name', 'Unknown'))
            
            elif chunk['chunk_type'] == 'function':
                functions += 1
        
        total_contracts += len(contracts)
        total_functions += functions
        
        # Create a file summary
        file_name = os.path.basename(file_path)
        file_summaries.append({
            'file': file_name,
            'contracts': len(contracts),
            'functions': functions,
            'chunks': len(chunks)
        })
    
    # Print the overall statistics
    print("\n" + "=" * 70)
    print("Project Analysis Summary (Async)")
    print("=" * 70)
    print(f"Total files processed: {len(chunked_files)}")
    print(f"Total contracts found: {total_contracts}")
    print(f"Total functions found: {total_functions}")
    
    # Print file summaries
    print("\nFile summaries:")
    for summary in sorted(file_summaries, key=lambda x: x['file']):
        print(f"  - {summary['file']}: {summary['contracts']} contracts, {summary['functions']} functions, {summary['chunks']} chunks")
    
    return chunked_files

def export_results(chunked_files, output_file):
    """Export analysis results to a JSON file"""
    print(f"Exporting results to {output_file}")
    
    # Prepare a serializable summary
    summary = {
        "files": {},
        "total_contracts": 0,
        "total_functions": 0,
        "total_chunks": 0,
    }
    
    for file_path, chunks in chunked_files.items():
        file_name = os.path.basename(file_path)
        
        # Count contracts and functions
        contracts = set()
        functions = []
        
        for chunk in chunks:
            if chunk['chunk_type'] == 'contract':
                contracts.add(chunk.get('contract_name', 'Unknown'))
            
            elif chunk['chunk_type'] == 'function':
                functions.append({
                    "name": chunk.get('function_name', 'Unknown'),
                    "contract": chunk.get('contract_name', 'Unknown'),
                    "calls": [f"{c['contract']}.{c['function']}" for c in chunk.get('function_calls', [])],
                    "called_by": [f"{c['contract']}.{c['function']}" for c in chunk.get('called_by', [])]
                })
        
        # Add to summary
        summary["files"][file_name] = {
            "contracts": list(contracts),
            "functions": functions,
            "chunk_count": len(chunks)
        }
        
        summary["total_contracts"] += len(contracts)
        summary["total_functions"] += len(functions)
        summary["total_chunks"] += len(chunks)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results exported to {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Finite Monkey Engine Chunking Demo with CallGraph')
    
    # File or project selection
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument('--file', '-f', type=str, help='Path to a single Solidity file to analyze')
    file_group.add_argument('--project', '-p', type=str, help='Path to a project directory to analyze')
    
    # Mode selection
    parser.add_argument('--async', '-a', action='store_true', help='Use async version for analysis')
    parser.add_argument('--no-call-graph', action='store_true', help='Disable CallGraph integration')
    
    # Output options
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed information including content snippets')
    parser.add_argument('--output', '-o', type=str, help='Output file for results (JSON format)')
    
    # Performance options
    parser.add_argument('--max-size', type=int, default=4000, help='Maximum chunk size in characters')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    
    args = parser.parse_args()
    
    # Set default to analyze examples/src if no file or project specified
    if not args.file and not args.project:
        examples_src = os.path.join(parent_dir, "examples", "src")
        if os.path.exists(examples_src):
            print(f"No file or project specified, defaulting to examples/src")
            args.project = examples_src
        else:
            print("Error: examples/src directory not found and no file or project specified")
            return 1
    
    # File validation
    if args.file and not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return 1
    
    if args.project and not os.path.exists(args.project):
        print(f"Error: Project directory not found: {args.project}")
        return 1
    
    include_call_graph = not args.no_call_graph
    
    # Print header information
    print("=" * 70)
    print(f"Finite Monkey Engine - Contract Chunking Demo")
    print("=" * 70)
    print(f"Mode: {'Async' if getattr(args, 'async', False) else 'Sync'}")
    print(f"Call Graph: {'Enabled' if include_call_graph else 'Disabled'}")
    print(f"Max Chunk Size: {args.max_size} chars")
    print("=" * 70)
    
    # Run analysis
    chunked_files = {}
    
    if args.file:
        # Single file analysis
        if getattr(args, 'async', False):
            chunked_files = {args.file: asyncio.run(analyze_file_async(args.file, args.max_size, args.detailed))}
        else:
            chunked_files = {args.file: analyze_file(args.file, args.max_size, args.detailed, include_call_graph)}
    
    elif args.project:
        # Project analysis
        if getattr(args, 'async', False):
            chunked_files = asyncio.run(analyze_project_async(args.project, args.max_size, args.detailed, args.limit))
        else:
            chunked_files = analyze_project(args.project, args.max_size, args.detailed, args.limit, include_call_graph)
    
    # Export results if requested
    if args.output:
        export_results(chunked_files, args.output)
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())