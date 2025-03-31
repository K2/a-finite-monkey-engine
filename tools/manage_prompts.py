#!/usr/bin/env python3
"""
Utility for managing document prompts separately from the vector store.
This allows prompts to be stored, retrieved, and modified without affecting the vector embeddings.
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

async def export_prompts(collection_name: str, output_file: str, vector_store_dir: str = "./vector_store"):
    """
    Export prompts from a collection to a standalone JSONL file.
    
    Args:
        collection_name: Name of the collection
        output_file: Path to save the prompts
        vector_store_dir: Vector store directory
    
    Returns:
        Success status
    """
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    prompts_path = os.path.join(vector_store_dir, collection_name, "document_prompts.json")
    
    # First check if we have a separate prompts file
    if os.path.exists(prompts_path):
        try:
            import aiofiles
            
            # Load prompts from the separate file
            async with aiofiles.open(prompts_path, 'r') as f:
                content = await f.read()
                prompts = json.loads(content)
            
            # Export to JSONL
            async with aiofiles.open(output_file, 'w') as f:
                for doc_id, prompt_data in prompts.items():
                    # Create a record with doc_id and prompt data
                    record = {"doc_id": doc_id, "prompts": prompt_data}
                    await f.write(json.dumps(record) + "\n")
            
            logger.info(f"Exported {len(prompts)} document prompts to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting prompts from separate file: {e}")
    
    # If no separate prompts file, try from metadata
    if os.path.exists(metadata_path):
        try:
            import aiofiles
            
            # Load documents with metadata
            async with aiofiles.open(metadata_path, 'r') as f:
                content = await f.read()
                documents = json.loads(content)
            
            # Extract prompts
            prompts_count = 0
            async with aiofiles.open(output_file, 'w') as f:
                for doc in documents:
                    doc_id = doc.get('id')
                    metadata = doc.get('metadata', {})
                    prompt_data = {}
                    
                    if 'prompt' in metadata:
                        prompt_data['single'] = metadata['prompt']
                    
                    if 'multi_llm_prompts' in metadata:
                        prompt_data['multi'] = metadata['multi_llm_prompts']
                    
                    if prompt_data:
                        # Create a record with doc_id and prompt data
                        record = {"doc_id": doc_id, "prompts": prompt_data}
                        await f.write(json.dumps(record) + "\n")
                        prompts_count += 1
            
            logger.info(f"Exported {prompts_count} document prompts to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting prompts from metadata: {e}")
    
    logger.error(f"No prompts found for collection {collection_name}")
    return False

async def import_prompts(collection_name: str, input_file: str, vector_store_dir: str = "./vector_store"):
    """
    Import prompts from a JSONL file into a collection.
    
    Args:
        collection_name: Name of the collection
        input_file: Path to the JSONL file with prompts
        vector_store_dir: Vector store directory
    
    Returns:
        Success status
    """
    collection_dir = os.path.join(vector_store_dir, collection_name)
    prompts_path = os.path.join(collection_dir, "document_prompts.json")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return False
    
    try:
        import aiofiles
        
        # Load prompts from JSONL
        prompts = {}
        async with aiofiles.open(input_file, 'r') as f:
            lines = await f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                record = json.loads(line)
                doc_id = record.get('doc_id')
                prompt_data = record.get('prompts', {})
                
                if doc_id and prompt_data:
                    prompts[doc_id] = prompt_data
        
        if not prompts:
            logger.error(f"No valid prompts found in {input_file}")
            return False
        
        # Ensure collection directory exists
        os.makedirs(collection_dir, exist_ok=True)
        
        # Save prompts
        async with aiofiles.open(prompts_path, 'w') as f:
            await f.write(json.dumps(prompts))
        
        logger.info(f"Imported {len(prompts)} document prompts to collection {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Error importing prompts: {e}")
        return False

async def analyze_prompts(collection_name: str, vector_store_dir: str = "./vector_store"):
    """
    Analyze prompts in a collection.
    
    Args:
        collection_name: Name of the collection
        vector_store_dir: Vector store directory
    
    Returns:
        Success status
    """
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    prompts_path = os.path.join(vector_store_dir, collection_name, "document_prompts.json")
    
    prompts = {}
    prompts_source = "none"
    
    # First try the separate prompts file
    if os.path.exists(prompts_path):
        try:
            import aiofiles
            
            async with aiofiles.open(prompts_path, 'r') as f:
                content = await f.read()
                prompts = json.loads(content)
            
            prompts_source = "separate"
        except Exception as e:
            logger.error(f"Error loading separate prompts file: {e}")
    
    # If no prompts from separate file, try metadata
    if not prompts and os.path.exists(metadata_path):
        try:
            import aiofiles
            
            async with aiofiles.open(metadata_path, 'r') as f:
                content = await f.read()
                documents = json.loads(content)
            
            for doc in documents:
                doc_id = doc.get('id')
                metadata = doc.get('metadata', {})
                prompt_data = {}
                
                if 'prompt' in metadata:
                    prompt_data['single'] = metadata['prompt']
                
                if 'multi_llm_prompts' in metadata:
                    prompt_data['multi'] = metadata['multi_llm_prompts']
                
                if prompt_data and doc_id:
                    prompts[doc_id] = prompt_data
            
            prompts_source = "metadata"
        except Exception as e:
            logger.error(f"Error loading prompts from metadata: {e}")
    
    if not prompts:
        logger.error(f"No prompts found for collection {collection_name}")
        return False
    
    # Count prompt types
    single_count = sum(1 for p in prompts.values() if 'single' in p)
    multi_count = sum(1 for p in prompts.values() if 'multi' in p)
    multi_types = set()
    for p in prompts.values():
        if 'multi' in p:
            multi_types.update(p['multi'].keys())
    
    # Calculate statistics
    prompt_lengths = [len(p['single']) for p in prompts.values() if 'single' in p]
    avg_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    min_length = min(prompt_lengths) if prompt_lengths else 0
    max_length = max(prompt_lengths) if prompt_lengths else 0
    
    # Create a stats table
    table = Table(title=f"Prompt Analysis for Collection: {collection_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Prompts Source", prompts_source)
    table.add_row("Total Documents with Prompts", str(len(prompts)))
    table.add_row("Documents with Single Prompts", str(single_count))
    table.add_row("Documents with Multi-LLM Prompts", str(multi_count))
    
    if multi_types:
        table.add_row("Multi-LLM Prompt Types", ", ".join(sorted(multi_types)))
    
    if prompt_lengths:
        table.add_row("Average Prompt Length", f"{avg_length:.1f} chars")
        table.add_row("Shortest Prompt", f"{min_length} chars")
        table.add_row("Longest Prompt", f"{max_length} chars")
    
    console.print(table)
    
    # Show sample prompts
    if single_count > 0:
        console.print("\n[bold cyan]Sample Single Prompts:[/]")
        
        sample_count = 0
        for doc_id, prompt_data in prompts.items():
            if 'single' in prompt_data and sample_count < 2:
                prompt = prompt_data['single']
                preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
                console.print(Panel(preview, title=f"Document: {doc_id}", border_style="blue"))
                sample_count += 1
    
    if multi_count > 0 and multi_types:
        console.print("\n[bold cyan]Sample Multi-LLM Prompts:[/]")
        
        # Show one sample for each type
        for prompt_type in sorted(multi_types)[:2]:
            for doc_id, prompt_data in prompts.items():
                if 'multi' in prompt_data and prompt_type in prompt_data['multi']:
                    prompt = prompt_data['multi'][prompt_type]
                    if isinstance(prompt, str):
                        preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
                        console.print(Panel(preview, title=f"Type: {prompt_type}, Document: {doc_id}", border_style="green"))
                        break
    
    return True

async def migrate_prompts(collection_name: str, vector_store_dir: str = "./vector_store"):
    """
    Migrate prompts from metadata to a separate file.
    
    Args:
        collection_name: Name of the collection
        vector_store_dir: Vector store directory
    
    Returns:
        Success status
    """
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    prompts_path = os.path.join(vector_store_dir, collection_name, "document_prompts.json")
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
    
    try:
        import aiofiles
        import copy
        
        # Load documents with metadata
        async with aiofiles.open(metadata_path, 'r') as f:
            content = await f.read()
            documents = json.loads(content)
        
        # Extract prompts and remove them from metadata
        prompts = {}
        documents_updated = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn()
        ) as progress:
            task = progress.add_task("[green]Migrating prompts", total=len(documents))
            
            for doc in documents:
                doc_id = doc.get('id')
                if not doc_id:
                    documents_updated.append(doc)
                    progress.update(task, advance=1)
                    continue
                
                # Make a copy so we don't modify the original
                doc_copy = copy.deepcopy(doc)
                metadata = doc_copy.get('metadata', {})
                prompt_data = {}
                
                # Extract and remove prompt
                if 'prompt' in metadata:
                    prompt_data['single'] = metadata.pop('prompt')
                
                # Extract and remove multi-LLM prompts
                if 'multi_llm_prompts' in metadata:
                    prompt_data['multi'] = metadata.pop('multi_llm_prompts')
                
                # Only add to prompts if we extracted something
                if prompt_data:
                    prompts[doc_id] = prompt_data
                
                # Update metadata
                doc_copy['metadata'] = metadata
                documents_updated.append(doc_copy)
                
                progress.update(task, advance=1)
        
        if not prompts:
            logger.warning("No prompts found in metadata to migrate")
            return False
        
        # Save updated metadata
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(documents_updated))
        
        # Save prompts to separate file
        async with aiofiles.open(prompts_path, 'w') as f:
            await f.write(json.dumps(prompts))
        
        logger.info(f"Migrated {len(prompts)} document prompts to separate file")
        return True
    except Exception as e:
        logger.error(f"Error migrating prompts: {e}")
        return False

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Manage document prompts separately from vector store')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export prompts to a standalone file')
    export_parser.add_argument('-c', '--collection', required=True, help='Collection name')
    export_parser.add_argument('-o', '--output', required=True, help='Output JSONL file')
    export_parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import prompts from a standalone file')
    import_parser.add_argument('-c', '--collection', required=True, help='Collection name')
    import_parser.add_argument('-i', '--input', required=True, help='Input JSONL file')
    import_parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze prompts in a collection')
    analyze_parser.add_argument('-c', '--collection', required=True, help='Collection name')
    analyze_parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate prompts from metadata to separate file')
    migrate_parser.add_argument('-c', '--collection', required=True, help='Collection name')
    migrate_parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    if args.command == 'export':
        await export_prompts(args.collection, args.output, args.dir)
    elif args.command == 'import':
        await import_prompts(args.collection, args.input, args.dir)
    elif args.command == 'analyze':
        await analyze_prompts(args.collection, args.dir)
    elif args.command == 'migrate':
        await migrate_prompts(args.collection, args.dir)
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())
