#!/usr/bin/env python3
"""
Diagnostic tool to check if prompts are properly saved in the vector store.
This script examines metadata at different stages to track prompt persistence.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

console = Console()

def analyze_metadata_file(metadata_path):
    """Analyze document metadata to check for prompts."""
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            documents = json.load(f)
        
        # Count stats
        prompt_counts = {
            'with_prompt': 0,
            'with_multi_prompts': 0,
            'without_prompt': 0,
            'total': len(documents)
        }
        
        prompt_lengths = []
        prompt_samples = []
        
        # Analyze each document
        for doc in documents:
            metadata = doc.get('metadata', {})
            if 'prompt' in metadata and metadata['prompt']:
                prompt_counts['with_prompt'] += 1
                prompt_lengths.append(len(metadata['prompt']))
                if len(prompt_samples) < 3:
                    prompt_samples.append({
                        'id': doc.get('id', 'unknown'),
                        'prompt': metadata['prompt'][:100] + ('...' if len(metadata['prompt']) > 100 else '')
                    })
            elif 'multi_llm_prompts' in metadata and metadata['multi_llm_prompts']:
                prompt_counts['with_multi_prompts'] += 1
            else:
                prompt_counts['without_prompt'] += 1
        
        # Display stats
        table = Table(title=f"Prompt Analysis for {Path(metadata_path).name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Percentage", style="yellow")
        
        total = prompt_counts['total']
        
        for key, count in prompt_counts.items():
            if key != 'total':
                percentage = f"{count/total*100:.1f}%" if total > 0 else "0%"
                table.add_row(key.replace('_', ' ').title(), str(count), percentage)
        
        table.add_row("Total Documents", str(total), "100%")
        
        if prompt_lengths:
            avg_length = sum(prompt_lengths) / len(prompt_lengths)
            table.add_row("Average Prompt Length", f"{avg_length:.1f} chars", "-")
            table.add_row("Shortest Prompt", f"{min(prompt_lengths)} chars", "-")
            table.add_row("Longest Prompt", f"{max(prompt_lengths)} chars", "-")
        
        console.print(table)
        
        # Show samples
        if prompt_samples:
            console.print("\n[bold cyan]Sample Prompts:[/]")
            for sample in prompt_samples:
                console.print(f"[bold]Document {sample['id']}:[/] {sample['prompt']}")
        
        return True
    except Exception as e:
        logger.error(f"Error analyzing metadata: {e}")
        return False

def check_persistence_in_collection(collection_name, vector_store_dir="./vector_store"):
    """Check prompt persistence in a specific collection."""
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    return analyze_metadata_file(metadata_path)

def scan_all_collections(vector_store_dir="./vector_store"):
    """Scan all collections in the vector store directory."""
    if not os.path.exists(vector_store_dir):
        logger.error(f"Vector store directory not found: {vector_store_dir}")
        return False
    
    collections = [d for d in os.listdir(vector_store_dir) 
                   if os.path.isdir(os.path.join(vector_store_dir, d)) and
                   os.path.exists(os.path.join(vector_store_dir, d, "document_metadata.json"))]
    
    if not collections:
        logger.error(f"No collections found in {vector_store_dir}")
        return False
    
    logger.info(f"Found {len(collections)} collections: {', '.join(collections)}")
    
    for collection in collections:
        logger.info(f"\nAnalyzing collection: {collection}")
        check_persistence_in_collection(collection, vector_store_dir)
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Check prompt persistence in vector store')
    parser.add_argument('-c', '--collection', help='Specific collection to check')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    parser.add_argument('--scan-all', action='store_true', help='Scan all collections')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    if args.scan_all:
        scan_all_collections(args.dir)
    elif args.collection:
        check_persistence_in_collection(args.collection, args.dir)
    else:
        logger.error("Please specify either a collection name with -c or --scan-all")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
