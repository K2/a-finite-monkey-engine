#!/usr/bin/env python3
"""
Utility script to view and analyze prompts in a vector store collection.
"""
import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

console = Console()

async def view_prompts(collection_name, vector_store_dir="./vector_store", limit=10):
    """View prompts from documents in a collection."""
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.error(f"Collection not found: {collection_name}")
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            documents = json.load(f)
        
        prompt_count = 0
        multi_prompt_count = 0
        no_prompt_count = 0
        
        # Count documents with different prompt types
        for doc in documents:
            metadata = doc.get('metadata', {})
            if 'prompt' in metadata:
                prompt_count += 1
            elif 'multi_llm_prompts' in metadata:
                multi_prompt_count += 1
            else:
                no_prompt_count += 1
        
        # Create a table with statistics
        table = Table(title=f"Prompt Statistics for Collection: {collection_name}")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        total = len(documents)
        table.add_row("Documents with single prompt", str(prompt_count), f"{prompt_count/total*100:.1f}%" if total > 0 else "0%")
        table.add_row("Documents with multi-LLM prompts", str(multi_prompt_count), f"{multi_prompt_count/total*100:.1f}%" if total > 0 else "0%")
        table.add_row("Documents without prompts", str(no_prompt_count), f"{no_prompt_count/total*100:.1f}%" if total > 0 else "0%")
        table.add_row("Total documents", str(total), "100%")
        
        console.print(table)
        
        # Display some sample prompts
        documents_with_prompts = [doc for doc in documents if 'prompt' in doc.get('metadata', {})]
        documents_with_multi = [doc for doc in documents if 'multi_llm_prompts' in doc.get('metadata', {})]
        
        if documents_with_prompts and limit > 0:
            console.print("\n[bold cyan]Sample Single Prompts:[/]")
            for i, doc in enumerate(documents_with_prompts[:limit]):
                prompt = doc.get('metadata', {}).get('prompt', '')
                console.print(Panel(prompt, title=f"Prompt {i+1}", title_align="left", border_style="blue"))
        
        if documents_with_multi and limit > 0:
            console.print("\n[bold cyan]Sample Multi-LLM Prompts:[/]")
            for i, doc in enumerate(documents_with_multi[:limit]):
                multi_prompts = doc.get('metadata', {}).get('multi_llm_prompts', {})
                if not multi_prompts:
                    continue
                
                # Display a few prompt types
                multi_table = Table(title=f"Multi-LLM Prompts for Document {i+1}")
                multi_table.add_column("Type", style="cyan")
                multi_table.add_column("Prompt", style="green")
                
                for prompt_type, prompt in multi_prompts.items():
                    if isinstance(prompt, str):
                        # Truncate long prompts for display
                        display_prompt = prompt if len(prompt) < 100 else prompt[:100] + "..."
                        multi_table.add_row(prompt_type, display_prompt)
                    elif isinstance(prompt, list) and prompt:
                        multi_table.add_row(prompt_type, f"{len(prompt)} additional prompts")
                
                console.print(multi_table)
        
        return True
    except Exception as e:
        logger.error(f"Error viewing prompts: {e}")
        return False

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='View prompts in a vector store collection')
    parser.add_argument('-c', '--collection', required=True, help='Name of the collection')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    parser.add_argument('-l', '--limit', type=int, default=5, help='Maximum number of prompts to show')
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    await view_prompts(args.collection, args.dir, args.limit)

if __name__ == "__main__":
    asyncio.run(main())
