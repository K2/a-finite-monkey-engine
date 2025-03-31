#!/usr/bin/env python3
"""
Command-line utility to query the vector store and retrieve associated prompts.
This tool demonstrates how to find semantically similar documents in the vector store
and use their prompts to better understand the context.
"""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import vector store
from tools.vector_store_util import SimpleVectorStore

console = Console()

async def query_and_get_prompts(collection_name: str, query: str, top_k: int = 20, vector_store_dir: str = "./vector_store"):
    """
    Query the vector store and get prompts for the top matching documents.
    
    Args:
        collection_name: Name of the collection to query
        query: Query text
        top_k: Number of top results to consider (default: 20)
        vector_store_dir: Vector store directory
        
    Returns:
        Query results with prompts
    """
    try:
        # Initialize vector store
        store = SimpleVectorStore(
            storage_dir=vector_store_dir,
            collection_name=collection_name
        )
        
        # Ensure index is initialized
        if not store._index:
            logger.error(f"Failed to initialize vector store for collection: {collection_name}")
            return None
        
        # Query with prompts
        result = await store.query_with_prompts(query, top_k=top_k)
        return result
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        return None

def display_results(result):
    """Display query results in a rich format."""
    if not result:
        console.print("[red]No results to display[/]")
        return
    
    # Display query
    console.print(Panel(f"Query: [cyan]{result['query']}[/]", border_style="blue"))
    
    # Display top results
    top_results = result['results'][:5]  # Show top 5 for display clarity
    if top_results:
        result_table = Table(title=f"Top {len(top_results)} Results")
        result_table.add_column("Rank", style="cyan", justify="right")
        result_table.add_column("ID", style="green")
        result_table.add_column("Score", style="yellow")
        result_table.add_column("Text Preview", style="white")
        
        for i, res in enumerate(top_results):
            text_preview = res['text'][:100] + "..." if len(res['text']) > 100 else res['text']
            result_table.add_row(
                str(i+1),
                res['id'],
                f"{res['score']:.4f}" if res['score'] is not None else "N/A",
                text_preview
            )
        
        console.print(result_table)
    
    # Display prompts
    prompts = result.get('prompts', {})
    if 'individual' in prompts and prompts['individual']:
        prompt_table = Table(title="Top Prompts")
        prompt_table.add_column("Rank", style="cyan", justify="right")
        prompt_table.add_column("ID", style="green")
        prompt_table.add_column("Score", style="yellow")
        prompt_table.add_column("Type", style="magenta")
        prompt_table.add_column("Prompt Preview", style="white")
        
        for i, prompt_info in enumerate(prompts['individual'][:3]):  # Show top 3 for clarity
            prompt_text = prompt_info['prompt']
            prompt_preview = prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
            prompt_type = prompt_info.get('type', 'single')
            
            prompt_table.add_row(
                str(i+1),
                str(prompt_info.get('rank', i+1)),
                f"{prompt_info.get('score', 0):.4f}" if prompt_info.get('score') is not None else "N/A",
                prompt_type,
                prompt_preview
            )
        
        console.print(prompt_table)
    
    # Display combined prompt
    if 'combined' in prompts and prompts['combined']:
        console.print("\n[bold cyan]Combined Prompt for LLM:[/]")
        combined_prompt = prompts['combined']
        # Show a preview if it's too long
        if len(combined_prompt) > 1000:
            preview = combined_prompt[:1000] + "\n[...truncated for display...]"
            console.print(Panel(preview, border_style="green", title="Combined Prompt (Preview)"))
        else:
            console.print(Panel(combined_prompt, border_style="green", title="Combined Prompt"))
    
    # Option to save the full results to a file
    console.print("\n[yellow]Would you like to save the full results to a file? (y/n)[/]")
    choice = input().strip().lower()
    if choice == 'y':
        filename = f"query_results_{result['query'][:20].replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        console.print(f"[green]Results saved to {filename}[/]")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Query vector store and retrieve associated prompts')
    parser.add_argument('-c', '--collection', required=True, help='Collection name')
    parser.add_argument('-q', '--query', required=True, help='Query text')
    parser.add_argument('-k', '--top-k', type=int, default=20, help='Number of top results to consider')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    result = await query_and_get_prompts(args.collection, args.query, args.top_k, args.dir)
    if result:
        display_results(result)
    else:
        console.print("[red]Query failed. Check logs for details.[/]")

if __name__ == "__main__":
    asyncio.run(main())
