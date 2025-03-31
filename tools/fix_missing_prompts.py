#!/usr/bin/env python3
"""
Utility to regenerate missing prompts in a vector store collection.
"""
import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

console = Console()

async def fix_prompts(collection_name, vector_store_dir="./vector_store", dry_run=False):
    """
    Fix missing prompts in a collection by generating new ones.
    
    Args:
        collection_name: Name of the collection to fix
        vector_store_dir: Directory where vector store collections are stored
        dry_run: Whether to just report issues without fixing them
    
    Returns:
        Success status
    """
    from vector_store_util import SimpleVectorStore
    
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
    
    try:
        # Load the documents
        with open(metadata_path, 'r') as f:
            documents = json.load(f)
        
        total_docs = len(documents)
        missing_prompts = []
        
        # Find documents missing prompts
        for i, doc in enumerate(documents):
            metadata = doc.get('metadata', {})
            if not ('prompt' in metadata or 'multi_llm_prompts' in metadata):
                # Add to list of documents needing prompts
                missing_prompts.append((i, doc))
        
        logger.info(f"Collection {collection_name}: {len(missing_prompts)}/{total_docs} documents missing prompts")
        
        if not missing_prompts:
            logger.info("No missing prompts to fix")
            return True
        
        if dry_run:
            logger.info("Dry run - not fixing prompts")
            return True
        
        # Initialize vector store with prompt generation enabled
        store = SimpleVectorStore(
            storage_dir=vector_store_dir,
            collection_name=collection_name,
            embedding_model="local"  # Use local embedding for safety
        )
        
        # Force prompt generation settings
        store.generate_prompts = True
        store.use_ollama_for_prompts = True
        
        # Initialize prompt generator if needed
        if not hasattr(store, 'prompt_generator'):
            from vector_store_prompts import PromptGenerator
            store.prompt_generator = PromptGenerator(
                generate_prompts=True,
                use_ollama_for_prompts=True,
                prompt_model=store.prompt_model,
                ollama_url=store.ollama_url,
                multi_llm_prompts=store.multi_llm_prompts
            )
        
        # Process documents with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(
                "[green]Generating missing prompts",
                total=len(missing_prompts),
                completed=0
            )
            
            for i, (doc_idx, doc) in enumerate(missing_prompts):
                try:
                    if store.multi_llm_prompts:
                        multi_prompts = await store.prompt_generator.generate_multi_llm_prompts(doc)
                        documents[doc_idx]['metadata']['multi_llm_prompts'] = multi_prompts
                        logger.debug(f"Added multi-LLM prompts to document {doc_idx}")
                    else:
                        prompt = await store.prompt_generator.generate_prompt(doc)
                        documents[doc_idx]['metadata']['prompt'] = prompt
                        logger.debug(f"Added prompt to document {doc_idx}: {prompt[:50]}...")
                    
                    # Update progress
                    progress.update(task, advance=1, description=f"[green]Generating prompt {i+1}/{len(missing_prompts)}")
                except Exception as e:
                    logger.error(f"Error generating prompt for document {doc_idx}: {e}")
                    continue
        
        # Save updated documents
        backup_path = f"{metadata_path}.bak"
        logger.info(f"Creating backup at {backup_path}")
        import shutil
        shutil.copy2(metadata_path, backup_path)
        
        logger.info(f"Saving updated metadata with {len(missing_prompts)} new prompts")
        with open(metadata_path, 'w') as f:
            json.dump(documents, f)
        
        return True
    except Exception as e:
        logger.error(f"Error fixing prompts: {e}")
        return False

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fix missing prompts in a vector store collection')
    parser.add_argument('-c', '--collection', required=True, help='Name of the collection')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    parser.add_argument('--dry-run', action='store_true', help="Don't actually fix prompts, just report issues")
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    await fix_prompts(args.collection, args.dir, args.dry_run)

if __name__ == "__main__":
    asyncio.run(main())
