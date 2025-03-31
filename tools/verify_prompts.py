#!/usr/bin/env python3
"""
Utility to verify that prompts are properly saved in the vector store metadata.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

def verify_prompts(collection_name, vector_store_dir="./vector_store", verbose=False):
    """
    Verify that prompts are properly saved in the vector store metadata.
    
    Args:
        collection_name: Name of the collection to check
        vector_store_dir: Directory where vector store collections are stored
        verbose: Whether to show detailed diagnostic information
    
    Returns:
        Success status
    """
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    
    # Check if collection directory exists
    collection_dir = os.path.join(vector_store_dir, collection_name)
    if not os.path.exists(collection_dir):
        logger.error(f"Collection directory not found: {collection_dir}")
        return False
    
    # List all files in collection directory for diagnostics
    if verbose:
        logger.info(f"Files in collection directory {collection_dir}:")
        for file in os.listdir(collection_dir):
            file_path = os.path.join(collection_dir, file)
            size = os.path.getsize(file_path)
            logger.info(f"  {file} ({size} bytes)")
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
    
    try:
        # Check metadata file size
        file_size = os.path.getsize(metadata_path)
        logger.info(f"Metadata file size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("Metadata file is empty!")
            return False
        
        # Load the metadata
        with open(metadata_path, 'r') as f:
            try:
                documents = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse metadata file as JSON: {e}")
                
                # Show a snippet of the file content for diagnosis
                with open(metadata_path, 'r') as f2:
                    content = f2.read(1000)  # Read first 1000 chars
                    logger.info(f"First 1000 characters of metadata file:\n{content}")
                return False
        
        # Count documents with prompts
        prompt_count = 0
        multi_prompt_count = 0
        no_prompt_count = 0
        
        largest_prompt = 0
        smallest_prompt = float('inf')
        
        if not documents:
            logger.warning("Metadata file contains an empty document list")
            return False
        
        # Check if it's a list or some other structure
        if not isinstance(documents, list):
            logger.error(f"Metadata doesn't contain a document list! Found type: {type(documents)}")
            logger.info(f"Content structure: {documents if len(str(documents)) < 100 else str(documents)[:100] + '...'}")
            return False
        
        # Analyze each document
        metadata_keys_seen = set()
        missing_metadata_count = 0
        
        for i, doc in enumerate(documents):
            # Track metadata keys for diagnostic purposes
            if verbose:
                if i < 5:  # Show first 5 docs in detail
                    logger.debug(f"Document {i}: {doc}")
                
                if 'metadata' in doc:
                    metadata_keys_seen.update(doc['metadata'].keys())
                else:
                    missing_metadata_count += 1
            
            metadata = doc.get('metadata', {})
            if 'prompt' in metadata and metadata['prompt']:
                prompt_count += 1
                prompt_len = len(metadata['prompt'])
                largest_prompt = max(largest_prompt, prompt_len)
                smallest_prompt = min(smallest_prompt, prompt_len)
            elif 'multi_llm_prompts' in metadata and metadata['multi_llm_prompts']:
                multi_prompt_count += 1
            else:
                no_prompt_count += 1
        
        # Print statistics
        total = len(documents)
        logger.info(f"Collection: {collection_name} ({total} documents)")
        percentage_with_prompts = (prompt_count / total * 100) if total > 0 else 0
        percentage_with_multi = (multi_prompt_count / total * 100) if total > 0 else 0
        percentage_without = (no_prompt_count / total * 100) if total > 0 else 0
        
        logger.info(f"Documents with prompts: {prompt_count} ({percentage_with_prompts:.1f}%)")
        logger.info(f"Documents with multi-LLM prompts: {multi_prompt_count} ({percentage_with_multi:.1f}%)")
        logger.info(f"Documents without prompts: {no_prompt_count} ({percentage_without:.1f}%)")
        
        if verbose:
            logger.info(f"Metadata keys found: {sorted(metadata_keys_seen)}")
            logger.info(f"Documents missing metadata field: {missing_metadata_count}")
        
        if prompt_count > 0:
            logger.info(f"Prompt sizes - Largest: {largest_prompt} chars, Smallest: {smallest_prompt} chars")
            
            # Show sample prompts
            logger.info("\nSample prompts:")
            samples = 0
            for doc in documents:
                metadata = doc.get('metadata', {})
                if 'prompt' in metadata and metadata['prompt']:
                    prompt = metadata['prompt']
                    preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
                    logger.info(f"Document {doc.get('id', 'unknown')} prompt: {preview}")
                    samples += 1
                    if samples >= 3:
                        break
        
        # Create a table with statistics
        table = Table(title=f"Prompt Statistics for Collection: {collection_name}")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        table.add_row("Documents with single prompt", str(prompt_count), f"{percentage_with_prompts:.1f}%")
        table.add_row("Documents with multi-LLM prompts", str(multi_prompt_count), f"{percentage_with_multi:.1f}%")
        table.add_row("Documents without prompts", str(no_prompt_count), f"{percentage_without:.1f}%")
        table.add_row("Total documents", str(total), "100%")
        
        console.print(table)
        
        return True
    except Exception as e:
        logger.error(f"Error verifying prompts: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Verify that prompts are properly saved in the vector store metadata.')
    parser.add_argument('-c', '--collection', required=True, help='Name of the collection')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed diagnostic information')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")
    
    verify_prompts(args.collection, args.dir, args.verbose)

if __name__ == "__main__":
    main()
