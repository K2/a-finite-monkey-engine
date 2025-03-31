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

def verify_prompts(collection_name, vector_store_dir="./vector_store"):
    """
    Verify that prompts are properly saved in the vector store metadata.
    """
    metadata_path = os.path.join(vector_store_dir, collection_name, "document_metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.error(f"Collection not found: {collection_name}")
        return False
    
    try:
        # Load the metadata
        with open(metadata_path, 'r') as f:
            documents = json.load(f)
        
        # Count documents with prompts
        prompt_count = 0
        multi_prompt_count = 0
        no_prompt_count = 0
        
        largest_prompt = 0
        smallest_prompt = float('inf')
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            if 'prompt' in metadata:
                prompt_count += 1
                prompt_len = len(metadata['prompt'])
                largest_prompt = max(largest_prompt, prompt_len)
                smallest_prompt = min(smallest_prompt, prompt_len)
            elif 'multi_llm_prompts' in metadata:
                multi_prompt_count += 1
            else:
                no_prompt_count += 1
        
        # Print statistics
        total = len(documents)
        logger.info(f"Collection: {collection_name} ({total} documents)")
        logger.info(f"Documents with prompts: {prompt_count} ({prompt_count/total*100:.1f}%)")
        logger.info(f"Documents with multi-LLM prompts: {multi_prompt_count} ({multi_prompt_count/total*100:.1f}%)")
        logger.info(f"Documents without prompts: {no_prompt_count} ({no_prompt_count/total*100:.1f}%)")
        
        if prompt_count > 0:
            logger.info(f"Prompt sizes - Largest: {largest_prompt} chars, Smallest: {smallest_prompt} chars")
            
            # Show sample prompts
            logger.info("\nSample prompts:")
            samples = 0
            for doc in documents:
                metadata = doc.get('metadata', {})
                if 'prompt' in metadata:
                    prompt = metadata['prompt']
                    logger.info(f"Document {doc.get('id', 'unknown')} prompt: {prompt[:100]}..." if len(prompt) > 100 else prompt)
                    samples += 1
                    if samples >= 3:
                        break
        
        return True
    except Exception as e:
        logger.error(f"Error verifying prompts: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Verify that prompts are properly saved in the vector store metadata.')
    parser.add_argument('-c', '--collection', required=True, help='Name of the collection')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    verify_prompts(args.collection, args.dir)

if __name__ == "__main__":
    main()
