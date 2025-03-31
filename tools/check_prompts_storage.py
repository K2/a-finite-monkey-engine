#!/usr/bin/env python3
"""
Diagnostic script to check if prompts are being properly stored and retrieved from vector store.
"""
import os
import json
import asyncio
from pathlib import Path
from loguru import logger

async def check_prompts_storage(collection_name="default", storage_dir="./vector_store"):
    """Check if prompts are being properly stored and retrieved."""
    logger.info(f"Checking prompt storage for collection: {collection_name}")
    
    # First check if the collection directory exists
    collection_dir = os.path.join(storage_dir, collection_name)
    if not os.path.exists(collection_dir):
        logger.error(f"Collection directory not found: {collection_dir}")
        return
    
    # Check if document_metadata.json exists
    metadata_path = os.path.join(collection_dir, "document_metadata.json")
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return
    
    # Check the size of the metadata file
    metadata_size = os.path.getsize(metadata_path)
    logger.info(f"Metadata file size: {metadata_size} bytes")
    
    # Check if document_prompts.json exists
    prompts_path = os.path.join(collection_dir, "document_prompts.json")
    if not os.path.exists(prompts_path):
        logger.error(f"Prompts file not found: {prompts_path}")
        return
    
    # Check the size of the prompts file
    prompts_size = os.path.getsize(prompts_path)
    logger.info(f"Prompts file size: {prompts_size} bytes")
    
    # Load metadata to count documents
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        doc_count = len(metadata)
        logger.info(f"Found {doc_count} documents in metadata")
        
        # Check a sample document for metadata
        if doc_count > 0:
            sample_doc = metadata[0]
            logger.info(f"Sample document ID: {sample_doc.get('id')}")
            logger.info(f"Sample document metadata keys: {list(sample_doc.get('metadata', {}).keys())}")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
    
    # Load prompts to count documents with prompts
    try:
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        
        prompt_count = len(prompts)
        logger.info(f"Found {prompt_count} documents with prompts")
        
        # Check permission of the prompts file
        import stat
        file_stats = os.stat(prompts_path)
        permissions = stat.filemode(file_stats.st_mode)
        logger.info(f"Prompts file permissions: {permissions}")
        
        # If we have any prompts, check a sample
        if prompt_count > 0:
            sample_doc_id = list(prompts.keys())[0]
            sample_prompt = prompts[sample_doc_id]
            logger.info(f"Sample prompt document ID: {sample_doc_id}")
            logger.info(f"Sample prompt types: {list(sample_prompt.keys())}")
            
            # Check if this document is in the metadata
            doc_in_metadata = any(d.get('id') == sample_doc_id for d in metadata)
            logger.info(f"Document {sample_doc_id} exists in metadata: {doc_in_metadata}")
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        
        # If loading failed, try to check the file format
        try:
            with open(prompts_path, 'r') as f:
                content = f.read(500)  # Read first 500 chars
            logger.info(f"Prompts file starts with: {content[:100]}")
            
            if content.strip().startswith('{') and '"' in content:
                logger.info("File appears to be in JSON format")
            else:
                logger.error("File does not appear to be valid JSON")
        except Exception as e2:
            logger.error(f"Error reading prompts file content: {e2}")
    
    logger.info("Prompts storage check complete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check vector store prompts storage")
    parser.add_argument("--collection", "-c", default="default", help="Collection name to check")
    parser.add_argument("--dir", "-d", default="./vector_store", help="Vector store directory")
    args = parser.parse_args()
    
    asyncio.run(check_prompts_storage(args.collection, args.dir))
