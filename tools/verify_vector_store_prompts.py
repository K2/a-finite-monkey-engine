#!/usr/bin/env python3
"""
Utility to verify that prompts are properly removed from vector store nodes
but correctly saved to separate files.
"""
import os
import json
import asyncio
from pathlib import Path
from loguru import logger
from typing import Dict, List, Any, Optional

async def verify_prompt_separation(collection_name="default", storage_dir="./vector_store"):
    """Verify that prompts are properly separated from vector store nodes."""
    logger.info(f"Verifying prompt separation for collection: {collection_name}")
    
    collection_dir = os.path.join(storage_dir, collection_name)
    
    # Check if docstore.json exists (contains the nodes)
    docstore_path = os.path.join(collection_dir, "docstore.json")
    if not os.path.exists(docstore_path):
        logger.error(f"Docstore file not found: {docstore_path}")
        return
    
    # Check if document_prompts.json exists (contains the prompts)
    prompts_path = os.path.join(collection_dir, "document_prompts.json")
    if not os.path.exists(prompts_path):
        logger.error(f"Prompts file not found: {prompts_path}")
        return
    
    # Load docstore to check node metadata
    prompt_field_count = 0
    other_field_count = 0
    total_nodes = 0
    
    try:
        with open(docstore_path, 'r') as f:
            docstore = json.load(f)
            
        # Check nodes for prompt fields
        for node_id, node_data in docstore.get("docstore", {}).get("docs", {}).items():
            total_nodes += 1
            
            # Check if node has metadata
            if "metadata" in node_data:
                metadata = node_data["metadata"]
                
                # Count prompt-related fields
                for field in ['prompt', 'multi_llm_prompts', 'invariant_analysis', 
                             'general_flaw_pattern', 'quick_checks', 'api_interactions',
                             'call_flow', 'vulnerable_paths']:
                    if field in metadata:
                        prompt_field_count += 1
                        logger.warning(f"Node {node_id} contains prompt field: {field}")
                
                # Count other fields
                other_field_count += len(metadata) - prompt_field_count
        
        # Load prompts file
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        
        prompt_count = len(prompts)
        prompt_types = {}
        
        # Count types of prompts
        for doc_id, doc_prompts in prompts.items():
            for prompt_type in doc_prompts:
                if prompt_type not in prompt_types:
                    prompt_types[prompt_type] = 0
                prompt_types[prompt_type] += 1
        
        # Report results
        logger.info(f"Analyzed {total_nodes} nodes in the vector store")
        logger.info(f"Found {prompt_field_count} prompt-related fields in node metadata (should be 0)")
        logger.info(f"Found {other_field_count} other metadata fields in nodes")
        logger.info(f"Found {prompt_count} documents with prompts in document_prompts.json")
        for prompt_type, count in prompt_types.items():
            logger.info(f"  - {prompt_type}: {count} prompts")
        
        # Final status
        if prompt_field_count == 0:
            logger.info("✅ Prompts are properly separated from vector store nodes")
        else:
            logger.warning("❌ Some prompts are still in vector store nodes!")
            
    except Exception as e:
        logger.error(f"Error verifying prompt separation: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify vector store prompt separation")
    parser.add_argument("--collection", "-c", default="default", help="Collection name to check")
    parser.add_argument("--dir", "-d", default="./vector_store", help="Vector store directory")
    args = parser.parse_args()
    
    asyncio.run(verify_prompt_separation(args.collection, args.dir))
