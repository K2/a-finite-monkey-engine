#!/usr/bin/env python3
"""
Script to inspect prompts in the vector store documents.
"""
import asyncio
import json
from vector_store_util import SimpleVectorStore
from loguru import logger

async def inspect_prompts():
    """Inspect prompts in the vector store documents."""
    store = SimpleVectorStore(collection_name="gitset1")  # Use your actual collection name
    
    # Get all documents
    docs = store._documents
    
    # Check how many have prompts
    basic_prompts = [doc for doc in docs if 'prompt' in doc.get('metadata', {})]
    multi_prompts = [doc for doc in docs if 'multi_llm_prompts' in doc.get('metadata', {})]
    
    print(f"Total documents: {len(docs)}")
    print(f"Documents with basic prompts: {len(basic_prompts)}")
    print(f"Documents with multi-LLM prompts: {len(multi_prompts)}")
    
    # Print a sample of each type
    if basic_prompts:
        print("\n=== SAMPLE BASIC PROMPT ===")
        sample_doc = basic_prompts[0]
        print(f"Document ID: {sample_doc.get('id', 'unknown')}")
        print(f"Prompt: {sample_doc['metadata']['prompt']}")
    
    if multi_prompts:
        print("\n=== SAMPLE MULTI-LLM PROMPTS ===")
        sample_doc = multi_prompts[0]
        print(f"Document ID: {sample_doc.get('id', 'unknown')}")
        for prompt_type, prompt in sample_doc['metadata']['multi_llm_prompts'].items():
            # Skip list items like additional_security
            if isinstance(prompt, str):
                print(f"{prompt_type}: {prompt}")

if __name__ == "__main__":
    asyncio.run(inspect_prompts())