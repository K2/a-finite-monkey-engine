#!/usr/bin/env python3
"""
Standalone test for the chunking utilities
"""

import asyncio
import os
import sys
from loguru import logger
from pathlib import Path

from finite_monkey.utils.chunking import AsyncContractChunker, async_chunk_solidity_file

async def main():
    # Get file path from command line or use example
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Check for example contracts
        example_dirs = [
            "./examples",
            "./tests/contracts",
            "./contracts",
        ]
        
        for example_dir in example_dirs:
            if os.path.exists(example_dir):
                files = [f for f in os.listdir(example_dir) if f.endswith(".sol")]
                if files:
                    file_path = os.path.join(example_dir, files[0])
                    break
        else:
            print("No example contract found. Please specify a Solidity file path.")
            return 1
    
    print(f"Testing chunking on: {file_path}")
    
    try:
        # Test direct file chunking
        chunks = await async_chunk_solidity_file(file_path, include_call_graph=False)
        print(f"Successfully chunked file into {len(chunks)} chunks:")
        
        for i, chunk in enumerate(chunks):
            chunk_type = chunk.get("chunk_type", "unknown")
            content_length = len(chunk.get("content", ""))
            print(f"  Chunk {i+1}: {chunk_type}, {content_length} chars")
            
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
