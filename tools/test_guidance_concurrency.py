#!/usr/bin/env python
"""
Test script to diagnose concurrency issues with Guidance and Ollama.
"""
import asyncio
import sys
import os
from loguru import logger

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finite_monkey.utils.guidance_version_utils import (
    create_guidance_program, 
    GUIDANCE_AVAILABLE,
    OLLAMA_SEMAPHORE
)
from finite_monkey.nodes_config import config
from pydantic import BaseModel
from typing import List, Optional

# Simple model for testing
class TestResult(BaseModel):
    message: str
    items: Optional[List[str]] = None

async def test_concurrent_calls(contract_count: int = 5, concurrency_limit: int = 2):
    """Test how guidance handles concurrent calls"""
    # Ensure we're using Ollama
    if config.DEFAULT_PROVIDER.lower() != "ollama":
        print(f"This test is designed for Ollama, current provider is {config.DEFAULT_PROVIDER}")
        return
    
    # Check if guidance is available
    if not GUIDANCE_AVAILABLE:
        print("Guidance is not available. Install it with 'pip install guidance'")
        return
    
    print(f"Testing {contract_count} concurrent calls with semaphore limit {concurrency_limit}")
    
    # Set the semaphore limit
    global OLLAMA_SEMAPHORE
    OLLAMA_SEMAPHORE = asyncio.Semaphore(concurrency_limit)
    
    # Simple prompt template
    prompt_template = """
You are a helpful assistant.

Analyze the following code:
