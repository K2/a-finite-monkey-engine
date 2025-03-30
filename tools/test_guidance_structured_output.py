#!/usr/bin/env python
"""
Tool to test Guidance's structured output capabilities.
"""
import sys
import os
import asyncio
import argparse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from loguru import logger

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finite_monkey.nodes_config import config
from finite_monkey.utils.guidance_version_utils import (
    create_guidance_program, 
    GUIDANCE_AVAILABLE,
    GUIDANCE_VERSION
)

# Define a simple test model
class TestFlow(BaseModel):
    name: str = Field(..., description="Name of the flow")
    description: str = Field(..., description="Description of what the flow does")
    steps: List[str] = Field(..., description="Steps in the flow")

class TestResult(BaseModel):
    flows: List[TestFlow] = Field(..., description="List of flows")
    summary: str = Field(..., description="Summary of the analysis")

async def main():
    parser = argparse.ArgumentParser(description="Test Guidance structured output")
    parser.add_argument("--model", default=None, help="Model to test (default: from config)")
    parser.add_argument("--provider", default=None, help="Provider to use (default: from config)")
    parser.add_argument("--prompt", default="Analyze this simple code snippet: function hello() { return 'world'; }", 
                       help="Prompt to use for testing")
    
    args = parser.parse_args()
    
    # Use configured values if not specified
    model = args.model or config.ANALYSIS_MODEL
    provider = args.provider or config.ANALYSIS_MODEL_PROVIDER
    
    print(f"=== Guidance Structured Output Test ===")
    print(f"Guidance Available: {GUIDANCE_AVAILABLE}")
    print(f"Guidance Version: {GUIDANCE_VERSION}")
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    
    if not GUIDANCE_AVAILABLE:
        print("\n❌ Guidance is not available. Please install it with: pip install guidance")
        return
    
    # Create a simple test prompt with schema
    prompt_template = f"""
You are a code analyzer that identifies key workflows in code.

Code to analyze:
