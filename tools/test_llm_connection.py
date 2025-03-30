#!/usr/bin/env python
"""
Utility script to test LLM connection and diagnose issues.
"""
import asyncio
import sys
import os
import json
from loguru import logger

# Add the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finite_monkey.utils.llm_diagnostics import diagnose_llm_issues
from finite_monkey.llm.llama_index_adapter import LlamaIndexAdapter
from finite_monkey.nodes_config import config

async def main():
    """Run diagnostics on the configured LLMs"""
    logger.info("Testing LLM connection")
    
    # Test with default model
    default_adapter = LlamaIndexAdapter(
        model_name=config.DEFAULT_MODEL,
        provider=config.DEFAULT_PROVIDER,
        timeout=config.REQUEST_TIMEOUT
    )
    
    # Run diagnostics
    default_diagnostics = await diagnose_llm_issues(default_adapter, detailed=True)
    
    # Print results
    print("\n=== DEFAULT MODEL DIAGNOSTICS ===")
    print(f"Model: {config.DEFAULT_MODEL}")
    print(f"Provider: {config.DEFAULT_PROVIDER}")
    print(f"Basic connectivity: {'✅ PASS' if default_diagnostics['basic_connectivity']['success'] else '❌ FAIL'}")
    
    if 'error' in default_diagnostics['basic_connectivity']:
        print(f"Error: {default_diagnostics['basic_connectivity']['error']}")
    
    # Print full JSON for detailed analysis
    with open("llm_diagnostics.json", "w") as f:
        json.dump(default_diagnostics, f, indent=2)
    
    print(f"Full diagnostics saved to llm_diagnostics.json")
    
    # Test vulnerability scan model if different
    if config.SCAN_MODEL != config.DEFAULT_MODEL or config.SCAN_MODEL_PROVIDER != config.DEFAULT_PROVIDER:
        scan_adapter = LlamaIndexAdapter(
            model_name=config.SCAN_MODEL,
            provider=config.SCAN_MODEL_PROVIDER,
            timeout=config.REQUEST_TIMEOUT
        )
        
        # Run diagnostics
        scan_diagnostics = await diagnose_llm_issues(scan_adapter)
        
        print("\n=== VULNERABILITY SCAN MODEL DIAGNOSTICS ===")
        print(f"Model: {config.SCAN_MODEL}")
        print(f"Provider: {config.SCAN_MODEL_PROVIDER}")
        print(f"Basic connectivity: {'✅ PASS' if scan_diagnostics['basic_connectivity']['success'] else '❌ FAIL'}")
        
        if 'error' in scan_diagnostics['basic_connectivity']:
            print(f"Error: {scan_diagnostics['basic_connectivity']['error']}")

if __name__ == "__main__":
    asyncio.run(main())
