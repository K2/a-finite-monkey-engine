#!/usr/bin/env python3
"""
A minimal test script for the Claude adapter
"""

import os
import asyncio
import argparse
from finite_monkey.adapters.claude import Claude

async def main():
    """
    Main entry point for testing the Claude adapter
    """
    parser = argparse.ArgumentParser(
        description="Test the Claude adapter",
    )
    
    parser.add_argument(
        "--api-key",
        help="Claude API key (defaults to CLAUDE_API_KEY environment variable)",
    )
    
    parser.add_argument(
        "--model",
        default="claude-3-5-sonnet",
        help="Claude model to use (default: claude-3-5-sonnet)",
    )
    
    parser.add_argument(
        "--prompt",
        default="Tell me about smart contract security in 3 sentences.",
        help="Prompt to send to Claude",
    )
    
    args = parser.parse_args()
    
    # Initialize Claude adapter
    api_key = args.api_key or os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        print("Error: Claude API key not provided and CLAUDE_API_KEY environment variable not set")
        return 1
    
    try:
        print(f"Initializing Claude adapter with model {args.model}...")
        claude = Claude(api_key=api_key, model=args.model)
        
        print(f"Sending prompt to Claude: '{args.prompt}'")
        response = await claude.acomplete(prompt=args.prompt)
        
        print("\nResponse from Claude:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        
        # Clean up
        await claude.close()
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exitcode = asyncio.run(main())
    exit(exitcode)