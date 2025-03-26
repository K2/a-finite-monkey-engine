"""
CLI integration module for direct LlamaIndex structured output with A Finite Monkey Engine.
"""

import argparse
import json
import logging
import sys
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

from .business_flow_analyzer import BusinessFlowAnalyzer
from .models import BusinessFlowData, SecurityAnalysisResult

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for CLI commands.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Structured output analyzer for A Finite Monkey Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Business flow analysis command
    flow_parser = subparsers.add_parser("flow", help="Analyze business flow in smart contracts")
    flow_parser.add_argument("--contract", "-c", type=str, required=True, help="Contract file path")
    flow_parser.add_argument("--model", "-m", type=str, help="Model identifier (provider:model_name)")
    flow_parser.add_argument("--output", "-o", type=str, help="Output file path")
    flow_parser.add_argument("--name", "-n", type=str, help="Contract name")
    
    # Security analysis command
    security_parser = subparsers.add_parser("security", help="Security analysis for smart contracts")
    security_parser.add_argument("--contract", "-c", type=str, required=True, help="Contract file path")
    security_parser.add_argument("--flow", "-f", type=str, help="Flow data JSON file path")
    security_parser.add_argument("--model", "-m", type=str, help="Model identifier (provider:model_name)")
    security_parser.add_argument("--output", "-o", type=str, help="Output file path")
    security_parser.add_argument("--name", "-n", type=str, help="Contract name")
    
    return parser

async def handle_flow_command(args) -> None:
    """
    Handle the business flow analysis command.
    
    Args:
        args: Command line arguments
    """
    # Load contract code
    with open(args.contract, 'r') as f:
        contract_code = f.read()
    
    # Get contract name from args or extract from file path
    contract_name = args.name
    if not contract_name:
        contract_name = Path(args.contract).stem
    
    # Create analyzer with specified model or default
    model = args.model or "openai:gpt-4o"
    analyzer = BusinessFlowAnalyzer(model_name=model)
    
    # Analyze flow
    flow_data = await analyzer.analyze_contract_flow(
        contract_code=contract_code,
        contract_name=contract_name
    )
    
    # Convert the Pydantic model to a dictionary
    flow_dict = flow_data.model_dump(
        exclude_none=True,
        exclude_defaults=False,
        mode='json'
    )
    
    # Print or save output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(flow_dict, f, indent=2, default=str)
        print(f"Flow analysis saved to {args.output}")
    else:
        print(json.dumps(flow_dict, indent=2, default=str))

async def handle_security_command(args) -> None:
    """
    Handle the security analysis command.
    
    Args:
        args: Command line arguments
    """
    # Load contract code
    with open(args.contract, 'r') as f:
        contract_code = f.read()
    
    # Get contract name from args or extract from file path
    contract_name = args.name
    if not contract_name:
        contract_name = Path(args.contract).stem
    
    # Load flow data if provided
    flow_data = None
    if args.flow and Path(args.flow).exists():
        with open(args.flow, 'r') as f:
            flow_dict = json.load(f)
            flow_data = BusinessFlowData.model_validate(flow_dict)
    
    # Create analyzer with specified model or default
    model = args.model or "openai:gpt-4o"
    analyzer = BusinessFlowAnalyzer(model_name=model)
    
    # Analyze security
    security_data = await analyzer.analyze_security(
        contract_code=contract_code, 
        flow_data=flow_data, 
        contract_name=contract_name
    )
    
    # Convert the Pydantic model to a dictionary
    security_dict = security_data.model_dump(
        exclude_none=True,
        exclude_defaults=False,
        mode='json'
    )
    
    # Print or save output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(security_dict, f, indent=2, default=str)
        print(f"Security analysis saved to {args.output}")
    else:
        print(json.dumps(security_dict, indent=2, default=str))

async def main_async():
    """
    Async main entry point for the CLI
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "flow":
        await handle_flow_command(args)
    elif args.command == "security":
        await handle_security_command(args)
    else:
        parser.print_help()
        return 1
    
    return 0

def main():
    """
    Main entry point for the CLI
    """
    return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main())
