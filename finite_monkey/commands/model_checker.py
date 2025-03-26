"""
Command-line utility for checking model configurations
"""

import asyncio
import argparse
import json
from loguru import logger

from ..pipeline.model_verifier import ModelVerifier
from ..nodes_config import get_model_status, SUPPORTED_MODELS

async def check_models(args):
    """Check model configurations and verify their availability"""
    # Print current configuration
    logger.info("Current model configuration:")
    model_status = get_model_status()
    
    # Pretty print configuration
    for name, config in model_status.items():
        logger.info(f"  {name.upper()}: {config['model']} ({config['provider']})")
    
    # Print supported models
    if args.list:
        logger.info("\nSupported models:")
        for provider, models in SUPPORTED_MODELS.items():
            logger.info(f"  {provider}:")
            for model in models:
                logger.info(f"    - {model}")
    
    # Verify models if requested
    if args.verify:
        logger.info("\nVerifying model availability...")
        verifier = ModelVerifier()
        results = await verifier.verify_all_models()
        
        # Print results
        for model_name, success in results.items():
            if success:
                logger.info(f"  ✅ {model_name}: Available")
            else:
                logger.error(f"  ❌ {model_name}: Not available")
        
        # Write results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Verification results written to {args.output}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Check model configurations")
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List supported models"
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify model availability"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for verification results"
    )
    
    args = parser.parse_args()
    asyncio.run(check_models(args))

if __name__ == "__main__":
    main()
