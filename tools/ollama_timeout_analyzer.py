#!/usr/bin/env python3
"""
Tool to analyze and debug Ollama timeout issues in the vector store prompt generation.
This script can be used to identify patterns in timeouts and suggest optimal timeout settings.
"""
import os
import sys
import json
import asyncio
import time
from datetime import datetime
from pathlib import Path
from loguru import logger
import argparse

# Add parent directory to path so we can import the modules
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import application config
try:
    from finite_monkey.nodes_config import config as app_config
except ImportError:
    logger.warning("Could not import application config, using defaults")
    app_config = type('DefaultConfig', (), {
        'OLLAMA_URL': "http://localhost:11434",
        'OLLAMA_TIMEOUT': 900,
        'PROMPT_MODEL': "gemma:2b"
    })

async def test_ollama_connection(url=None, retries=3, timeout=60):
    """Test Ollama connection with the specified timeout."""
    import aiohttp
    
    url = url or getattr(app_config, "OLLAMA_URL", "http://localhost:11434")
    logger.info(f"Testing Ollama connection to {url} with {timeout}s timeout...")
    
    start_time = time.time()
    
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                # First test a simple ping to the Ollama API
                async with session.get(f"{url}/api/tags", timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        model_count = len(data.get('models', []))
                        elapsed = time.time() - start_time
                        logger.info(f"Successfully connected to Ollama. Found {model_count} models. Time: {elapsed:.2f}s")
                        return True, elapsed, model_count
                    else:
                        logger.warning(f"Ollama returned status {response.status}")
        except asyncio.TimeoutError:
            logger.warning(f"Connection to Ollama timed out after {timeout}s (attempt {attempt}/{retries})")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
        
        if attempt < retries:
            logger.info(f"Retrying in 5 seconds...")
            await asyncio.sleep(5)
    
    elapsed = time.time() - start_time
    logger.error(f"Failed to connect to Ollama after {retries} attempts. Total time: {elapsed:.2f}s")
    return False, elapsed, 0

async def benchmark_ollama_response_times(url=None, model=None, sample_size=5, timeout=120):
    """Benchmark Ollama response times for a given model."""
    import aiohttp
    import json
    
    url = url or getattr(app_config, "OLLAMA_URL", "http://localhost:11434")
    model = model or getattr(app_config, "PROMPT_MODEL", "gemma:2b")
    
    logger.info(f"Benchmarking Ollama response times for model {model} with {timeout}s timeout...")
    
    # Create a sample prompt that's representative of the workload
    sample_prompt = """Analyze the following code snippet and provide a concise summary:
    
    ```python
    def process_data(data):
        results = []
        for item in data:
            if item.get('valid'):
                results.append(item['value'] * 2)
        return results
    ```
    
    Provide a brief description of what this function does and any potential issues.
    """
    
    response_times = []
    success_count = 0
    failure_count = 0
    
    for i in range(1, sample_size + 1):
        logger.info(f"Running benchmark test {i}/{sample_size}...")
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": sample_prompt,
                    "stream": False
                }
                
                async with session.post(
                    f"{url}/api/generate",
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        elapsed = time.time() - start_time
                        response_times.append(elapsed)
                        success_count += 1
                        logger.info(f"Test {i}: Response received in {elapsed:.2f}s")
                    else:
                        failure_count += 1
                        logger.warning(f"Test {i}: Ollama returned status {response.status}")
        except asyncio.TimeoutError:
            failure_count += 1
            elapsed = time.time() - start_time
            logger.warning(f"Test {i}: Request timed out after {elapsed:.2f}s")
        except Exception as e:
            failure_count += 1
            elapsed = time.time() - start_time
            logger.error(f"Test {i}: Error during request: {e} (after {elapsed:.2f}s)")
        
        # Add a short delay between requests
        await asyncio.sleep(2)
    
    # Calculate statistics
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        p95 = sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) >= 20 else max_time
        
        logger.info(f"Benchmark results for {model}:")
        logger.info(f"Success rate: {success_count}/{sample_size} ({success_count/sample_size*100:.1f}%)")
        logger.info(f"Average response time: {avg_time:.2f}s")
        logger.info(f"Min/Max response time: {min_time:.2f}s / {max_time:.2f}s")
        logger.info(f"P95 response time: {p95:.2f}s")
        
        # Suggest a good timeout value (2x the P95 or max time)
        suggested_timeout = max(p95 * 2, max_time * 1.5)
        logger.info(f"Suggested timeout setting: {suggested_timeout:.0f} seconds")
        
        return {
            "success_rate": success_count/sample_size,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "p95": p95,
            "suggested_timeout": suggested_timeout
        }
    else:
        logger.warning("No successful responses to calculate statistics")
        return None

async def main():
    """Main entry point for the Ollama timeout analyzer."""
    parser = argparse.ArgumentParser(description="Analyze and debug Ollama timeout issues")
    parser.add_argument("--url", "-u", help="Ollama URL", default=getattr(app_config, "OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--model", "-m", help="Ollama model to test", default=getattr(app_config, "PROMPT_MODEL", "gemma:2b"))
    parser.add_argument("--timeout", "-t", type=int, help="Timeout in seconds for tests", default=getattr(app_config, "OLLAMA_TIMEOUT", 120))
    parser.add_argument("--samples", "-s", type=int, help="Number of samples for benchmarking", default=5)
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark tests")
    
    args = parser.parse_args()
    
    # Test basic connection
    success, connection_time, model_count = await test_ollama_connection(args.url, timeout=30)
    
    if not success:
        logger.error("Could not connect to Ollama server. Please check the server is running and accessible.")
        return 1
    
    if args.benchmark:
        # Run benchmark tests
        results = await benchmark_ollama_response_times(
            url=args.url,
            model=args.model,
            sample_size=args.samples,
            timeout=args.timeout
        )
        
        if results:
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"ollama_benchmark_{args.model}_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "model": args.model,
                    "url": args.url,
                    "timestamp": timestamp,
                    "results": results
                }, f, indent=2)
            logger.info(f"Saved benchmark results to {results_file}")
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())
