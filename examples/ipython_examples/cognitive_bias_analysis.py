"""
Example script for using the Cognitive Bias Analyzer on smart contracts
via the IPython terminal in the web interface.

Usage:
1. Copy and paste this code into the IPython terminal
2. The script will analyze potential cognitive biases in the contract design
3. Results will be displayed in the terminal

This demonstrates the cognitive bias analysis capabilities.
"""

import os
import asyncio
from pathlib import Path

print("Cognitive Bias Analysis Demonstration")
print("=" * 50)

# Get the example contract path
examples_dir = Path(os.getcwd()) / "examples"
vault_path = str(examples_dir / "Vault.sol")

print(f"Analyzing contract at: {vault_path}")

# Read the contract code
with open(vault_path, "r") as f:
    code = f.read()

print("\nContract Code Preview:")
print("-" * 40)
print(code[:500] + "...\n")  # Just show the first 500 characters

# Import cognitive bias analyzer
from finite_monkey.agents.cognitive_bias_analyzer import CognitiveBiasAnalyzer

# Create a cognitive bias analyzer
analyzer = CognitiveBiasAnalyzer(
    model_name=config.WORKFLOW_MODEL,  # Use the configured model
    base_dir=os.getcwd()
)

print("Running cognitive bias analysis...\n")

# Run analysis on the contract
result = await analyzer.analyze_cognitive_biases(
    file_path=vault_path,
    code=code
)

print("\nCognitive Bias Analysis Results:")
print("=" * 50)
print(f"Number of cognitive biases detected: {len(result.biases)}")

# Display detected biases
for i, bias in enumerate(result.biases, 1):
    print(f"\n{i}. {bias.name}")
    print(f"   Type: {bias.type}")
    print(f"   Description: {bias.description}")
    print(f"   Evidence in Code: {bias.evidence}")
    print(f"   Impact: {bias.impact}")
    print(f"   Recommendations: {bias.recommendations}")

# Obtain an overall assessment
print("\nOverall Cognitive Bias Assessment:")
print("-" * 40)
print(result.summary)

print("\nRecommendations to mitigate cognitive biases:")
print("-" * 40)
for i, recommendation in enumerate(result.recommendations, 1):
    print(f"{i}. {recommendation}")

print("\nCognitive Bias Analysis completed!")