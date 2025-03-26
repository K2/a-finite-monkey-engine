"""
Example script for using the Validator agent to analyze smart contracts
via the IPython terminal in the web interface.

Usage:
1. Copy and paste this code into the IPython terminal
2. The script will validate the example Vault.sol contract
3. Results will be displayed in the terminal

This demonstrates how to work with the validator agent directly.
"""

import os
from pathlib import Path

# Get the validator agent from the terminal namespace
agent = validator  # This is already available in the terminal namespace

print("Validator Agent Demonstration")
print("=" * 40)
print(f"Agent type: {type(agent).__name__}")
print(f"Model: {agent.model_name}")

# Get the example contract path
examples_dir = Path(os.getcwd()) / "examples"
vault_path = str(examples_dir / "Vault.sol")

print(f"\nValidating contract at: {vault_path}")

# Read the contract code
with open(vault_path, "r") as f:
    code = f.read()

print("\nContract Code Preview:")
print("-" * 40)
print(code[:500] + "...\n")  # Just show the first 500 characters

print("Running basic validation...\n")

# Run validation on the contract
result = await agent.validate_contract(
    file_path=vault_path,
    code=code,
    specific_concerns=["reentrancy", "access control", "integer overflow"]
)

print("\nValidation Results:")
print("-" * 40)
print(f"Issues found: {len(result.issues)}")

for i, issue in enumerate(result.issues, 1):
    print(f"\n{i}. {issue.title} (Severity: {issue.severity})")
    print(f"   Location: {issue.location}")
    print(f"   Description: {issue.description}")
    print(f"   Recommendation: {issue.recommendation}")

print("\nNow let's run a counterfactual analysis...\n")

# Run a counterfactual analysis on what would happen if a specific vulnerability existed
counterfactual_result = await agent.generate_counterfactual(
    file_path=vault_path,
    code=code,
    hypothetical_scenario="What if the withdraw function doesn't check the user's balance before withdrawing?"
)

print("\nCounterfactual Analysis:")
print("-" * 40)
print(f"Scenario: {counterfactual_result.scenario}")
print(f"Impact: {counterfactual_result.impact}")
print(f"Likelihood: {counterfactual_result.likelihood}")
print(f"Suggested Fix: {counterfactual_result.suggested_fix}")

print("\nValidation completed!")