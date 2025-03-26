"""
Example script for using the Researcher agent directly
via the IPython terminal in the web interface.

Usage:
1. Copy and paste this code into the IPython terminal
2. The script will analyze vulnerabilities in DeFi protocols
3. Results will be displayed in the terminal

This demonstrates how to work with individual agents directly.
"""

import asyncio
from pprint import pprint

# Get access to the researcher agent from the terminal namespace
agent = researcher  # This is already available in the terminal namespace

print("Researcher Agent Demonstration")
print("=" * 40)
print(f"Agent type: {type(agent).__name__}")
print(f"Model: {agent.model_name}")
print("\nRunning research query...\n")

# Request research on a specific DeFi security topic
query = """
Research the top 3 security vulnerabilities found in DeFi protocols 
in the past year, with examples of exploits and prevention strategies.
"""

# Execute the query
result = await agent.research(query)

print("\nResearch Results:")
print("-" * 40)

# Print the research findings
print(result.findings)

print("\nNow let's try a more specific query about smart contract vulnerabilities...\n")

# Query about reentrancy attacks
reentrancy_query = "Explain reentrancy vulnerabilities in DeFi contracts and how to prevent them"
reentrancy_result = await agent.research(reentrancy_query)

print("\nReentrancy Research:")
print("-" * 40)
print(reentrancy_result.findings)

print("\nResearch completed!")
print("\nTrying a code pattern query...\n")

# Ask about a specific code pattern
code_pattern_query = """
Identify secure patterns for implementing access control in 
Solidity smart contracts with examples.
"""

code_pattern_result = await agent.research(code_pattern_query)

print("\nSecure Access Control Patterns:")
print("-" * 40)
print(code_pattern_result.findings)