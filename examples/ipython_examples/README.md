# IPython Terminal Examples

This directory contains example scripts for demonstrating the capabilities of the Finite Monkey Engine via the IPython terminal in the web interface.

## How to Use These Examples

1. Start the Finite Monkey web interface:
   ```
   python -m finite_monkey.web.app
   ```

2. Navigate to the IPython Terminal page at `http://localhost:8000/terminal/main`

3. Copy and paste the contents of any example script into the terminal

4. Execute the script to see the demonstration

## Available Examples

### 1. Simple Analysis (`simple_analysis.py`)
- Basic analysis of a single smart contract
- Demonstrates core analysis workflow
- Uses the orchestrator to run a complete audit

### 2. Researcher Agent (`researcher_agent.py`)
- Demonstrates using the Researcher agent directly
- Performs research on DeFi security topics
- Shows how to query for specific security patterns

### 3. Validator Agent (`validator_agent.py`)
- Shows how to use the Validator agent for contract validation
- Runs targeted validation on a contract
- Demonstrates counterfactual analysis

### 4. Cognitive Bias Analysis (`cognitive_bias_analysis.py`)
- Uses the Cognitive Bias Analyzer 
- Identifies potential cognitive biases in contract design
- Provides recommendations to mitigate these biases

### 5. End-to-End Analysis (`end_to_end_analysis.py`)
- Comprehensive multi-stage analysis of a DeFi project
- Combines multiple agents and analysis types
- Generates a complete report

## Tips for Customization

- Modify the file paths to analyze your own smart contracts
- Adjust the queries and concerns to focus on specific vulnerabilities
- Try combining different analysis techniques for your specific needs

## Notes

- All examples use async/await syntax since they run in an asyncio environment
- The scripts use the agents already initialized in the terminal environment
- Some scripts may take a few minutes to complete depending on contract complexity