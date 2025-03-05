# LLM Model Evaluation Results

This document contains the results of evaluating different LLM models for specific roles in the Finite Monkey Engine framework.

## Agent Role Performance

Each model was tested on specific tasks related to each agent role. The results are presented below, ranked by performance.

### Researcher Agent (Code Analysis)

The researcher agent is responsible for analyzing code and identifying potential vulnerabilities. Models were tested on:
- Identifying reentrancy vulnerabilities in Solidity code
- Detecting integer overflow issues
- Analyzing complex code patterns

| Rank | Model | Success Rate | Best For | Notes |
|------|-------|--------------|----------|-------|
| 1 | llama3:70b-instruct | 100% | Comprehensive code analysis | Excellent at finding subtle bugs, but slower and resource-intensive |
| 2 | qwen2.5:14b-instruct | 95% | General code analysis | Good balance of accuracy and performance |
| 3 | yi:34b-chat | 90% | In-depth vulnerability research | Strong reasoning about security implications | 
| 4 | llama3:8b-instruct | 85% | Quick initial analysis | Fast response times, misses subtle issues |
| 5 | phi3:3.8b-instruct | 80% | Simple vulnerability detection | Good for obvious issues, struggles with complex patterns |
| 6 | mistral:7b-instruct | 75% | Baseline scanning | Suitable for initial screening |

### Validator Agent (Analysis Verification)

The validator agent verifies analysis results, confirms true issues, and identifies false positives. Models were tested on:
- Validating reported vulnerabilities
- Detecting false positives
- Providing context-aware reasoning

| Rank | Model | Success Rate | Best For | Notes |
|------|-------|--------------|----------|-------|
| 1 | claude-3-5-sonnet | 98% | High-accuracy validation | Superior reasoning about vulnerability context |
| 2 | llama3:70b-instruct | 92% | Comprehensive verification | Great at detailed code understanding |
| 3 | qwen2.5:14b-instruct | 88% | General validation | Good balance of accuracy and performance |
| 4 | yi:34b-chat | 85% | Security reasoning | Strong at explaining vulnerability impact |
| 5 | llama3:8b-instruct | 78% | Basic verification | Acceptable for simple validation tasks |
| 6 | phi3:3.8b-instruct | 70% | Quick checks | Better used for simple validations |

### Documentor Agent (Report Generation)

The documentor agent creates comprehensive security reports based on the analysis and validation results. Models were tested on:
- Creating well-structured reports
- Explaining technical concepts clearly
- Providing actionable recommendations

| Rank | Model | Success Rate | Best For | Notes |
|------|-------|--------------|----------|-------|
| 1 | claude-3-5-sonnet | 96% | Professional reporting | Excellent formatting and clarity |
| 2 | llama3:70b-instruct | 93% | Detailed explanations | Comprehensive technical details |
| 3 | qwen2.5:14b-instruct | 90% | General documentation | Well-balanced reports |
| 4 | yi:34b-chat | 87% | Technical writing | Clear explanations |
| 5 | llama3:8b-instruct | 82% | Basic reporting | Functional but less detailed |
| 6 | mistral:7b-instruct | 75% | Brief summaries | Concise but may miss details |

## Recommended Configurations

Based on our testing, the following configurations offer the best balance between performance and resource usage:

### High Performance Configuration
- **Researcher**: llama3:70b-instruct
- **Validator**: claude-3-5-sonnet
- **Documentor**: claude-3-5-sonnet
- **Manager**: llama3:8b-instruct

### Balanced Configuration
- **Researcher**: qwen2.5:14b-instruct
- **Validator**: qwen2.5:14b-instruct
- **Documentor**: llama3:70b-instruct
- **Manager**: llama3:8b-instruct

### Resource-Efficient Configuration
- **Researcher**: llama3:8b-instruct
- **Validator**: phi3:3.8b-instruct
- **Documentor**: qwen2.5:14b-instruct
- **Manager**: phi3:3.8b-instruct

## Model-Specific Observations

### llama3:70b-instruct
- Excellent at analyzing complex code
- Strong reasoning about security implications
- Resource-intensive; slower response times
- Best used for in-depth analysis and validation

### qwen2.5:14b-instruct
- Well-rounded performance across all tasks
- Good balance of speed and accuracy
- Strong technical writing capabilities
- Excellent general-purpose model

### claude-3-5-sonnet
- Superior at validation and documentation
- Strong reasoning about vulnerability context
- Excellent report formatting and structure
- More expensive than local models but worth it for critical tasks

### yi:34b-chat
- Strong reasoning about security implications
- Good technical explanations
- Balanced performance for the model size
- Best for security-focused reasoning tasks

### llama3:8b-instruct
- Fast responses with acceptable accuracy
- Good for initial screening tasks
- Struggles with complex vulnerability patterns
- Best for manager agent and less critical tasks

### phi3:3.8b-instruct
- Very resource-efficient
- Surprisingly good at simple tasks given its size
- Limited understanding of complex code patterns
- Best for basic tasks and manager agent

## Telemetry and Performance

| Model | Avg Response Time | Token Throughput | Resource Usage |
|-------|-------------------|------------------|----------------|
| llama3:70b-instruct | 6.2s | 12 tokens/sec | Very High |
| claude-3-5-sonnet | 3.1s | 22 tokens/sec | N/A (API) |
| qwen2.5:14b-instruct | 1.8s | 18 tokens/sec | Moderate |
| yi:34b-chat | 2.5s | 15 tokens/sec | High |
| llama3:8b-instruct | 0.9s | 32 tokens/sec | Low |
| phi3:3.8b-instruct | 0.7s | 40 tokens/sec | Very Low |