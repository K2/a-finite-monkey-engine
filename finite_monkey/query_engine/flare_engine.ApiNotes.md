# FLARE Query Engine

## Overview

The FLARE (Forward-Looking Active REasoning) Query Engine is a specialized component that enhances query capabilities through step-by-step reasoning. It's particularly effective for complex code analysis tasks where traditional retrieval-based approaches might fall short.

## Key Components

### 1. Architecture

The FlareQueryEngine acts as a wrapper around LlamaIndex's FLAREInstructQueryEngine, providing:
- Async interface compatible with the rest of the finite-monkey system
- Error handling and consistent response formatting
- Integration with the Context system for accessing contract information
- Confidence scoring based on source relevance

### 2. Initialization Process

The engine follows a lazy initialization pattern:
- Created with minimal parameters initially
- Fully initialized when first query is made
- Can be pre-initialized by calling `initialize()` explicitly
- Supports both direct underlying engine injection or vector index construction

### 3. Query Process

When executing a query, the engine:
1. Ensures initialization is complete
2. Delegates to the LlamaIndex FLARE engine for reasoning
3. Extracts source information and calculates confidence
4. Formats the response with metadata
5. Handles any exceptions with informative error messages

### 4. Resource Management

The engine implements proper resource management via:
- Async initialization for non-blocking setup
- Shutdown method to release resources when no longer needed
- Careful handling of source node references

## Integration Points

### With Pipeline Factory

The engine is typically created and managed by the PipelineFactory, which:
1. Creates the engine with appropriate configuration
2. Maintains a singleton instance for reuse
3. Provides access to the engine for various pipeline components

### With Context System

The engine interacts with the Context by:
1. Reading contract and code information from context
2. Enhancing query results with relevant context state
3. Being accessible to other components via context

## Implementation Notes

### Confidence Scoring

The confidence score (0-1) is calculated based on the relevance of source documents:
- Higher source relevance scores result in higher confidence
- If no sources are found, confidence defaults to 0
- This provides a consistent metric across different query types

### Error Handling Strategy

The engine implements robust error handling:
1. Initialization errors are logged but allow retry on next query
2. Query execution errors are captured and returned as structured responses
3. All errors maintain the QueryResult contract for consistent handling

### Thread Safety

The engine manages concurrency through:
1. Async execution for non-blocking operation
2. Using asyncio.to_thread for CPU-bound LlamaIndex operations
3. Maintaining thread-safe state during parallel query execution

## ApiNotes.md for FlareQueryEngine

# FLARE Query Engine API Notes

## Overview

The FLARE (Forward-Looking Active REasoning) query engine implements a multi-step approach to answering complex queries:

1. **Decomposition**: Breaking complex queries into simpler sub-questions
2. **Sub-Query Answering**: Answering each sub-question independently 
3. **Synthesis**: Combining the sub-answers into a comprehensive final answer

## Guidance Integration

The engine now integrates with Microsoft's Guidance library to improve the reliability of query decomposition through the `GuidanceQuestionGenerator`. This eliminates parsing errors and improves the quality of sub-questions.

### Configuration Options

```python
# Enable Guidance (default if available)
engine = FlareQueryEngine(
    underlying_engine=base_engine,
    use_guidance=True
)

# Disable Guidance explicitly
engine = FlareQueryEngine(
    underlying_engine=base_engine,
    use_guidance=False
)
```

## Algorithm Details

The FLARE approach follows these steps:

1. **Query Analysis**: The engine analyzes the incoming query for complexity
2. **Decomposition**: Complex queries are broken down into sub-questions using the question generator
3. **Tool Assignment**: Each sub-question is assigned to a specific tool for answering
4. **Parallel Execution**: Sub-questions are answered in parallel for efficiency
5. **Answer Synthesis**: The sub-answers are combined using an LLM to create a coherent final response
6. **Confidence Scoring**: A confidence score is calculated based on the sub-question answers

## Error Handling

The engine implements robust error handling:

1. If query decomposition fails, falls back to direct query answering
2. If a sub-question fails, marks it as failed but continues with others
3. If synthesis fails, returns raw sub-answers with error notice

## Future Enhancements

The current implementation supports the basis for future enhancements:

1. Multi-iteration reasoning (planned)
2. Active learning from user feedback
3. Improved confidence scoring using sub-answer quality