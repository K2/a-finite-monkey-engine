# ThreatDetector with Vector DB and TreeSitter Integration

## Architecture Overview

The `ThreatDetectorVDB` component combines the power of vector similarity search with TreeSitter's accurate code parsing to identify security vulnerabilities in code:

1. **TreeSitter Integration**:
   - Uses the existing TreeSitter parser (`finite_monkey.parsers.sitter.CodeParser`)
   - Defers all code parsing to the TreeSitter implementation
   - Falls back to basic extraction methods only when TreeSitter is unavailable

2. **Vector Store Usage**:
   - Leverages the vector database to find similar patterns to known vulnerabilities
   - Uses adaptive threshold mechanism to identify relevant matches
   - Extracts associated prompts to enhance threat assessments

3. **Centralized Pipeline Component**:
   - Implements the standard PipelineComponent interface
   - Integrates seamlessly with the existing finite-monkey-engine pipeline
   - Maintains separation of concerns between parsing and threat detection

## Implementation Details

### TreeSitter Integration

The component does not implement any TreeSitter parsing logic directly, instead:
- Imports and instantiates `finite_monkey.parsers.sitter.CodeParser`
- Calls its parsing methods to extract code blocks
- Maintains the TreeSitter parser's output format and structure

### Input Handling

The component accepts multiple input formats through the PipelineContext:
1. Direct code strings or blocks
2. File paths pointing to source code files
3. AST structures (which TreeSitter can process)

### Threat Detection Process

1. **Code Extraction**: Uses TreeSitter to parse code into semantic blocks
2. **Pattern Matching**: Queries the vector database for similar vulnerability patterns
3. **Threat Assessment**: Analyzes matches against confidence thresholds
4. **Enhanced Analysis**: Uses prompt templates with LLM for detailed explanations

## Usage Notes

### Required Components

- The TreeSitter parser (`finite_monkey.parsers.sitter.CodeParser`)
- A populated vector store containing threat patterns
- Optional prompt templates for enhanced analysis

### Pipeline Integration

Example pipeline configuration:
```python
from finite_monkey.pipelines.threat_detector import ThreatDetectorVDB

# Create the pipeline with the threat detector
pipeline = Pipeline([
    SourceCodeLoader(),
    ThreatDetectorVDB(
        vector_store_dir="./vector_store",
        collection_name="threats",
        similarity_threshold=0.65
    ),
    ThreatReporter()
])

# Use the pipeline
results = pipeline.process({"file_paths": ["app.py", "utils.js"]})
```

### Error Handling

The component gracefully handles TreeSitter unavailability by:
1. Logging the import failure
2. Falling back to basic code extraction methods
3. Continuing with threat detection using available code blocks
