# Security Pattern Extractor

## Purpose and Architecture

The `SecurityPatternExtractor` is designed to enhance security vulnerability analysis by extracting patterns from code and generating specialized security-focused prompts. It serves as a bridge between the vector database system and LLM-powered security analysis.

### Key Components

1. **Pattern Extraction Engine**:
   - Extracts security-relevant patterns from code
   - Identifies API interactions and misuse patterns
   - Recognizes execution flows that might indicate vulnerabilities

2. **Prompt Generation System**:
   - Creates specialized prompts for different aspects of security analysis
   - Generates invariant analysis prompts that focus on underlying security principles
   - Produces general vulnerability pattern prompts that abstract specific flaws into classes

3. **Call Flow Analysis**:
   - Extracts call flow information from vulnerability descriptions 
   - Identifies potentially vulnerable execution paths
   - Maps entry points to vulnerable code patterns

## Integration Points

The `SecurityPatternExtractor` integrates with:

1. **Vector Store**: Uses `SimpleVectorStore` for saving and retrieving security patterns
2. **Ollama/LLM Backend**: Leverages LLMs to generate enhanced security prompts
3. **Threat Detector Pipeline**: Provides data for threat analysis in the pipeline

## Implementation Details

### Template System

The class uses a template-based system with these key templates:
- `invariant_template`: For identifying invariant properties that are violated in vulnerable code
- `pattern_template`: For extracting general vulnerability patterns
- `quick_check_template`: For creating simple checks that can identify similar vulnerabilities
- `api_template`: For analyzing API interactions and potential misuse
- `flow_template`: For extracting call flows from vulnerability descriptions
- `paths_template`: For identifying vulnerable execution paths

### LLM Integration

The system is designed to work with:
- Ollama API for local LLM execution
- Fallback pattern extraction when LLMs aren't available
- Structured response parsing to convert LLM outputs to structured data

### Response Processing

Specialized parsers for different response types:
- JSON response parsing for structured data
- List extraction for quick check patterns
- Flow extraction for execution paths

## Usage Patterns

### Pattern Extraction Flow

1. **Input**: Document containing vulnerable code or description
2. **Processing**: 
   - Generate invariant analysis
   - Extract general vulnerability pattern
   - Identify quick checks
   - Map API interactions
   - Extract call flows
3. **Output**: Enriched security metadata for vector store

### Command Line Interface

The CLI supports three main modes:
- `process`: Process a single document
- `batch`: Batch process multiple documents
- `generate`: Generate patterns for new documents

## Extension Points

The system is designed to be extended in these ways:
- Add new template types for specialized security analyses
- Integrate with different LLM backends
- Add new response parsers for different output formats
- Enhance fallback mechanisms for offline operation
