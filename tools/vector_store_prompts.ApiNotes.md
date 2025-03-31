# Vector Store Prompts System

## Purpose and Architecture

The `PromptGenerator` class is designed to enhance the vector store by generating specialized prompts for different types of documents and security analyses. It works in tandem with the `SimpleVectorStore` to ensure that documents stored in the vector database have rich, contextual prompts that make retrieval and analysis more effective.

## Key Components

### Template System

The prompt generator uses a flexible template system with several specialized templates:

1. **General Template**: For general documents without specific type characteristics
2. **Code Template**: Specifically for code snippets and programming-related content
3. **Security Template**: For security-related content such as vulnerabilities and threats
4. **Multi-LLM Templates**: Specialized prompts optimized for different LLM architectures (Gemma, Llama, Mistral, etc.)

Templates are loaded from files in the `templates` directory when available, or defaults are used and saved for future reference.

### LLM Integration

The system supports two primary modes of LLM integration:

1. **Ollama Integration**: Uses the Ollama API to generate prompts with local models
2. **Fallback Generation**: When LLM access is unavailable, uses heuristic rules to generate basic prompts

### Security Analysis Features

For security-related documents, the generator provides several specialized analyses:

1. **Invariant Analysis**: Identifies invariant properties that should be maintained to prevent vulnerabilities
2. **General Pattern Extraction**: Abstracts specific vulnerabilities into general classes and patterns
3. **Quick Check Generation**: Creates simple patterns and rules to detect similar vulnerabilities
4. **API Interaction Analysis**: Identifies APIs and their potential security implications
5. **Call Flow Extraction**: Maps execution flows that might lead to vulnerabilities
6. **Vulnerable Path Identification**: Pinpoints specific execution paths that can trigger vulnerabilities

## Implementation Details

### Prompt Generation Process

The prompt generation follows this general process:

1. **Template Selection**: Choose the appropriate template based on document metadata and content
2. **Template Filling**: Fill the template with document text and context
3. **LLM Generation**: Submit to Ollama for prompt generation
4. **Response Processing**: Parse and clean the LLM's response
5. **Fallback Generation**: If LLM generation fails, use rules-based fallbacks

### Response Parsing

The system includes specialized parsers for different types of responses:

- **JSON parsing**: For structured data like API interactions
- **List extraction**: For extracting items from bullet points or numbered lists
- **Flow parsing**: For extracting execution paths with arrow notation

### Integration with Vector Store

This class is designed to be used by `SimpleVectorStore` during document ingestion. When documents are added to the vector store:

1. The vector store calls `generate_prompt()` to create basic prompts
2. For security documents, additional specialized analyses are generated
3. These prompts and analyses are stored separately from the vector embeddings
4. During retrieval, the prompts are reattached to provide rich context

## Usage Patterns

### Basic Prompt Generation

```python
# Create a prompt generator
generator = PromptGenerator()

# Generate a prompt for a document
prompt = await generator.generate_prompt(document)
```

### Security Analysis

```python
# Generate invariant analysis for a security vulnerability
invariant = await generator.generate_invariant_analysis(document)

# Extract quick checks for vulnerability detection
checks = await generator.extract_quick_checks(document)

# Analyze API interactions
api_interactions = await generator.extract_api_interactions(document)

# Extract execution flows
flows = await generator.extract_call_flow(document)

# Identify vulnerable paths
vulnerable_paths = await generator.identify_vulnerable_paths(document, flows)
```

### Multi-LLM Support

```python
# Generate prompts optimized for different LLMs
multi_prompts = await generator.generate_multi_llm_prompts(document)
```

## Extension Points

The system can be extended in several ways:

1. **New Template Types**: Add new templates for specialized document types
2. **Additional Analysis Methods**: Implement new security analysis techniques
3. **Alternative LLM Backends**: Replace Ollama with other LLM providers
4. **Enhanced Parsing**: Improve response parsing for specific output formats

# Vector Store Prompts API Notes

## Configurable Parameters

The PromptGenerator class now supports the following configurable parameters:

### LLM Generation Controls

- `temperature` (float, default: 0.2): Controls the randomness of the LLM outputs
  - Lower values (0.0-0.3): More focused, deterministic responses
  - Higher values (0.7-1.0): More creative, varied responses

- `max_tokens` (int, default: 4096): Maximum response length
  - Increased from previous 2048 limit to allow for more comprehensive prompts
  - Especially important for detailed code analysis and security evaluations

### Provider Settings

- `provider_type` (str, default: "ollama"): Type of LLM provider to use
  - "ollama": Local Ollama instance
  - "openai": OpenAI's API (partial implementation)
  - "hosted": Generic hosted provider (future implementation)

- `ollama_timeout` (float, default: 900.0): Specific timeout for Ollama requests
  - Used directly in the httpx.AsyncClient configuration

- `timeout` (float, default: None): General timeout for all operations
  - Defaults to ollama_timeout if not specified

## Configuration Sources

These settings can be configured from multiple sources, in order of precedence:

1. **Direct instance parameters**: When creating a PromptGenerator directly
2. **nodes_config settings**: Automatically read from the app_config object
3. **Environment variables**: For runtime configuration
4. **Default values**: As specified in the constructor

## Integration with SimpleVectorStore

The SimpleVectorStore class uses introspection to detect available parameters:

```python
import inspect
prompt_gen_params = inspect.signature(PromptGenerator.__init__).parameters
prompt_gen_kwargs = {
    # Base parameters...
}

# Add parameters if supported by PromptGenerator
if 'temperature' in prompt_gen_params:
    prompt_gen_kwargs['temperature'] = getattr(app_config, "PROMPT_TEMPERATURE", 0.2)
if 'max_tokens' in prompt_gen_params:
    prompt_gen_kwargs['max_tokens'] = getattr(app_config, "PROMPT_MAX_TOKENS", 4096)
```

## Provider Flexibility

The architecture now supports different provider backends:

1. **Ollama**: For local deployment (fully implemented)
2. **OpenAI**: For hosted API access (partial implementation)
3. **Generic hosted**: Abstract base for other providers (future implementation)

Each provider implementation handles:
- Authentication
- Request formatting
- Response parsing
- Error handling

## URL Configuration

Provider URLs are now designed to be configurable:
- `ollama_url`: For Ollama API endpoint (default: "http://localhost:11434")
- Future: `openai_url`, `anthropic_url`, etc. for other hosted providers

URLs can be configured via:
1. Direct parameter
2. nodes_config setting
3. Environment variable (e.g., OLLAMA_URL)

## Timeout Configuration

### Previous Implementation Issues

The original code had a hardcoded timeout of 30.0 seconds in the `_query_ollama` method:

```python
async with httpx.AsyncClient(timeout=30.0) as client:  # Reduced timeout from 60s to 30s
```

This hardcoded value was overriding any timeout settings passed from other parts of the application, causing LLM requests to time out for larger documents or when using more complex models.

### Current Implementation

The current implementation properly handles timeout configuration:

1. The `PromptGenerator` constructor now accepts two timeout parameters:
   - `ollama_timeout`: Specifically for Ollama API requests (default: 900 seconds)
   - `timeout`: A general timeout for all operations (defaults to ollama_timeout if not specified)

2. These timeout values are then used in the `_query_ollama` method:
   ```python
   async with httpx.AsyncClient(timeout=self.ollama_timeout) as client:
   ```

### Configuration Guidelines

- For large models (>7B parameters), use timeouts of 900-1800 seconds (15-30 minutes)
- For smaller models (<7B parameters), timeouts of 300-600 seconds (5-10 minutes) are usually sufficient
- If processing large batches of documents, consider using longer timeouts
- The timeout should be proportional to the complexity of the prompts and the size of the model

## Interacting with PromptGenerator

The `PromptGenerator` class is used by `SimpleVectorStore` to generate prompts for documents added to the vector store. The `SimpleVectorStore` will automatically pass the appropriate timeout settings when creating an instance of `PromptGenerator`.

For standalone usage, make sure to provide the appropriate timeout values:

```python
# Example of standalone usage with proper timeout configuration
generator = PromptGenerator(
    generate_prompts=True,
    use_ollama_for_prompts=True,
    prompt_model="gemma:7b",
    ollama_url="http://localhost:11434",
    multi_llm_prompts=False,
    ollama_timeout=900.0  # 15 minutes
)
```

## Related Components

This component connects with:
- `vector_store_util.py`: Uses PromptGenerator to create semantic prompts for vector search
- Ollama API: External dependency for LLM inference