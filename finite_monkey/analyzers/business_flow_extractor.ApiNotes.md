# Business Flow Extractor API Notes

## Overview
The BusinessFlowExtractor performs semantic analysis of smart contracts to identify and extract business flow patterns. It uses an LLM to analyze contract code and detect key workflows, processes, and interactions.

## Flow Types
The extractor can be configured with specific flow types to focus the analysis. These flow types include:
- token_transfer - Flows related to token transfers and token economics
- access_control - Flows related to permissions and authorization
- state_transition - Flows that change contract state through defined sequences
- external_call - Flows that interact with external contracts
- fund_management - Flows that handle ETH or other asset movements

Additional specialized flow types can be provided during initialization.

## Analysis Levels

The `BusinessFlowExtractor` supports three different analysis levels:

1. **contract**: Analyze business flows at the contract level only
2. **function**: Analyze business flows at the function level only
3. **method_first**: First analyze at function level, then at contract level (recommended)

The "method_first" approach is recommended as it:
1. Captures fine-grained flows at the function level
2. Also identifies broader flows that span multiple functions
3. Provides more comprehensive analysis with both detailed and overview perspectives

## Choosing Between Levels

- Use contract-level for initial overview analysis
- Use function-level for more detailed security auditing
- Function-level is better when contracts have many functions (>5)
- Contract-level is better when functions are tightly coupled

## Derived vs. Acquired Analysis Taxonomy

The BusinessFlowExtractor implements a clear conceptual distinction between two primary approaches to vulnerability analysis:

### Derived Flows
- **Definition**: Flows that are algorithmically derived through code path traversal and static analysis
- **Source**: Direct analysis of smart contract code
- **Characteristics**: 
  - More technical and precise about code paths
  - Higher confidence in function call sequences
  - Less context about developer intent or business logic
  - Can identify vulnerabilities that aren't discussed in issues

### Acquired Flows
- **Definition**: Flows that are acquired from human descriptions in issues, pull requests, or discussions
- **Source**: Natural language descriptions of issues, bugs or vulnerabilities
- **Characteristics**:
  - Richer in contextual understanding and intent
  - May contain insights not obvious from code alone
  - Often includes specific attack scenarios
  - Highlights concerns raised by security researchers

### Hybrid Flows
- **Definition**: Combined analysis leveraging both derived and acquired approaches
- **Characteristics**:
  - Most comprehensive vulnerability detection
  - Technical precision from code analysis
  - Contextual understanding from human descriptions
  - Highest confidence when both sources align

## Implementation Details

### Flow Source Detection
The system automatically determines which analysis approach to use based on available inputs:
- If only code is available: Uses derived approach
- If only issue text is available: Uses acquired approach
- If both are available: Uses hybrid approach

### Merging Logic
When merging derived and acquired flows:
1. Start with the derived flow as the base (for technical accuracy)
2. Add unique flow functions from acquired analysis
3. Add unique attack surfaces from acquired analysis
4. Combine notes and analysis sections to retain all context
5. Use the higher confidence score from either analysis

### Metadata Tracking
Every analyzed flow includes metadata indicating its source:
- `flow_source`: One of "derived", "acquired", or "hybrid"
- This metadata is preserved throughout the analysis pipeline and reporting

## Example Usage

```python
# Contract-level analysis (default)
extractor = BusinessFlowExtractor(
    flow_types=["token_transfer", "access_control"],
    analysis_level="contract"
)

# Function-level analysis
extractor = BusinessFlowExtractor(
    flow_types=["token_transfer", "access_control"],
    analysis_level="function"
)

# Method-first analysis
extractor = BusinessFlowExtractor(
    flow_types=["token_transfer", "access_control"],
    analysis_level="method_first"
)
```

## LLM Integration
The extractor uses a structured LLM approach to generate consistent business flow data:
1. The contract code is sent to the LLM with specific instructions
2. The LLM responds with a structured JSON format matching the BusinessFlowAnalysisResult model
3. The response is parsed into flow objects and stored in the context

## Debugging Tips
If business flow extraction isn't working:
1. Verify the LLM adapter is properly initialized
2. Check that the ANALYSIS_MODEL is correctly configured in config
3. Examine the timeout settings if large contracts are failing
4. Look for detailed error logs during the _analyze_contract method execution

## Usage Example
```python
# Initialize with custom flow types
flow_extractor = BusinessFlowExtractor(
    flow_types=["token_transfer", "access_control", "state_transition"]
)

# Set a custom LLM adapter
flow_extractor.set_llm_adapter(my_llm_adapter)

# Process a context
context = await flow_extractor.process(context)

# Access results
for contract_name, flows in context.business_flows.items():
    print(f"Contract {contract_name} has {len(flows)} flows")
```

## Key Methods

### set_llm_adapter(llm_adapter)

Sets or updates the LLM adapter used by the extractor after initialization. This is useful when the extractor is created without an adapter and needs to be configured later, or when the adapter needs to be changed during runtime.

```python
# Example usage
flow_extractor = BusinessFlowExtractor()
flow_extractor.set_llm_adapter(llm_adapter)
```

### analyze_function_flow(contract_name, function_name, contract_code)

Analyzes the business flow of a specific function within a contract. Returns a structured BusinessFlow object with:
- Function call sequence
- Variables involved in the flow
- Potential attack surfaces
- Confidence score
- Additional notes and analysis

### process(context)

Processes all functions in the provided context, extracting business flows and updating the context with analysis results. Uses a semaphore to limit concurrent LLM calls and avoid resource exhaustion.

## Integration with PipelineFactory

The PipelineFactory creates a BusinessFlowExtractor instance and configures it with:
1. Predefined flow types to analyze
2. LLM adapter if available
3. Error handling for resilient processing

The flow extractor integrates with the pipeline by:
- Taking and returning a Context object
- Updating context.business_flows with analysis results
- Adding any errors to context.errors with appropriate metadata

## Troubleshooting

If the business flow extractor continues to complete in zero seconds:

1. Verify the config values in `nodes_config.py` for ANALYSIS_MODEL are correctly set
2. Check if the LlamaIndexAdapter is being properly initialized
3. Add more detailed logging to track the execution path
4. Consider adding a delay or dummy response for testing

## LLM Request Timeout Handling

The LLM adapter in this project requires special handling for timeout settings. Instead of setting `request_timeout` directly on the LLM object, pass the timeout as a parameter to the `achat()` method:

```python
# INCORRECT - Will cause AttributeError
self.llm_adapter.llm.request_timeout = config.REQUEST_TIMEOUT
response = await structured_llm.achat(messages)

# CORRECT - Pass timeout as a parameter
timeout = getattr(config, "REQUEST_TIMEOUT", 60)
response = await structured_llm.achat(
    messages,
    timeout=timeout
)
```

This is because the `llm` property on the adapter is a method object that doesn't have a `request_timeout` attribute.

## Structured Output Format

The BusinessFlowExtractor uses structured output from the LLM to ensure consistent data format. The expected response structure is:

```json
{
  "flows": [
    {
      "name": "Token Transfer",
      "description": "Transfer tokens between accounts",
      "steps": ["Check balances", "Deduct from sender", "Add to recipient"],
      "functions": ["transfer", "transferFrom"],
      "actors": ["sender", "recipient"]
    }
  ],
  "contract_summary": "ERC20 token implementation with standard transfer functionality"
}
```

This structure is defined in the `BusinessFlowAnalysisResult` model.

## Flow Type Configuration

The extractor can be initialized with a list of flow types to identify. These are used to guide the LLM's analysis and ensure it focuses on the most relevant business logic patterns in the contract.

## LLM Integration Pattern

The BusinessFlowExtractor uses a direct approach with the LLM adapter to get structured output. Instead of using `as_structured_llm()` (which may not be available in all LLM implementations), it:

1. Constructs a prompt that explicitly requests JSON output in a specific format
2. Uses the LLM adapter's llm method directly to generate a response
3. Parses the response string as JSON using Python's standard json module

This approach is more versatile and compatible with different LLM adapters.

## Error Handling for LLM Responses

The implementation includes robust error handling:

1. JSON parsing errors are caught and logged
2. Timeout errors are handled gracefully
3. The raw response is logged when parsing fails, to help with debugging

## Debugging LLM Integration

If the LLM responses aren't being properly structured as JSON:

1. Check the system prompt to ensure it emphasizes JSON formatting
2. Review the debug logs for the raw LLM responses
3. Try simplifying the requested JSON structure
4. Consider using a regex-based parser as a fallback for partially-formatted responses

## Flow Types Configuration

The analyzer can be configured with custom flow types to focus the LLM's analysis. These flow types are included directly in the prompt to guide the LLM toward identifying specific types of smart contract business flows.

## Guidance Integration

The Business Flow Extractor now supports Microsoft's Guidance library for structured output generation. This significantly improves the reliability of JSON parsing by forcing the LLM to adhere to the expected schema.

### Benefits of Guidance

- Eliminates JSON parsing errors completely
- Improves output consistency, especially with weaker LLMs
- Allows LLMs to focus on content rather than syntax
- Generates properly structured Pydantic objects directly

### Implementation Strategy

The implementation provides a hybrid approach with automatic fallback:

1. **Primary Method**: Guidance-based structured output
2. **Fallback 1**: Direct LLM with JSON instructions if Guidance unavailable
3. **Fallback 2**: Empty result if both methods fail

This ensures robustness when dependencies aren't available or when specific LLMs have issues.

### Usage Options

```python
# Use with Guidance (default if available)
extractor = BusinessFlowExtractor(use_guidance=True)

# Force standard approach even if Guidance is available
extractor = BusinessFlowExtractor(use_guidance=False)
```

## Handlebars Templates

Guidance uses handlebars-style templates which are different from Python format strings:

- Handlebars: `{{variable_name}}`
- Python: `{variable_name}`

The implementation handles this conversion automatically.

## Dependencies

To use the Guidance approach, additional packages must be installed:

```bash
pip install guidance llama-index-llms
```

The system gracefully degrades to the standard approach if these dependencies are not available.

## Analysis Levels

The extractor supports both contract-level and function-level analysis as previously implemented.

## Response Mapping Issues

When processing LLM responses, several structure patterns need to be handled:

1. **Single flow object**: LLM returns a single flow object with fields like `name`, `description`, etc.
   ```json
   {
     "name": "Token Transfer Flow",
     "description": "...",
     "steps": ["..."],
     "functions": ["..."]
   }
   ```

2. **Flows array**: LLM returns an object with a `flows` array
   ```json
   {
     "flows": [
       {
         "name": "Flow 1",
         "description": "..."
       }
     ]
   }
   ```

The wrapper handles these variations by:
- Converting single flow objects to a flows array
- Handling the flows array directly
- Providing fallbacks for other structures

## Timeout Configuration

Timeout is set from `nodes_config.REQUEST_TIMEOUT` and passed to:

1. The `LlamaIndexAdapter` during initialization
2. The LLM instance creation in `_create_llm`

The default timeout is 300 seconds (5 minutes) if not specified in config.

## LLM Adapter Interface Handling

The `BusinessFlowExtractor` needs to interact with different LLM adapters that might have inconsistent interfaces. The key challenge is handling the variations in method names and signatures across different adapter implementations:

### Common Interface Variations

1. **Async vs Sync Methods**:
   - Some adapters provide `achat()` (async) methods
   - Others only provide `chat()` (sync) methods that need to be run in an executor

2. **Method Naming**:
   - `achat()` vs `async_chat()` vs `chat_async()`
   - `chat()` vs `complete()` vs `generate()`

3. **Parameter Naming**:
   - `messages` vs `chat_messages` vs `prompt`
   - `max_tokens` vs `max_new_tokens` vs `token_limit`

### Robust Resolution Strategy

The `_get_llm_response` method implements a cascading fallback mechanism:

1. First tries the standard async methods (`achat`, `acomplete`)
2. Then checks for methods on the underlying LLM object (`llm.achat`)
3. Falls back to running synchronous methods in an executor if available
4. As a last resort, uses direct API calls with the `DirectOllamaClient`

This approach ensures maximum compatibility with different adapter implementations without requiring changes to the adapter classes themselves.

## DirectOllamaClient as Fallback

The `DirectOllamaClient` bypasses all the LlamaIndex and adapter abstractions to make direct HTTP calls to the Ollama API. This serves as both:

1. A diagnostic tool to check if Ollama itself is working correctly
2. A robust fallback when the standard adapter methods fail

When all other methods fail, the extractor falls back to:
1. Extracting the model name from the adapter or config
2. Formatting the messages into a prompt
3. Making a direct API call to Ollama
4. Parsing the response

This ensures we can still get results even when there are compatibility issues with the adapter interfaces.

## Interface Recommendation

For future LLM adapters, implementing the following methods would ensure compatibility:

```python
async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
    """Async method for chat-based completions"""
    
def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
    """Sync method for chat-based completions"""
```

The extractor's fallback mechanism will handle cases where these aren't available, but implementing them directly would improve performance and reliability.

## LLM Parameter Handling

### Parameter Name Inconsistencies

Different LLM adapters use different parameter names for the same concepts:

1. **Token Limit Parameters**:
   - `max_tokens`: Used by OpenAI and most LlamaIndex adapters
   - `max_new_tokens`: Used by some Hugging Face and transformers-based adapters
   - `token_limit`: Used by some custom adapters
   - `max_length`: Used by transformers and some other adapters

2. **Temperature Parameters**:
   - `temperature`: Standard across most adapters
   - `temp`: Occasionally used as an abbreviation
   - Some adapters don't support temperature at all

3. **Chat Message Parameters**:
   - `messages`: Standard for most chat interfaces
   - `chat_messages`: Used by some custom adapters
   - Some adapters expect a different format entirely

### Robust Parameter Strategy

The `_get_llm_response` method implements a fallback strategy to handle these inconsistencies:

1. **Defensive Parameter Passing**:
   - Try to use the most common parameter names first
   - Fall back to more specific ones if the common ones fail
   - When all else fails, try passing no parameters at all

2. **Error Handling**:
   - Catch `TypeError` exceptions which indicate parameter mismatches
   - Try alternative method calls with different parameter combinations
   - Log detailed information about what was attempted

3. **Configuration-Based Parameters**:
   - Get parameter values from `nodes_config.py` instead of hardcoding them
   - This makes it easy to standardize parameters across the application
   - Examples: `MAX_TOKENS`, `TEMPERATURE`

### Method Availability Checks

The code handles different method naming conventions:

1. First checks for async methods (`achat`, `acomplete`)
2. Then falls back to synchronous methods in an executor
3. As a last resort, tries the direct Ollama API

This approach ensures maximum compatibility with different adapter implementations and LLM backends.

## Output Structure Handling

The business flow extractor needs to handle multiple possible output structures from LLMs:

1. Expected structure with `business_flows` attribute:
```json
{
  "business_flows": [
    {
      "name": "Flow name",
      "description": "Flow description",
      "steps": ["Step 1", "Step 2"]
    }
  ]
}
```

2. Common alternative with `flows` attribute:
```json
{
  "flows": [
    {
      "name": "Flow name",
      "description": "Flow description",
      "steps": ["Step 1", "Step 2"]
    }
  ]
}
```

The implementation should gracefully handle both structures and convert them to BusinessFlow objects.

## BusinessFlowAnalysisResult Structure

The `BusinessFlowAnalysisResult` Pydantic model should have a field compatible with both naming conventions:

```python
class BusinessFlowAnalysisResult(BaseModel):
    # Accept either business_flows or flows
    business_flows: Optional[List[BusinessFlow]] = None
    flows: Optional[List[BusinessFlow]] = None
    
    # Custom validator to ensure at least one is populated
    @validator('business_flows', 'flows', pre=True)
    def ensure_flows(cls, v, values):
        # If no business_flows and no flows, initialize empty list
        if v is None and 'business_flows' not in values and 'flows' not in values:
            return []
        return v
        
    # Property to get flows regardless of which field is used
    @property
    def get_flows(self) -> List[BusinessFlow]:
        return self.business_flows or self.flows or []
```

Always check first for `business_flows`, then `flows`, and provide proper error handling for malformed results.

## BusinessFlow Model Requirements

The `BusinessFlow` model has the following required fields that must be included when creating instances:

- `name`: Name of the business flow
- `description`: Description of the business flow
- `steps`: List of steps in the business flow 
- `contract_name`: Name of the contract containing the flow
- `functions`: List of functions involved in the flow (REQUIRED - cannot be omitted)
- `inputs`: List of inputs to the flow
- `outputs`: List of outputs from the flow

The `functions` field is required and must be included even if empty (as an empty list `[]`).

## Handling Object Creation

The helper method `_create_business_flow` ensures all required fields are properly set:

```python
def _create_business_flow(self, flow_data, contract_name, is_dict=False):
    """Create a BusinessFlow object with proper field handling"""
    # See implementation for details
```

This centralized method handles both object and dictionary inputs and ensures all required fields are included, even if they're missing in the input data.
