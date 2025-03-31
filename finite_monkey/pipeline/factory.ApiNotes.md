# Pipeline Factory API Notes

## BusinessFlowExtractor Stage

The `create_business_flow_extractor` method creates a pipeline stage that extracts business logic flows from smart contracts. It identifies and maps out logical patterns, state transitions, and interactions between different contract components.

### Configuration

The BusinessFlowExtractor can be configured with:

- **flow_types**: List of flow types to extract. Default flow types include:
  - `token_transfer`: Flows related to token transfers and management
  - `access_control`: Permission and authorization patterns
  - `state_transition`: Contract state changes and transitions
  - `external_call`: Interactions with external contracts
  - `fund_management`: ETH and fund management operations

- **LLM Integration**: When an LLM adapter is available, it's used to enhance extraction with more nuanced understanding of code semantics.

### Context Transformations

The business flow extractor updates the context with:

- `context.business_flows`: Dictionary mapping contract names to lists of extracted flows
- Each flow contains:
  - `name`: Descriptive name of the flow
  - `description`: Detailed description of what the flow does
  - `steps`: Ordered list of steps in the flow
  - `actors`: Entities involved in the flow
  - `preconditions`: Required conditions for the flow to execute
  - `postconditions`: Resulting state after flow execution

### Error Handling

Errors during business flow extraction are captured in `context.errors` with the stage identifier "business_flow_extraction". The extraction process is designed to continue with best-effort extraction even if certain contracts or functions cannot be fully analyzed.

## VulnerabilityScanner Stage

The `create_vulnerability_scanner` method creates a pipeline stage for identifying security vulnerabilities in smart contracts. It integrates with the VulnerabilityScanner component from the analyzers module.

### Configuration

The VulnerabilityScanner can operate in two modes:

1. **LLM-enhanced mode**: When an LLM adapter is available, the scanner uses it to perform semantic analysis of the code, understanding the business context and detecting complex vulnerabilities that may not be identifiable through pattern matching alone.

2. **Pattern-based mode**: When no LLM adapter is available, the scanner falls back to a pattern-based approach using predefined vulnerability patterns.

### Context Transformations

The vulnerability scanner updates the context with:

- `context.vulnerabilities`: Dictionary mapping contract names to lists of detected vulnerabilities
- Each vulnerability contains:
  - `name`: Name or type of the vulnerability
  - `description`: Detailed description of the vulnerability
  - `severity`: The severity level (Critical, High, Medium, Low, Informational)
  - `location`: Where in the code the vulnerability was found
  - `confidence`: Confidence level in the finding
  - `recommendation`: Suggested fix or mitigation strategy

### Error Handling

The stage implements robust error handling:
1. Exceptions during scanner creation result in a fallback no-op stage
2. Exceptions during scanning are caught, logged, and added to context errors
3. The stage always returns a context object, even in error cases

### Integration with Pipeline

The vulnerability scanning stage typically runs after business flow analysis, as it can leverage the business flow information to provide more contextually relevant security analysis.

## DataFlowAnalyzer Stage

The `create_dataflow_analyzer` method creates a pipeline stage for analyzing data flows within smart contracts. It identifies sources of data, how data moves through the system, sinks where data is consumed, and potential vulnerabilities related to data handling.

### Configuration

The DataFlowAnalyzer can operate in two modes:

1. **LLM-enhanced mode**: When an LLM adapter is available, the analyzer uses it to perform semantic analysis of the code, understanding complex data relationships and identifying subtle data flow issues.

2. **Pattern-based mode**: When no LLM adapter is available, the analyzer falls back to a pattern-based approach using predefined data flow patterns.

### Context Transformations

The data flow analyzer updates the context with:

- `context.dataflows`: Dictionary mapping contract names to lists of identified data flows
- Each data flow contains:
  - `source`: Where the data originates (function parameters, external calls, etc.)
  - `target`: Where the data is used or consumed
  - `type`: Type of data being passed
  - `impact`: Potential impact or risk of this data flow
  - `tainted`: Whether the data may be controlled by untrusted parties
  - `validation`: Whether the data undergoes validation

### Error Handling

The stage implements robust error handling:
1. Exceptions during analyzer creation result in a fallback no-op stage
2. Exceptions during analysis are caught, logged, and added to context errors
3. The stage always returns a context object, even in error cases

### Integration with Pipeline

The data flow analysis stage typically runs after vulnerability scanning, as it can provide additional context about how vulnerabilities might be exploited through data manipulation.

## CognitiveBiasAnalyzer Stage

The `create_cognitive_bias_analyzer` method creates a pipeline stage for analyzing cognitive biases in smart contract design and implementation. It helps identify assumptions, biases, and blind spots in contract logic that could introduce security vulnerabilities or unexpected behaviors.

### Configuration

The CognitiveBiasAnalyzer can operate in two modes:

1. **LLM-enhanced mode**: When an LLM adapter is available, the analyzer uses it to perform sophisticated analysis of code patterns that might indicate cognitive biases like optimism bias, anchoring, or confirmation bias.

2. **Basic mode**: When no LLM adapter is available, the analyzer falls back to a pattern-based approach using predefined cognitive bias patterns.

### Context Transformations

The cognitive bias analyzer updates the context with:

- `context.cognitive_biases`: Dictionary mapping contract names to lists of identified biases
- Each bias contains:
  - `type`: The type or name of the cognitive bias
  - `description`: Explanation of how the bias manifests in the code
  - `impact`: Potential impact of this bias on contract behavior
  - `location`: Where in the code the bias was detected
  - `recommendation`: Suggested approaches to mitigate the bias

### Integration with Pipeline

The cognitive bias analysis typically runs after vulnerability scanning and data flow analysis, as it can build on the insights from these earlier stages to provide more contextual bias detection.

## DocumentationAnalyzer Stage

The `create_documentation_analyzer` method creates a pipeline stage for analyzing documentation quality in smart contracts. It evaluates NatSpec comments, inline documentation, and overall documentation completeness relative to the code complexity.

### Configuration

The DocumentationAnalyzer can operate in two modes:

1. **LLM-enhanced mode**: When an LLM adapter is available, the analyzer uses it to perform semantic analysis of code and documentation, detecting inconsistencies and gaps between implementation and documentation.

2. **Basic mode**: When no LLM adapter is available, the analyzer falls back to simple metrics like documentation coverage percentages and presence of key documentation elements.

### Context Transformations

The documentation analyzer updates the context with:

- `context.documentation_issues`: Dictionary mapping contract names to lists of identified documentation issues
- Each issue contains:
  - `type`: The type of documentation issue (missing, incomplete, inconsistent)
  - `description`: Description of the documentation problem
  - `severity`: How severe the documentation gap is
  - `location`: Where in the code the issue occurs
  - `recommendation`: Suggested improvements for documentation

### Integration with Pipeline

The documentation analysis typically runs near the end of the pipeline, as it benefits from having context about contract functionality, security considerations, and business logic that's established by earlier stages.

## DocumentationAnalyzer Stage

The `create_documentation_analyzer` method creates a pipeline stage for analyzing consistency between code and documentation in smart contracts. Its primary purpose is to identify discrepancies where the actual implementation diverges from what is documented.

### Primary Focus Areas

1. **Code-Comment Consistency**: Identifies where comments describe behavior that doesn't match the actual code implementation
2. **NatSpec Accuracy**: Verifies that NatSpec comments (especially @param, @return, @notice) accurately reflect function parameters, return values, and behavior
3. **Behavioral Mismatch**: Detects cases where documented invariants, guarantees, or requirements in comments don't align with code logic
4. **Security Claim Verification**: Validates whether security-related claims in documentation (e.g., "this function is reentrancy-safe") are actually implemented 

### Configuration

The DocumentationAnalyzer can operate in two modes:

1. **LLM-enhanced mode**: When an LLM adapter is available, the analyzer uses semantic understanding to identify subtle inconsistencies between implementation and documentation.

2. **Basic mode**: When no LLM adapter is available, the analyzer falls back to pattern matching that looks for common inconsistency indicators.

### Context Transformations

The documentation analyzer updates the context with:

- `context.documentation_issues`: Dictionary mapping contract names to lists of identified documentation inconsistencies
- Each issue contains:
  - `type`: The type of inconsistency (parameter mismatch, behavior mismatch, etc.)
  - `description`: Description of the inconsistency
  - `severity`: How critical the inconsistency is
  - `location`: Where in the code the inconsistency occurs
  - `recommendation`: Suggested approaches to resolve the inconsistency

### Integration with Pipeline

The documentation analysis typically runs near the end of the pipeline, as it needs context from other analyzers to properly evaluate the consistency between implementation and documentation.

## VectorMatchAnalyzer Stage

The `create_vector_match_analyzer` method creates a pipeline stage for identifying semantically similar code patterns using vector embeddings. This analyzer employs sophisticated similarity matching with decay functions to optimize the groupings of similar code fragments.

### Database Configuration

The VectorMatchAnalyzer uses pgvector with PostgreSQL for efficient vector similarity search:

- **Database Name**: postgres_vector (derived from ASYNC_DB_URL configuration)
- **Table Name**: issues
- **Vector Column**: embedding
- **Connection**: Uses psycopg2 to connect to the PostgreSQL database

It transforms the standard database connection string from `config.ASYNC_DB_URL` by replacing "/postgres" with "/postgres_vector" to point to the vector-specific database.

### Core Functionality

1. **Vector Similarity with Decay**: Uses multiple decay functions to find the most efficient set of matches, optimizing for the best possible grouping of similar code patterns.

2. **Source Matching**: When a similarity match is found, it identifies the original source text and uses vector dbscan to analyze it against known vulnerability patterns from sources like GitHub issues.

3. **Semantic Grouping**: Groups semantically similar code patterns that may share the same underlying vulnerability or implementation pattern.

### Configuration

The VectorMatchAnalyzer can be configured with:

- `similarity_threshold`: Controls how similar code must be to be considered a match
- `use_decay_functions`: Whether to use adaptive decay functions for optimal grouping
- `match_sources`: Sources of known vulnerability patterns to match against
- `llm_adapter`: Optional LLM adapter for enhanced semantic understanding

### Context Transformations

The vector match analyzer updates the context with:

- `context.vector_matches`: Dictionary mapping contract names to lists of detected matches
- Each match contains:
  - `source_fragment`: Original code fragment being matched
  - `matched_fragments`: Similar code fragments found in the codebase
  - `similarity_score`: How similar the matches are
  - `potential_issue`: Inferred potential issue based on similarity matching
  - `reference_sources`: References to known vulnerability patterns that were matched

## ThreatDetector Stage

The `create_threat_detector` method creates a pipeline stage for detecting potential security threats based on code patterns and vector matches. This detector uses the results from vector matching to generate targeted LLM prompts that probe each function for specific vulnerability characteristics.

### Core Functionality

1. **Targeted LLM Prompting**: Uses vector match results to generate specific prompts for LLMs to analyze functions for characteristics of identified flaws.

2. **Vulnerability Characterization**: Determines why particular code patterns represent vulnerabilities (bad math, exposed functionality to calling contracts, etc.)

3. **Exploit Path Analysis**: For identified threats, analyzes possible exploitation paths to assess impact and likelihood.

### Configuration

The ThreatDetector can be configured with:

- `generate_mitigation`: Whether to generate mitigation suggestions for detected threats
- `analyze_exploit_paths`: Whether to analyze and document potential exploit paths
- `threat_categories`: Categories of threats to detect (arithmetic flaws, access control issues, etc.)
- `llm_adapter`: Optional LLM adapter for enhanced threat analysis

### Context Transformations

The threat detector updates the context with:

- `context.threats`: Dictionary mapping contract names to lists of detected threats
- Each threat contains:
  - `category`: Category of the threat (e.g., "arithmetic_flaw", "access_control")
  - `description`: Detailed description of the threat
  - `severity`: Estimated severity (Critical, High, Medium, Low)
  - `location`: Where in the code the threat was found
  - `exploit_path`: Potential exploit path if analysis is enabled
  - `mitigation`: Suggested mitigation if generation is enabled

### Integration with Pipeline

The threat detector optimally runs after vector match analysis, as it can leverage the identified similar code patterns to perform more targeted threat detection. It typically comes before other high-level analyses like cognitive bias or documentation analysis.

## LLM Adapter Initialization

The `PipelineFactory` needs to initialize the appropriate LLM adapter based on configuration settings:

1. **Provider Selection**: The factory supports multiple LLM providers:
   - OpenAI (gpt-3.5-turbo, gpt-4, etc.)
   - Anthropic (claude-2, etc.)
   - LlamaIndex (supports various backend models)
   - Ollama (local models like llama2, mistral, etc.)
   - Mock (for testing)

2. **Initialization Process**:
   ```python
   # Main initialization method
   def _initialize_llm_adapter(self):
       # Determine provider and call appropriate initializer
       
   # Provider-specific initializers
   def _initialize_openai_adapter(self):
       # Set up OpenAI client with API key, model, temperature
   
   def _initialize_anthropic_adapter(self):
       # Set up Anthropic client
   
   # etc.
   ```

3. **Error Handling**:
   - Falls back to mock adapter if initialization fails
   - Logs detailed error information

4. **Configuration Parameters**:
   - `LLM_PROVIDER`: Which provider to use (openai, anthropic, etc.)
   - `LLM_MODEL`: Model name to use (gpt-4, claude-2, etc.)
   - `TEMPERATURE`: Randomness parameter for generation
   - Provider-specific keys (OPENAI_API_KEY, etc.)

## Vector Store Integration

The `engine_dir` attribute specifies where vector indices are stored and loaded from. This is used by:

1. `_initialize_existing_query_engine()`: To load existing vector indices or create new ones
2. `get_query_engine()`: To provide query capabilities over the vector data

The directory path can be customized via the `VECTOR_STORE_DIR` configuration setting, with a default of "./vector_store" if not specified.
