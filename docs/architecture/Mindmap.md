# High/mid level overview 
# 
## Component Groups

### Entry Point (Blue)
- **WorkflowOrchestrator**: The main entry point and coordinator for the entire analysis process
  - Implements dual-ring architecture with outer ring (management) and inner ring (detailed analysis)
  - Uses asyncio for non-blocking orchestration of multiple agent workflows
  - Provides callback hooks for real-time analysis tracking
  - Maintains workflow lineage for auditable execution history

### Pipeline Management (Green)
- **PipelineManager**: Creates and manages pipeline execution
  - State machine implementation using FlowState enum for predictable transitions
  - Records workflow events with UUID-based tracking for distributed tracing
  - Implements retry mechanism with configurable backoff strategy
  - Uses structured event logging compatible with ELK stack analysis
- **Pipeline**: Container for processing stages
  - Directed acyclic graph (DAG) implementation for stage dependencies
  - Supports conditional execution paths based on previous stage outputs
  - Provides progress monitoring via event emission
  - Uses cached results for idempotent operations to improve performance
- **Stage**: Individual processing unit
  - Function-as-argument design pattern for pluggable processing logic
  - Supports both synchronous and asynchronous processing functions
  - Implements detailed error handling with contextual information
  - Memory and performance profiling integration points
- **Context**: Data container that flows through the pipeline
  - Immutable core with transactional state updates
  - Progressive data enrichment pattern with type annotations
  - Cross-reference capabilities between different data entities
  - Efficient memory management using lazy loading for large datasets

### Data Sources (Purple)
- **FileSource**: Loads individual files
  - Uses memory-mapped I/O for efficient handling of large files
  - Implements content detection for binary vs text formats
  - Streaming parser for incremental processing of large contracts
  - Multi-encoding support (UTF-8, UTF-16, ISO-8859)
- **DirectorySource**: Loads files from directories
  - Recursive directory traversal with configurable depth limits
  - File filtering using glob patterns with Python's pathlib
  - Parallel processing for large directories with thread pooling
  - Progress reporting with file counts and sizes
- **GithubSource**: Loads files from GitHub repositories
  - GitHub API integration with rate limiting and pagination handling
  - Caches repository content locally with content-aware invalidation
  - Supports authentication for private repositories
  - Branch and tag specific checkout capabilities

### Transformers (Orange)
- **ContractChunker**: Breaks contracts into manageable chunks
  - Tree-sitter based parsing for accurate syntax-aware chunking
  - Implements structural boundary detection (contract, library definitions)
  - Maintains context references between related chunks
  - Contextual window sizing based on token complexity
- **FunctionExtractor**: Extracts functions from contracts
  - Uses AST parsing with Solidity-specific grammar rules
  - Captures function signatures, modifiers, and visibility
  - Detects and classifies function types (view, pure, payable)
  - Cross-references inherited functions from base contracts
- **BusinessFlowExtractor**: Identifies business flows within functions
  - Pattern-based analysis for common business flow types
  - Flow identification uses both AST and semantic analysis
  - Per-function analysis that gets aggregated up to contract level
  - Implements token flow tracking across function boundaries
  - JSON output compatible with visualization tools
- **CallGraphBuilder**: Builds function call relationships
  - Static analysis to map caller-callee relationships
  - Identifies external contract interactions (cross-contract calls)
  - Handles delegate calls and low-level function invocations
  - Exports call graph in DOT format for visualization
  - Uses custom Kotlin JAR for complex call resolution
- **ASTAnalyzer**: Analyzes abstract syntax trees
  - Tree-sitter integration for language-specific parsing
  - Pattern matching against known vulnerability patterns
  - Symbol resolution for variable usage analysis
  - Data flow analysis for tracking value movements
  - Control flow graph generation for path analysis

### Analysis (Red)
- **LlamaIndexAgent**: Provides vector search capabilities
  - LlamaIndex integration for semantic document search
  - Vector embedding generation using configurable LLM models
  - Hierarchical document chunking for context preservation
  - Query optimization with custom ranking algorithms
- **AsyncIndexProcessor**: Indexes and searches documents
  - Asynchronous document processing for non-blocking operation
  - Uses HNSW algorithm for efficient similarity search
  - Persistent vector storage with SQLite/FAISS backends
  - Metadata filtering capabilities for targeted searches

### State Tracking (Yellow)
- **FlowState**: Enumeration of pipeline states
  - Defined state machine with strict transition rules
  - Support for parallel state tracking in subcomponents
  - Provides hooks for state-change notifications
  - Includes detailed state descriptions for reporting
- **WorkflowEvent**: Events during pipeline execution
  - UUID-based event tracking for correlation
  - Hierarchical event structure for parent-child relationships
  - Time-series compatible format for dashboard visualization
  - Supports structured metadata for detailed event context

## Detailed Flow

1. The **WorkflowOrchestrator** receives a request to analyze a smart contract source
   - Input validation and normalization occurs at this stage
   - Configuration parameters are resolved with defaults where needed
   - Project ID generation for tracking the complete analysis workflow

2. The orchestrator calls the **PipelineManager** to run the pipeline
   - Sets up initial context with configuration parameters
   - Determines which pipeline stages to include based on config

3. The **PipelineManager** uses the appropriate source to load data
   - **FileSource** for single file analysis
   - **DirectorySource** for project directory analysis (uses glob patterns)
   - **GithubSource** for GitHub repository analysis (handles authentication)

4. The **PipelineManager** creates and executes the **Pipeline**
   - Creates a DAG of processing stages based on dependencies
   - Sets up monitoring for the pipeline execution

5. The **Pipeline** runs each stage in sequence:
   - **ContractChunker** to split contracts (tree-sitter based parsing)
     - Identifies logical boundaries in contracts
     - Maintains cross-references between related chunks

   - **FunctionExtractor** to extract functions
     - Builds function signatures and metadata
     - Maps visibility, modifiers, and access patterns
     - Extracts documentation comments for each function

   - **BusinessFlowExtractor** (optional) to identify business flows
     - Per-function analysis of business logic patterns
     - Merges related flows up to contract level
     - Identifies token transfers, access control patterns, state transitions
     - Uses pattern matching against known business logic templates

   - **CallGraphBuilder** (optional) to build call graphs
     - Uses Kotlin JAR for complex resolution of call hierarchies
     - Outputs JSON representation of call relationships
     - Identifies external contract interactions
     - Maps modifier applications to functions

   - **ASTAnalyzer** (optional) to analyze AST
     - Uses tree-sitter for parsing and tree generation
     - Performs static analysis checks against known patterns
     - Variable usage and data flow tracking
     - Control flow analysis for execution paths

6. The **PipelineManager** returns to the **WorkflowOrchestrator**
   - Complete context with analysis results
   - Execution metrics and performance data
   - Error reports from any failed stages

7. The **WorkflowOrchestrator** calls the **LlamaIndexAgent** to index and analyze the data
   - Converts analysis artifacts into vector embeddings
   - Creates searchable index from contract components
   - Maps relationships between contract elements

8. The **LlamaIndexAgent** uses **AsyncIndexProcessor** for vector search capabilities
   - Non-blocking search operations
   - Semantic similarity matching
   - Metadata filtering for targeted analysis

9. The **LlamaIndexAgent** returns results to the **WorkflowOrchestrator**
   - Relevance-ranked search results
   - Contextual snippets for findings
   - Confidence scores for identified patterns

10. The **WorkflowOrchestrator** generates a summary report
    - Aggregates findings by severity and category
    - Provides contract risk assessment scores
    - Generates visualization-ready output (JSON)
    - Creates markdown reports for human consumption

Throughout this process, state changes and events are tracked using **FlowState** and **WorkflowEvent** objects, creating a complete audit trail of the analysis workflow with timestamps and execution metrics.

# Finite Monkey Engine Architecture Mindmap

## System Overview
- Dual-ring architecture
  - Outer ring: Manager agents for coordination and reporting
  - Inner ring: Detail agents for specialized analysis
- Event-driven workflow tracking
- Pipeline-based processing
- Context-centric data flow

## Core Components

### WorkflowOrchestrator (Line 400)
- **Purpose**: Main entry point and coordinator
- **Key Functions**:
  - `run_analysis` (Line 420): Orchestrates complete analysis flow
  - `track_workflow` (Line 415): Records workflow events
  - `_execute_callback` (Line 500): Handles both sync and async callbacks
  - `_extract_findings_from_results` (Line 510): Identifies potential issues
  - `_generate_summary` (Line 525): Creates final analysis report
- **Technologies**: 
  - Asyncio for non-blocking execution
  - Structured event logging
  - UUID-based workflow tracking

### PipelineManager (Line 70)
- **Purpose**: Manages pipeline creation and execution
- **Key Functions**:
  - `create_pipeline` (Line 118): Factory method for pipeline creation
  - `run_pipeline` (Line 170): Executes pipeline with specified source
  - `record_event` (Line 91): Tracks workflow events
  - `set_flow_state` (Line 105): Updates workflow state with event tracking
- **State Machine**: 
  - PREPARING → LOADING → PROCESSING → ANALYZING → VALIDATING → REPORTING → COMPLETED/FAILED
- **Implementation Details**:
  - EventEmitter pattern for state changes
  - Configurable pipeline stages
  - Error handling with graceful degradation

### Data Sources
- **FileSource**
  - `process` (in external module): Loads single file content
  - Memory-mapped I/O for large files
  - Content-type detection
- **DirectorySource**
  - `process` (in external module): Recursively loads directory content
  - Glob pattern matching with pathlib
  - Parallel loading for large directories  
- **GithubSource**
  - `process` (in external module): Clones and loads GitHub repositories
  - Handles authentication and rate limiting
  - Caches repository content

### Transformers

#### ContractChunker
- **Purpose**: Splits contracts into manageable chunks
- **Implementation**:
  - Tree-sitter based parsing for accurate syntax boundaries
  - Chunk size optimization for analysis
  - Maintains references between chunks
- **Output**: Annotated contract chunks with positional metadata

#### FunctionExtractor (Line referenced in pipeline)
- **Purpose**: Extracts function definitions and metadata
- **Key Processing Steps**:
  - AST traversal for function identification
  - Signature parsing and normalization
  - Visibility and modifier analysis
  - Documentation extraction
- **Output Format**:
  - Function ID
  - Name, signature, visibility
  - Parameter types and names
  - Line range (start_line, end_line)
  - Full source text

#### BusinessFlowExtractor (New Component)
- **Purpose**: Identifies business logic flows within functions
- **Key Functions**:
  - `process` (Line 40 in transformers.py): Main entry point
  - `_extract_flows_from_function` (Line 70 in transformers.py): Per-function analysis
  - `_is_token_transfer`, `_is_access_control`, etc.: Pattern detection methods
- **Flow Types Detected**:
  - Token transfers
  - Access control patterns
  - State transitions
  - External calls
  - Fund management operations
- **Technical Approach**:
  - Pattern-based detection using function text
  - Function name heuristics
  - Bottom-up aggregation (function → contract → project)
  - JSON output format compatible with visualization tools

#### CallGraphBuilder
- **Purpose**: Builds function call relationships
- **Implementation**:
  - Uses Kotlin JAR for complex resolution (external process)
  - Static analysis of function calls
  - Handles inheritance and interfaces
  - Maps modifier applications
- **Output**: 
  - Caller-callee relationships
  - Call type classification
  - External contract interactions
  - DOT format export for visualization

#### ASTAnalyzer
- **Purpose**: Deep analysis of contract structure
- **Implementation**:
  - Tree-sitter integration for AST generation
  - Pattern matching against vulnerability signatures
  - Control flow analysis
  - Data flow tracking
- **Analysis Types**:
  - Variable usage and scope analysis
  - Control flow path enumeration
  - Value tracking across operations
  - Security pattern detection

### LlamaIndex Integration

#### LlamaIndexAgent (Line 237)
- **Purpose**: Vector search capabilities for semantic analysis
- **Key Functions**:
  - `initialize` (Line 249): Sets up LlamaIndex environment
  - `process_context` (Line 271): Indexes pipeline data
  - `search` (Line 370): Semantic search in indexed data
- **Technical Details**:
  - LLM-based embedding generation
  - Hierarchical document structure
  - Custom ranking algorithms
  - Metadata filtering

#### AsyncIndexProcessor (External)
- **Purpose**: Asynchronous document indexing and search
- **Implementation**:
  - Non-blocking operations with asyncio
  - HNSW algorithm for similarity search
  - Persistent storage with FAISS backend
- **Features**:
  - Incremental indexing
  - Batch processing for efficiency
  - Query optimization

## Data Flow

### Context Object
- **Structure**:
  - `pipeline_id`: Unique workflow identifier
  - `files`: Dictionary of file data
  - `chunks`: Contract chunks
  - `functions`: Extracted functions
  - `findings`: Analysis findings
  - `state`: Internal execution state
- **Lifecycle**:
  1. Created by WorkflowOrchestrator
  2. Populated by data source
  3. Transformed by pipeline stages
  4. Analyzed by LlamaIndexAgent
  5. Results extracted for reporting

### State Tracking

#### FlowState (Line 13)
- **States**:
  - PREPARING: Initial pipeline setup
  - LOADING: Data source loading
  - PROCESSING: Pipeline execution
  - ANALYZING: LlamaIndex processing
  - VALIDATING: Result verification
  - REPORTING: Report generation
  - COMPLETED: Successful execution
  - FAILED: Error state
- **Implementation**: Enum with string representation

#### WorkflowEvent (Line 22)
- **Structure**:
  - `event_id`: UUID for event
  - `timestamp`: Event time
  - `event_type`: Category of event
  - `from_state`/`to_state`: For state transitions
  - `description`: Human-readable description
  - `metadata`: Additional contextual data
- **Usage**: Comprehensive audit trail of execution

## Integration Points

### LLM Integration
- Vector embedding generation
- Semantic search capabilities
- Natural language query processing
- Finding description enhancement

### Visualization Outputs
- Call graphs in DOT format
- Business flow diagrams (JSON for D3.js)
- Finding severity heatmaps
- Interactive report generation

### External Tools
- Solidity compiler integration
- Slither static analyzer hooks
- Mythril symbolic execution
- Custom Kotlin analyzer for complex patterns

## Execution Workflow
1. Initialize WorkflowOrchestrator (Line 400)
2. Configure and start analysis (Line 420)
3. PipelineManager creates pipeline (Line 118)
4. Load source data (Lines 170-200)
5. Execute pipeline stages (Lines 200-230)
   - ContractChunker processing
   - FunctionExtractor processing
   - BusinessFlowExtractor processing (optional)
   - CallGraphBuilder processing (optional)
   - ASTAnalyzer processing (optional)
6. Create LlamaIndex representations (Line 271)
7. Perform semantic search (if query provided) (Line 370)
8. Generate findings and report (Line 525)
9. Return results to caller

## Performance Considerations
- Memory management for large contracts
- Caching strategies for repeated analysis
- Parallel processing of independent files
- Incremental updates for modified contracts
- Resource limiting for external tool invocations

## Future Enhancements
- Real-time analysis during contract development
- Expanded business flow pattern library
- Integration with testing frameworks
- Interactive query capabilities
- Machine learning for pattern recognition
- Customizable reporting templates
