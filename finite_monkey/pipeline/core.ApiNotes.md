# Pipeline Core Module

## Architecture Overview

The pipeline core module implements a flexible, composable analysis pipeline architecture for the Finite Monkey Engine. The system is designed with a clear data flow hierarchy and comprehensive state tracking.

## Data Flow Hierarchy

The primary data flow follows a three-level hierarchy:

1. **Files**: Raw source files loaded from the input path
   - Stored in `context.files`
   - Each file contains metadata and content

2. **Contracts**: Smart contracts extracted from files
   - Stored in `context.contracts` (previously named "chunks")
   - Each contract is associated with a parent file
   - Contains contract code and metadata

3. **Functions**: Function definitions extracted from contracts
   - Stored in `context.functions`
   - Each function is associated with a parent contract
   - Contains function signatures, bodies, and metadata

## Pipeline Stage State Machine

The pipeline uses the `PipelineStageState` enum to implement a state machine approach:

- `INIT`: Initial state when context is created
- `FILES_LOADED`: Raw files have been loaded from disk
- `CONTRACTS_EXTRACTED`: Contracts have been extracted from files
- `FUNCTIONS_EXTRACTED`: Functions have been extracted from contracts
- `ANALYSIS_COMPLETE`: All analysis steps completed
- `ERROR`: Error occurred during processing

## Context Class

The `Context` class serves as the primary data structure flowing through the pipeline. Key components:

1. **State Tracking**: Current stage, processing history, stage transitions
2. **Data Containers**: 
   - `files`: Source code files
   - `contracts`: Smart contracts (renamed from "chunks")
   - `functions`: Function definitions
3. **Metadata**: Processing timestamps, counts, statistics
4. **Analysis Results**: Findings, metrics, errors

### Helper Methods

The Context class includes methods for consistent state management:

- `update_stage()`: Transition to a new pipeline stage
- `set_files_loaded()`: Mark file loading complete
- `set_contracts_extracted()`: Mark contract extraction complete
- `set_functions_extracted()`: Mark function extraction complete
- `add_contract()`: Add a contract to the context
- `add_function()`: Add a function to the context
- `add_finding()`: Add an analysis finding
- `add_error()`: Record an error without halting the pipeline

## Pipeline Architecture 

The pipeline consists of stages that can be composed and combined:

1. **Document Processing Stage**: Combines the first three steps:
   - Loading files from input path
   - Extracting contracts from files
   - Parsing functions from contracts

2. **Analysis Stages**: Remain separate and specialized:
   - Business flow extraction
   - Vulnerability scanning
   - Data flow analysis
   - And other analysis components

## Implementation Notes

- The combined document processing stage improves efficiency for input processing
- State transitions ensure all components can track pipeline progress
- The metadata system provides insights into processing statistics
- Error handling permits graceful degradation when components fail