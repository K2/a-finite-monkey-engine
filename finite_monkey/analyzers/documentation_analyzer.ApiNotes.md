# Documentation Analyzer API Notes

## Documentation Quality Assessment

The documentation analyzer includes a quality assessment feature that evaluates how well code is documented against standard practices. This assessment is performed by the `_assess_quality` method which:

1. Creates a specialized prompt using `_create_quality_assessment_prompt`
2. Calls the LLM using the universal interface from `llm_interface`
3. Parses the response to extract quality metrics

### Prompt Construction

The quality assessment prompt is constructed by:

1. **File Type Detection**: Different documentation standards are applied based on file type (e.g., Solidity uses NatSpec)
2. **Standards Inclusion**: Relevant documentation standards are included in the prompt
3. **Rating Scale**: A clear 0-10 rating scale is provided to guide the LLM assessment
4. **Response Format**: A JSON response format is requested for consistent parsing

### Response Parsing

The analyzer handles two response scenarios:

1. **Valid JSON Response**: Parses directly using `json.loads()`
2. **Text Response**: Uses regex fallbacks to extract quality scores and issues

## Method Dependencies

The quality assessment has these method dependencies:

- `_assess_quality`: Main method that orchestrates the assessment
- `_create_quality_assessment_prompt`: Creates the user prompt with file content
- `_create_quality_assessment_system_prompt`: Creates the system prompt that instructs the LLM

All three methods are required for proper functioning of the documentation quality assessment.
