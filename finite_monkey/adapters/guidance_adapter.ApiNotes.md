# Guidance Adapter API Notes

## Overview

The `GuidanceAdapter` provides integration with Microsoft's Guidance library for structured output generation from LLMs. This adapter enables the generation of outputs that strictly adhere to predefined schemas, eliminating parsing errors and improving output reliability.

## Key Features

1. **Forced Schema Compliance**: Unlike regular LLM outputs which merely suggest output structures, Guidance forces the LLM to follow the exact schema.
2. **Improved Output Consistency**: Especially valuable for weaker LLMs that might struggle with complex structured outputs.
3. **Error Elimination**: JSON parsing errors are eliminated because the output strictly adheres to the schema.

## Usage Pattern

The adapter follows this usage pattern:

1. **Create adapter**: Initialize with model settings
2. **Define schema**: Use Pydantic models to define output structure
3. **Create prompt**: Use handlebars-style templates with schema blocks
4. **Generate output**: Call the adapter to get structured outputs

## Handlebars Format

Guidance uses handlebars-style templates:
- Variables: `{{variable_name}}`
- Schema blocks: `{{#schema}} {...} {{/schema}}`

This differs from Python's f-string format (`{variable_name}`). The adapter includes automatic conversion from Python format to handlebars.

## Integration Points

- **BusinessFlowExtractor**: Used for structured business flow analysis
- **FLARE Query Engine**: Used for reliable query decomposition
- **Can be used independently**: For any component needing structured LLM outputs

## Error Handling

The adapter implements robust error handling with fallbacks:
1. Tries to use the Guidance library
2. Falls back to standard LLM JSON generation if Guidance fails
3. Returns empty objects as last resort to prevent pipeline failures
```
