# LlamaIndex Adapter API Notes

## Parameter Naming Consistency

There's an important inconsistency in parameter naming that can cause issues:

- The parameter for controlling request timeouts should be consistently named `request_timeout` (not `timeout`).
- This parameter is passed all the way through from LlamaIndexAdapter to the underlying LLM implementations.
- Different LLM providers may handle timeout parameters differently:
  - OpenAI uses `request_timeout`
  - Ollama uses `request_timeout`
  - Anthropic uses `timeout`

## Ollama-Specific Considerations

When working with Ollama:

1. The `request_timeout` parameter is especially important as Ollama runs locally and may be slower than cloud APIs.
2. Too short timeouts can cause generation failures, especially with longer prompts.
3. Default timeout values that work well with cloud APIs may be too short for local models.

## Configuration Flow

The timeout parameter flows through the system like this:

1. `nodes_config.py` defines `REQUEST_TIMEOUT` (typically 60-120 seconds)
2. LlamaIndexAdapter is initialized with `request_timeout=config.REQUEST_TIMEOUT`
3. The adapter passes this to the underlying LLM implementation
4. For guidance integration, this setting is applied when creating the LLM

## Troubleshooting Timeouts

If you encounter timeout issues:

1. Check the `REQUEST_TIMEOUT` value in nodes_config.py
2. Ensure the parameter name is consistent (`request_timeout`, not `timeout`)
3. For Ollama, try increasing the timeout to 120-180 seconds for complex prompts
4. Use the tools/check_ollama_timeout.py script to find optimal timeout values

