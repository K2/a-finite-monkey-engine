# Configuration in A Finite Monkey Engine

## Overview

A Finite Monkey Engine uses a centralized configuration system through the `nodes_config` module.
This document outlines how configuration works in the codebase.

## Configuration Sources

Configuration values are loaded from multiple sources in the following order of precedence:

1. Default values defined in the code
2. Environment variables (prefixed with `FM_`)
3. Configuration file (if specified via the `CONFIG_PATH` environment variable)

## Using Configuration

Always import the `config` object from `nodes_config`:

```python
from finite_monkey.nodes_config import config

# Access configuration
model_name = config.BUSINESS_FLOW_MODEL
timeout = config.REQUEST_TIMEOUT
```

## Available Configuration Options

| Config Name | Environment Variable | Description | Default |
|-------------|---------------------|-------------|---------|
| `ANALYSIS_MODEL` | `FM_ANALYSIS_MODEL` | Main model used for analysis | `gpt-4o` |
| `BUSINESS_FLOW_MODEL` | `FM_BUSINESS_FLOW_MODEL` | Model used for business flow extraction | `gpt-4o` |
| `BUSINESS_FLOW_MODEL_PROVIDER` | `FM_BUSINESS_FLOW_MODEL_PROVIDER` | Provider for business flow model | `openai` |
| `BUSINESS_FLOW_MODEL_BASE_URL` | `FM_BUSINESS_FLOW_MODEL_BASE_URL` | Base URL for business flow model | `None` |
| `SCAN_MODEL` | `FM_SCAN_MODEL` | Model used for vulnerability scanning | `gpt-4o` |
| `SCAN_MODEL_PROVIDER` | `FM_SCAN_MODEL_PROVIDER` | Provider for scan model | `openai` |
| `SCAN_MODEL_BASE_URL` | `FM_SCAN_MODEL_BASE_URL` | Base URL for scan model | `None` |
| `EMBEDDING_MODEL` | `FM_EMBEDDING_MODEL` | Model used for embeddings | `text-embedding-ada-002` |
| `VALIDATOR_MODEL` | `FM_VALIDATOR_MODEL` | Model used for validation | `gpt-3.5-turbo` |
| `REQUEST_TIMEOUT` | `FM_REQUEST_TIMEOUT` | Timeout for API requests (seconds) | `60` |

## Model Parameters

Model-specific parameters can be configured in the `MODEL_PARAMS` dictionary. The default values are:

```python
{
    "default": {
        "temperature": 0.7,
        "max_tokens": 4000
    },
    "gpt-4o": {
        "temperature": 0.7,
        "max_tokens": 4000
    }
}
```

## Deprecated Configuration

The `finite_monkey.config` package and its components (including `loader.py`) are deprecated. 
Always use `finite_monkey.nodes_config` as the single source of truth for configuration.
