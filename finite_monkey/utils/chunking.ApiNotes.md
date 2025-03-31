# Chunking Module API Notes

## AsyncContractChunker.chunk_code

The `chunk_code` method has been standardized to always return a dictionary with a consistent structure. This simplifies consumption in pipelines and reduces the need for type checking and branching logic.

### Return Structure

```python
{
    "chunk_id": "filename_full",    # Unique identifier for the chunk
    "content": "full source code",   # Complete source code of the file
    "start_char": 0,                 # Starting character position
    "end_char": len(code),           # Ending character position
    "source_file": file_path,        # Path to the original file
    "chunk_type": "file",            # Type: 'file', 'error', etc.
    "name": file_name,               # Name of the file
    "contracts": [                   # List of contract chunks
        {
            "chunk_id": "filename_ContractName",
            "content": "contract source code",
            "start_char": start_pos,
            "end_char": end_pos,
            "source_file": file_path,
            "chunk_type": "contract",
            "name": "ContractName",
            "contract_name": "ContractName",
            "functions": [           # List of function chunks
                {
                    "chunk_id": "filename_ContractName_functionName",
                    "content": "function source code",
                    "start_char": 0,
                    "end_char": length,
                    "source_file": file_path,
                    "chunk_type": "function",
                    "name": "functionName",
                    "function_name": "functionName",
                    "contract_name": "ContractName",
                    "full_name": "ContractName.functionName"
                }
            ]
        }
    ]
}
```

### Error Handling

If an error occurs during processing, the method will return a minimal error dictionary:

```python
{
    "chunk_id": "filename_error",
    "content": "// Error processing code",
    "start_char": 0,
    "end_char": 0,
    "source_file": file_path,
    "chunk_type": "error",
    "error": str(e),
    "contracts": []  # Empty list to prevent None checks
}
```

### Usage Notes

1. Always access the "contracts" list, even in error cases (it will be empty if there's an error)
2. Check the "chunk_type" to determine if an error occurred during processing
3. When processing a contract, check that it has a "functions" list before iterating
