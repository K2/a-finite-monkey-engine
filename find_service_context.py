#!/usr/bin/env python3
"""
Utility script to find instances of ServiceContext in the codebase
"""

import os
import re
import sys
from pathlib import Path

def find_service_context_references(root_dir):
    """Find files containing ServiceContext references"""
    patterns = [
        r'ServiceContext',
        r'service_context',
        r'from llama_index.core import ServiceContext',
    ]
    
    found_files = []
    
    for path in Path(root_dir).rglob('*.py'):
        if 'venv' in str(path) or '.git' in str(path):
            continue
            
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        for pattern in patterns:
            if re.search(pattern, content):
                found_files.append(str(path))
                print(f"Found reference in: {path}")
                break
                
    return found_files

if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Searching for ServiceContext references in {root_dir}...")
    files = find_service_context_references(root_dir)
    
    print(f"\nFound {len(files)} files with ServiceContext references")
    for file in files:
        print(f" - {file}")
