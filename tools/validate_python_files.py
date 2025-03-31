#!/usr/bin/env python3
"""
Utility script to check Python files for corruption or structural issues.
"""
import os
import sys
import ast
from pathlib import Path

def check_file_structure(file_path):
    """Check if a Python file has valid syntax and structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file to check for syntax errors
        tree = ast.parse(content)
        
        # Check indentation consistency
        lines = content.split('\n')
        indent_levels = []
        
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indent_levels.append(indent)
        
        # Check for inconsistent indentation
        if indent_levels:
            unique_indents = set(indent_levels)
            if len(unique_indents) > 4:  # Allow for nested indentation
                print(f"Warning: Possible inconsistent indentation in {file_path}")
                print(f"Found indentation levels: {sorted(unique_indents)}")
        
        print(f"✓ File {file_path} is structurally valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking {file_path}: {e}")
        return False

def main():
    """Check Python files for corruption."""
    # Get the project root directory
    root_dir = Path(__file__).parent.parent
    
    # Get paths to check
    paths_to_check = []
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            paths_to_check.append(Path(path))
    else:
        # Default to checking the tools directory
        paths_to_check = [root_dir / 'tools']
    
    # Find all Python files
    python_files = []
    for path in paths_to_check:
        if path.is_file() and path.suffix == '.py':
            python_files.append(path)
        elif path.is_dir():
            python_files.extend(path.glob('**/*.py'))
    
    # Check each file
    valid_count = 0
    invalid_count = 0
    
    for file_path in python_files:
        if check_file_structure(file_path):
            valid_count += 1
        else:
            invalid_count += 1
    
    print(f"\nSummary: {valid_count} valid files, {invalid_count} invalid files")
    if invalid_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
