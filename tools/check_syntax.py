#!/usr/bin/env python
"""
Simple utility to check Python syntax without executing code.
Useful for finding syntax errors in files.
"""
import os
import sys
import py_compile
import argparse
from pathlib import Path

def check_file_syntax(file_path):
    """Check a single file for syntax errors"""
    try:
        py_compile.compile(file_path, doraise=True)
        print(f"✅ {file_path}: Syntax OK")
        return True
    except py_compile.PyCompileError as e:
        line_no = getattr(e.exc_value, 'lineno', '?')
        offset = getattr(e.exc_value, 'offset', '?')
        error_msg = str(getattr(e.exc_value, 'msg', str(e.exc_value)))
        print(f"❌ {file_path}:{line_no}:{offset}: {error_msg}")
        return False
    except Exception as e:
        print(f"❌ {file_path}: Error checking syntax: {e}")
        return False

def check_directory(directory, recursive=True):
    """Check all Python files in a directory"""
    success_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(directory):
        if not recursive and root != directory:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if check_file_syntax(file_path):
                    success_count += 1
                else:
                    error_count += 1
    
    print(f"\nSummary: {success_count} files OK, {error_count} files with errors")
    return error_count == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Python files for syntax errors")
    parser.add_argument("path", help="File or directory to check")
    parser.add_argument("-r", "--recursive", action="store_true", help="Check directories recursively")
    
    args = parser.parse_args()
    path = args.path
    
    if os.path.isfile(path):
        sys.exit(0 if check_file_syntax(path) else 1)
    elif os.path.isdir(path):
        sys.exit(0 if check_directory(path, args.recursive) else 1)
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)
