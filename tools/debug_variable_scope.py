#!/usr/bin/env python3
"""
Utility to detect potential variable scope issues in Python code.
Particularly useful for finding issues with imports inside function/method blocks.
"""
import os
import sys
import ast
import argparse
from typing import List, Dict, Set, Tuple
from pathlib import Path
from loguru import logger

def find_scope_issues(file_path: str) -> List[Dict]:
    """
    Analyze a Python file for potential variable scope issues.
    
    Args:
        file_path: Path to the Python file to analyze
        
    Returns:
        List of detected issues with line numbers and descriptions
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Parse the code
        tree = ast.parse(code)
        
        # Track imports at different levels
        top_level_imports = set()
        
        # First pass: collect top-level imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and isinstance(node.parent, ast.Module):
                for name in node.names:
                    top_level_imports.add(name.name)
            elif isinstance(node, ast.ImportFrom) and isinstance(node.parent, ast.Module):
                top_level_imports.add(node.module)

        # Second pass: find function-level imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_name = node.name
                function_imports = set()
                
                # Find imports in this function
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Import):
                        for name in subnode.names:
                            function_imports.add(name.name)
                    elif isinstance(subnode, ast.ImportFrom):
                        function_imports.add(subnode.module)
                
                # Find names that are referred to but not imported within the function
                referenced_names = set()
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load):
                        referenced_names.add(subnode.id)
                
                # Check for used names that might cause scope issues
                for name in referenced_names:
                    if name in top_level_imports and name in function_imports:
                        issues.append({
                            'line': node.lineno,
                            'function': function_name,
                            'name': name,
                            'description': f"Module '{name}' is imported both at top level and inside function '{function_name}'"
                        })
                
                # Check for potential shadowing of common module names (like os, sys, etc.)
                common_modules = {'os', 'sys', 'json', 'pickle', 'datetime', 're', 'math', 'random'}
                for var_node in ast.walk(node):
                    if isinstance(var_node, ast.Assign):
                        for target in var_node.targets:
                            if isinstance(target, ast.Name) and target.id in common_modules:
                                issues.append({
                                    'line': var_node.lineno,
                                    'function': function_name,
                                    'name': target.id,
                                    'description': f"Variable '{target.id}' shadows a common module name in function '{function_name}'"
                                })
        
        return issues
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        return []

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Find potential variable scope issues in Python code')
    parser.add_argument('path', help='Path to a Python file or directory to analyze')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively analyze directories')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    path = Path(args.path)
    
    if path.is_file() and path.suffix == '.py':
        # Analyze a single file
        issues = find_scope_issues(str(path))
        if issues:
            logger.warning(f"Found {len(issues)} potential issues in {path}")
            for issue in issues:
                logger.warning(f"Line {issue['line']} - {issue['description']}")
        else:
            logger.info(f"No issues found in {path}")
    elif path.is_dir():
        # Analyze a directory
        py_files = list(path.glob('**/*.py' if args.recursive else '*.py'))
        logger.info(f"Analyzing {len(py_files)} Python files in {path}")
        
        total_issues = 0
        for py_file in py_files:
            issues = find_scope_issues(str(py_file))
            if issues:
                total_issues += len(issues)
                logger.warning(f"Found {len(issues)} potential issues in {py_file}")
                for issue in issues:
                    logger.warning(f"Line {issue['line']} - {issue['description']}")
        
        if total_issues > 0:
            logger.warning(f"Found a total of {total_issues} potential issues")
        else:
            logger.info("No issues found")
    else:
        logger.error(f"Path does not exist or is not a Python file: {path}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
