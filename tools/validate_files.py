#!/usr/bin/env python3
"""
Utility to check Python files for syntax errors and potential corruption.
"""
import os
import sys
import ast
import tokenize
from pathlib import Path
from loguru import logger

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'rb') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking {filepath}: {e}")
        return False

def check_file_indentation(filepath):
    """Check file for indentation issues."""
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # Check for mixed tabs and spaces
            if '\t' in line and ' ' in line.lstrip('\t'):
                issues.append(f"Line {i+1}: Mixed tabs and spaces")
            
            # Check for trailing whitespace
            if line.rstrip('\n').endswith(' '):
                issues.append(f"Line {i+1}: Trailing whitespace")
        
        return issues
    except Exception as e:
        logger.error(f"Error checking indentation in {filepath}: {e}")
        return [f"Error: {e}"]

def main():
    """Check all Python files in the project."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    root_dir = Path(__file__).parent.parent
    python_files = list(root_dir.glob('**/*.py'))
    
    logger.info(f"Checking {len(python_files)} Python files")
    
    syntax_errors = []
    indentation_issues = []
    
    for filepath in python_files:
        # Check syntax
        if not check_file_syntax(filepath):
            syntax_errors.append(filepath)
        
        # Check indentation
        issues = check_file_indentation(filepath)
        if issues:
            indentation_issues.append((filepath, issues))
    
    # Report results
    if syntax_errors:
        logger.error(f"Found {len(syntax_errors)} files with syntax errors:")
        for filepath in syntax_errors:
            logger.error(f"  - {filepath}")
    else:
        logger.success("No syntax errors found")
    
    if indentation_issues:
        logger.warning(f"Found {len(indentation_issues)} files with indentation issues:")
        for filepath, issues in indentation_issues:
            logger.warning(f"  - {filepath}:")
            for issue in issues[:5]:  # Limit to first 5 issues
                logger.warning(f"    * {issue}")
            if len(issues) > 5:
                logger.warning(f"    * ... and {len(issues) - 5} more issues")
    else:
        logger.success("No indentation issues found")

if __name__ == "__main__":
    main()
