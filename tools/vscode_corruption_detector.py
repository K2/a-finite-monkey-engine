#!/usr/bin/env python3
"""
VSCode corruption detector for Python files.

This utility scans Python files for common patterns of VSCode-induced corruption
and reports or fixes the issues.
"""

import os
import sys
import re
import ast
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from loguru import logger
from datetime import datetime
import shutil

def backup_file(file_path: str) -> str:
    """Create a backup of the file before modifying it."""
    backup_dir = os.path.join(os.path.dirname(file_path), "backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(file_path)
    backup_path = os.path.join(backup_dir, f"{base_name}.{timestamp}.bak")
    
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    return backup_path

def detect_syntax_errors(file_path: str) -> List[Dict]:
    """Use AST to detect syntax errors in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    errors = []
    try:
        ast.parse(content)
    except SyntaxError as e:
        errors.append({
            'line': e.lineno,
            'column': e.offset,
            'message': e.msg,
            'text': content.split('\n')[e.lineno-1] if e.lineno <= len(content.split('\n')) else ""
        })
    
    return errors

def detect_corruption_patterns(file_path: str) -> List[Dict]:
    """
    Detect common VSCode corruption patterns including:
    1. Mixed/interleaved code fragments
    2. Duplicate lines with slight offsets
    3. Missing closing braces, quotes, etc.
    4. Method declarations that don't follow proper whitespace
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    issues = []
    
    # Check for interleaved code fragments
    for i in range(1, len(lines)-1):
        line = lines[i].strip()
        prev_line = lines[i-1].strip()
        
        # Check for try/except blocks without proper indentation
        if re.match(r'^\s*except\s+.*:', line) and not re.match(r'^\s*try\s*:', prev_line) and not re.match(r'^\s*except\s+.*:', prev_line):
            leading_whitespace = len(lines[i]) - len(lines[i].lstrip())
            expected_try_whitespace = leading_whitespace
            
            # Look backward for a matching 'try'
            try_found = False
            for j in range(i-1, max(0, i-10), -1):
                if re.match(r'^\s*try\s*:', lines[j].strip()):
                    try_found = True
                    break
            
            if not try_found:
                issues.append({
                    'line': i + 1,
                    'pattern': 'interleaved',
                    'message': 'except without matching try block nearby',
                    'text': line
                })
    
        # Check for code fragments from different methods mixed together
        if re.match(r'^\s*def\s+\w+\s*\(', line) and not re.match(r'^$', prev_line) and not re.match(r'^#', prev_line):
            indentation = len(lines[i]) - len(lines[i].lstrip())
            if indentation > 0:
                continue  # It's an indented method definition, probably a nested function
                
            issues.append({
                'line': i + 1,
                'pattern': 'interleaved',
                'message': 'method definition without preceding blank line',
                'text': line
            })
    
        # Check for corrupted error handling
        if "logger.error" in line and "except" in prev_line:
            # Make sure logger.error is inside a properly formatted string
            if not re.search(r'logger\.error\s*\(\s*[frf]?[\'"]', line):
                issues.append({
                    'line': i + 1,
                    'pattern': 'corrupted-error',
                    'message': 'malformed logger.error statement',
                    'text': line
                })
    
    # Check for unmatched brackets and quotes
    bracket_pairs = {
        '(': ')',
        '[': ']',
        '{': '}',
        '"': '"',
        "'": "'"
    }
    
    for i, line in enumerate(lines):
        for bracket, match in bracket_pairs.items():
            # Skip comments
            if line.strip().startswith('#'):
                continue
                
            if bracket in line and match not in line and bracket != match:
                # This is a simplistic check - it might have false positives
                # For quotes, we need to count occurrences
                if bracket in ['"', "'"]:
                    count = line.count(bracket)
                    if count % 2 == 1:  # Odd number of quotes
                        issues.append({
                            'line': i + 1,
                            'pattern': 'unmatched',
                            'message': f'unmatched {bracket}',
                            'text': line.rstrip()
                        })
                else:
                    issues.append({
                        'line': i + 1,
                        'pattern': 'unmatched',
                        'message': f'potential unmatched {bracket}',
                        'text': line.rstrip()
                    })
    
    return issues

def fix_corruption(file_path: str, auto_fix: bool = False) -> bool:
    """
    Detect and optionally fix corruption in the given file.
    
    Args:
        file_path: Path to the file to check and fix
        auto_fix: Whether to automatically apply fixes
        
    Returns:
        Whether the file was fixed
    """
    syntax_errors = detect_syntax_errors(file_path)
    corruption_patterns = detect_corruption_patterns(file_path)
    
    if not syntax_errors and not corruption_patterns:
        logger.info(f"No corruption detected in {file_path}")
        return True
    
    logger.warning(f"Found issues in {file_path}:")
    if syntax_errors:
        logger.warning("Syntax errors:")
        for error in syntax_errors:
            logger.warning(f"  Line {error['line']}: {error['message']}")
            logger.warning(f"    {error['text']}")
    
    if corruption_patterns:
        logger.warning("Corruption patterns:")
        for issue in corruption_patterns:
            logger.warning(f"  Line {issue['line']} ({issue['pattern']}): {issue['message']}")
            logger.warning(f"    {issue['text']}")
    
    if auto_fix:
        logger.info("Attempting to fix issues...")
        backup_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        
        # Fix corrupted error handling
        for issue in [i for i in corruption_patterns if i['pattern'] == 'corrupted-error']:
            line_idx = issue['line'] - 1
            line = lines[line_idx]
            if "logger.error" in line and not re.search(r'logger\.error\s*\(\s*[frf]?[\'"]', line):
                # Try to fix by adding proper formatting
                if "Error" in line:
                    fixed_line = re.sub(r'logger\.error\s*(.*?)([\'"]?Error.*?[\'"]?)',
                                         r'logger.error(f"\2")', line)
                else:
                    fixed_line = re.sub(r'logger\.error\s*(.*?)$',
                                         r'logger.error(f"Error: \1")', line)
                
                lines[line_idx] = fixed_line
                modified = True
                logger.info(f"Fixed logger.error at line {issue['line']}")
        
        # Fix unmatched quotes
        for issue in [i for i in corruption_patterns if i['pattern'] == 'unmatched' and "'" in i['message'] or '"' in i['message']]:
            line_idx = issue['line'] - 1
            line = lines[line_idx]
            
            for quote in ['"', "'"]:
                if line.count(quote) % 2 == 1:
                    # Add the missing quote at the end
                    if not line.rstrip().endswith(quote):
                        lines[line_idx] = line.rstrip() + quote + '\n'
                        modified = True
                        logger.info(f"Fixed unmatched {quote} at line {issue['line']}")
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            logger.success(f"Fixed issues in {file_path}")
        
        # Check if the file is now syntactically valid
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            logger.success(f"File is now syntactically valid: {file_path}")
            return True
        except SyntaxError as e:
            logger.error(f"File still has syntax errors: {e}")
            return False
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Detect and fix VSCode corruption in Python files")
    parser.add_argument("path", help="File or directory to check")
    parser.add_argument("--fix", action="store_true", help="Automatically fix issues")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively scan directories")
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        fix_corruption(str(path), args.fix)
    elif path.is_dir():
        if args.recursive:
            for py_file in path.glob("**/*.py"):
                logger.info(f"Checking {py_file}...")
                fix_corruption(str(py_file), args.fix)
        else:
            for py_file in path.glob("*.py"):
                logger.info(f"Checking {py_file}...")
                fix_corruption(str(py_file), args.fix)
    else:
        logger.error(f"Path not found: {path}")
        return 1
    
    return 0

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    sys.exit(main())
