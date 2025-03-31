#!/usr/bin/env python3
"""
Utility script to fix corrupted vector_store_util.py file.
This script reads the original file, performs basic syntax validation,
and generates a fixed version if syntax errors are found.
"""
import os
import sys
import ast
import shutil
from pathlib import Path
from datetime import datetime
from loguru import logger

def fix_corrupted_file(filepath):
    """
    Fix corrupted Python file by validating syntax and making backup copies.
    
    Args:
        filepath: Path to the Python file to fix
    
    Returns:
        Boolean indicating success of the operation
    """
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return False
    
    # Create backup directory
    backup_dir = filepath.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{filepath.name}.{timestamp}.bak"
    shutil.copy2(filepath, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    # Read the file content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for syntax errors
    try:
        ast.parse(content)
        logger.info("File passes syntax check, no corruption detected.")
        return True
    except SyntaxError as e:
        logger.warning(f"Syntax error detected at line {e.lineno}, col {e.offset}: {e.msg}")
        
        # Extract the problematic lines for context
        lines = content.split('\n')
        start_line = max(0, e.lineno - 5)
        end_line = min(len(lines), e.lineno + 5)
        
        logger.info("Context of the error:")
        for i in range(start_line, end_line):
            prefix = "→ " if i + 1 == e.lineno else "  "
            logger.info(f"{prefix}{i+1}: {lines[i]}")
        
        # Try to fix the most common corruption issues
        fixed_content = fix_common_issues(content, e.lineno)
        
        # Save the fixed file
        fixed_path = filepath.parent / f"{filepath.name}.fixed.py"
        with open(fixed_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Validate the fixed file
        try:
            ast.parse(fixed_content)
            logger.success(f"Fixed file saved to {fixed_path}")
            logger.info("You can replace the original file with the fixed version after verification.")
            return True
        except SyntaxError as e2:
            logger.error(f"Failed to fix the file: {e2}")
            return False

def fix_common_issues(content, error_line):
    """Fix common corruption issues in Python files."""
    lines = content.split('\n')
    
    # Look for misplaced error handling blocks
    for i in range(max(0, error_line - 10), min(len(lines), error_line + 10)):
        # Check for incomplete except blocks
        if "except Exception as e:" in lines[i] and i+1 < len(lines):
            if "logger.error" in lines[i+1] and not lines[i+1].strip().endswith(":"):
                # Fix the line ending with a properly formatted error message
                if ")" not in lines[i+1] or lines[i+1].count(")") < lines[i+1].count("("):
                    lines[i+1] = lines[i+1].rstrip() + ")"
    
    # Look for misplaced string literals
    for i in range(max(0, error_line - 5), min(len(lines), error_line + 5)):
        # Check for lines with unmatched quotes
        if (lines[i].count("'") % 2 != 0) or (lines[i].count('"') % 2 != 0):
            # Try to fix by adding a closing quote
            if lines[i].count("'") % 2 != 0:
                lines[i] = lines[i] + "'"
            if lines[i].count('"') % 2 != 0:
                lines[i] = lines[i] + '"'
    
    # Look for misplaced function/method definitions
    for i in range(max(0, error_line - 10), min(len(lines), error_line + 10)):
        # Check for lines that look like a method definition but are misplaced
        if (i > 0 and "def " in lines[i] and 
            "def " not in lines[i-1] and 
            not lines[i-1].strip().endswith(":") and
            not lines[i].startswith(" ")):
            # Add a newline before the method definition
            lines[i] = "\n" + lines[i]
    
    return '\n'.join(lines)

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        logger.error("Usage: python fix_corrupted_file.py <python_file_path>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    success = fix_corrupted_file(filepath)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
