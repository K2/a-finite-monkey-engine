#!/usr/bin/env python3
"""
Emergency syntax error fix - this will try to load the file,
find syntax errors, and fix them automatically.
"""
import ast
import sys

FILE_PATH = "/home/files/git/a-finite-monkey-engine/finite_monkey/utils/chunking.py"

# Make a backup
with open(FILE_PATH, 'r') as f:
    original_content = f.read()

with open(FILE_PATH + '.bak', 'w') as f:
    f.write(original_content)

print(f"Backup created at {FILE_PATH}.bak")

# Try to parse and find syntax errors
try:
    ast.parse(original_content)
    print("No syntax errors found!")
    sys.exit(0)
except SyntaxError as e:
    print(f"Syntax error found at line {e.lineno}, column {e.offset}: {e.text}")
    
    # Get the lines
    lines = original_content.split('\n')
    
    if e.lineno <= len(lines):
        problem_line = lines[e.lineno-1]
        print(f"Problem line: {problem_line}")
        
        # If it's an unmatched parenthesis, try to fix it
        if "unmatched" in str(e) and ")" in str(e):
            # Remove the last closing parenthesis in the line
            fixed_line = problem_line.rsplit(')', 1)[0]
            lines[e.lineno-1] = fixed_line
            print(f"Fixed line: {fixed_line}")
            
            # Write the fixed content back
            with open(FILE_PATH, 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"File fixed and saved to {FILE_PATH}")
            sys.exit(0)

print("Could not automatically fix the error. Please check the file manually.")
sys.exit(1)
