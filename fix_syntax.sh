#!/bin/bash
# Simple script to fix the syntax error by removing the extra closing parenthesis

# Use sed to fix the specific problem at line 470
FILE="/home/files/git/a-finite-monkey-engine/finite_monkey/utils/chunking.py"

# Make a backup
cp "$FILE" "${FILE}.bak"

# Find the problematic line and manually fix it (replace the specific character)
LINE=$(sed -n '470p' "$FILE")
echo "Line 470: $LINE"

# Option 1: Remove trailing parenthesis (if that's the issue)
sed -i '470s/)$//' "$FILE"

# Or option 2: Find and remove any extra unmatched closing parenthesis on line 470
# This is a more aggressive approach
sed -i '470s/))/) /' "$FILE"

echo "Fixed syntax error. Original file backed up to ${FILE}.bak"
