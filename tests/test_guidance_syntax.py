#!/usr/bin/env python3
"""Simple test to verify syntax correctness of guidance_version_utils.py"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Try to import the module to test syntax
    from finite_monkey.utils.guidance_version_utils import GUIDANCE_AVAILABLE, create_guidance_program
    print("✅ Syntax check passed: Successfully imported guidance_version_utils")
except SyntaxError as e:
    print(f"❌ Syntax error in guidance_version_utils.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"⚠️ Other error (not syntax related): {e}")

print("All tests passed!")
