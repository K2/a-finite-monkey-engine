#!/usr/bin/env python3
"""
Test script to verify imports are working correctly
"""

import sys

def test_imports():
    """Test importing key modules"""
    import_errors = []
    
    # Core modules
    modules_to_test = [
        "finite_monkey.nodes_config",
        "finite_monkey.db.models",
        "finite_monkey.db.manager",
        "finite_monkey.agents.orchestrator",
        "finite_monkey.workflow.agent_controller",
        "finite_monkey.adapters.ollama",
        "finite_monkey.web.app"
    ]
    
    print("Testing module imports...")
    
    for module_name in modules_to_test:
        print(f"  - {module_name}...", end="")
        try:
            __import__(module_name)
            print("OK")
        except ImportError as e:
            print(f"FAILED ({str(e)})")
            import_errors.append((module_name, str(e)))
    
    print("\nResults:")
    if import_errors:
        print(f"✗ {len(import_errors)} import errors detected:")
        for module, error in import_errors:
            print(f"  - {module}: {error}")
        return 1
    else:
        print("✓ All imports successful!")
        return 0

if __name__ == "__main__":
    sys.exit(test_imports())