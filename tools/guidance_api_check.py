#!/usr/bin/env python
"""
Utility to identify the correct Guidance API pattern for the installed version.
"""
import sys
import os
import inspect

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    try:
        import guidance
        
        print(f"=== Guidance API Check ===")
        print(f"Guidance version: {getattr(guidance, '__version__', 'unknown')}")
        
        # Check for Program class
        if hasattr(guidance, "Program"):
            print("✅ guidance.Program is available")
            print(f"   Signature: {str(inspect.signature(guidance.Program))}")
        else:
            print("❌ guidance.Program is NOT available")
        
        # Check for program function
        if hasattr(guidance, "program"):
            print("✅ guidance.program is available")
            print(f"   Signature: {str(inspect.signature(guidance.program))}")
        else:
            print("❌ guidance.program is NOT available")
        
        # Check for LLM class
        if hasattr(guidance, "LLM"):
            print("✅ guidance.LLM is available")
            print(f"   Signature: {str(inspect.signature(guidance.LLM))}")
        else:
            print("❌ guidance.LLM is NOT available")
        
        # Check if guidance is callable
        if callable(guidance):
            print("✅ guidance itself is callable")
            try:
                sig = inspect.signature(guidance)
                print(f"   Signature: {sig}")
            except (ValueError, TypeError):
                print("   Cannot determine signature")
        else:
            print("❌ guidance itself is NOT callable")
        
        # Check for llms attribute
        if hasattr(guidance, "llms"):
            print("✅ guidance.llms module is available")
            llm_classes = [name for name in dir(guidance.llms) 
                          if inspect.isclass(getattr(guidance.llms, name))]
            print(f"   LLM classes: {', '.join(llm_classes)}")
        else:
            print("❌ guidance.llms module is NOT available")
        
        # Show available top-level attributes
        print("\nTop-level guidance attributes:")
        for name in dir(guidance):
            if not name.startswith("_"):  # Skip private attributes
                attr = getattr(guidance, name)
                attr_type = type(attr).__name__
                print(f"- {name}: {attr_type}")
                
        print("\nRecommended API pattern based on available features:")
        if hasattr(guidance, "program"):
            print("guidance.program(template, llm=llm_instance)")
        elif hasattr(guidance, "LLM"):
            print("g_llm = guidance.LLM(llm_instance)\nprogram = g_llm(template)")
        elif callable(guidance):
            print("guidance(template, api=llm_instance)")
        elif hasattr(guidance, "Program"):
            print("guidance.Program(template, llm=llm_instance)")
        else:
            print("Unable to determine the appropriate pattern")
        
    except ImportError:
        print("Guidance is not installed. Install with 'pip install guidance'.")
    except Exception as e:
        print(f"Error checking Guidance API: {e}")

if __name__ == "__main__":
    main()
