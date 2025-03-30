#!/usr/bin/env python
"""
Diagnostic tool to check LlamaIndex LLM import paths.

This script helps identify the correct import paths for LLMs in the installed
LlamaIndex version, which can be useful when debugging import errors.
"""
import sys
import importlib
from pathlib import Path
import inspect

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def check_module_for_llms(module_path):
    """Check if a module has LLM classes and print available classes"""
    try:
        module = importlib.import_module(module_path)
        print(f"\n✅ Successfully imported: {module_path}")
        
        # Get version if available
        version = getattr(module, "__version__", "unknown")
        print(f"   Version: {version}")
        
        # Find classes that might be LLM implementations
        llm_classes = []
        llm_functions = []
        
        for name in dir(module):
            if name.startswith('_'):
                continue
                
            attr = getattr(module, name)
            if inspect.isclass(attr) and name[0].isupper():
                llm_classes.append(name)
            elif inspect.isfunction(attr) and (name.lower().endswith('llm') or 'llm' in name.lower()):
                llm_functions.append(name)
        
        if llm_classes:
            print(f"   Found potential LLM classes: {', '.join(llm_classes)}")
        else:
            print("   No potential LLM classes found")
            
        if llm_functions:
            print(f"   Found potential LLM functions: {', '.join(llm_functions)}")
            
        # Check for submodules that might contain LLMs
        for submodule_name in dir(module):
            if submodule_name.startswith('_'):
                continue
                
            submodule_attr = getattr(module, submodule_name)
            if inspect.ismodule(submodule_attr) and ('llm' in submodule_name.lower() or 'model' in submodule_name.lower()):
                print(f"   Found potential LLM submodule: {module_path}.{submodule_name}")
                
        return True
    except ImportError as e:
        print(f"❌ Could not import: {module_path}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Error inspecting {module_path}: {e}")
        return False


def main():
    """Main function to check LlamaIndex imports"""
    print("LlamaIndex LLM Import Path Checker")
    print("==================================")
    
    # Check if LlamaIndex is installed
    try:
        import llama_index
        print(f"Found LlamaIndex version: {getattr(llama_index, '__version__', 'unknown')}")
    except ImportError:
        print("❌ LlamaIndex is not installed")
        return
    
    # List of potential import paths to check
    paths_to_check = [
        "llama_index",
        "llama_index.llms",
        "llama_index.llm",
        "llama_index.core",
        "llama_index.core.llms",
        "llama_index.core.llm",
        "llama_index.program",
        "llama_index.program.guidance",
    ]
    
    # Check each path
    for path in paths_to_check:
        check_module_for_llms(path)
    
    print("\nRecommended Import Paths:")
    print("========================")
    
    # Try to provide concrete recommendations
    try:
        # Check OpenAI import
        openai_paths = []
        for path in ["llama_index.llms", "llama_index.core.llms", "llama_index.llm"]:
            try:
                module = importlib.import_module(path)
                if hasattr(module, "OpenAI"):
                    openai_paths.append(path)
            except ImportError:
                continue
        
        if openai_paths:
            print(f"OpenAI LLM:  from {openai_paths[0]} import OpenAI")
        else:
            print("OpenAI LLM:  Not found in standard locations")
            
        # Check Anthropic import
        anthropic_paths = []
        for path in ["llama_index.llms", "llama_index.core.llms", "llama_index.llm"]:
            try:
                module = importlib.import_module(path)
                if hasattr(module, "Anthropic"):
                    anthropic_paths.append(path)
            except ImportError:
                continue
        
        if anthropic_paths:
            print(f"Anthropic LLM: from {anthropic_paths[0]} import Anthropic")
        else:
            print("Anthropic LLM: Not found in standard locations")
            
        # Check Guidance integration
        guidance_paths = []
        for path in ["llama_index.program.guidance", "llama_index.core.program.guidance", "llama_index.prompts.guidance"]:
            try:
                module = importlib.import_module(path)
                if hasattr(module, "GuidancePydanticProgram"):
                    guidance_paths.append(path)
            except ImportError:
                continue
        
        if guidance_paths:
            print(f"Guidance:     from {guidance_paths[0]} import GuidancePydanticProgram")
        else:
            print("Guidance:     Not found in standard locations")
            
    except Exception as e:
        print(f"Error generating recommendations: {e}")
    
    print("\nSuggested Fix:")
    print("==============")
    print("Based on the detected import paths, update your code to use the correct imports.")
    print("If you're encountering import errors, you may need to:")
    print("1. Update LlamaIndex to the latest version")
    print("2. Update your import paths to match the detected modules")
    print("3. Check the LlamaIndex documentation for API changes")


if __name__ == "__main__":
    main()
