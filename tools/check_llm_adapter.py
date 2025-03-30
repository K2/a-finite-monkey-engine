#!/usr/bin/env python
"""
Tool to check what methods are available on the LLMAdapter.
"""
import sys
import os
import inspect
import asyncio
from loguru import logger

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finite_monkey.nodes_config import config
from finite_monkey.llm.llama_index_adapter import LlamaIndexAdapter

def inspect_object(obj, name="object"):
    """Print detailed information about an object"""
    print(f"\n=== Inspecting {name} ===")
    
    # Get all attributes
    attributes = dir(obj)
    print(f"Total attributes: {len(attributes)}")
    
    # Separate methods from other attributes
    methods = []
    properties = []
    
    for attr in attributes:
        if attr.startswith("_"):
            continue  # Skip private/magic methods
            
        value = getattr(obj, attr)
        if callable(value):
            methods.append((attr, value))
        else:
            properties.append((attr, value))
    
    # Print properties
    if properties:
        print("\nProperties:")
        for name, value in properties:
            print(f"  {name}: {value}")
    
    # Print methods
    if methods:
        print("\nMethods:")
        for name, method in methods:
            sig = ""
            try:
                sig = str(inspect.signature(method))
            except (ValueError, TypeError):
                sig = "(unknown signature)"
                
            is_async = asyncio.iscoroutinefunction(method)
            print(f"  {name}{sig}" + (" [async]" if is_async else ""))
    
    return methods, properties

async def main():
    # Create adapter
    print(f"Creating LlamaIndexAdapter with model={config.ANALYSIS_MODEL}, provider={config.ANALYSIS_MODEL_PROVIDER}")
    adapter = LlamaIndexAdapter(
        model_name=config.ANALYSIS_MODEL,
        provider=config.ANALYSIS_MODEL_PROVIDER,
        base_url=config.ANALYSIS_MODEL_BASE_URL,
        request_timeout=config.REQUEST_TIMEOUT
    )
    
    # Inspect adapter
    adapter_methods, adapter_props = inspect_object(adapter, "LlamaIndexAdapter")
    
    # Inspect LLM if available
    if hasattr(adapter, "llm"):
        llm_methods, llm_props = inspect_object(adapter.llm, "adapter.llm")
    
    # Summarize chat methods
    print("\n=== Chat Methods Summary ===")
    chat_methods = [m for m in adapter_methods if "chat" in m[0].lower()]
    if chat_methods:
        print("Adapter chat methods:")
        for name, method in chat_methods:
            is_async = asyncio.iscoroutinefunction(method)
            print(f"  {name}" + (" [async]" if is_async else ""))
    else:
        print("No chat methods found on adapter!")
    
    if hasattr(adapter, "llm"):
        llm_chat_methods = [m for m in llm_methods if "chat" in m[0].lower()]
        if llm_chat_methods:
            print("\nLLM chat methods:")
            for name, method in llm_chat_methods:
                is_async = asyncio.iscoroutinefunction(method)
                print(f"  {name}" + (" [async]" if is_async else ""))
        else:
            print("\nNo chat methods found on LLM!")
    
    # Try to use any identified chat method
    print("\n=== Testing Chat Method ===")
    from llama_index.core.llms import ChatMessage, MessageRole
    
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Say hello in one word.")
    ]
    
    # Try chat methods in order of preference
    methods_to_try = [
        # Adapter methods
        ("adapter.achat", lambda: adapter.achat(messages=messages) if hasattr(adapter, "achat") else None),
        ("adapter.chat", lambda: adapter.chat(messages=messages) if hasattr(adapter, "chat") else None),
        # LLM methods
        ("adapter.llm.achat", lambda: adapter.llm.achat(messages=messages) if hasattr(adapter.llm, "achat") else None),
        ("adapter.llm.chat", lambda: adapter.llm.chat(messages=messages) if hasattr(adapter.llm, "chat") else None),
        # Completion methods as fallback
        ("adapter.acomplete", lambda: adapter.acomplete("Hello!") if hasattr(adapter, "acomplete") else None),
        ("adapter.complete", lambda: adapter.complete("Hello!") if hasattr(adapter, "complete") else None),
    ]
    
    for method_name, method_func in methods_to_try:
        if "a" in method_name and method_name.startswith("a"):
            # Async method
            try:
                print(f"Trying {method_name}...")
                response = await method_func()
                if response:
                    print(f"✅ Success with {method_name}: {response}")
                    break
                else:
                    print(f"❌ Method returned None")
            except AttributeError:
                print(f"❌ Method doesn't exist")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            # Sync method
            try:
                print(f"Trying {method_name}...")
                response = method_func()
                if response:
                    print(f"✅ Success with {method_name}: {response}")
                    break
                else:
                    print(f"❌ Method returned None")
            except AttributeError:
                print(f"❌ Method doesn't exist")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("\nDiagnostics complete!")

if __name__ == "__main__":
    asyncio.run(main())
