"""
Test the integrity of the guidance_version_utils module to prevent corruption.
"""
import os
import sys
import importlib
import hashlib
import inspect
import asyncio
from typing import Dict, List, Any, Optional

async def test_module_imports():
    """Test that the module can be imported without errors."""
    try:
        # Remove module from sys.modules if it's already loaded
        if "finite_monkey.utils.guidance_version_utils" in sys.modules:
            del sys.modules["finite_monkey.utils.guidance_version_utils"]
        
        # Import the module
        import finite_monkey.utils.guidance_version_utils as gvu
        
        # Check critical attributes
        assert hasattr(gvu, "DirectTemplateHandler"), "DirectTemplateHandler not found"
        assert hasattr(gvu, "create_guidance_program"), "create_guidance_program not found"
        assert hasattr(gvu, "GuidanceProgramWrapper"), "GuidanceProgramWrapper not found"
        
        print("✅ Module imports test passed")
        return True
    except Exception as e:
        print(f"❌ Module imports test failed: {e}")
        return False

async def test_direct_template_handler_integrity():
    """Test the integrity of the DirectTemplateHandler class."""
    try:
        from finite_monkey.utils.guidance_version_utils import DirectTemplateHandler
        
        # Create an instance
        handler = DirectTemplateHandler("template", None)
        
        # Check essential methods
        essential_methods = [
            "__call__", 
            "_process_template", 
            "_process_conditionals", 
            "_call_llm", 
            "_extract_text_from_response", 
            "_extract_structured_data"
        ]
        
        for method in essential_methods:
            assert hasattr(handler, method), f"Method {method} not found"
            assert callable(getattr(handler, method)), f"Method {method} is not callable"
        
        print("✅ DirectTemplateHandler integrity test passed")
        return True
    except Exception as e:
        print(f"❌ DirectTemplateHandler integrity test failed: {e}")
        return False

async def test_sync_async_compatibility():
    """Test synchronous and asynchronous compatibility."""
    try:
        from finite_monkey.utils.guidance_version_utils import DirectTemplateHandler
        import inspect
        
        # Create a mock synchronous LLM
        class MockSyncLLM:
            def complete(self, template):
                return f"Completed: {template}"
                
        # Create a mock asynchronous LLM
        class MockAsyncLLM:
            async def complete(self, template):
                return f"Async completed: {template}"
        
        # Test with sync LLM
        sync_handler = DirectTemplateHandler("Hello {{name}}", MockSyncLLM())
        assert not inspect.iscoroutinefunction(sync_handler.llm.complete), "Should detect sync method"
        
        # Test with async LLM
        async_handler = DirectTemplateHandler("Hello {{name}}", MockAsyncLLM())
        assert inspect.iscoroutinefunction(async_handler.llm.complete), "Should detect async method"
        
        print("✅ Sync/async compatibility test passed")
        return True
    except Exception as e:
        print(f"❌ Sync/async compatibility test failed: {e}")
        return False

async def main():
    """Run all integrity tests."""
    print("Running guidance_version_utils integrity tests...")
    
    tests = [
        test_module_imports(),
        test_direct_template_handler_integrity(),
        test_sync_async_compatibility()
    ]
    
    results = await asyncio.gather(*tests)
    
    if all(results):
        print("All integrity tests passed! The module appears to be intact.")
    else:
        print("Some integrity tests failed. The module might be corrupted.")

if __name__ == "__main__":
    asyncio.run(main())
