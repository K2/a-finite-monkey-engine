"""
Test the enhanced template processing capabilities.
"""
import asyncio
import re
import json
from typing import Dict, Any, List
from loguru import logger
from pydantic import BaseModel

from finite_monkey.utils.guidance_version_utils import DirectTemplateHandler

class TestResponse(BaseModel):
    message: str
    items: List[str] = []

async def test_variable_replacement():
    """Test basic variable replacement."""
    template = "Hello {{name}}!"
    handler = DirectTemplateHandler(template, None)
    
    # Process template without calling LLM
    processed = handler._process_template({"name": "World"})
    
    assert processed == "Hello World!"
    print("✅ Variable replacement test passed")

async def test_conditional_blocks():
    """Test conditional block processing."""
    template = """
    {{#if question}}
    Q: {{question}}
    {{/if}}
    {{#if context}}
    Context: {{context}}
    {{/if}}
    """
    handler = DirectTemplateHandler(template, None)
    
    # With both variables
    processed1 = handler._process_template({
        "question": "What is the capital of France?",
        "context": "France is in Europe."
    })
    
    # With only question
    processed2 = handler._process_template({
        "question": "What is the capital of France?"
    })
    
    # Empty data
    processed3 = handler._process_template({})
    
    assert "Q: What is the capital of France?" in processed1
    assert "Context: France is in Europe." in processed1
    assert "Q: What is the capital of France?" in processed2
    assert "Context:" not in processed2
    assert "Q:" not in processed3
    assert "Context:" not in processed3
    
    print("✅ Conditional blocks test passed")

async def test_loop_blocks():
    """Test loop block processing."""
    template = """
    Items:
    {{#each items}}
    - {{this}}
    {{/each}}
    
    People:
    {{#each people}}
    - {{this.name}} ({{this.age}})
    {{/each}}
    """
    handler = DirectTemplateHandler(template, None)
    
    processed = handler._process_template({
        "items": ["Apple", "Banana", "Cherry"],
        "people": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
    })
    
    assert "- Apple" in processed
    assert "- Banana" in processed
    assert "- Cherry" in processed
    assert "- Alice (30)" in processed
    assert "- Bob (25)" in processed
    
    print("✅ Loop blocks test passed")

async def test_json_extraction():
    """Test JSON extraction from responses."""
    handler = DirectTemplateHandler("", None)
    
    # Test with clean JSON
    text1 = '{"message": "Hello", "items": ["A", "B", "C"]}'
    data1 = handler._extract_structured_data(text1)
    assert data1["message"] == "Hello"
    assert len(data1["items"]) == 3
    
    # Test with JSON embedded in text
    text2 = """
    The answer is:
    {"message": "Hello", "items": ["A", "B", "C"]}
    Thank you!
    """
    data2 = handler._extract_structured_data(text2)
    assert data2["message"] == "Hello"
    assert len(data2["items"]) == 3
    
    # Test with invalid JSON
    text3 = "No JSON here"
    data3 = handler._extract_structured_data(text3)
    assert "result" in data3
    assert data3["result"] == "No JSON here"
    
    print("✅ JSON extraction test passed")

async def test_complex_template():
    """Test a template with multiple features."""
    template = """
    {{#if title}}
    # {{title}}
    {{/if}}
    
    {{#if intro}}
    {{intro}}
    {{/if}}
    
    {{#each sections}}
    ## {{this.heading}}
    
    {{this.content}}
    
    {{#each this.items}}
    - {{this}}
    {{/each}}
    {{/each}}
    """
    
    handler = DirectTemplateHandler(template, None)
    
    processed = handler._process_template({
        "title": "Test Document",
        "intro": "This is a test of complex template processing.",
        "sections": [
            {
                "heading": "First Section", 
                "content": "This is the first section content.",
                "items": ["Item 1.1", "Item 1.2", "Item 1.3"]
            },
            {
                "heading": "Second Section", 
                "content": "This is the second section content.",
                "items": ["Item 2.1", "Item 2.2"]
            }
        ]
    })
    
    assert "# Test Document" in processed
    assert "This is a test of complex template processing." in processed
    assert "## First Section" in processed
    assert "## Second Section" in processed
    assert "- Item 1.1" in processed
    assert "- Item 2.2" in processed
    
    print("✅ Complex template test passed")

async def main():
    """Run all tests."""
    print("Running template processing tests...")
    await test_variable_replacement()
    await test_conditional_blocks()
    await test_loop_blocks()
    await test_json_extraction()
    await test_complex_template()
    print("All tests passed! ✅")

if __name__ == "__main__":
    asyncio.run(main())
