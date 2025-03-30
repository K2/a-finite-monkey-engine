"""
Test direct LLM integration with template handling as an alternative to guidance decorators.
"""
import asyncio
import json
import re
from pydantic import BaseModel
from typing import List, Any, Dict

class TestOutput(BaseModel):
    message: str
    items: List[str] = []

async def test_direct_template_approach():
    """Test the direct template approach without guidance decorators."""
    try:
        from finite_monkey.utils.guidance_version_utils import _create_llm
        
        # Create a simple template with handlebars syntax
        template = """
        You are a helpful assistant.
        
        {{#if question}}
        Question: {{question}}
        {{/if}}
        
        Please provide a response in the following JSON format:
        {
            "message": "Your response to the question",
            "items": ["item1", "item2", "item3"]
        }
        """
        
        # Create an LLM instance
        llm = await _create_llm(model="dolphin3:8b-llama3.1-q8_0", provider="ollama")
        
        if not llm:
            print("❌ Failed to create LLM")
            return
        
        print("✅ Created LLM successfully")
        
        # Replace variables in the template
        question = "List three programming languages"
        filled_template = template.replace("{{question}}", question)
        filled_template = filled_template.replace("{{#if question}}", "")
        filled_template = filled_template.replace("{{/if}}", "")
        
        print(f"📝 Filled template:\n{filled_template}")
        
        # Send to LLM - handle both synchronous and asynchronous APIs
        print("🔄 Sending to LLM...")
        
        response = None
        response_text = None
        
        try:
            if hasattr(llm, "complete") and callable(llm.complete):
                print("Using LLM.complete method")
                # Try calling without await first (for synchronous APIs)
                try:
                    response = llm.complete(filled_template)
                    print(f"  → Sync response type: {type(response)}")
                    # If we got here, it's a synchronous API
                except Exception as sync_e:
                    print(f"  → Sync call failed: {sync_e}, trying async...")
                    # If that failed, it might be an async API
                    try:
                        import inspect
                        if inspect.iscoroutinefunction(llm.complete):
                            response = await llm.complete(filled_template)
                            print(f"  → Async response type: {type(response)}")
                    except Exception as async_e:
                        print(f"  → Async call also failed: {async_e}")
            elif hasattr(llm, "generate") and callable(llm.generate):
                print("Using LLM.generate method")
                # Same approach for generate
                try:
                    response = llm.generate(filled_template)
                except:
                    try:
                        import inspect
                        if inspect.iscoroutinefunction(llm.generate):
                            response = await llm.generate(filled_template)
                    except Exception as e:
                        print(f"  → Generate failed: {e}")
            elif callable(llm):
                print("Using direct LLM call")
                response = llm(filled_template)
        except Exception as e:
            print(f"❌ All LLM call methods failed: {e}")
            return
            
        if response is None:
            print("❌ Failed to get a response from the LLM")
            return
            
        # Extract the text from the response, which might be an object
        if hasattr(response, "text"):
            response_text = response.text
        elif hasattr(response, "response"):
            response_text = response.response
        elif hasattr(response, "content"):
            response_text = response.content
        elif hasattr(response, "completion"):
            response_text = response.completion
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict) and "text" in response:
            response_text = response["text"]
        elif isinstance(response, dict) and "content" in response:
            response_text = response["content"]
        elif isinstance(response, dict) and "completion" in response:
            response_text = response["completion"]
        else:
            # Last resort, convert to string
            response_text = str(response)
        
        print(f"✅ Got response type: {type(response)}")
        print(f"✅ Got response text: {response_text[:100]}...")
        
        # Extract JSON
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(json_pattern, response_text)
        
        if not matches:
            print("❌ No JSON found in response")
            return
        
        for match in matches:
            try:
                parsed = json.loads(match)
                print(f"✅ Parsed JSON: {parsed}")
                
                # Validate with pydantic
                result = TestOutput(**parsed)
                print(f"✅ Validated result: {result}")
                return
            except json.JSONDecodeError:
                continue
        
        print("❌ Failed to parse any JSON from response")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_template_approach())
