"""
Tests for the universal LLM interface adapter.
"""
import pytest
import asyncio
from typing import List, Dict

from finite_monkey.utils.llm_interface import call_llm, extract_text_from_response

class MockAsyncLLM:
    """Mock LLM with async chat interface"""
    async def achat(self, messages: List[Dict[str, str]]):
        return {"message": {"content": f"Response to: {messages[-1]['content']}"}}

class MockSyncLLM:
    """Mock LLM with sync complete interface"""
    def complete(self, prompt: str):
        return f"Completion for: {prompt}"

class MockCallableLLM:
    """Mock LLM that's directly callable"""
    def __call__(self, prompt: str):
        return {"text": f"Called with: {prompt}"}

class MockAdapter:
    """Mock LLM adapter that wraps another LLM"""
    def __init__(self, llm):
        self.llm = llm

@pytest.mark.asyncio
async def test_call_llm_with_async_chat():
    """Test calling LLM with async chat interface"""
    llm = MockAsyncLLM()
    response = await call_llm(llm, "Test prompt", as_chat=True)
    assert "Response to: Test prompt" in extract_text_from_response(response)

@pytest.mark.asyncio
async def test_call_llm_with_sync_complete():
    """Test calling LLM with sync complete interface"""
    llm = MockSyncLLM()
    response = await call_llm(llm, "Test prompt", as_chat=False)
    assert "Completion for: Test prompt" in extract_text_from_response(response)

@pytest.mark.asyncio
async def test_call_llm_with_callable():
    """Test calling LLM that's directly callable"""
    llm = MockCallableLLM()
    response = await call_llm(llm, "Test prompt")
    assert "Called with: Test prompt" in extract_text_from_response(response)

@pytest.mark.asyncio
async def test_call_llm_with_adapter():
    """Test calling LLM through an adapter"""
    llm = MockAdapter(MockAsyncLLM())
    response = await call_llm(llm, "Test prompt", as_chat=True)
    assert "Response to: Test prompt" in extract_text_from_response(response)

@pytest.mark.asyncio
async def test_extract_text_from_response_formats():
    """Test extracting text from various response formats"""
    responses = [
        "Plain text",
        {"content": "Dict content"},
        {"message": {"content": "Nested content"}},
        {"text": "Dict text"},
        {"completion": "Dict completion"},
        type("ObjWithText", (), {"text": "Object text"})(),
        type("ObjWithMessage", (), {"message": type("Message", (), {"content": "Message content"})()})()
    ]
    
    for response in responses:
        text = extract_text_from_response(response)
        assert isinstance(text, str)
        assert len(text) > 0
