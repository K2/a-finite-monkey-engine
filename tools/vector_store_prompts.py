"""
Prompt generation for vector store issue analysis.

This module specializes in generating prompts that guide LLMs to discover issues in code
similar to those found in embedded GitHub issues, without explicitly revealing the issue.
"""

import re
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import aiohttp
import asyncio


class PromptGenerator:
    """
    Generate analysis prompts based on GitHub issues.
    
    These prompts are designed to guide LLMs to discover issues in new code
    similar to those found in the original GitHub issues, without explicitly
    stating what the issue is.
    """

    def __init__(
        self,
        generate_prompts: bool = True,
        use_ollama_for_prompts: bool = True,
        prompt_model: str = "gemma:2b",
        ollama_url: str = "http://localhost:11434",
        multi_llm_prompts: bool = False
    ):
        """Initialize the prompt generator with configuration settings."""
        self.generate_prompts = generate_prompts
        self.use_ollama_for_prompts = use_ollama_for_prompts
        self.prompt_model = prompt_model
        self.ollama_url = ollama_url
        self.multi_llm_prompts = multi_llm_prompts
        
        # Check if Ollama client is available and configure it
        try:
            import ollama
            # Configure client with custom base URL if needed
            if self.ollama_url != "http://localhost:11434":
                ollama.client.base_url = self.ollama_url
            self.ollama_client_available = True
            logger.info(f"Official Ollama client initialized with URL: {self.ollama_url}")
        except ImportError:
            self.ollama_client_available = False
            logger.warning("Official Ollama client not available. Install with: pip install ollama")
        logger.info(f"Prompt generation settings: enabled={self.generate_prompts}, "
                   f"use_ollama={self.use_ollama_for_prompts}, model={self.prompt_model}")

    async def generate_prompt(self, document: Dict[str, Any]) -> str:
        """
        Generate a prompt for a document.
        
        Args:
            document: Document with metadata
            
        Returns:
            Generated prompt
        """
        try:
            prompt = await self._generate_prompt_from_metadata(document)
            logger.info(f"Generated prompt [{len(prompt)} chars]: {prompt[:100]}..." if len(prompt) > 100 else prompt)
            return prompt
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return "Analyze this code for potential issues."

    async def generate_multi_llm_prompts(self, document: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate multiple specialized prompts for different LLM types.
        
        Args:
            document: Document with metadata
            
        Returns:
            Dictionary of prompt types to prompts
        """
        try:
            prompts = await self._generate_multi_llm_prompts_from_metadata(document)
            
            # Log the generated prompts
            logger.info("Generated multi-LLM prompts:")
            for prompt_type, prompt in prompts.items():
                if isinstance(prompt, str):
                    shortened = f"{prompt[:100]}..." if len(prompt) > 100 else prompt
                    logger.info(f"  - {prompt_type}: [{len(prompt)} chars] {shortened}")
                elif isinstance(prompt, list) and prompt:
                    logger.info(f"  - {prompt_type}: {len(prompt)} additional prompts")
                    
            return prompts
        except Exception as e:
            logger.error(f"Error generating multi-LLM prompts: {e}")
            return {
                "general": "Analyze this code for potential issues.",
                "security": "Check this code for security vulnerabilities."
            }

    async def _generate_prompt_from_metadata(self, document: Dict[str, Any]) -> str:
        """
        Generate a prompt that guides an LLM to analyze code and discover issues.
        Instead of describing the bug, we want to prompt the LLM to examine
        the code segments and discover the issue independently, similar to
        how the original reporter found it.
        
        Args:
            document: Document with metadata
        Returns:
            Generated prompt string
        """
        metadata = document.get('metadata', {})
        text = document.get('text', '')
        
        # Extract code blocks from the text
        code_segments = self._extract_code_segments(text)
        # Extract non-code text (for context understanding)
        non_code_text = self._extract_non_code_text(text, code_segments)
        # If no code segments found, use a small sample of the text
        if not code_segments:
            text_sample = text[:200] + "..." if len(text) > 200 else text
        else:
            # Use the first code segment as a sample
            text_sample = f"Code sample: {code_segments[0][:200]}..." if len(code_segments[0]) > 200 else code_segments[0]
        
        # Extract key metadata
        title = metadata.get('title', '')
        source = metadata.get('source', '')
        category = metadata.get('category', '')
        # Extract key issue information without revealing the specific issue
        issue_context = self._extract_issue_context(non_code_text, title)
        
        # Use Ollama for sophisticated prompt generation if available
        if self.use_ollama_for_prompts:
            try:
                # Build a system message explaining the goal
                system_message = (
                    "You are a prompt engineer specializing in creating prompts for code analysis. "
                    "Your task is to create a prompt that will guide an LLM to examine the code and "
                    "discover any issues or vulnerabilities independently, without explicitly stating what the issue is. "
                    "The prompt should encourage deep analysis of the code patterns and structures."
                )
                
                # Build a user message focused on code analysis
                user_message = f"""
Given this GitHub issue information:
- Title: {title}
- Category: {category}
- Source: {source}
- Context: {issue_context}

And this code sample:
```
{text_sample}
```

Generate a prompt that will guide an LLM to analyze the code and discover issues independently.
"""
                return await self._call_ollama_completion(system_message, user_message)
            except Exception as e:
                logger.error(f"Error using Ollama API: {e}")
                return "Analyze this code for potential issues and vulnerabilities."
        else:
            # Fallback prompt if Ollama is not used
            return f"""
Analyze this code for potential issues and vulnerabilities. Consider the following context:
- Title: {title}
- Category: {category}
- Source: {source}
- Context: {issue_context}
- Example: {text_sample}
"""

    async def _call_ollama_completion(self, system: str, user: str, model: str = None) -> Optional[str]:
        """
        Call Ollama API for completion.
        
        Args:
            system: System message
            user: User message
            model: Model to use (optional, defaults to self.prompt_model)
            
        Returns:
            Generated text or None if error
        """
        model_name = model or self.prompt_model
        logger.info(f"Generating prompt using Ollama model: {model_name}")
        
        # First try the official Python client
        try:
            # Import here to avoid dependency for those not using Ollama
            import ollama
            
            # Configure client with custom base URL if needed
            if self.ollama_url != "http://localhost:11434":
                ollama.set_host(self.ollama_url)
            
            logger.info(f"Using official Ollama client with model: {model_name}")
            
            # Use run_in_executor to make the synchronous API call asynchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ]
                )
            )
            
            if response and "message" in response and "content" in response["message"]:
                logger.info(f"Received response from Ollama Python client")
                return response["message"]["content"]
            
            logger.warning(f"Unexpected response format from Ollama client: {response}")
        except ImportError:
            logger.warning("Official Ollama client not available, trying direct API")
        except Exception as e:
            logger.warning(f"Error using official Ollama client: {e}, trying direct API")
        
        # Use direct API as fallback
        try:
            import aiohttp
            
            # Format the request payload correctly according to Ollama API docs
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            }
            
            logger.info(f"Calling Ollama API at {self.ollama_url}/api/chat with model {model_name}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "message" in result and "content" in result["message"]:
                            logger.info("Received response from Ollama API")
                            return result["message"]["content"]
                        logger.warning(f"Unexpected API response format: {result}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        # Suggest pulling the model if not found
                        if "model not found" in error_text.lower():
                            logger.error(f"Model {model_name} not found. Try running: ollama pull {model_name}")
        except Exception as e:
            logger.error(f"Error calling Ollama API directly: {e}")
        
        return None

    def _extract_code_segments(self, text: str) -> List[str]:
        """
        Extract code segments from the text.
        
        Args:
            text: Full document text
        Returns:
            List of code segments
        """
        return re.findall(r"```(?:python|java|javascript|c\+\+|c|solidity)?\n(.*?)```", text, re.DOTALL)

    def _extract_non_code_text(self, text: str, code_segments: List[str]) -> str:
        """
        Extract the non-code portions of the text.
        This is useful for extracting issue descriptions without code examples.
        
        Args:
            text: Full document text
            code_segments: Already extracted code segments
        Returns:
            Text with code segments removed
        """
        non_code = text
        # Remove markdown code blocks
        non_code = re.sub(r"```(?:python|java|javascript|c\+\+|c|solidity)?\n.*?```", "", non_code, flags=re.DOTALL)
        # Remove inline code segments
        for code in code_segments:
            non_code = non_code.replace(code, "")
        return non_code.strip()

    def _extract_issue_context(self, text: str, title: str) -> str:
        """
        Extract context information from the text and title.
        
        Args:
            text: Non-code text
            title: Issue title
        Returns:
            Context string
        """
        # Extract sentences containing keywords
        keywords = ['issue', 'problem', 'bug', 'vulnerability', 'error']
        context_sentences = [sentence for sentence in text.split('. ') if any(keyword in sentence.lower() for keyword in keywords)]
        
        # Combine context sentences with the title
        return f"{title}. {' '.join(context_sentences)}"