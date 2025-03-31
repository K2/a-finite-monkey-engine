"""
Question generator for decomposing complex queries using Guidance.

This module provides a reliable way to decompose queries into sub-questions
using structured output via Guidance.
"""
from typing import List, Dict, Any, Optional, Callable
import asyncio
import json
from loguru import logger

from .core import create_structured_program, GUIDANCE_AVAILABLE
from .models import SubQuestion, QuestionDecompositionResult
from ..nodes_config import config


class GuidanceQuestionGenerator:
    """
    Question generator using Guidance for structured decomposition of complex queries.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the question generator.
        
        Args:
            model: Model name to use
            provider: Model provider
            verbose: Whether to enable verbose logging
        """
        self.model = model or getattr(config, "REASONING_MODEL", config.DEFAULT_MODEL)
        self.provider = provider or getattr(config, "REASONING_MODEL_PROVIDER", config.DEFAULT_PROVIDER)
        self.verbose = verbose
        
        if not GUIDANCE_AVAILABLE:
            logger.warning("Guidance not available, question generator will use fallbacks")
    
    async def generate(
        self, 
        query: str,
        tools: List[Dict[str, Any]],
        fallback_fn: Optional[Callable] = None
    ) -> List[SubQuestion]:
        """
        Generate sub-questions for a complex query using available tools.
        
        Args:
            query: The main query to decompose
            tools: List of available tools with their metadata
            fallback_fn: Optional fallback function to use if Guidance fails
            
        Returns:
            List of structured SubQuestion objects
        """
        if self.verbose:
            logger.debug(f"Generating sub-questions for query: {query}")
            logger.debug(f"Available tools: {len(tools)}")
            
        if GUIDANCE_AVAILABLE:
            result = await self._generate_with_guidance(query, tools, fallback_fn)
            if result:
                return result
        
        # If we get here, either Guidance is not available or it failed
        if fallback_fn:
            try:
                logger.info("Using fallback method for question generation")
                return await fallback_fn(query, tools)
            except Exception as e:
                logger.error(f"Fallback method failed: {e}")
                
        # Last resort: use our standard generator
        return await self._generate_standard(query, tools)
    
    async def _generate_with_guidance(
        self, 
        query: str,
        tools: List[Dict[str, Any]],
        fallback_fn: Optional[Callable] = None
    ) -> Optional[List[SubQuestion]]:
        """Generate sub-questions using Guidance for structured output"""
        try:
            # Format tools for prompt
            tools_str = "\n".join([
                f"- {tool.get('name', '')}: {tool.get('description', '')}" 
                for tool in tools
            ])
            
            # Create handlebars template for guidance
            prompt = f"""
You are an expert at breaking down complex queries into simpler sub-questions.

Main Query: {{{{query}}}}

Available Tools:
{tools_str}

Break down the main query into specific sub-questions that can be answered using 
the available tools. Each sub-question should target a specific aspect of the 
main query and should be answerable by one of the tools.

Generate a structured decomposition with sub-questions and tool assignments.

{{{{#schema}}}}
{{{{
  "sub_questions": [
    {{{{
      "text": "What is X?",
      "tool_name": "tool_name",
      "reasoning": "This sub-question helps address the main query by..."
    }}}}
  ],
  "reasoning": "Explanation of why the query was decomposed this way"
}}}}
{{{{/schema}}}}
            """
            
            # Create structured program
            program = await create_structured_program(
                output_cls=QuestionDecompositionResult,
                prompt_template=prompt,
                model=self.model,
                provider=self.provider,
                fallback_fn=fallback_fn,
                verbose=self.verbose
            )
            
            if not program:
                logger.warning("Failed to create Guidance program for question generation")
                return None
                
            # Execute the program
            result = await program(query=query)
            
            if isinstance(result, QuestionDecompositionResult):
                if self.verbose:
                    logger.debug(f"Generated {len(result.sub_questions)} sub-questions with guidance")
                    if result.reasoning:
                        logger.debug(f"Reasoning: {result.reasoning}")
                
                return result.sub_questions
            elif isinstance(result, dict) and "sub_questions" in result:
                # Try to convert dict to SubQuestion objects
                sub_questions = []
                for sq_data in result["sub_questions"]:
                    try:
                        sub_questions.append(SubQuestion(**sq_data))
                    except Exception as e:
                        logger.error(f"Error converting sub-question data: {e}")
                
                return sub_questions
            else:
                logger.warning(f"Unexpected result type: {type(result)}")
                return None
                
        except Exception as e:
            logger.error(f"Error in guidance-based question generation: {e}")
            return None
    
    async def _generate_standard(
        self, 
        query: str,
        tools: List[Dict[str, Any]]
    ) -> List[SubQuestion]:
        """Standard fallback method for generating sub-questions without guidance"""
        try:
            # Use LLM to generate sub-questions
            from ..llm.llama_index_adapter import LlamaIndexAdapter
            
            # Create LLM adapter
            llm_adapter = LlamaIndexAdapter(
                model_name=self.model,
                provider=self.provider
            )
            
            # Format tools for prompt
            tools_str = "\n".join([
                f"- {tool.get('name', '')}: {tool.get('description', '')}" 
                for tool in tools
            ])
            
            # Create the prompt
            prompt = f"""
You are an expert at breaking down complex queries into simpler sub-questions.

Main Query: {query}

Available Tools:
{tools_str}

Break down the main query into specific sub-questions that can be answered using 
the available tools. Each sub-question should target a specific aspect of the 
main query and should be answerable by one of the tools.

Respond with a JSON array of sub-questions, each with a 'text' field for the question 
and a 'tool_name' field specifying which tool should be used to answer it.

For example:
```json
[
  {{"text": "What is X?", "tool_name": "tool_name"}},
  {{"text": "How does Y work?", "tool_name": "other_tool_name"}}
]
```
            """
            
            # Get response from LLM
            from llama_index.core.llms import ChatMessage, MessageRole
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant that decomposes queries."),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            response = await llm_adapter.llm.achat(messages)
            response_text = response.message.content
            
            # Extract JSON array from the response
            import json
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                sub_questions_data = json.loads(json_str)
                
                # Convert to SubQuestion objects
                sub_questions = []
                for sq_data in sub_questions_data:
                    sub_questions.append(SubQuestion(
                        text=sq_data.get("text", ""),
                        tool_name=sq_data.get("tool_name", "")
                    ))
                
                if self.verbose:
                    logger.debug(f"Generated {len(sub_questions)} sub-questions with standard method")
                
                return sub_questions
            else:
                logger.warning("Could not extract JSON array from LLM response")
                
                # Last resort: create a generic sub-question for each tool
                logger.warning("Creating generic sub-questions for tools")
                return [
                    SubQuestion(
                        text=f"What information about {tool.get('name', '')} is relevant to the query: {query}?",
                        tool_name=tool.get('name', '')
                    )
                    for tool in tools[:3]  # Limit to first 3 tools to avoid overwhelming
                ]
            
        except Exception as e:
            logger.error(f"Error in standard question generation: {e}")
            # Return minimal fallback questions
            return [
                SubQuestion(
                    text=query,
                    tool_name=tools[0].get('name', '') if tools else ""
                )
            ]
