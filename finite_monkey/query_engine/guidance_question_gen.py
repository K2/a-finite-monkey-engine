"""
Guidance-based question generator for FLARE query engine.

This module provides a structured way to decompose complex queries
into sub-questions using Guidance for reliable structured output.
"""
from typing import List, Dict, Any, Optional, Callable
import asyncio
from loguru import logger

from ..models.structured_query import SubQuestion, QueryDecomposition
from ..utils.guidance_version_utils import (
    GUIDANCE_AVAILABLE,
    create_guidance_program
)
from ..nodes_config import config


class GuidanceQuestionGenerator:
    """
    Question generator using Guidance for reliable decomposition of complex queries.
    
    This generator forces the output to follow a specific schema, eliminating
    parsing errors and improving the robustness of the FLARE engine.
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
            provider: Provider to use
            verbose: Whether to enable verbose output
        """
        self.model = model or getattr(config, "REASONING_MODEL", config.DEFAULT_MODEL)
        self.provider = provider or getattr(config, "REASONING_MODEL_PROVIDER", config.DEFAULT_PROVIDER)
        self.verbose = verbose
        
        # Check for guidance availability
        if not GUIDANCE_AVAILABLE:
            logger.warning("Guidance not available, will use fallback methods")
    
    async def generate(
        self, 
        query: str,
        tools: List[Dict[str, Any]],
        fallback_fn: Optional[Callable] = None
    ) -> List[SubQuestion]:
        """
        Generate sub-questions for a complex query using available tools.
        
        Args:
            query: Main query to decompose
            tools: List of available tools with metadata
            fallback_fn: Optional fallback function if guidance fails
            
        Returns:
            List of structured SubQuestion objects
        """
        if self.verbose:
            logger.debug(f"Generating sub-questions for: {query}")
            logger.debug(f"Available tools: {len(tools)}")
        
        if GUIDANCE_AVAILABLE:
            sub_questions = await self._generate_with_guidance(query, tools)
            if sub_questions:
                return sub_questions
        
        # Try fallback if guidance failed or is unavailable
        if fallback_fn is not None:
            try:
                logger.info("Using fallback function for question generation")
                return await fallback_fn(query, tools)
            except Exception as e:
                logger.error(f"Fallback function failed: {e}")
        
        # Last resort: use standard generation
        return await self._generate_standard(query, tools)
    
    async def _generate_with_guidance(self, query: str, tools: List[Dict[str, Any]]) -> Optional[List[SubQuestion]]:
        """Generate sub-questions using Guidance for structured output"""
        try:
            # Format tools for the prompt
            tools_str = "\n".join([
                f"- {tool.get('name', '')}: {tool.get('description', '')}" 
                for tool in tools
            ])
            
            # Create prompt template with schema
            prompt_template = f"""
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
            
            # Create guidance program
            program = await create_guidance_program(
                output_cls=QueryDecomposition,
                prompt_template=prompt_template,
                model=self.model,
                provider=self.provider,
                verbose=self.verbose
            )
            
            if not program:
                logger.warning("Failed to create guidance program")
                return None
            
            # Run the program
            try:
                result = await self._run_program(program, query=query)
                
                if isinstance(result, QueryDecomposition):
                    if self.verbose:
                        logger.debug(f"Generated {len(result.sub_questions)} sub-questions with guidance")
                    return result.sub_questions
                elif isinstance(result, dict) and "sub_questions" in result:
                    # Try to convert dict result to SubQuestion objects
                    sub_questions = []
                    for sq_data in result["sub_questions"]:
                        sub_questions.append(SubQuestion(**sq_data))
                    return sub_questions
                else:
                    logger.warning(f"Unexpected result type: {type(result)}")
                    return None
            except Exception as e:
                logger.error(f"Error running guidance program: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error in guidance-based question generation: {e}")
            return None
    
    async def _run_program(self, program, **kwargs):
        """Run a guidance program with proper async handling"""
        if asyncio.iscoroutinefunction(program.__call__):
            # Program is already async
            return await program(**kwargs)
        else:
            # Run sync program in executor to not block
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: program(**kwargs)
            )
    
    async def _generate_standard(self, query: str, tools: List[Dict[str, Any]]) -> List[SubQuestion]:
        """Standard fallback method for generating sub-questions without guidance"""
        try:
            # Use LLM directly
            from ..llm.llama_index_adapter import LlamaIndexAdapter
            
            # Create adapter
            llm_adapter = LlamaIndexAdapter(
                model_name=self.model,
                provider=self.provider
            )
            
            # Format tools for prompt
            tools_str = "\n".join([
                f"- {tool.get('name', '')}: {tool.get('description', '')}" 
                for tool in tools
            ])
            
            # Create prompt
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

Example format:
```json
[
  {"text": "What is X?", "tool_name": "tool_name"},
  {"text": "How does Y work?", "tool_name": "other_tool_name"}
]
```
            """
            
            # Ask LLM
            from llama_index.core.llms import ChatMessage, MessageRole
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant that decomposes queries."),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            
            response = await llm_adapter.llm.achat(messages)
            response_text = response.message.content
            
            # Extract JSON from response
            import json
            import re
            
            # Find JSON array
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
                
                # Create generic sub-questions as last resort
                return [
                    SubQuestion(
                        text=f"What information about {tool.get('name', '')} is relevant to the query: {query}?",
                        tool_name=tool.get('name', '')
                    )
                    for tool in tools[:3]  # Limit to avoid overwhelming
                ]
            
        except Exception as e:
            logger.error(f"Error in standard question generation: {e}")
            # Return minimal fallback
            return [
                SubQuestion(
                    text=query,
                    tool_name=tools[0].get('name', '') if tools else ""
                )
            ]
