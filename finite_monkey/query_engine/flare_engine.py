"""
Implementation of a FLARE Instruction Query Engine for enhanced
reasoning capabilities in code analysis tasks.
"""
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import BaseModel, Field
from loguru import logger

from finite_monkey.pipeline.core import Context
from finite_monkey.nodes_config import config
from .base_engine import BaseQueryEngine, QueryResult
from .guidance_question_gen import GuidanceQuestionGenerator, GUIDANCE_AVAILABLE
from ..models.query import SubQuestion


class FlareQueryEngine(BaseQueryEngine):
    """
    FLARE (Forward-Looking Active REasoning) query engine implementation.
    
    This engine breaks complex queries into sub-questions, answers those
    sub-questions, and then synthesizes a final answer.
    """
    
    def __init__(
        self,
        underlying_engine: BaseQueryEngine,
        max_iterations: int = 3,
        verbose: bool = False,
        use_guidance: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the FLARE query engine.
        
        Args:
            underlying_engine: Base query engine for answering sub-questions
            max_iterations: Maximum number of reasoning iterations
            verbose: Whether to enable verbose logging
            use_guidance: Whether to use Guidance for structured outputs
            config: Additional configuration options
        """
        super().__init__()
        self.underlying_engine = underlying_engine
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.config = config or {}
        
        # Set up the question generator
        self.use_guidance = use_guidance and GUIDANCE_AVAILABLE
        if self.use_guidance:
            try:
                self.question_generator = GuidanceQuestionGenerator(
                    model=getattr(config, "REASONING_MODEL", config.DEFAULT_MODEL),
                    provider=getattr(config, "REASONING_MODEL_PROVIDER", config.DEFAULT_PROVIDER),
                    verbose=verbose
                )
                logger.info("Using Guidance-based question generator for FLARE engine")
            except Exception as e:
                logger.error(f"Failed to initialize Guidance question generator: {e}")
                self.use_guidance = False
                # Will fall back to standard question generator below
        
        # Create standard question generator as fallback or default
        if not self.use_guidance or not hasattr(self, 'question_generator'):
            self.question_generator = GuidanceQuestionGenerator(verbose=verbose)
            logger.info("Using standard question generator for FLARE engine")
        
        # Initialize LLM for synthesis
        try:
            from ..llm.llama_index_adapter import LlamaIndexAdapter
            
            model_name = self.config.get("model", config.REASONING_MODEL)
            self.llm_adapter = LlamaIndexAdapter(
                model_name=model_name,
                provider=config.REASONING_MODEL_PROVIDER,
                base_url=config.REASONING_MODEL_BASE_URL,
                request_timeout=config.REQUEST_TIMEOUT
            )
            logger.info(f"Initialized synthesis LLM with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize synthesis LLM: {e}")
            self.llm_adapter = None
    
    async def initialize(self) -> None:
        """Initialize the FLARE engine and its components"""
        logger.info("Initializing FLARE query engine")
        
        # Initialize underlying engine if needed
        if hasattr(self.underlying_engine, 'initialize'):
            await self.underlying_engine.initialize()
    
    async def query(self, query: str, context: Optional[Context] = None) -> QueryResult:
        """
        Execute a query using the FLARE approach
        
        Args:
            query: The query to answer
            context: Optional context with additional information
            
        Returns:
            QueryResult containing the response and metadata
        """
        logger.info(f"FLARE engine processing query: {query}")
        
        # Initialize result with default values
        result = QueryResult(
            query=query,
            response="",
            confidence=0.0,
            metadata={}
        )
        
        # Check if we have the required components
        if not self.llm_adapter or not self.underlying_engine:
            logger.error("Missing required components for FLARE query execution")
            result.response = "Error: FLARE engine is not properly initialized"
            return result
        
        try:
            # Step 1: Decompose the query into sub-questions
            sub_questions = await self._decompose_query(query, self._get_tools())
            
            if not sub_questions:
                logger.warning("Could not decompose query into sub-questions")
                # Fall back to direct query
                direct_result = await self.underlying_engine.query(query, context)
                return direct_result
            
            # Store sub-questions for metadata
            sq_data = [sq.dict() for sq in sub_questions]
            result.sub_questions = sq_data
            
            # Step 2: Answer each sub-question
            sub_answers = await self._answer_sub_questions(sub_questions, context)
            
            # Step 3: Synthesize the final answer
            final_answer = await self._synthesize_answer(query, sub_questions, sub_answers)
            
            # Step 4: Set result fields
            result.response = final_answer
            result.confidence = self._calculate_confidence(sub_questions, sub_answers)
            result.metadata = {
                "sub_questions": sq_data,
                "sub_answers": sub_answers,
                "iterations": 1  # For future multi-iteration implementation
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in FLARE query execution: {e}")
            result.response = f"Error processing query: {str(e)}"
            return result
    
    async def _decompose_query(
        self, 
        query: str, 
        tools: List[Dict[str, Any]]
    ) -> List[SubQuestion]:
        """
        Decompose a complex query into sub-questions using available tools
        
        Args:
            query: The main query to decompose
            tools: List of available tools with their metadata
            
        Returns:
            List of structured SubQuestion objects
        """
        logger.debug(f"Decomposing query: {query}")
        
        if self.use_guidance and hasattr(self, 'question_generator'):
            # Use Guidance-based generator
            sub_questions = await self.question_generator.generate(
                query=query, 
                tools=tools,
                fallback_fn=self._standard_decompose_query
            )
            
            # Convert to expected format
            return [sq.dict() for sq in sub_questions]
        else:
            # Use standard approach
            return await self._standard_decompose_query(query, tools)
    
    async def _standard_decompose_query(self, query: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standard method for decomposing queries (no Guidance)"""
        # Use the question generator to decompose the query
        sub_questions = await self.question_generator.generate(query, tools)
        
        if self.verbose:
            logger.debug(f"Generated {len(sub_questions)} sub-questions")
            for i, sq in enumerate(sub_questions):
                logger.debug(f"  {i+1}. {sq.text} (Tool: {sq.tool_name})")
        
        return sub_questions
    
    async def _answer_sub_questions(
        self,
        sub_questions: List[SubQuestion],
        context: Optional[Context]
    ) -> Dict[str, str]:
        """
        Answer each sub-question using the underlying engine
        
        Args:
            sub_questions: List of sub-questions to answer
            context: Optional context with additional information
            
        Returns:
            Dictionary mapping sub-question IDs to answers
        """
        logger.debug(f"Answering {len(sub_questions)} sub-questions")
        
        answers = {}
        
        # Answer each sub-question concurrently
        async def answer_question(idx: int, sq: SubQuestion):
            logger.debug(f"Answering sub-question {idx+1}: {sq.text}")
            try:
                # Use the underlying engine to answer the sub-question
                result = await self.underlying_engine.query(sq.text, context)
                return idx, sq.text, result.response
            except Exception as e:
                logger.error(f"Error answering sub-question {idx}: {e}")
                return idx, sq.text, f"Error: {str(e)}"
        
        # Create tasks for all sub-questions
        tasks = [answer_question(i, sq) for i, sq in enumerate(sub_questions)]
        results = await asyncio.gather(*tasks)
        
        # Process results
        for idx, question, answer in results:
            answers[f"question_{idx}"] = {
                "question": question,
                "answer": answer
            }
        
        return answers
    
    async def _synthesize_answer(
        self,
        query: str,
        sub_questions: List[SubQuestion],
        sub_answers: Dict[str, Dict[str, str]]
    ) -> str:
        """
        Synthesize a final answer from the sub-question answers
        
        Args:
            query: The original query
            sub_questions: List of sub-questions
            sub_answers: Dictionary of answers to sub-questions
            
        Returns:
            Synthesized final answer
        """
        logger.debug("Synthesizing final answer")
        
        # Format the sub-questions and answers for the prompt
        qa_pairs = []
        for i, sq in enumerate(sub_questions):
            answer_data = sub_answers.get(f"question_{i}")
            if answer_data:
                qa_pairs.append(f"Question: {sq.text}\nAnswer: {answer_data['answer']}")
        
        qa_text = "\n\n".join(qa_pairs)
        
        # Create synthesis prompt
        prompt = f"""
I need to answer the following query:
{query}

To help answer this query, I've broken it down into sub-questions and found answers for each:

{qa_text}

Based on these sub-answers, provide a comprehensive response to the original query.
Synthesize the information coherently, cite relevant details from the sub-answers,
and ensure the response directly addresses the original query.
        """
        
        from llama_index.core.llms import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an expert at synthesizing information."),
            ChatMessage(role=MessageRole.USER, content=prompt)
        ]
        
        # Get synthesized answer from LLM
        response = await self.llm_adapter.llm.achat(messages)
        synthesized_answer = response.message.content
        
        return synthesized_answer
    
    def _calculate_confidence(
        self,
        sub_questions: List[SubQuestion],
        sub_answers: Dict[str, Dict[str, str]]
    ) -> float:
        """
        Calculate a confidence score for the answer
        
        Args:
            sub_questions: List of sub-questions
            sub_answers: Dictionary of answers to sub-questions
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Simple heuristic: confidence is proportional to the number of 
        # sub-questions that were successfully answered
        answered_count = sum(1 for i in range(len(sub_questions)) 
                             if f"question_{i}" in sub_answers 
                             and "Error" not in sub_answers[f"question_{i}"].get("answer", ""))
        
        if not sub_questions:
            return 0.0
            
        return min(1.0, answered_count / len(sub_questions))
    
    def _get_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools for query decomposition
        
        Returns:
            List of tool metadata
        """
        # Simple tool representing the underlying engine
        return [{
            "name": "knowledge_base",
            "description": "Answers questions about the smart contracts and code"
        }]
    
    async def shutdown(self) -> None:
        """
        Clean up resources when the engine is no longer needed.
        """
        self.flare_engine = None
        self.underlying_engine = None
        self.logger.info("FLARE engine resources released")