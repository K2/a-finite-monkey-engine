"""
Cognitive bias analyzer for smart contracts.

This module analyzes smart contracts for potential cognitive biases in the
code structure, developer assumptions, and design decisions.
"""

import os
import re
import json
import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger

from ..pipeline.core import Context
from ..models.analysis import BiasAnalysisResult, AssumptionAnalysis
from ..utils.json_repair import safe_parse_json, extract_json_from_text
from ..llm.llama_index_adapter import LlamaIndexAdapter
from finite_monkey.nodes_config import config

class CognitiveBiasAnalyzer:
    """
    Analyzer for cognitive biases in smart contracts.
    
    Identifies potential cognitive biases in contract design and implementation
    that could lead to security vulnerabilities or logic errors.
    """
    
    def __init__(self, llm_adapter: Optional[LlamaIndexAdapter] = None):
        """
        Initialize the cognitive bias analyzer
        
        Args:
            llm_adapter: LlamaIndex adapter for LLM access
        """     
        # Check if we need to create a default LLM adapter
        if llm_adapter is None:
            try:
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                
                # Use cognitive bias model configuration
                self.llm_adapter = LlamaIndexAdapter(
                    model_name=config.COGNITIVE_BIAS_MODEL,
                    provider=config.COGNITIVE_BIAS_MODEL_PROVIDER,
                    base_url=config.COGNITIVE_BIAS_MODEL_BASE_URL
                )
                logger.info(f"Created cognitive bias LLM adapter with model: {config.COGNITIVE_BIAS_MODEL}")
                logger.info(f"Using provider: {config.COGNITIVE_BIAS_MODEL_PROVIDER}, base URL: {config.COGNITIVE_BIAS_MODEL_BASE_URL}")
            except Exception as e:
                logger.error(f"Failed to create cognitive bias LLM adapter: {e}")
                self.llm_adapter = None
        else:
            self.llm_adapter = llm_adapter
        
        # Common cognitive biases in smart contract development
        self.bias_patterns = {
            "optimism_bias": [
                r"(\bassume\b|\bassuming\b|\bshould\b)",
                r"(\balways\b|\bnever\b|\bimpossible\b)",
                r"(will not|won't|cannot|can't)\s+happen"
            ],
            "authority_bias": [
                r"(standard|recommended|best practice)",
                r"((ERC|EIP)-\d+)",
                r"(OpenZeppelin|Uniswap|Compound)"
            ],
            "confirmation_bias": [
                r"(obviously|clearly|certainly)",
                r"(we know|we're sure|we expect)"
            ],
            "anchoring_bias": [
                r"(hardcode|constant|fixed|default)",
                r"(MAX_[A-Z_]+|MIN_[A-Z_]+)"
            ],
            "status_quo_bias": [
                r"(legacy|traditional|existing|current)",
                r"(maintain|preserve|keep)"
            ]
        }
        
    async def process(self, context: Context) -> Context:
        """
        Process the context to detect cognitive biases in the smart contracts.
        
        Args:
            context: The context containing the contracts to analyze
            
        Returns:
            Updated context with cognitive bias analysis
        """
        logger.info("Starting cognitive bias analysis")
        
        # Initialize cognitive biases dict in context if not present
        if not hasattr(context, 'cognitive_biases'):
            context.cognitive_biases = {}
        
        # Check if contracts are available
        if not hasattr(context, 'contracts') or not context.contracts:
            logger.warning("No contracts found for cognitive bias analysis")
            return context
            
        # Analyze each contract
        for contract in context.contracts:
            contract_name = getattr(contract, 'name', None)
            contract_path = getattr(contract, 'file_path', None)
            contract_content = getattr(contract, 'content', None) or getattr(contract, 'code', None)
            
            if not contract_name or not contract_content:
                logger.warning(f"Missing name or content for contract: {contract_path}")
                continue
                
            try:
                # Analyze the contract code for cognitive biases
                contract_biases = await self._analyze_contract(contract_name, contract_content)
                
                # Add to context
                if contract_biases:
                    context.cognitive_biases[contract_name] = contract_biases
                    logger.info(f"Found {len(contract_biases)} cognitive biases in {contract_name}")
            except Exception as e:
                # Fix: Ensure we're not using format specifiers in the error message
                error_msg = str(e).replace("%", "%%")  # Escape percent signs
                logger.error(f"Error analyzing cognitive biases in {contract_path}: {error_msg}")
                context.add_error(f"Failed to analyze file: {contract_path}", e)
        
        return context
    
    async def _analyze_contract(self, contract_name: str, contract_content: str) -> List[Dict[str, Any]]:
        """
        Analyze a contract for cognitive biases.
        
        Args:
            contract_name: Name of the contract
            contract_content: Solidity code of the contract
            
        Returns:
            List of detected cognitive biases
        """
        # For simplicity, we'll use pattern matching to detect common biases
        biases = []
        
        # Check for overconfidence bias (no error handling)
        overconfidence_matches = re.finditer(
            r'(\w+\s*\.\s*(?:transfer|call|send|delegatecall)\s*\([^;]+\))\s*(?![^;]*(?:require|assert|revert|if\s*\([^)]*(?:success|fail)))',
            contract_content
        )
        
        for match in overconfidence_matches:
            line_number = contract_content[:match.start()].count('\n') + 1
            biases.append({
                "type": "Overconfidence Bias",
                "description": "Function assumes external call will succeed without proper error handling",
                "impact": "High - May lead to transaction failures or lock funds",
                "location": f"Line {line_number}",
                "code": match.group(1).strip()
            })
            
        # Check for authority bias (overreliance on onlyOwner)
        authority_matches = re.finditer(
            r'modifier\s+onlyOwner|require\s*\(\s*(?:msg\.sender\s*==\s*owner|owner\s*==\s*msg\.sender)',
            contract_content
        )
        
        for match in authority_matches:
            line_number = contract_content[:match.start()].count('\n') + 1
            
            # Avoid duplicate reporting - only add if this is a new location
            if not any(b["location"] == f"Line {line_number}" and b["type"] == "Authority Bias" for b in biases):
                biases.append({
                    "type": "Authority Bias",
                    "description": "Centralized control pattern may indicate authority bias",
                    "impact": "Medium - May lead to centralization risks",
                    "location": f"Line {line_number}",
                    "code": match.group(0).strip()
                })
                
        # Check for Availability Bias (comments indicating specific concerns)
        availability_matches = re.finditer(
            r'//\s*(?:TODO|FIXME|WARNING|DANGER|SECURITY)\s*:([^\n]*)',
            contract_content,
            re.IGNORECASE
        )
        
        for match in availability_matches:
            line_number = contract_content[:match.start()].count('\n') + 1
            biases.append({
                "type": "Availability Bias",
                "description": f"Comment highlights specific concern: {match.group(1).strip()}",
                "impact": "Medium - May indicate focus on known issues at expense of unknown issues",
                "location": f"Line {line_number}",
                "code": match.group(0).strip()
            })
            
        return biases
    
    def _format_code_snippet(self, content: str, line_number: int, context_lines: int = 2) -> str:
        """
        Extract a code snippet around the specified line number.
        
        Args:
            content: Full contract content
            line_number: Line number to center the snippet around
            context_lines: Number of lines of context to include
            
        Returns:
            Formatted code snippet
        """
        lines = content.split('\n')
        
        # Calculate start and end lines with bounds checking
        start_line = max(0, line_number - context_lines - 1)
        end_line = min(len(lines), line_number + context_lines)
        
        # Build the snippet
        snippet = []
        for i in range(start_line, end_line):
            prefix = '>' if i == line_number - 1 else ' '
            snippet.append(f"{prefix} {i+1}: {lines[i]}")
            
        return '\n'.join(snippet)
