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
        Process the context to analyze cognitive biases
        
        Args:
            context: Processing context with contract files
            
        Returns:
            Updated context
        """
        logger.info(f"Starting cognitive bias analysis with model {config.COGNITIVE_BIAS_MODEL}")
        logger.info(f"Provider: {config.COGNITIVE_BIAS_MODEL_PROVIDER}, Base URL: {config.COGNITIVE_BIAS_MODEL_BASE_URL}")
        
        if self.llm_adapter is None:
            logger.error("No LLM adapter available for cognitive bias analysis")
            context.add_error(
                stage="cognitive_bias_analysis",
                message="No LLM adapter available for cognitive bias analysis"
            )
            return context
        
        # Initialize cognitive biases dictionary in context if not present
        if not hasattr(context, "cognitive_biases"):
            context.cognitive_biases = {}
            
        # Analyze each file for cognitive biases
        for file_id, file_data in context.files.items():
            # Skip non-Solidity files
            if not file_data.get("is_solidity", False):
                continue
                
            try:
                # Extract contract name from file path
                contract_name = os.path.basename(file_data.get("path", file_id))
                if contract_name.endswith(".sol"):
                    contract_name = contract_name[:-4]
                
                # Initial pattern-based bias detection
                content = file_data["content"]
                biases = self._detect_biases_with_patterns(content)
                
                # Apply LLM-based bias detection
                llm_biases = await self._detect_biases_with_llm(content, contract_name)
                
                # Merge results, prioritizing LLM findings
                for bias_type, instances in llm_biases.items():
                    if bias_type in biases:
                        # Add only non-duplicate instances
                        existing_locations = set(i["location"] for i in biases[bias_type])
                        for instance in instances:
                            if instance["location"] not in existing_locations:
                                biases[bias_type].append(instance)
                    else:
                        biases[bias_type] = instances
                
                # Create bias analysis result
                result = BiasAnalysisResult(
                    contract_name=contract_name,
                    file_path=file_data.get("path", file_id),
                    biases=biases,
                    analysis_date=None  # Will be set automatically
                )
                
                # Store in context
                context.cognitive_biases[file_id] = result
                
                # Log findings
                total_biases = sum(len(instances) for instances in biases.values())
                logger.info(f"Found {total_biases} potential cognitive biases in {contract_name}")
                
            except Exception as e:
                logger.error(f"Error analyzing cognitive biases in {file_id}: {str(e)}")
                context.add_error(
                    stage="cognitive_bias_analysis",
                    message=f"Failed to analyze file: {file_id}",
                    exception=e
                )
        
        # Cross-contract analysis
        await self._analyze_cross_contract_biases(context)
        
        return context
    
    def _detect_biases_with_patterns(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect cognitive biases using regex patterns
        
        Args:
            content: Contract code content
            
        Returns:
            Dictionary of biases by type
        """
        biases = {}
        lines = content.split("\n")
        
        for bias_type, patterns in self.bias_patterns.items():
            instances = []
            
            for i, line in enumerate(lines):
                line_number = i + 1
                
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Extract surrounding context
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        context_lines = lines[start:end]
                        
                        instances.append({
                            "location": f"Line {line_number}",
                            "line_number": line_number,
                            "context": "\n".join(context_lines),
                            "description": f"Potential {bias_type.replace('_', ' ')} detected",
                            "pattern": pattern,
                            "confidence": "low"  # Pattern-based detection has lower confidence
                        })
            
            if instances:
                biases[bias_type] = instances
        
        return biases
    
    async def _detect_biases_with_llm(self, content: str, contract_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect cognitive biases using LLM analysis
        
        Args:
            content: Contract code content
            contract_name: Name of the contract
            
        Returns:
            Dictionary of biases by type
        """
        biases = {}
        
        if not self.llm_adapter or not hasattr(self.llm_adapter, 'llm'):
            logger.warning("No LLM available for cognitive bias detection")
            return biases
            
        llm = self.llm_adapter.llm
        
        # Create prompt for cognitive bias detection
        prompt = f"""
        You are a smart contract auditor specializing in cognitive bias detection. Analyze the following Solidity contract for potential cognitive biases in its design and implementation.
        
        Contract: {contract_name}
        
        ```solidity
        {content}
        ```
        
        Identify cognitive biases including but not limited to:
        1. Optimism bias - assuming things will always work correctly
        2. Anchoring bias - relying too heavily on initial information
        3. Confirmation bias - interpreting information to confirm existing beliefs
        4. Authority bias - deferring to standards without understanding them
        5. Status quo bias - resistance to change
        
        Format your response as a JSON object with bias types as keys and arrays of instances as values:
        
        ```json
        {
          "optimism_bias": [
            {
              "location": "Line 42",
              "description": "Assumes transaction will always succeed without error handling"
            }
          ],
          "anchoring_bias": [
            {
              "location": "Line 105",
              "description": "Hardcoded gas limits could be problematic with future Ethereum upgrades"
            }
          ]
        }
        ```
        
        Only include biases that you have high or medium confidence in. Focus on substantive issues that could impact contract security or functionality.
        """
        
        try:
            # Get response from LLM
            response = await llm.acomplete(prompt)
            response_text = response.text
            
            # Extract JSON from response
            json_str = extract_json_from_text(response_text)
            
            # Parse JSON
            results = safe_parse_json(json_str)
                
            # Process each bias type
            for bias_type, instances in results.items():
                if not isinstance(instances, list):
                    continue
                
                # Process each instance
                valid_instances = []
                for instance in instances:
                    if not isinstance(instance, dict):
                        continue
                    
                    # Extract line number
                    line_number = None
                    if "location" in instance:
                        line_match = re.search(r'Line\s+(\d+)', instance["location"])
                        if line_match:
                            line_number = int(line_match.group(1))
                    
                    instance["line_number"] = line_number
                    instance["confidence"] = "medium"  # LLM-based detection has medium confidence
                    valid_instances.append(instance)
                
                if valid_instances:
                    biases[bias_type] = valid_instances
                
        except Exception as e:
            logger.error(f"Error in LLM-based bias detection for {contract_name}: {str(e)}")
        
        return biases
    
    async def _analyze_cross_contract_biases(self, context: Context):
        """
        Analyze cognitive biases across multiple contracts
        
        Args:
            context: Processing context
        """
        # Skip if no LLM is available or not enough contracts
        if not hasattr(context, "cognitive_biases") or len(context.cognitive_biases) < 2:
            return
        
        # Collect contract names and most common biases
        contracts = []
        common_biases = {}
        
        for file_id, bias_result in context.cognitive_biases.items():
            contracts.append(bias_result.contract_name)
            # Collect bias types and instances
            for bias_type, instances in bias_result.biases.items():
                if bias_type not in common_biases:
                    common_biases[bias_type] = 0
                common_biases[bias_type] += len(instances)
        
        # Find most common biases
        top_biases = sorted(common_biases.items(), key=lambda x: x[1], reverse=True)[:3]
        top_bias_types = [bias_type for bias_type, _ in top_biases]
        
        # Add cross-contract bias analysis
        context.cross_contract_biases = {
            "contracts_analyzed": contracts,
            "common_biases": top_bias_types,
            "description": f"Project-wide tendency toward {', '.join(top_bias_types)}"
        }
        
        logger.info(f"Completed cross-contract bias analysis across {len(contracts)} contracts")
    
    async def _generate_assumption_analysis(self, context: Context):
        """
        Generate analysis of developer assumptions
        
        Args:
            context: Processing context
        """
        # Skip if no LLM is available
        llm = None
        if self.llm_adapter and hasattr(self.llm_adapter, 'llm'):
            llm = self.llm_adapter.llm
        else:
            llm = Settings.llm
        
        if not llm:
            logger.warning("No LLM available for assumption analysis")
            return
        
        # Collect all biases
        all_biases = {}
        for file_id, bias_result in context.cognitive_biases.items():
            for bias_type, instances in bias_result.biases.items():
                if bias_type not in all_biases:
                    all_biases[bias_type] = []
                all_biases[bias_type].extend(instances)
        
        # Skip if no biases found
        if not all_biases:
            return
        
        # Format biases for LLM
        bias_summary = ""
        for bias_type, instances in all_biases.items():
            bias_summary += f"\n{bias_type.replace('_', ' ').title()}:\n"
            for i, instance in enumerate(instances[:5]):  # Limit to 5 examples per bias
                bias_summary += f"- {instance.get('description', 'No description')}\n"
        
        # Create a prompt for assumption analysis
        prompt = f"""
        You are a smart contract security expert analyzing cognitive biases and developer assumptions in a project.
        
        Based on the cognitive biases detected in the codebase, identify the key assumptions that developers are making
        that could lead to security vulnerabilities or logical errors.
        
        Detected Biases:
        {bias_summary}
        
        For each underlying assumption, provide:
        1. A clear statement of the assumption
        2. Why this assumption is problematic
        3. How this assumption might be exploited or lead to failures
        4. A recommendation to address this assumption
        
        Format your response as valid JSON:
        ```json
        {{
          "assumptions": [
            {{
              "statement": "The contract assumes that token transfers always succeed",
              "risk": "Some tokens return false instead of reverting on failure",
              "potential_exploit": "An attacker could use a non-standard token that doesn't revert to drain funds",
              "recommendation": "Always check the return value of token transfers or use SafeERC20"
            }},
            ...
          ],
          "summary": "Brief summary of the overall assumption patterns"
        }}
        ```
        """
        
        try:
            # Get response from LLM
            response = await llm.acomplete(prompt)
            response_text = response.text
            
            # Extract JSON from response
            json_str = extract_json_from_text(response_text)
            
            # Parse JSON
            results = safe_parse_json(json_str)
                
            # Create assumption analysis
            assumption_analysis = AssumptionAnalysis(
                assumptions=results.get("assumptions", []),
                summary=results.get("summary", "No summary available")
            )
            
            # Store in context
            context.assumption_analysis = assumption_analysis
            
            logger.info(f"Generated assumption analysis with {len(assumption_analysis.assumptions)} key assumptions")
                
        except Exception as e:
            logger.error(f"Error in assumption analysis: {str(e)}")

    async def _extract_flows_with_llm(self, func: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract business flows using LLM"""
        # Define schema for LLM response
        schema = {
            "type": "object",
            "properties": {
                "businessFlows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "confidence": {"type": "number"}
                        }
                    }
                },
                "notes": {"type": "string"},
                "confidence": {"type": "number"},
                "f1_score": {"type": "number"}
            }
        }
        
        # Create prompt for LLM
        function_text = func.get("full_text", "")
        function_name = func.get("name", "")
        contract_name = func.get("contract_name", "")
        
        prompt = f"""
    Analyze the following Solidity function and identify the business flows it implements.
    Focus on operations like token transfers, access control, state changes, etc.

    Contract: {contract_name}
    Function: {function_name}

    ```solidity
    {function_text}
    ```

    Identify the main business flows present in this function and respond in JSON format with the following structure:
    {{
        "businessFlows": [
            {{
                "type": "string - e.g. token_transfer, access_control, state_change",
                "description": "Detailed description of the business flow",
                "confidence": 0.0-1.0
            }}
        ],
        "notes": "Add any notes about information that would be helpful but is missing, or tools you would like access to",
        "confidence": 0.0-1.0,
        "f1_score": 0.0-1.0
    }}

    In the notes field, please indicate any missing information or tools that would help improve your analysis.
    For confidence, rate your certainty in the analysis from 0.0 to 1.0.
    For f1_score, estimate your precision/recall balance from 0.0 to 1.0.
    """
        
        # ...existing code...
