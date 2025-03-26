"""
Counterfactual analyzer for smart contracts.

This module analyzes smart contracts by generating "what if" scenarios
to identify potential vulnerabilities and edge cases.
"""

import json
import re
import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger

from llama_index.core.settings import Settings
from ..pipeline.core import Context
from ..models.analysis import CounterfactualScenario
from ..utils.json_repair import safe_parse_json, extract_json_from_text
from ..llm.llama_index_adapter import LlamaIndexAdapter
from finite_monkey.nodes_config import config


class CounterfactualAnalyzer:
    """
    Analyzer for counterfactual scenarios in smart contracts.
    
    Generates and evaluates "what if" scenarios to identify edge cases
    and potential vulnerabilities in contract logic.
    """
    
    def __init__(self, llm_adapter=None):
        """
        Initialize the counterfactual analyzer
        
        Args:
            llm_adapter: LlamaIndex adapter for LLM access
        """
        # Check if we need to create a default LLM adapter
        if llm_adapter is None:
            try:
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                
                # Use configuration explicitly to ensure all parameters are set
                self.llm_adapter = LlamaIndexAdapter(
                    model_name=config.COUNTERFACTUAL_MODEL,
                    provider=config.COUNTERFACTUAL_MODEL_PROVIDER,
                    base_url=config.COUNTERFACTUAL_MODEL_BASE_URL
                )
                logger.info(f"Created counterfactual analysis LLM adapter with model: {config.COUNTERFACTUAL_MODEL}")
                logger.info(f"Provider: {config.COUNTERFACTUAL_MODEL_PROVIDER}, Base URL: {config.COUNTERFACTUAL_MODEL_BASE_URL}")
            except Exception as e:
                logger.error(f"Failed to create counterfactual LLM adapter: {e}")
                self.llm_adapter = None
        else:
            self.llm_adapter = llm_adapter
        
        # Define common counterfactual scenario types
        self.scenario_types = [
            "parameter_extremes",  # What if parameters reach min/max values?
            "state_divergence",    # What if contract state diverges from expectations?
            "external_failure",    # What if external calls fail?
            "reordering_attacks",  # What if transactions are reordered?
            "front_running",       # What if transactions are front-run?
            "permission_changes",  # What if permissions change unexpectedly?
            "market_conditions"    # What if market conditions change drastically?
        ]
    
    async def process(self, context: Context) -> Context:
        """
        Process the context to generate and analyze counterfactual scenarios
        
        Args:
            context: Processing context with contract files
            
        Returns:
            Updated context with counterfactual scenarios
        """
        logger.info("Starting counterfactual analysis")
        
        if self.llm_adapter is None:
            logger.error("No LLM adapter available for counterfactual analysis")
            context.add_error(
                stage="counterfactual_analysis",
                message="No LLM adapter available for counterfactual analysis"
            )
            # Return the context even if we can't process
            return context
        
        # Initialize counterfactuals dict if it doesn't exist
        if not hasattr(context, "counterfactuals"):
            context.counterfactuals = {}
        
        # Get list of solidity files to analyze
        solidity_files = [(file_id, file_data) for file_id, file_data in context.files.items() 
                          if file_data.get("is_solidity", False)]
        
        logger.info(f"Analyzing counterfactual scenarios in {len(solidity_files)} Solidity files")
        
        # Process files in chunks to manage resources
        chunk_size = 3  # Smaller chunk size for more intensive analysis
        for i in range(0, len(solidity_files), chunk_size):
            chunk = solidity_files[i:i+chunk_size]
            
            # Process this chunk of files concurrently
            tasks = [self._analyze_file(context, file_id, file_data) for file_id, file_data in chunk]
            await asyncio.gather(*tasks)
            
            # Small delay to prevent resource exhaustion
            await asyncio.sleep(0.2)
        
        # If available, integrate with dataflow analysis
        await self._integrate_with_dataflows(context)
        
        # If available, integrate with business flows
        await self._integrate_with_business_flows(context)
        
        logger.info("Counterfactual analysis complete")
        return context
    
    async def _analyze_file(self, context: Context, file_id: str, file_data: Dict[str, Any]):
        """
        Generate and analyze counterfactual scenarios for a file
        
        Args:
            context: Processing context
            file_id: File ID
            file_data: File data
        """
        try:
            # Extract contract name
            contract_name = file_data.get("name", file_id)
            if contract_name.endswith(".sol"):
                contract_name = contract_name[:-4]
            
            # Extract contract content
            content = file_data["content"]
            
            # Generate scenarios using LLM
            scenarios = await self._generate_scenarios(content, contract_name)
            
            # Store scenarios in context
            context.counterfactuals[file_id] = scenarios
            
            logger.info(f"Generated {len(scenarios)} counterfactual scenarios for {contract_name}")
            
        except Exception as e:
            logger.error(f"Error in counterfactual analysis for {file_id}: {str(e)}")
            context.add_error(
                stage="counterfactual_analysis",
                message=f"Failed to analyze file: {file_id}",
                exception=e
            )
    
    async def _generate_scenarios(self, content: str, contract_name: str) -> List[Dict[str, Any]]:
        """
        Generate counterfactual scenarios using LLM
        
        Args:
            content: Contract source code
            contract_name: Name of the contract
            
        Returns:
            List of counterfactual scenarios
        """
        scenarios = []
        
        # Skip if no LLM is available
        llm = None
        if self.llm_adapter and hasattr(self.llm_adapter, 'llm'):
            llm = self.llm_adapter.llm
        else:
            llm = Settings.llm
        
        if not llm:
            logger.warning("No LLM available for counterfactual analysis")
            return scenarios
        
        # Extract function definitions for more focused analysis
        functions = self._extract_functions(content)
        
        # If contract is too large, analyze only public/external functions
        if len(content) > 10000:
            functions = [f for f in functions if "public" in f["modifiers"] or "external" in f["modifiers"]]
        
        # If still too many functions, prioritize state-changing functions
        if len(functions) > 5:
            state_changing = [f for f in functions if "view" not in f["modifiers"] and "pure" not in f["modifiers"]]
            functions = state_changing[:5] if state_changing else functions[:5]
        
        # Generate scenarios for each function
        for function in functions:
            # Create a prompt for the LLM
            prompt = f"""
            You are a smart contract security analyst specializing in counterfactual analysis. For the given Solidity function,
            generate "what if" scenarios that could expose vulnerabilities or edge cases.

            Contract: {contract_name}
            Function: {function['name']}

            ```solidity
            {function['code']}
            ```

            Generate counterfactual scenarios for this function covering these categories:
            1. Parameter Extremes: What if inputs reach min/max values or unexpected states?
            2. State Divergence: What if contract state differs from expectations?
            3. External Calls: What if external interactions fail or behave unexpectedly?
            4. Transaction Ordering: What if transactions are executed in unexpected order?
            5. Permission Changes: What if permissions or roles change during execution?

            For each scenario:
            - Provide a descriptive title
            - Describe the scenario conditions
            - Explain what might go wrong
            - Rate the likelihood (Low/Medium/High)
            - Rate the impact (Low/Medium/High)
            - Provide a recommended mitigation

            Format your response as valid JSON:
            ```json
            [
              {{
                "title": "Integer Overflow in Token Amount",
                "category": "parameter_extremes",
                "description": "What if the token amount exceeds the maximum uint256 value?",
                "impact": "Contract state corruption leading to fund loss",
                "likelihood": "Low",
                "severity": "High",
                "mitigation": "Use SafeMath or Solidity 0.8+ built-in overflow checks"
              }}
            ]
            ```

            Generate 2-4 high-quality scenarios focusing on realistic security concerns.
            """
            
            try:
                # Get response from LLM
                response = await llm.acomplete(prompt)
                response_text = response.text
                
                # Extract JSON from response
                function_scenarios = []
                try:
                    # First try to extract JSON
                    json_data = extract_json_from_text(response_text)
                    
                    # Then parse it safely with a default empty list
                    function_scenarios = safe_parse_json(json_data, [])
                    
                    # Convert to list if a dictionary was returned
                    if isinstance(function_scenarios, dict):
                        if "scenarios" in function_scenarios:
                            function_scenarios = function_scenarios["scenarios"]
                        else:
                            function_scenarios = [function_scenarios]
                    
                    # Ensure we have a list
                    if not isinstance(function_scenarios, list):
                        logger.warning(f"Invalid response format for {contract_name}:{function['name']} - expected list, got {type(function_scenarios)}")
                        function_scenarios = []
                    
                    # Add function information to scenarios
                    for scenario in function_scenarios:
                        if not isinstance(scenario, dict):
                            continue
                            
                        scenario["function"] = function["name"]
                        scenario["function_signature"] = function["signature"]
                        scenario["location"] = f"Function {function['name']}"
                    
                    # Add to collected scenarios
                    scenarios.extend([s for s in function_scenarios if isinstance(s, dict)])
                    
                except Exception as e:
                    logger.warning(f"Failed to parse JSON from LLM response for {contract_name}:{function['name']}: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error generating scenarios for {contract_name}:{function['name']}: {str(e)}")
        
        # After generating and collecting scenarios, assess them in more depth
        if scenarios:
            scenarios = await self._assess_scenarios(content, contract_name, scenarios)
        
        return scenarios
    
    async def _assess_scenarios(self, content: str, contract_name: str, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assess counterfactual scenarios in more depth using LLM
        
        Args:
            content: Contract source code
            contract_name: Name of the contract
            scenarios: Generated counterfactual scenarios
            
        Returns:
            Enhanced scenarios with deeper assessment
        """
        # Skip if no LLM is available
        llm = None
        if self.llm_adapter and hasattr(self.llm_adapter, 'llm'):
            llm = self.llm_adapter.llm
        else:
            llm = Settings.llm
        
        if not llm:
            logger.warning("No LLM available for counterfactual assessment")
            return scenarios
        
        enhanced_scenarios = []
        
        for scenario in scenarios:
            try:
                # Extract relevant function code
                function_name = scenario.get("function")
                function_code = ""
                
                for func in self._extract_functions(content):
                    if func["name"] == function_name:
                        function_code = func["code"]
                        break
                
                if not function_code:
                    # If function code not found, use the scenario as is
                    enhanced_scenarios.append(scenario)
                    continue
                
                # Create a prompt for deeper assessment
                prompt = f"""
                You are a smart contract security expert evaluating a counterfactual security scenario.
                Given the contract code and a potential "what if" scenario, determine how realistic and
                impactful this scenario is, and provide details on exactly how it could be exploited.

                CONTRACT: {contract_name}

                FUNCTION CODE:
                ```solidity
                {function_code}
                ```

                COUNTERFACTUAL SCENARIO:
                Title: {scenario.get("title", "Untitled")}
                Description: {scenario.get("description", "No description")}
                Category: {scenario.get("category", "Unknown")}
                Initial Assessment:
                - Likelihood: {scenario.get("likelihood", "Unknown")}
                - Severity: {scenario.get("severity", "Unknown")}
                - Initial Mitigation: {scenario.get("mitigation", "No mitigation provided")}

                Provide a deeper analysis of this scenario, answering:
                1. What specific code path would be taken in this scenario?
                2. What exact conditions must be true for this to occur?
                3. Could an attacker deliberately trigger this scenario?
                4. What would be the step-by-step exploit process?
                5. Are there specific blockchain mechanics (e.g., gas, timestamps) involved?

                Return your detailed assessment as JSON:
                ```json
                {{
                    "detailed_exploit_path": "Step by step walkthrough of how the exploit would work",
                    "required_conditions": ["Condition 1", "Condition 2"],
                    "attacker_control": "High|Medium|Low",
                    "technical_difficulty": "High|Medium|Low",
                    "specific_vulnerable_patterns": ["Pattern 1", "Pattern 2"],
                    "enhanced_mitigation": "More detailed mitigation strategy",
                    "real_world_relevance": "Commentary on whether this has been seen in real exploits"
                }}
                ```

                Provide ONLY the JSON response, with no additional text.
                """
                
                # Get response from LLM
                response = await llm.acomplete(prompt)
                response_text = response.text
                
                # Extract and parse JSON safely
                try:
                    # First extract JSON
                    json_data = extract_json_from_text(response_text)
                    
                    # Then parse safely with default empty dict
                    assessment = safe_parse_json(json_data, {})
                    
                    # Only use non-empty assessments
                    if assessment:
                        # Enhance the scenario with deeper assessment
                        enhanced_scenario = {**scenario, **assessment}
                        enhanced_scenarios.append(enhanced_scenario)
                    else:
                        # Use original if assessment is empty
                        enhanced_scenarios.append(scenario)
                        
                except Exception as e:
                    # If JSON parsing fails, keep original scenario
                    logger.warning(f"Failed to parse assessment JSON for {scenario.get('title')}: {str(e)}")
                    enhanced_scenarios.append(scenario)
                    
            except Exception as e:
                logger.error(f"Error assessing scenario {scenario.get('title')}: {str(e)}")
                enhanced_scenarios.append(scenario)
        
        return enhanced_scenarios
    
    async def _analyze_contract(self, contract_content: str, contract_name: str) -> Dict[str, Any]:
        """
        Perform counterfactual analysis on a single contract
        
        Args:
            contract_content: Contract source code
            contract_name: Name of the contract
            
        Returns:
            Dictionary with counterfactual scenarios and analyses
        """
        # Create prompt for counterfactual analysis
        prompt = f"""
        You are a smart contract security auditor specializing in counterfactual analysis. 
        Analyze the following Solidity contract and identify edge cases, unexpected input combinations, 
        or state conditions that could lead to vulnerabilities.
        
        Contract: {contract_name}
        
        ```solidity
        {contract_content}
        ```
        
        For each function, consider:
        1. What happens with extreme input values?
        2. What if functions are called in an unexpected order?
        3. What if there are race conditions or front-running attacks?
        4. What assumptions are being made about the contract state?
        
        Provide your analysis in JSON format with the following structure:
        {{
          "counterfactual_scenarios": [
            {{
              "function": "function_name",
              "scenario": "Description of counterfactual scenario",
              "impact": "Description of potential impact",
              "likelihood": "high|medium|low",
              "prevention_measures": "How to prevent this issue"
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
    
    def _extract_functions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract function definitions from contract code
        
        Args:
            content: Contract source code
            
        Returns:
            List of function information
        """
        functions = []
        
        # Simple regex to find function definitions
        # This is a simplified version; a robust parser would be better
        pattern = r"function\s+(\w+)\s*\(([\s\S]*?)\)\s*(public|private|internal|external)?\s*(view|pure|payable|virtual|override)?\s*(view|pure|payable|virtual|override)?\s*(returns\s*\([\s\S]*?\))?\s*{([\s\S]*?)}"
        
        matches = re.finditer(pattern, content)
        
        for match in matches:
            name = match.group(1)
            params = match.group(2)
            modifiers = []
            
            # Extract modifiers
            if match.group(3):
                modifiers.append(match.group(3))
            if match.group(4):
                modifiers.append(match.group(4))
            if match.group(5):
                modifiers.append(match.group(5))
            
            # Extract function code
            full_match = match.group(0)
            
            functions.append({
                "name": name,
                "params": params,
                "modifiers": modifiers,
                "signature": f"function {name}({params})",
                "code": full_match
            })
        
        return functions
    
    async def _integrate_with_dataflows(self, context: Context):
        """
        Integrate counterfactual analysis with dataflow findings
        
        Args:
            context: Processing context
        """
        # Skip if dataflow analysis is not available
        if not hasattr(context, "dataflows"):
            return
        
        logger.info("Integrating counterfactual analysis with dataflow findings")
        
        for file_id, dataflows in context.dataflows.items():
            if file_id not in context.counterfactuals:
                continue
            
            counterfactuals = context.counterfactuals[file_id]
            
            # Flag dataflows that are covered by counterfactual scenarios
            for flow in dataflows:
                relevant_scenarios = []
                
                # Check which scenarios apply to this flow
                for scenario in counterfactuals:
                    # Match by function
                    flow_functions = set()
                    for node in flow.get("path", []):
                        if "name" in node and "type" in node and node["type"] == "function":
                            flow_functions.add(node["name"])
                    
                    if "function" in scenario and scenario["function"] in flow_functions:
                        relevant_scenarios.append(scenario)
                
                # Add relevant scenarios to the flow
                if relevant_scenarios:
                    flow["counterfactual_scenarios"] = relevant_scenarios
                    flow["scenario_count"] = len(relevant_scenarios)
    
    async def _integrate_with_business_flows(self, context: Context):
        """
        Integrate counterfactual analysis with business flow findings
        
        Args:
            context: Processing context
        """
        # Skip if business flow analysis is not available
        if not hasattr(context, "business_flows"):
            return
        
        logger.info("Integrating counterfactual analysis with business flows")
        
        for file_id, business_flows in context.business_flows.items():
            if file_id not in context.counterfactuals:
                continue
            
            counterfactuals = context.counterfactuals[file_id]
            
            # Add counterfactual scenarios to business flows
            for flow in business_flows:
                if not isinstance(flow, dict) or "name" not in flow:
                    continue
                
                relevant_scenarios = []
                
                # Check which scenarios apply to this business flow
                for scenario in counterfactuals:
                    # Match by function or description
                    if "function" in scenario and flow.get("name", "").lower().find(scenario["function"].lower()) >= 0:
                        relevant_scenarios.append(scenario)
                
                # Add relevant scenarios to the business flow
                if relevant_scenarios:
                    flow["counterfactual_scenarios"] = relevant_scenarios
                    flow["scenario_count"] = len(relevant_scenarios)
