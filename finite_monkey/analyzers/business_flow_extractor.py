import asyncio
import os
from typing import Dict, List, Any, Optional, AsyncIterator
import traceback

from finite_monkey.models.business_flow import BusinessFlow
from finite_monkey.utils import guidance_program
from ..llm.llama_index_adapter import LlamaIndexAdapter
import json
import logging
from loguru import logger
from llama_index.core.llms import ChatMessage
from finite_monkey.pipeline.core import Context
from finite_monkey.nodes_config import config

# Import the new Guidance utilities
from ..utils.guidance_version_utils import GUIDANCE_AVAILABLE, create_guidance_program
from ..models.business_flow import BusinessFlowAnalysisResult

class BusinessFlowExtractor:
    """Extracts business flows from smart contracts using LlamaIndex"""
    
    def __init__(self, llm_adapter: Optional[LlamaIndexAdapter] = None, flow_types: Optional[List[str]] = None, analysis_level="contract", use_guidance=True):
        """
        Initialize with LLM adapter and flow types to extract
        
        Args:
            llm_adapter: LLM adapter for analysis
            flow_types: List of business flow types to identify
            analysis_level: Level of analysis - "contract", "function", or "method_first"
            use_guidance: Whether to use Guidance for structured output
        """
        # Add detailed logging about LLM initialization
        logger.debug("Initializing BusinessFlowExtractor")
        
        # Store flow types
        self.flow_types = flow_types or [
            "token_transfer",
            "access_control", 
            "state_transition",
            "external_call",
            "fund_management"
        ]
        
        # Validate analysis_level
        if analysis_level not in ["contract", "function", "method_first"]:
            logger.warning(f"Invalid analysis_level: {analysis_level}. Using 'method_first' as default.")
            self.analysis_level = "method_first"
        else:
            self.analysis_level = analysis_level
            
        logger.debug(f"Business flow extractor configured with {len(self.flow_types)} flow types at {self.analysis_level} level")
        
        self.use_guidance = use_guidance and GUIDANCE_AVAILABLE
        
        # Add a semaphore to limit concurrent analysis tasks
        self._analysis_semaphore = asyncio.Semaphore(3)  # Allow max 3 concurrent analyses
        
        # Initialize LLM adapter
        if llm_adapter is None:
            try:
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                
                # Use analysis model for business flow extraction
                self.llm_adapter = LlamaIndexAdapter(
                    model_name=config.ANALYSIS_MODEL,
                    provider=config.ANALYSIS_MODEL_PROVIDER,
                    base_url=config.ANALYSIS_MODEL_BASE_URL,
                    request_timeout=config.REQUEST_TIMEOUT  # Ensure timeout is passed correctly
                )
                logger.info(f"Created business flow LLM adapter with model: {config.ANALYSIS_MODEL}, timeout: {config.REQUEST_TIMEOUT}")
            except Exception as e:
                logger.error(f"Failed to create business flow LLM adapter: {e}")
                self.llm_adapter = None
        else:
            self.llm_adapter = llm_adapter
            logger.info(f"Using provided LLM adapter for business flow extraction: {type(llm_adapter).__name__}")
    
    def set_llm_adapter(self, llm_adapter):
        """Set the LLM adapter after initialization"""
        self.llm_adapter = llm_adapter
        logger.info(f"Updated LLM adapter: {type(llm_adapter).__name__}")
        
    async def process(self, context: Context) -> Context:
        """Process the context to extract business flows"""
        logger.info(f"Starting business flow extraction at {self.analysis_level} level")
        
        # Check if LLM adapter is available
        if not self.llm_adapter or not hasattr(self.llm_adapter, 'llm'):
            logger.error("No LLM adapter available for business flow extraction - early exit")
            return context
        
        # Initialize business_flows attribute if needed
        if not hasattr(context, 'business_flows'):
            context.business_flows = {}
        
        # Implement the method-first, then contract approach
        if self.analysis_level == "method_first":
            # First analyze at function level
            await self._process_function_level(context)
            
            # Then analyze at contract level to catch any flows that span multiple functions
            await self._process_contract_level(context)
        elif self.analysis_level == "function":
            # Only analyze at function level
            await self._process_function_level(context)
        else:
            # Default is contract level only
            await self._process_contract_level(context)
                
        logger.info(f"Business flow extraction complete. Found flows for {len(context.business_flows)} contracts")
        return context
        
    async def _process_contract_level(self, context: Context) -> None:
        """Process flows at contract level with concurrency control"""
        # Process contracts with detailed logging
        contract_count = len(context.contracts)
        logger.debug(f"Processing {contract_count} contracts for business flow extraction")
        
        # Process contracts in smaller batches to avoid overwhelming the system
        batch_size = 3  # Process 3 contracts at a time
        for i in range(0, len(context.contracts), batch_size):
            batch = context.contracts[i:i+batch_size]
            
            # Create tasks for this batch
            tasks = []
            for contract in batch:
                contract_name = getattr(contract, 'name', f"Contract_{i}")
                contract_code = getattr(contract, 'content', getattr(contract, 'code', ''))
                
                # Skip if no code available
                if not contract_code:
                    logger.warning(f"No code available for contract: {contract_name}")
                    continue
                    
                # Create a task for this contract
                task = asyncio.create_task(self._process_single_contract(context, contract, i))
                tasks.append(task)
            
            # Wait for all tasks in this batch to complete
            if tasks:
                await asyncio.gather(*tasks)
                
                # Small delay between batches
                await asyncio.sleep(0.5)
        
        logger.info(f"Contract-level analysis complete")

    async def _process_single_contract(self, context: Context, contract, index: int):
        """Process a single contract with semaphore protection"""
        try:
            # Extract contract information
            contract_name = getattr(contract, 'name', f"Contract_{index}")
            contract_code = getattr(contract, 'content', getattr(contract, 'code', ''))
            
            logger.debug(f"Analyzing contract: {contract_name}")
            
            # Use semaphore to limit concurrent analyses
            async with self._analysis_semaphore:
                # Analyze contract for business flows
                contract_flows = await self._analyze_contract(contract_name, contract_code)
            
            # Add to context
            if contract_flows:
                context.business_flows[contract_name] = contract_flows
                logger.info(f"Extracted {len(contract_flows)} business flows from {contract_name}")
            else:
                logger.warning(f"No business flows extracted from {contract_name}")
                    
        except Exception as e:
            logger.error(f"Error analyzing contract for business flows: {str(e)}")
            traceback.print_exc()
    
    async def _process_function_level(self, context: Context) -> None:
        """Process flows at function level"""
        # Verify functions are available
        if not hasattr(context, 'functions') or not context.functions:
            logger.warning("No functions found in context for business flow extraction")
            return
            
        function_count = len(context.functions)
        logger.debug(f"Processing {function_count} functions for business flow extraction")
        
        # Group functions by contract for better organization
        contract_functions = {}
        for func_id, func in context.functions.items():
            contract_name = getattr(func, 'contract_name', 'Unknown')
            if contract_name not in contract_functions:
                contract_functions[contract_name] = []
            contract_functions[contract_name].append(func)
            
        # Process each contract's functions
        for contract_name, functions in contract_functions.items():
            if contract_name not in context.business_flows:
                context.business_flows[contract_name] = []
                
            # Process each function
            for func in functions:
                try:
                    func_name = getattr(func, 'name', getattr(func, 'function_name', 'Unknown'))
                    func_code = getattr(func, 'content', getattr(func, 'code', ''))
                    
                    if not func_code:
                        logger.warning(f"No code available for function: {func_name}")
                        continue
                        
                    # Analyze the function
                    func_flows = await self._analyze_function(contract_name, func_name, func_code)
                    
                    # Add to context
                    if func_flows:
                        for flow in func_flows:
                            flow['function_name'] = func_name
                            context.business_flows[contract_name].append(flow)
                        logger.debug(f"Extracted {len(func_flows)} flows from function {func_name}")
                except Exception as e:
                    logger.error(f"Error analyzing function {getattr(func, 'name', 'Unknown')}: {str(e)}")
                    continue
                    
            # Log summary for this contract
            flow_count = len(context.business_flows[contract_name])
            logger.info(f"Extracted {flow_count} flows from {len(functions)} functions in {contract_name}")
    
    async def _analyze_function(self, contract_name: str, function_name: str, function_code: str) -> List[Dict[str, Any]]:
        """
        Analyze a function for business flows
        
        Args:
            contract_name: Name of the contract
            function_name: Name of the function
            function_code: Solidity code of the function
            
        Returns:
            List of business flow dictionaries
        """
        logger.debug(f"Starting business flow analysis for function: {contract_name}.{function_name}")
        
        # Check if LLM adapter is available
        if not self.llm_adapter:
            logger.error("No LLM adapter available for business flow analysis")
            return []
            
        try:
            # Use serializable message format for LLM
            messages = [
                {"role": "system", "content": """You are an expert smart contract analyst with deep understanding 
of blockchain business logic. Identify the key workflows and processes in this function.
Always respond with properly formatted JSON that can be parsed by Python's json.loads()."""},
                {"role": "user", "content": self._create_function_analysis_prompt(contract_name, function_name, function_code)}
            ]
            
            # Get LLM response
            response = await self._get_llm_response(messages)
            
            # Parse response
            try:
                result = json.loads(response)
                flows = result.get("flows", [])
                logger.info(f"Successfully generated business flow analysis with {len(flows)} flows for function {function_name}")
                
                # Add contract and function name to each flow
                for flow in flows:
                    flow["contract_name"] = contract_name
                    flow["function_name"] = function_name
                
                return flows
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error in function business flow analysis: {str(e)}")
            return []
    
    def _create_function_analysis_prompt(self, contract_name: str, function_name: str, function_code: str) -> str:
        """Create a prompt for function-level flow analysis"""
        # Format flow types for the prompt
        flow_type_text = "\n".join([f"- {flow_type}" for flow_type in self.flow_types])
        
        return f"""
Analyze the following Solidity function from a smart contract and identify specific business flows.
Focus on these types of flows:
{flow_type_text}

Function Name: {function_name}
Contract Name: {contract_name}

Function code:
```solidity
{function_code}
```

Return your analysis as a JSON object with the following structure:
{{
  "flows": [
    {{
      "name": "Flow name",
      "description": "Flow description",
      "steps": ["Step 1", "Step 2", ...],
      "relevant_variables": ["var1", "var2", ...],
      "interaction_points": ["external call", "state change", ...]
    }},
    ...
  ]
}}
        """
    
    async def _analyze_contract(self, contract_name: str, contract_code: str) -> List[Dict[str, Any]]:
        """
        Analyze a contract for business flows
        
        Args:
            contract_name: Name of the contract
            contract_code: Solidity code of the contract
            
        Returns:
            List of business flow dictionaries
        """
        logger.debug(f"Starting business flow analysis for contract: {contract_name}")
        
        # Check if we should use Guidance
        if self.use_guidance and GUIDANCE_AVAILABLE:
            return await self._analyze_contract_with_guidance(contract_name, contract_code)
        
        # Otherwise use the standard approach
        return await self._analyze_contract_standard(contract_name, contract_code)
    
    async def _analyze_contract_with_guidance(self, contract_name: str, contract_source: str) -> List[BusinessFlow]:
        """Analyze a contract using Guidance for structured output"""
        logger.info(f"Analyzing contract with Guidance: {contract_name}")
        
        # Create a guidance program with the appropriate output class
        guidance_program = await create_guidance_program(
            output_cls=BusinessFlowAnalysisResult,
            prompt_template=self._create_contract_analysis_prompt(contract_name, contract_source),
            model=getattr(config, "ANALYSIS_MODEL", None),
            provider=getattr(config, "ANALYSIS_MODEL_PROVIDER", None),
            verbose=True
        )
        
        if not guidance_program:
            logger.error(f"Failed to create guidance program for contract: {contract_name}")
            return []
        
        try:
            # Execute the guidance program (no additional parameters needed since they're in the template)
            extraction_result = await guidance_program()
            
            logger.info(f"Successfully generated business flow analysis using Guidance for contract: {contract_name}")
            
            # Add debug logging to see what's returned
            logger.debug(f"Extraction result type: {type(extraction_result)}")
            if hasattr(extraction_result, "__dict__"):
                logger.debug(f"Extraction result attributes: {extraction_result.__dict__.keys()}")
            else:
                logger.debug(f"Extraction result repr: {repr(extraction_result)[:200]}")
            
            # Handle both 'flows' and 'business_flows' attribute names
            flows = []
            
            # Check for business_flows attribute (expected from BusinessFlowAnalysisResult)
            if hasattr(extraction_result, "business_flows"):
                flow_data_list = extraction_result.business_flows
                logger.debug(f"Found business_flows attribute with {len(flow_data_list)} items")
                
                # Convert to BusinessFlow objects
                for flow_data in flow_data_list:
                    flow = self._create_business_flow(flow_data, contract_name)
                    flows.append(flow)
            
            # Check for 'flows' attribute (common in LLM responses)
            elif hasattr(extraction_result, "flows"):
                flow_data_list = extraction_result.flows
                logger.debug(f"Found flows attribute with {len(flow_data_list)} items")
                
                # Convert to BusinessFlow objects
                for flow_data in flow_data_list:
                    flow = self._create_business_flow(flow_data, contract_name)
                    flows.append(flow)
            
            # Support dict response
            elif isinstance(extraction_result, dict) and ('flows' in extraction_result or 'business_flows' in extraction_result):
                flow_data_list = extraction_result.get('business_flows', extraction_result.get('flows', []))
                logger.debug(f"Found flows in dict with {len(flow_data_list)} items")
                
                # Convert to BusinessFlow objects
                for flow_data in flow_data_list:
                    flow = self._create_business_flow(flow_data, contract_name, is_dict=True)
                    flows.append(flow)
            
            else:
                logger.warning(f"No valid extraction result structure for contract: {contract_name}")
                return []
                
            if not flows:
                logger.warning(f"No business flows extracted from {contract_name}")
                
            return flows
            
        except Exception as e:
            logger.error(f"Error analyzing contract {contract_name}: {e}")
            logger.exception("Stack trace:")
            return []
    
    async def _analyze_contract_standard(self, contract_name: str, contract_code: str) -> List[Dict[str, Any]]:
        """Analyze a contract using standard LLM approach
        
        Args:
            contract_name: Name of the contract
            contract_code: Solidity code of the contract
            
        Returns:
            List of business flow dictionaries
        """
        logger.debug(f"Starting standard business flow analysis for contract: {contract_name}")
        
        # Check if LLM adapter is available
        if not self.llm_adapter:
            logger.error("No LLM adapter available for business flow analysis")
            return []
            
        try:
            # Use serializable message format for LLM
            messages = [
                {"role": "system", "content": """You are an expert smart contract analyst with deep understanding 
of blockchain business logic. Identify the key workflows and processes in this contract.
Always respond with properly formatted JSON that can be parsed by Python's json.loads()."""},
                {"role": "user", "content": self._create_contract_analysis_prompt(contract_name, contract_code)}
            ]
            
            # Get LLM response
            response = await self._get_llm_response(messages)
            
            # Use our robust JSON extractor
            from ..utils.json_extractor import extract_json_from_complex_response
            result, debug_info = extract_json_from_complex_response(response)
            
            # Log debug info at debug level
            logger.debug(f"JSON extraction debug info:\n{debug_info}")
            if result:
                flows = result.get("flows", [])
                logger.info(f"Successfully extracted business flow analysis with {len(flows)} flows for contract {contract_name}")
                
                # Add contract name to each flow
                for flow in flows:
                    flow["contract_name"] = contract_name
                      
                return flows
            else:
                logger.error(f"Failed to extract JSON from response for {contract_name}")
                # Log a portion of the raw response for debugging
                logger.debug(f"Raw response (first 500 chars): {response[:500]}")
                return []
        except Exception as e:
            logger.error(f"Error in standard business flow analysis: {str(e)}")
            return []
            
    def _create_contract_analysis_prompt(self, contract_name: str, contract_code: str) -> str:
        """Create a prompt for contract-level flow analysis"""
        # Format flow types for the prompt
        flow_type_text = "\n".join([f"- {flow_type}" for flow_type in self.flow_types])
        
        return f"""
Analyze the following Solidity smart contract and identify specific business flows.
Focus on these types of flows:
{flow_type_text}

Contract Name: {contract_name}

Contract code:
```solidity
{contract_code}
```

Return your analysis as a JSON object with the following structure:
{{
  "flows": [
    {{
      "name": "Flow name",
      "description": "Flow description",
      "steps": ["Step 1", "Step 2", ...],
      "functions": ["function1", "function2", ...],
      "actors": ["actor1", "actor2", ...],
      "flow_type": "{self.flow_types[0] if self.flow_types else 'token_transfer'}"
    }}
  ],
  "contract_summary": "Summary of the contract's business logic"
}}
        """
    
    async def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM adapter using chat messages
        
        Args:
            messages: List of chat messages with role and content
            
        Returns:
            Text response from the LLM
        """
        if not self.llm_adapter:
            logger.error("No LLM adapter available")
            return "{}"
            
        try:
            # Get token limit from config
            from finite_monkey.nodes_config import config
            max_tokens = getattr(config, "MAX_TOKENS", 1024)
            temperature = getattr(config, "TEMPERATURE", 0.1)
            
            # Convert to ChatMessage format if needed
            chat_messages = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in messages
            ]
            
            logger.debug(f"Attempting to call LLM with {len(messages)} messages")
            
            # Define common parameter dictionaries for different method types
            chat_params = {
                "messages": chat_messages,
            }
            
            # Try each method in sequence until one works
            response = None
            
            # 1. Try llm.achat if available 
            if hasattr(self.llm_adapter, 'llm') and hasattr(self.llm_adapter.llm, 'achat'):
                logger.debug("Using llm_adapter.llm.achat method")
                try:
                    response = await self.llm_adapter.llm.achat(messages=chat_messages)
                except Exception as e:
                    logger.warning(f"llm.achat method failed: {e}")
                    
            # 2. Try achat directly on adapter
            if response is None and hasattr(self.llm_adapter, 'achat'):
                logger.debug("Using llm_adapter.achat method")
                try:
                    response = await self.llm_adapter.achat(messages=chat_messages)
                except Exception as e:
                    logger.warning(f"achat method failed: {e}")
            
            # 3. Try direct Ollama client
            if response is None:
                logger.debug("Using direct Ollama client")
                try:
                    from ..utils.ollama_direct import DirectOllamaClient
                    
                    # Get model info from adapter if possible
                    model = getattr(self.llm_adapter, 'model_name', 
                           getattr(self.llm_adapter, 'model', 
                           getattr(self.llm_adapter.llm, 'model_name', 
                           getattr(self.llm_adapter.llm, 'model', config.ANALYSIS_MODEL))))
                    
                    # Format prompt from messages
                    prompt = "\n\n".join([
                        f"# {msg['role'].upper()}\n{msg['content']}" 
                        for msg in messages
                    ])
                    
                    # Use direct Ollama client
                    client = DirectOllamaClient()
                    result = await client.generate(
                        model=model,
                        prompt=prompt,
                        temperature=temperature
                    )
                    await client.close()
                    if "error" not in result:
                        return result.get("response", "{}")
                    else:
                        logger.error(f"Direct Ollama call failed: {result['error']}")
                        return "{}"
                except Exception as e:
                    logger.error(f"Direct Ollama call failed: {e}")
                    return "{}"
            
            # Extract response text from adapter response
            if response:
                if hasattr(response, "message") and hasattr(response.message, "content"):
                    return response.message.content
                elif hasattr(response, "text"):
                    return response.text
                elif hasattr(response, "response"):
                    return response.response
                elif isinstance(response, str):
                    return response
                else:
                    # Return string representation as last resort
                    return str(response)
            else:
                logger.error("All LLM methods failed")
                return "{}"
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            logger.debug(f"Available methods on llm_adapter: {dir(self.llm_adapter)}")
            return "{}"
    
    def _create_business_flow(self, flow_data, contract_name, is_dict=False):
        """
        Create a BusinessFlow object with proper handling for required fields
        
        Args:
            flow_data: The flow data (object or dict)
            contract_name: Name of the contract
            is_dict: Whether flow_data is a dict or object
            
        Returns:
            BusinessFlow object
        """
        try:
            if is_dict:
                # Add verbose logging to help diagnose issues
                logger.debug(f"Creating BusinessFlow from dict with keys: {flow_data.keys() if hasattr(flow_data, 'keys') else 'No keys'}")
                return BusinessFlow(
                    name=flow_data.get("name", "Unnamed Flow"),
                    description=flow_data.get("description", ""),
                    steps=flow_data.get("steps", []),
                    contract_name=contract_name,
                    inputs=flow_data.get("inputs", []),
                    outputs=flow_data.get("outputs", []),
                    functions=flow_data.get("functions", [])  # Required field
                )
            else:
                # Add verbose logging to help diagnose issues
                if hasattr(flow_data, "__dict__"):
                    logger.debug(f"Creating BusinessFlow from object with attributes: {flow_data.__dict__.keys()}")
                return BusinessFlow(
                    name=getattr(flow_data, "name", "Unnamed Flow"),
                    description=getattr(flow_data, "description", ""),
                    steps=getattr(flow_data, "steps", []),
                    contract_name=contract_name,
                    inputs=getattr(flow_data, "inputs", []),
                    outputs=getattr(flow_data, "outputs", []),
                    functions=getattr(flow_data, "functions", [])  # Required field
                )
        except Exception as e:
            logger.error(f"Error creating BusinessFlow object: {e}")
            # Create a minimal valid object as fallback
            return BusinessFlow(
                name="Unnamed Flow",
                description="Error creating flow",
                steps=[],
                functions=[],  # Required field
                contract_name=contract_name,
                inputs=[],
                outputs=[]
            )