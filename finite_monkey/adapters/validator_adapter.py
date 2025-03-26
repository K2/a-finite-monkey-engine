"""
Adapter for result validation using a separate LLM
"""

from typing import Dict, List, Any, Optional
import json
from loguru import logger

from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from ..config.model_config import ModelConfig
from ..utils.json_repair import safe_parse_json, extract_json_from_text

class ValidatorAdapter:
    """
    Adapter for validating results using a separate LLM
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the validator adapter
        
        Args:
            config: Model configuration, or None to use default
        """
        self.config = config or ModelConfig()
        self._initialize_validator()
    
    def _initialize_validator(self):
        """Initialize the validator LLM"""
        try:
            model_name = self.config.validator_model
            
            # Special handling based on model type
            if model_name.startswith("hf.co/"):
                # For Ollama models loaded from Hugging Face
                model_name = model_name.replace("hf.co/", "")
                self.validator = Ollama(
                    model=model_name,
                    **self.config.get_model_params(self.config.validator_model)
                )
                logger.info(f"Initialized validator with Ollama model: {model_name}")
            elif model_name.startswith("anthropic/"):
                # Initialize Anthropic model (will be used in production)
                try:
                    from llama_index.llms.anthropic import Anthropic
                    self.validator = Anthropic(
                        model=model_name,
                        **self.config.get_model_params(self.config.validator_model)
                    )
                    logger.info(f"Initialized validator with Anthropic model: {model_name}")
                except ImportError:
                    logger.warning("Anthropic package not installed, falling back to default LLM")
                    self.validator = Settings.llm
            else:
                # For other models, use Ollama
                self.validator = Ollama(
                    model=model_name,
                    **self.config.get_model_params(self.config.validator_model)
                )
                logger.info(f"Initialized validator with model: {model_name}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize validator LLM: {e}")
            logger.warning("Falling back to default LLM for validation")
            self.validator = Settings.llm
    
    async def validate_findings_with_reachability(
        self,
        findings: List[Dict[str, Any]],
        file_data: Dict[str, Any],
        reachability_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Validate findings with reachability analysis
        
        Args:
            findings: List of findings to validate
            file_data: Data for the file being analyzed
            reachability_info: Dataflows and business flows for reachability analysis
            
        Returns:
            Validated and enhanced findings
        """
        if not findings:
            return []
        
        try:
            # Extract file content for context
            file_content = file_data.get("content", "")
            file_name = file_data.get("name", "unknown")
            
            # Prepare validation prompt with reachability information
            prompt = self._prepare_validation_prompt_with_reachability(
                findings,
                file_name,
                file_content,
                reachability_info
            )
            
            # Get validation response
            response = await self.validator.acomplete(prompt)
            
            # Process validation response
            validated_findings = self._process_validation_response(
                response.text,
                findings,
                reachability_info
            )
            
            return validated_findings
            
        except Exception as e:
            logger.error(f"Error in validation with reachability: {e}")
            # Return original findings on error
            return findings
    
    def _prepare_validation_prompt_with_reachability(
        self,
        findings: List[Dict[str, Any]],
        file_name: str,
        file_content: str,
        reachability_info: Dict[str, Any]
    ) -> str:
        """
        Prepare validation prompt with reachability information
        
        Args:
            findings: Findings to validate
            file_name: Name of the file
            file_content: Content of the file
            reachability_info: Reachability information
            
        Returns:
            Validation prompt
        """
        # Extract relevant dataflows for context
        dataflow_summary = self._format_dataflows_for_prompt(reachability_info.get("dataflows", []))
        
        # Format control flow conditions
        conditions_summary = self._format_conditions_for_prompt(
            reachability_info.get("control_flow_conditions", [])
        )
        
        # Create a prompt that asks the validator to evaluate findings with reachability
        prompt = f"""
        You are an expert smart contract security validator. Analyze these findings for {file_name} and determine their reachability.
        
        CODE SUMMARY:
        ```solidity
        {file_content[:1000]}... (truncated for brevity)
        ```
        
        DATAFLOW PATHS:
        {dataflow_summary}
        
        CONTROL FLOW CONDITIONS:
        {conditions_summary}
        
        FINDINGS TO VALIDATE:
        ```json
        {json.dumps(findings, indent=2)}
        ```
        
        For each finding, perform deep validation with reachability analysis:
        
        1. Is this finding a true positive?
        2. Can the vulnerability actually be triggered based on data and control flows?
        3. What exact expression conditions must hold true/false/equal for the vulnerability to be exploited?
        4. What is the complete call flow path to trigger the vulnerability?
        5. Are there any limiting conditions that would prevent exploitation?
        
        For each validated finding, add these fields:
        - "reachable": true/false whether the vulnerability is actually reachable
        - "call_flow": Step-by-step list of function calls needed to trigger the vulnerability
        - "required_conditions": List of expressions that must be true/false/equal for exploitation
        - "validation_confidence": Value from 0.0-1.0 indicating your confidence in this validation
        
        Format your response as a JSON array of findings. Remove any findings that are clearly false positives.
        Each finding object should include all original fields plus the added validation fields.

        Return ONLY the JSON array with no explanatory text.
        """
        return prompt
    
    def _format_dataflows_for_prompt(self, dataflows: List[Dict[str, Any]]) -> str:
        """Format dataflows for inclusion in the prompt"""
        if not dataflows:
            return "No dataflow information available."
        
        # Limit to the most important dataflows
        max_flows = 5
        sorted_flows = sorted(
            dataflows,
            key=lambda f: f.get("exploitability", 0),
            reverse=True
        )[:max_flows]
        
        result = []
        for i, flow in enumerate(sorted_flows, 1):
            source = flow.get("source", {}).get("name", "unknown")
            sink = flow.get("sink", {}).get("name", "unknown")
            
            path_str = ""
            for node in flow.get("path", [])[:10]:  # Limit path nodes
                path_str += f"- {node.get('type', 'node')}: {node.get('name', 'unknown')} (line {node.get('start_line', '?')})\n"
            
            result.append(
                f"Flow {i}: {source} → {sink} (Exploitability: {flow.get('exploitability', 0)}/10)\n"
                f"Path:\n{path_str}"
            )
        
        return "\n".join(result)
    
    def _format_conditions_for_prompt(self, conditions: List[Dict[str, Any]]) -> str:
        """Format control flow conditions for inclusion in the prompt"""
        if not conditions:
            return "No control flow conditions available."
        
        result = []
        for condition in conditions:
            name = condition.get("name", "unknown")
            result.append(f"- {name}: {condition.get('type', 'variable')}, influences flow: {condition.get('influences_flow', False)}")
        
        return "\n".join(result)
    
    def _process_validation_response(
        self,
        response: str,
        original_findings: List[Dict[str, Any]],
        reachability_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process the validation response with reachability information
        
        Args:
            response: Response from validator
            original_findings: Original findings
            reachability_info: Reachability information
            
        Returns:
            Validated findings
        """
        try:
            # Try to extract JSON from response
            json_data = extract_json_from_text(response)
            validated = safe_parse_json(json_data, [])
            
            if isinstance(validated, list):
                # Post-process to ensure we have complete information
                processed_findings = self._post_process_findings(validated, original_findings, reachability_info)
                return processed_findings
            
            # If we couldn't parse JSON or the result isn't a list,
            # log warning and return original findings
            logger.warning("Failed to parse validator response, using original findings")
            return original_findings
            
        except Exception as e:
            logger.error(f"Error processing validation response: {e}")
            return original_findings
    
    def _post_process_findings(
        self, 
        validated_findings: List[Dict[str, Any]], 
        original_findings: List[Dict[str, Any]],
        reachability_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Post-process validated findings to ensure completeness
        
        Args:
            validated_findings: Findings from validator
            original_findings: Original findings
            reachability_info: Reachability information
            
        Returns:
            Complete validated findings
        """
        # Create mapping of original findings by title for reference
        original_map = {f.get("title", f.get("id", str(i))): f 
                        for i, f in enumerate(original_findings)}
        
        processed = []
        
        for finding in validated_findings:
            # Skip findings marked as unreachable (unless config says to keep them)
            if "reachable" in finding and finding["reachable"] is False:
                continue
                
            # Find corresponding original finding
            title = finding.get("title")
            original = original_map.get(title)
            
            if original:
                # Ensure all original fields are preserved
                for key, value in original.items():
                    if key not in finding:
                        finding[key] = value
            
            # Ensure we have the required validation fields
            if "reachable" not in finding:
                finding["reachable"] = True  # Default to reachable
                
            if "validation_confidence" not in finding:
                finding["validation_confidence"] = 0.7  # Default confidence
            
            # Add reachability information if not present
            if "required_conditions" not in finding:
                related_flows = self._find_related_dataflows_for_finding(finding, reachability_info)
                if related_flows:
                    conditions = self._extract_conditions_from_dataflows(related_flows)
                    finding["required_conditions"] = conditions
            
            processed.append(finding)
            
        return processed
    
    def _find_related_dataflows_for_finding(
        self, 
        finding: Dict[str, Any],
        reachability_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find dataflows related to a finding"""
        dataflows = reachability_info.get("dataflows", [])
        
        if not dataflows:
            return []
            
        # Extract location info
        finding_loc = finding.get("location", "")
        finding_lines = set()
        
        if ":" in finding_loc:
            try:
                parts = finding_loc.split(":", 1)[1]
                if "-" in parts:
                    start, end = map(int, parts.split("-"))
                    finding_lines = set(range(start, end + 1))
                else:
                    line = int(parts)
                    finding_lines = {line}
            except (ValueError, IndexError):
                pass
        
        # Find overlapping dataflows
        related = []
        for flow in dataflows:
            flow_lines = set()
            for node in flow.get("path", []):
                if "start_line" in node and "end_line" in node:
                    flow_lines.update(range(node["start_line"], node["end_line"] + 1))
            
            if finding_lines and flow_lines and finding_lines.intersection(flow_lines):
                related.append(flow)
        
        return related
    
    def _extract_conditions_from_dataflows(self, dataflows: List[Dict[str, Any]]) -> List[str]:
        """Extract control flow conditions from dataflows"""
        conditions = []
        
        for flow in dataflows:
            # Get conditions from affecting variables
            if "affecting_variables" in flow:
                for var in flow["affecting_variables"]:
                    name = var.get("name", "unknown")
                    condition = f"{name} must be controlled"
                    if condition not in conditions:
                        conditions.append(condition)
            
            # Try to extract conditions from summary
            if "summary" in flow and "attack_vector" in flow["summary"]:
                attack_vector = flow["summary"]["attack_vector"]
                if "if" in attack_vector.lower() or "when" in attack_vector.lower():
                    conditions.append(attack_vector)
        
        return conditions

    async def validate_findings(self, findings: List[Dict[str, Any]], context: str = "") -> List[Dict[str, Any]]:
        """
        Validate and potentially filter/enhance findings
        
        Args:
            findings: List of findings to validate
            context: Optional context about the findings
            
        Returns:
            Validated findings
        """
        # Use our more detailed method for validation
        return await self.validate_findings_with_reachability(
            findings=findings,
            file_data={"name": "unknown", "content": context},
            reachability_info={}
        )
