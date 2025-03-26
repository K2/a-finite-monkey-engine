"""
Pipeline stage for batch validation of findings using a separate LLM
"""

from typing import Dict, List, Any, Optional
import json
from loguru import logger

from ..pipeline.core import Context
from ..adapters.validator_adapter import ValidatorAdapter
from ..utils.path_analyzer import PathAnalyzer
from ..config.model_provider import ModelProvider

class ValidationStage:
    """
    Pipeline stage that validates findings in batch at the end of the pipeline
    """
    
    def __init__(self, validator_adapter: Optional[ValidatorAdapter] = None):
        """
        Initialize the validation stage
        
        Args:
            validator_adapter: Validator adapter for validating findings
        """
        self.validator_adapter = validator_adapter
        self.name = "validate_findings"
        self.path_analyzer = PathAnalyzer()
    
    async def __call__(self, context: Context) -> Context:
        """
        Process the context by validating all findings in batch
        
        Args:
            context: Context with findings from all analyzers
            
        Returns:
            Updated context with validated findings
        """
        # Lazily initialize validator if needed
        if not self.validator_adapter:
            from ..adapters.validator_adapter import ValidatorAdapter
            from ..nodes_config import config as node_config
            
            # Create validator adapter with lazy initialization
            try:
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                llm_adapter = LlamaIndexAdapter(analyzer_type="validator")
                self.validator_adapter = ValidatorAdapter(llm_adapter)
                logger.info("Lazily initialized validator adapter")
            except Exception as e:
                logger.error(f"Failed to initialize validator adapter: {e}")
                return context  # Skip validation if we can't initialize
        
        logger.info("Starting batch validation of findings")
        
        # Get all findings
        if not hasattr(context, "findings") or not context.findings:
            logger.info("No findings to validate")
            return context
        
        # Group findings by file for better context
        findings_by_file = {}
        for finding in context.findings:
            file_id = finding.get("file")
            if not file_id:
                file_id = "global"
                
            if file_id not in findings_by_file:
                findings_by_file[file_id] = []
                
            findings_by_file[file_id].append(finding)
        
        # Store original findings
        context.unvalidated_findings = context.findings.copy()
        context.findings = []
        
        # Process each file's findings
        total_validated = 0
        for file_id, file_findings in findings_by_file.items():
            # Get file data for context
            file_data = {}
            if file_id in context.files:
                file_data = context.files[file_id]
            
            # Prepare reachability information
            reachability_info = self._prepare_reachability_info(context, file_id)
            
            # Validate findings for this file
            validated_findings = await self.validator_adapter.validate_findings_with_reachability(
                findings=file_findings,
                file_data=file_data,
                reachability_info=reachability_info
            )
            
            # Process validated findings to add execution paths
            enhanced_findings = self._enhance_with_execution_paths(
                validated_findings, 
                context, 
                file_id
            )
            
            # Add to the final findings list
            context.findings.extend(enhanced_findings)
            total_validated += len(enhanced_findings)
            
            logger.info(f"Validated {len(enhanced_findings)} findings for {file_id}")
        
        logger.info(f"Completed batch validation. Final finding count: {total_validated}")
        return context
    
    def _prepare_reachability_info(self, context: Context, file_id: str) -> Dict[str, Any]:
        """
        Prepare reachability information from dataflows and business flows
        
        Args:
            context: Analysis context
            file_id: File ID to prepare information for
            
        Returns:
            Dictionary with reachability information
        """
        reachability_info = {
            "dataflows": [],
            "business_flows": [],
            "control_flow_conditions": []
        }
        
        # Add dataflow information
        if hasattr(context, "dataflows") and file_id in context.dataflows:
            reachability_info["dataflows"] = context.dataflows[file_id]
        
        # Add business flow information
        if hasattr(context, "business_flows") and file_id in context.business_flows:
            reachability_info["business_flows"] = context.business_flows[file_id]
        
        # Extract control flow conditions from dataflows
        if hasattr(context, "dataflows") and file_id in context.dataflows:
            for flow in context.dataflows[file_id]:
                if "affecting_variables" in flow:
                    for var in flow["affecting_variables"]:
                        if var not in reachability_info["control_flow_conditions"]:
                            reachability_info["control_flow_conditions"].append(var)
        
        return reachability_info
    
    def _enhance_with_execution_paths(
        self, 
        findings: List[Dict[str, Any]], 
        context: Context, 
        file_id: str
    ) -> List[Dict[str, Any]]:
        """
        Enhance findings with detailed execution paths
        
        Args:
            findings: Validated findings
            context: Analysis context
            file_id: File ID
            
        Returns:
            Enhanced findings with execution paths
        """
        for finding in findings:
            # Skip if no location information
            if "location" not in finding:
                continue
                
            # Get related dataflows
            related_flows = self._find_related_dataflows(context, file_id, finding)
            
            if related_flows:
                # Add execution path information
                execution_paths = self.path_analyzer.construct_execution_paths(
                    finding, related_flows, context.files.get(file_id, {})
                )
                
                if execution_paths:
                    finding["execution_paths"] = execution_paths
                    
                    # Add conditions that must hold for the vulnerability to be exploited
                    conditions = self.path_analyzer.extract_path_conditions(execution_paths)
                    if conditions:
                        finding["exploit_conditions"] = conditions
        
        return findings
    
    def _find_related_dataflows(
        self, 
        context: Context, 
        file_id: str,
        finding: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find dataflows related to a finding"""
        if not hasattr(context, "dataflows") or file_id not in context.dataflows:
            return []
            
        related_flows = []
        finding_location = finding.get("location", "")
        
        # Extract line numbers from location
        finding_lines = set()
        if ":" in finding_location:
            try:
                parts = finding_location.split(":", 1)[1]
                if "-" in parts:
                    start, end = map(int, parts.split("-"))
                    finding_lines = set(range(start, end + 1))
                else:
                    line = int(parts)
                    finding_lines = {line}
            except (ValueError, IndexError):
                pass
        
        # Find overlapping dataflows
        for flow in context.dataflows[file_id]:
            flow_lines = set()
            for node in flow.get("path", []):
                if "start_line" in node and "end_line" in node:
                    flow_lines.update(range(node["start_line"], node["end_line"] + 1))
            
            if finding_lines and flow_lines and finding_lines.intersection(flow_lines):
                related_flows.append(flow)
                
            # Also check for function name overlap
            if "function" in finding:
                for node in flow.get("path", []):
                    if node.get("name") == finding["function"]:
                        if flow not in related_flows:
                            related_flows.append(flow)
        
        return related_flows
