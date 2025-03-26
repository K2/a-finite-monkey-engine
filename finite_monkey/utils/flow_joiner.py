"""
Utility for joining dataflow and business flow information.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

class FlowJoiner:
    """
    Joins dataflow analysis with business flow information to provide comprehensive context.
    """
    
    @staticmethod
    def join_flows(dataflows: Dict[str, List[Dict[str, Any]]], business_flows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Join dataflows with business flows to provide comprehensive context
        
        Args:
            dataflows: Dictionary of dataflows by file
            business_flows: Dictionary of business flows by file
            
        Returns:
            Enriched dataflows with business context
        """
        logger.info("Joining data flows with business flows")
        enriched_flows = {}
        
        for file_id, flows in dataflows.items():
            enriched_flows[file_id] = []
            
            # Get business flows for this file
            file_business_flows = business_flows.get(file_id, [])
            
            for flow in flows:
                # Skip flows that already have business context
                if "business_context" in flow and flow["business_context"]:
                    enriched_flows[file_id].append(flow)
                    continue
                
                # Find business flows that overlap with this flow
                overlapping_flows = []
                flow_lines = set()
                
                # Extract lines in the flow path
                for node in flow.get("path", []):
                    if "start_line" in node and "end_line" in node:
                        for line in range(node["start_line"], node["end_line"] + 1):
                            flow_lines.add(line)
                
                # Find overlapping business flows
                for bflow in file_business_flows:
                    if not isinstance(bflow, dict):
                        continue
                    if "start_line" not in bflow or "end_line" not in bflow:
                        continue
                    
                    bflow_lines = set(range(bflow["start_line"], bflow["end_line"] + 1))
                    if flow_lines.intersection(bflow_lines):
                        overlapping_flows.append(bflow)
                
                # Add business context
                flow["business_context"] = overlapping_flows
                
                # Calculate business context strength
                if overlapping_flows:
                    # More business flows = stronger business context
                    flow["business_relevance"] = min(len(overlapping_flows) * 0.2, 1.0)
                else:
                    flow["business_relevance"] = 0.0
                
                enriched_flows[file_id].append(flow)
        
        logger.info(f"Joined data flows across {len(enriched_flows)} files")
        return enriched_flows
    
    @staticmethod
    def generate_joint_findings(dataflows: Dict[str, List[Dict[str, Any]]], threshold: float = 7.0) -> List[Dict[str, Any]]:
        """
        Generate findings from dataflows for reporting
        
        Args:
            dataflows: Dictionary of dataflows by file
            threshold: Minimum exploitability score to include
            
        Returns:
            List of findings
        """
        findings = []
        
        for file_id, flows in dataflows.items():
            for flow in flows:
                # Skip flows below threshold
                if flow.get("exploitability", 0) < threshold:
                    continue
                
                # Skip flows without summaries
                if "summary" not in flow:
                    continue
                
                # Create a finding
                finding = {
                    "title": f"Data Flow Vulnerability: {flow['sink'].get('vulnerability', 'Unknown Risk')}",
                    "severity": flow["summary"].get("severity", "Medium"),
                    "description": flow["summary"].get("attack_vector", "Dangerous data flow detected"),
                    "impact": flow["summary"].get("impact", "Potential security risk"),
                    "location": f"{file_id}:{flow['source'].get('start_line', 0)}-{flow['sink'].get('end_line', 0)}",
                    "recommendation": flow["summary"].get("mitigation", "Review and secure this data flow"),
                    "confidence": flow["summary"].get("confidence", "Medium"),
                    "file": file_id,
                    "type": "dataflow",
                    "business_relevance": flow.get("business_relevance", 0.0),
                    "path": [
                        {
                            "type": node.get("type", "unknown"),
                            "name": node.get("name", "unnamed"),
                            "line": node.get("start_line", 0)
                        } for node in flow.get("path", [])
                    ]
                }
                
                # Add business context if available
                if "business_context" in flow and flow["business_context"]:
                    finding["business_context"] = [
                        {
                            "name": bf.get("name", "Unnamed flow"),
                            "description": bf.get("description", "No description")
                        } for bf in flow["business_context"]
                    ]
                
                findings.append(finding)
        
        logger.info(f"Generated {len(findings)} findings from data flows")
        return findings
    
    @staticmethod
    def enhance_with_vulnerability_info(dataflows: Dict[str, List[Dict[str, Any]]], vulnerabilities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enhance dataflows with vulnerability information
        
        Args:
            dataflows: Dictionary of dataflows by file
            vulnerabilities: List of vulnerability findings
            
        Returns:
            Dataflows enhanced with vulnerability information
        """
        # Group vulnerabilities by file
        vuln_by_file = {}
        for vuln in vulnerabilities:
            file_id = vuln.get("file")
            if file_id:
                if file_id not in vuln_by_file:
                    vuln_by_file[file_id] = []
                vuln_by_file[file_id].append(vuln)
        
        # Enhance dataflows
        for file_id, flows in dataflows.items():
            # Get vulnerabilities for this file
            file_vulns = vuln_by_file.get(file_id, [])
            
            for flow in flows:
                # Find vulnerabilities that overlap with this flow
                flow_vulns = []
                
                # Get flow lines
                flow_lines = set()
                for node in flow.get("path", []):
                    if "start_line" in node and "end_line" in node:
                        for line in range(node["start_line"], node["end_line"] + 1):
                            flow_lines.add(line)
                
                # Check each vulnerability for overlap
                for vuln in file_vulns:
                    # Extract line numbers from location if possible
                    location = vuln.get("location", "")
                    if ":" in location and "-" in location.split(":", 1)[1]:
                        try:
                            range_part = location.split(":", 1)[1]
                            start_line = int(range_part.split("-")[0])
                            end_line = int(range_part.split("-")[1])
                            
                            # Check for overlap
                            vuln_lines = set(range(start_line, end_line + 1))
                            if flow_lines.intersection(vuln_lines):
                                flow_vulns.append(vuln)
                        except (ValueError, IndexError):
                            pass
                
                # Add vulnerability context to flow
                if flow_vulns:
                    flow["related_vulnerabilities"] = flow_vulns
        
        logger.info(f"Enhanced dataflows with vulnerability information across {len(dataflows)} files")
        return dataflows
    
    @staticmethod
    def correlate_with_cognitive_biases(dataflows: Dict[str, List[Dict[str, Any]]], cognitive_biases: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Correlate dataflows with cognitive biases
        
        Args:
            dataflows: Dictionary of dataflows by file
            cognitive_biases: Dictionary of cognitive biases by file
            
        Returns:
            Dataflows enhanced with cognitive bias information
        """
        for file_id, flows in dataflows.items():
            # Get biases for this file
            if file_id not in cognitive_biases:
                continue
            
            file_biases = cognitive_biases[file_id]
            
            for flow in flows:
                # Find biases that overlap with this flow
                flow_biases = []
                
                # Get flow lines
                flow_lines = set()
                for node in flow.get("path", []):
                    if "start_line" in node and "end_line" in node:
                        for line in range(node["start_line"], node["end_line"] + 1):
                            flow_lines.add(line)
                
                # Check each bias for overlap
                for bias_type, instances in file_biases.biases.items():
                    for instance in instances:
                        if "line_number" in instance and instance["line_number"] in flow_lines:
                            flow_biases.append({
                                "type": bias_type,
                                "description": instance.get("description", ""),
                                "line": instance["line_number"]
                            })
                
                # Add bias context to flow
                if flow_biases:
                    flow["cognitive_biases"] = flow_biases
        
        logger.info(f"Correlated dataflows with cognitive biases across {len(dataflows)} files")
        return dataflows
