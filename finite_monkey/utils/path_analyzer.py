"""
Utilities for analyzing execution paths in findings
"""

from typing import Dict, List, Any

class PathAnalyzer:
    """
    Analyzer for execution paths in findings
    """
    
    def construct_execution_paths(
        self,
        finding: Dict[str, Any],
        dataflows: List[Dict[str, Any]],
        file_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Construct detailed execution paths for a finding
        
        Args:
            finding: Finding to analyze
            dataflows: Related dataflows
            file_data: File data
            
        Returns:
            List of execution paths
        """
        execution_paths = []
        
        # If the finding already has a call_flow from validation, use it
        if "call_flow" in finding and isinstance(finding["call_flow"], list):
            execution_paths.append({
                "name": "Validator-identified path",
                "steps": finding["call_flow"],
                "source": "validator"
            })
        
        # Add paths from dataflows
        for i, flow in enumerate(dataflows):
            path = {
                "name": f"Dataflow path {i+1}",
                "source": "dataflow_analyzer",
                "steps": []
            }
            
            # Extract steps from path
            for node in flow.get("path", []):
                step = {
                    "type": node.get("type", "unknown"),
                    "name": node.get("name", "unknown"),
                    "line": node.get("start_line", 0)
                }
                
                # Add code snippet if available
                if "code" in node:
                    step["code"] = node["code"]
                
                path["steps"].append(step)
            
            # Add exploitability score
            if "exploitability" in flow:
                path["exploitability"] = flow["exploitability"]
                
            # Add sink vulnerability type
            if "sink" in flow and "vulnerability" in flow["sink"]:
                path["vulnerability_type"] = flow["sink"]["vulnerability"]
            
            execution_paths.append(path)
        
        return execution_paths
    
    def extract_path_conditions(self, execution_paths: List[Dict[str, Any]]) -> List[str]:
        """
        Extract conditions that must hold for paths to be exploitable
        
        Args:
            execution_paths: Execution paths
            
        Returns:
            List of conditions
        """
        conditions = []
        
        for path in execution_paths:
            # Check if the path has required conditions
            for step in path.get("steps", []):
                if "condition" in step:
                    cond = step["condition"]
                    if cond not in conditions:
                        conditions.append(cond)
                
                # Look for conditions in code
                if "code" in step:
                    code = step["code"]
                    if "if " in code or " if(" in code or "(if " in code:
                        # Extract simple condition from if statement
                        try:
                            condition_start = code.find("if") + 2
                            condition_end = code.find(")", condition_start)
                            if condition_end > condition_start:
                                condition = code[condition_start:condition_end+1].strip()
                                if condition and condition not in conditions:
                                    conditions.append(condition)
                        except:
                            pass
        
        return conditions
