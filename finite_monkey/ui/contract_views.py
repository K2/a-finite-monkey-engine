"""
Contract views for the Starlight UI

This module provides helper functions and classes for displaying Solidity contracts
in the UI with rich property grids and visualizations.
"""

from typing import Dict, List, Any, Optional
from ..models.contract import ContractDef, FunctionDef, BusinessFlow

class PropertyGridView:
    """Helper class for generating property grid data for UI"""
    
    @staticmethod
    def contract_properties(contract: ContractDef) -> List[Dict[str, Any]]:
        """Convert contract properties to grid-friendly format"""
        props = contract.properties
        return [{"key": key, "value": value} for key, value in props.items()]
    
    @staticmethod
    def function_properties(func: FunctionDef) -> List[Dict[str, Any]]:
        """Convert function properties to grid-friendly format"""
        props = func.properties
        return [{"key": key, "value": value} for key, value in props.items()]
    
    @staticmethod
    def business_flow_properties(flow: BusinessFlow) -> List[Dict[str, Any]]:
        """Convert business flow properties to grid-friendly format"""
        props = flow.properties
        return [{"key": key, "value": value} for key, value in props.items()]


class RelationshipView:
    """Helper class for visualizing code relationships"""
    
    @staticmethod
    def function_call_graph(contract: ContractDef) -> Dict[str, Any]:
        """Generate function call graph data"""
        nodes = []
        edges = []
        
        # Add nodes for all functions
        for func_name, func in contract.functions.items():
            nodes.append({
                "id": func_name,
                "label": func_name,
                "type": "function",
                "visibility": func.visibility if isinstance(func, FunctionDef) else func.get('visibility', 'internal')
            })
        
        # Add edges for function calls
        for caller, callees in contract.function_relationships.items():
            for callee in callees:
                edges.append({
                    "source": caller,
                    "target": callee,
                    "type": "calls"
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    @staticmethod
    def business_flow_graph(contract: ContractDef) -> Dict[str, Any]:
        """Generate business flow graph data"""
        nodes = []
        edges = []
        
        # Add nodes for all functions
        for func_name, func in contract.functions.items():
            if not isinstance(func, FunctionDef):
                func = FunctionDef(func)
                
            flows = func.business_flows
            
            # Create function node
            nodes.append({
                "id": func_name,
                "label": func_name,
                "type": "function"
            })
            
            # Add flow nodes and edges
            for i, flow in enumerate(flows):
                flow_id = f"{func_name}_flow_{i}"
                flow_type = flow.get('type', 'unknown')
                
                nodes.append({
                    "id": flow_id,
                    "label": flow_type,
                    "type": "flow",
                    "confidence": flow.get('confidence', 0)
                })
                
                edges.append({
                    "source": func_name,
                    "target": flow_id,
                    "type": "has_flow"
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        }


class SecurityDashboard:
    """Helper class for security analysis dashboard"""
    
    @staticmethod
    def risk_overview(contract: ContractDef) -> Dict[str, Any]:
        """Generate security risk overview"""
        risk_counts = {
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        risk_areas = {
            "value_transfer": [],
            "external_calls": [],
            "state_changes": [],
            "access_control": []
        }
        
        # Calculate risk metrics
        for func_name, func in contract.functions.items():
            if not isinstance(func, FunctionDef):
                func = FunctionDef(func)
                
            # Count risks by severity
            risk_factors = func.security_risk_factors
            
            if any(r in ["Contains selfdestruct", "Uses delegatecall"] for r in risk_factors):
                risk_counts["high"] += 1
            elif any(r in ["Accepts Ether", "Transfers value"] for r in risk_factors):
                risk_counts["medium"] += 1
            elif risk_factors:
                risk_counts["low"] += 1
            
            # Categorize functions by risk area
            for flow in func.business_flows:
                flow_type = flow.get('type', '')
                
                if flow_type in ['value_transfer', 'ether_transfer']:
                    risk_areas["value_transfer"].append(func_name)
                elif flow_type in ['external_call', 'cross_contract']:
                    risk_areas["external_calls"].append(func_name)
                elif flow_type in ['state_change', 'storage_write']:
                    risk_areas["state_changes"].append(func_name)
                elif flow_type in ['access_control', 'authorization']:
                    risk_areas["access_control"].append(func_name)
        
        return {
            "risk_counts": risk_counts,
            "risk_areas": risk_areas,
            "security_properties": contract.security_properties
        }
