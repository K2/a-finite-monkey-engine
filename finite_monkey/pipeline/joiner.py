"""
Pipeline stage for normalizing and joining results from various analyses.
"""
from typing import Dict, List, Any
from loguru import logger
from ..pipeline.core import Context

class FlowJoiner:
    """
    Pipeline stage that normalizes and joins results from previous stages.
    Ensures consistent data structure for reporting.
    """
    
    async def __call__(self, context: Context) -> Context:
        """
        Process the context by normalizing and joining results.
        Required for the class to be callable as a pipeline stage.
        
        Args:
            context: The pipeline context to process
            
        Returns:
            Updated context with normalized data
        """
        logger.info("Starting flow joiner to normalize pipeline results")
        
        try:
            # Normalize business flows
            if hasattr(context, 'business_flows'):
                context.business_flows = self._normalize_business_flows(context.business_flows)
            
            # Normalize vulnerabilities
            if hasattr(context, 'vulnerabilities'):
                context.vulnerabilities = self._normalize_vulnerabilities(context.vulnerabilities)
            
            # Normalize documentation quality
            if hasattr(context, 'documentation_quality'):
                context.documentation_quality = self._normalize_documentation_quality(context.documentation_quality)
            
            # Normalize cognitive biases
            if hasattr(context, 'cognitive_biases'):
                context.cognitive_biases = self._normalize_cognitive_biases(context.cognitive_biases)
            
            # Normalize data flows
            if hasattr(context, 'dataflows'):
                context.dataflows = self._normalize_dataflows(context.dataflows)
            
            logger.info("Flow joiner completed normalization of all pipeline results")
            return context
            
        except Exception as e:
            logger.error(f"Error in FlowJoiner: {e}")
            context.add_error("Error in result normalization", e)
            return context
    
    def _normalize_business_flows(self, business_flows: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize business flows to ensure consistent structure."""
        normalized_flows = {}
        
        for contract_name, flows in business_flows.items():
            if not flows:
                continue
                
            normalized_contract_flows = []
            
            for flow in flows:
                # Handle both dictionary and object instances
                if isinstance(flow, dict):
                    normalized_flow = {
                        'name': flow.get('name', 'Unnamed Flow'),
                        'description': flow.get('description', 'No description'),
                        'steps': flow.get('steps', []),
                        'functions': flow.get('functions', []),
                        'actors': flow.get('actors', []),
                        'flow_type': flow.get('flow_type', 'unknown')
                    }
                else:
                    normalized_flow = {
                        'name': getattr(flow, 'name', 'Unnamed Flow'),
                        'description': getattr(flow, 'description', 'No description'),
                        'steps': getattr(flow, 'steps', []),
                        'functions': getattr(flow, 'functions', []),
                        'actors': getattr(flow, 'actors', []),
                        'flow_type': getattr(flow, 'flow_type', 'unknown')
                    }
                
                normalized_contract_flows.append(normalized_flow)
            
            normalized_flows[contract_name] = normalized_contract_flows
        
        return normalized_flows
    
    def _normalize_vulnerabilities(self, vulnerabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize vulnerabilities to ensure consistent structure."""
        normalized_vulns = {}
        
        for contract_name, vulns in vulnerabilities.items():
            if not vulns:
                continue
                
            normalized_contract_vulns = []
            
            for vuln in vulns:
                normalized_vuln = {
                    'name': vuln.get('name', 'Unknown Vulnerability'),
                    'severity': vuln.get('severity', 'Medium'),
                    'description': vuln.get('description', 'No description'),
                    'location': vuln.get('location', 'Unknown'),
                    'code_snippet': vuln.get('code_snippet', ''),
                    'impact': vuln.get('impact', 'Unknown'),
                    'recommendation': vuln.get('recommendation', 'Unknown')
                }
                
                normalized_contract_vulns.append(normalized_vuln)
            
            normalized_vulns[contract_name] = normalized_contract_vulns
        
        return normalized_vulns
    
    def _normalize_documentation_quality(self, doc_quality: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize documentation quality metrics."""
        normalized_quality = {}
        
        for contract_name, quality in doc_quality.items():
            if not quality:
                continue
                
            normalized_contract_quality = {
                'quality_score': quality.get('quality_score', 0) if isinstance(quality, dict) else 0,
                'issues': quality.get('issues', []) if isinstance(quality, dict) else [],
                'strengths': quality.get('strengths', []) if isinstance(quality, dict) else [],
                'improvement_suggestions': quality.get('improvement_suggestions', []) if isinstance(quality, dict) else []
            }
            
            normalized_quality[contract_name] = normalized_contract_quality
        
        return normalized_quality
    
    def _normalize_cognitive_biases(self, biases: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize cognitive biases."""
        normalized_biases = {}
        
        for contract_name, contract_biases in biases.items():
            if not contract_biases:
                continue
                
            normalized_contract_biases = []
            
            for bias in contract_biases:
                normalized_bias = {
                    'type': bias.get('type', 'Unknown Bias'),
                    'description': bias.get('description', 'No description'),
                    'impact': bias.get('impact', 'Unknown'),
                    'location': bias.get('location', 'Unknown'),
                    'code': bias.get('code', '')
                }
                
                normalized_contract_biases.append(normalized_bias)
            
            normalized_biases[contract_name] = normalized_contract_biases
        
        return normalized_biases
    
    def _normalize_dataflows(self, dataflows: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data flows."""
        normalized_dataflows = {}
        
        for contract_name, flows in dataflows.items():
            if not flows:
                continue
                
            normalized_contract_flows = []
            
            for flow in flows:
                normalized_flow = {
                    'source': flow.get('source', 'Unknown'),
                    'target': flow.get('target', 'Unknown'),
                    'type': flow.get('type', 'Unknown'),
                    'impact': flow.get('impact', 'Unknown')
                }
                
                normalized_contract_flows.append(normalized_flow)
            
            normalized_dataflows[contract_name] = normalized_contract_flows
        
        return normalized_dataflows
