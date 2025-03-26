"""
Contract definition models for Solidity contracts.

This module provides classes to represent Solidity contracts and their components,
built on top of box.Box for convenient attribute access.
"""

from typing import Dict, List, Any, Optional, Set, Union
from box import Box

class ContractDef(Box):
    """
    Solidity contract definition class that extends box.Box for attribute access.
    
    This class provides a convenient way to work with Solidity contract definitions,
    with properties that match the structure of parsed contract data.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a ContractDef with box.Box capabilities
        
        Args:
            *args: Positional arguments passed to box.Box
            **kwargs: Keyword arguments passed to box.Box
        """
        # Initialize the box with default Box options that preserve key case
        super().__init__(*args, frozen_box=False, default_box=True, **kwargs)
    
    @property
    def functions(self) -> Dict[str, Any]:
        """Get contract functions"""
        return self.get('functions', {})
    
    @property
    def variables(self) -> Dict[str, Any]:
        """Get contract state variables"""
        return self.get('variables', {})
    
    @property
    def modifiers(self) -> Dict[str, Any]:
        """Get contract modifiers"""
        return self.get('modifiers', {})
    
    @property
    def events(self) -> Dict[str, Any]:
        """Get contract events"""
        return self.get('events', {})
    
    @property
    def inheritance(self) -> List[str]:
        """Get contract inheritance chain"""
        return self.get('inheritance', [])
    
    @property
    def is_abstract(self) -> bool:
        """Check if contract is abstract"""
        return self.get('is_abstract', False)
    
    @property
    def contract_type(self) -> str:
        """Get contract type (contract, library, interface)"""
        return self.get('contract_type', 'contract')
    
    @property
    def source_code(self) -> str:
        """Get contract source code"""
        return self.get('source_code', '')
    
    @property
    def external_functions(self) -> List[str]:
        """Get names of external functions"""
        return [name for name, func in self.functions.items() 
                if func.get('visibility') in ('public', 'external')]
    
    @property
    def payable_functions(self) -> List[str]:
        """Get names of payable functions"""
        return [name for name, func in self.functions.items() 
                if func.get('is_payable', False)]
    
    def get_function_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a function by name"""
        return self.functions.get(name)
    
    @property
    def name(self) -> str:
        """Get contract name"""
        return self.get('name', '')
    
    @property
    def file_path(self) -> str:
        """Get file path containing this contract"""
        return self.get('file_path', '')
    
    @property
    def full_text(self) -> str:
        """Get contract full text"""
        return self.get('full_text', self.get('source_code', ''))
    
    @property
    def docstring(self) -> str:
        """Get contract docstring"""
        return self.get('docstring', '')
    
    @property
    def start_line(self) -> int:
        """Get contract start line"""
        return self.get('start_line', 0)
    
    @property
    def end_line(self) -> int:
        """Get contract end line"""
        return self.get('end_line', 0)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContractDef':
        """Create a ContractDef from a dictionary"""
        contract = cls(data)
        
        # Convert functions to FunctionDef objects
        if 'functions' in data and isinstance(data['functions'], dict):
            contract.functions = {
                name: FunctionDef(func_data)
                for name, func_data in data['functions'].items()
            }
        
        # Convert variables to VariableDef objects
        if 'variables' in data and isinstance(data['variables'], dict):
            contract.variables = {
                name: VariableDef(var_data)
                for name, var_data in data['variables'].items()
            }
        
        return contract
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary"""
        return dict(self)
    
    @property
    def properties(self) -> Dict[str, Any]:
        """Get all contract properties suitable for UI display"""
        return {
            "Name": self.name,
            "Type": self.contract_type,
            "Location": f"{self.file_path}:{self.start_line}-{self.end_line}",
            "Functions Count": len(self.functions),
            "Variables Count": len(self.variables),
            "Events Count": len(self.events),
            "Modifiers Count": len(self.modifiers),
            "Is Abstract": self.is_abstract,
            "Inheritance": ", ".join(self.inheritance) if self.inheritance else "None",
            "External API Surface": ", ".join(self.external_functions) if self.external_functions else "None",
            "Payable Functions": ", ".join(self.payable_functions) if self.payable_functions else "None"
        }
    
    @property
    def ui_tree_data(self) -> Dict[str, Any]:
        """Get hierarchical tree data for UI display"""
        return {
            "id": self.name,
            "name": self.name,
            "type": "contract",
            "children": [
                *[{"id": f"{self.name}.functions.{name}", "name": name, "type": "function"} 
                  for name in self.functions],
                *[{"id": f"{self.name}.variables.{name}", "name": name, "type": "variable"} 
                  for name in self.variables],
                *[{"id": f"{self.name}.events.{name}", "name": name, "type": "event"}
                  for name in self.events],
                *[{"id": f"{self.name}.modifiers.{name}", "name": name, "type": "modifier"}
                  for name in self.modifiers]
            ]
        }
    
    @property
    def function_relationships(self) -> Dict[str, List[str]]:
        """Get function call relationships for graph visualization"""
        relationships = {}
        
        for name, func in self.functions.items():
            if isinstance(func, FunctionDef):
                called_funcs = func.get('calls', [])
                relationships[name] = called_funcs
        
        return relationships
    
    @property
    def security_properties(self) -> Dict[str, Any]:
        """Get security-related properties for security analysis view"""
        return {
            "HasExternalCalls": any(func.get('has_external_calls', False) for func in self.functions.values()),
            "UsesAssembly": any(func.get('uses_assembly', False) for func in self.functions.values()),
            "HasPayableFunctions": bool(self.payable_functions),
            "HasSelfDestruct": any("selfdestruct" in func.get('source_code', '') for func in self.functions.values()),
            "HasDelegateCall": any("delegatecall" in func.get('source_code', '') for func in self.functions.values())
        }


class FunctionDef(Box):
    """Definition of a Solidity function"""
    
    def __init__(self, *args, **kwargs):
        """Initialize with box.Box capabilities"""
        super().__init__(*args, frozen_box=False, default_box=True, **kwargs)
    
    @property
    def name(self) -> str:
        """Get function name"""
        return self.get('name', '')
    
    @property
    def visibility(self) -> str:
        """Get function visibility (public, external, internal, private)"""
        return self.get('visibility', 'internal')
    
    @property
    def is_payable(self) -> bool:
        """Check if function is payable"""
        return self.get('is_payable', False)
    
    @property
    def is_view(self) -> bool:
        """Check if function is view"""
        return self.get('is_view', False)
    
    @property
    def is_pure(self) -> bool:
        """Check if function is pure"""
        return self.get('is_pure', False)
    
    @property
    def source_code(self) -> str:
        """Get function source code"""
        return self.get('source_code', '')
    
    @property
    def parameters(self) -> List[Dict[str, str]]:
        """Get function parameters"""
        return self.get('parameters', [])
    
    @property
    def returns(self) -> List[Dict[str, str]]:
        """Get function return values"""
        return self.get('returns', [])
    
    @property
    def modifiers(self) -> List[str]:
        """Get function modifiers"""
        return self.get('modifiers', [])
    
    @property
    def variables_read(self) -> List[Dict[str, Any]]:
        """Get variables read by this function"""
        return self.get('variables_read', [])
    
    @property
    def variables_written(self) -> List[Dict[str, Any]]:
        """Get variables written by this function"""
        return self.get('variables_written', [])
    
    @property
    def start_line(self) -> int:
        """Get function start line"""
        return self.get('start_line', 0)
    
    @property
    def end_line(self) -> int:
        """Get function end line"""
        return self.get('end_line', 0)
    
    @property
    def file_path(self) -> str:
        """Get file path containing this function"""
        return self.get('file_path', '')
    
    @property
    def full_text(self) -> str:
        """Get function full text"""
        return self.get('full_text', self.get('source_code', ''))
    
    @property
    def contract_name(self) -> str:
        """Get the name of the contract containing this function"""
        return self.get('contract_name', '')
    
    @property
    def signature(self) -> str:
        """Get function signature"""
        return self.get('signature', '')
    
    @property
    def business_flows(self) -> List[Dict[str, Any]]:
        """Get business flows for this function"""
        return self.get('business_flows', [])
    
    def add_business_flow(self, flow):
        """
        Add a business flow to this function
        
        Args:
            flow: BusinessFlow object containing flow analysis
        """
        if not hasattr(self, "business_flows"):
            self.business_flows = []
        
        # Add flow to the list
        self.business_flows.append(flow)
        
        # Return self for method chaining
        return self
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunctionDef':
        """Create a FunctionDef from a dictionary"""
        return cls(data)
    
    @property
    def properties(self) -> Dict[str, Any]:
        """Get all function properties suitable for UI display"""
        return {
            "Name": self.name,
            "Signature": self.signature,
            "Contract": self.contract_name,
            "Visibility": self.visibility,
            "Is Payable": self.is_payable,
            "Is View": self.is_view,
            "Is Pure": self.is_pure,
            "Parameters": ", ".join([f"{p.get('type', '')} {p.get('name', '')}" for p in self.parameters]),
            "Returns": ", ".join([ret for ret in self.returns]),
            "Modifiers": ", ".join(self.modifiers) if self.modifiers else "None",
            "Location": f"{self.start_line}-{self.end_line}"
        }
    
    @property
    def flows_summary(self) -> Dict[str, Any]:
        """Get business flows summary for UI display"""
        if not self.business_flows:
            return {"count": 0}
        
        flow_types = {}
        confidence = 0
        
        for flow in self.business_flows:
            flow_type = flow.get('type', 'unknown')
            flow_types[flow_type] = flow_types.get(flow_type, 0) + 1
            confidence += flow.get('confidence', 0)
        
        avg_confidence = confidence / len(self.business_flows) if self.business_flows else 0
        
        return {
            "count": len(self.business_flows),
            "types": flow_types,
            "average_confidence": avg_confidence
        }
    
    @property
    def security_risk_factors(self) -> List[str]:
        """Get security risk factors for this function"""
        risks = []
        
        # Check for common risk patterns
        if self.is_payable:
            risks.append("Accepts Ether")
        
        if "selfdestruct" in self.full_text:
            risks.append("Contains selfdestruct")
            
        if "delegatecall" in self.full_text:
            risks.append("Uses delegatecall")
        
        if "assembly" in self.full_text:
            risks.append("Contains assembly code")
            
        # Add any business flow risks
        for flow in self.business_flows:
            if flow.get('type') == 'value_transfer':
                risks.append("Transfers value")
                
        return risks


class VariableDef(Box):
    """Definition of a Solidity variable"""
    
    def __init__(self, *args, **kwargs):
        """Initialize with box.Box capabilities"""
        super().__init__(*args, frozen_box=False, default_box=True, **kwargs)
    
    @property
    def name(self) -> str:
        """Get variable name"""
        return self.get('name', '')
    
    @property
    def type(self) -> str:
        """Get variable type"""
        return self.get('type', '')
    
    @property
    def visibility(self) -> str:
        """Get variable visibility"""
        return self.get('visibility', 'internal')
    
    @property
    def is_constant(self) -> bool:
        """Check if variable is constant"""
        return self.get('is_constant', False)
    
    @property
    def is_state_variable(self) -> bool:
        """Check if this is a state variable"""
        return self.get('is_state_variable', False)
    
    @property
    def default_value(self) -> Optional[str]:
        """Get variable default value if any"""
        return self.get('default_value', None)
    
    @property
    def location(self) -> str:
        """Get variable storage location (memory, storage, calldata)"""
        return self.get('location', '')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VariableDef':
        """Create a VariableDef from a dictionary"""
        return cls(data)


class BusinessFlow(Box):
    """Representation of a business flow in a smart contract"""
    
    def __init__(self, *args, **kwargs):
        """Initialize with box.Box capabilities"""
        super().__init__(*args, frozen_box=False, default_box=True, **kwargs)
        
    @property
    def link_to(self) -> str:
        """Get link information for this flow to here"""
        return self.get('link', '')
    
    @property
    def link_from(self) -> str:
        """Get link information away"""
        return 
    
    @property
    def name(self) -> str:
        """Get flow name"""
        return self.get('name', '')
    
    @property
    def flow_type(self) -> str:
        """Get flow type"""
        return self.get('flow_type', '')
    
    @property
    def description(self) -> str:
        """Get flow description"""
        return self.get('description', '')
    
    @property
    def extracted_code(self) -> str:
        """Get extracted code for this flow"""
        return self.get('extracted_code', '')
    
    @property
    def context(self) -> str:
        """Get context information for this flow"""
        return self.get('context', '')
    
    @property
    def lines(self) -> str:
        """Get line information for this flow"""
        return self.get('lines', '')
   
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessFlow':
        """Create a BusinessFlow from a dictionary"""
        return cls(data)
    
    @classmethod
    def from_analysis(cls, analysis: Dict[str, Any]) -> List['BusinessFlow']:
        """Create BusinessFlow objects from analysis result"""
        flows = []
        for flow_data in analysis.get('businessFlows', []):
            flows.append(cls(flow_data))
        return flows
    
    @property
    def properties(self) -> Dict[str, Any]:
        """Get all flow properties suitable for UI display"""
        return {
            "Type": self.flow_type,
            "Description": self.description,
            "Confidence": f"{self.confidence:.2f}",
            "F1 Score": f"{self.f1_score:.2f}" if hasattr(self, 'f1_score') else "N/A",
            "Notes": self.notes
        }


class CodebaseContext(Box):
    """Context for an entire codebase"""
    
    def __init__(self, *args, **kwargs):
        """Initialize with box.Box capabilities"""
        super().__init__(*args, frozen_box=False, default_box=True, **kwargs)
    
    @property
    def contracts(self) -> Dict[str, ContractDef]:
        """Get all contracts in the codebase"""
        return self.get('contracts', {})
    
    @property
    def files(self) -> Dict[str, Any]:
        """Get all files in the codebase"""
        return self.get('files', {})
    
    def get_contract(self, name: str) -> Optional[ContractDef]:
        """Get a contract by name"""
        return self.contracts.get(name)
    
    @classmethod
    def from_context(cls, context: Any) -> 'CodebaseContext':
        """Create a CodebaseContext from a pipeline context"""
        codebase = cls()
        
        # Convert files to contracts
        contracts = {}
        for file_id, file_data in getattr(context, 'files', {}).items():
            if file_data.get('is_solidity', False):
                for contract_data in file_data.get('contracts', []):
                    contract = ContractDef.from_dict(contract_data)
                    contracts[contract.name] = contract
        
        codebase.contracts = contracts
        codebase.files = getattr(context, 'files', {})
        return codebase
