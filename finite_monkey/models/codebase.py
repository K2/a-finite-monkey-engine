"""
Codebase structure models and relationship tracking for the Finite Monkey framework

This module provides classes for representing and managing the structure of a codebase,
including relationships between contracts, functions, variables, and other code entities.
"""

from typing import Dict, List, Optional, Set, Any, Union, Tuple
from pathlib import Path
import re
import os


class CodeEntity:
    """Base class for code entities"""
    
    def __init__(
        self,
        name: str,
        source_code: str,
        start_line: int,
        end_line: int,
        file_path: str,
    ):
        """
        Initialize a code entity
        
        Args:
            name: Name of the entity
            source_code: Source code of the entity
            start_line: Starting line number
            end_line: Ending line number
            file_path: Path to the file containing the entity
        """
        self.name = name
        self.source_code = source_code
        self.start_line = start_line
        self.end_line = end_line
        self.file_path = file_path
        self.parent: Optional['CodeEntity'] = None
        self.children: List['CodeEntity'] = []
        self.references: List['CodeEntity'] = []
        self.referenced_by: List['CodeEntity'] = []
        
    def add_child(self, child: 'CodeEntity') -> None:
        """
        Add a child entity
        
        Args:
            child: Child entity to add
        """
        self.children.append(child)
        child.parent = self
        
    def add_reference(self, reference: 'CodeEntity') -> None:
        """
        Add a reference to another entity
        
        Args:
            reference: Entity being referenced
        """
        if reference not in self.references:
            self.references.append(reference)
            if self not in reference.referenced_by:
                reference.referenced_by.append(self)
                
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "file_path": self.file_path,
            "parent": self.parent.name if self.parent else None,
            "children": [child.name for child in self.children],
            "references": [ref.name for ref in self.references],
            "referenced_by": [ref.name for ref in self.referenced_by],
        }
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, lines={self.start_line}-{self.end_line})"


class VariableDef(CodeEntity):
    """Variable definition"""
    
    def __init__(
        self,
        name: str,
        source_code: str,
        start_line: int,
        end_line: int,
        file_path: str,
        variable_type: str,
        visibility: str = "public",
        is_constant: bool = False,
        is_state_variable: bool = True,
        default_value: Optional[str] = None,
    ):
        """
        Initialize a variable definition
        
        Args:
            name: Name of the variable
            source_code: Source code of the variable declaration
            start_line: Starting line number
            end_line: Ending line number
            file_path: Path to the file containing the variable
            variable_type: Type of the variable
            visibility: Visibility of the variable
            is_constant: Whether the variable is constant
            is_state_variable: Whether the variable is a state variable
            default_value: Default value of the variable
        """
        super().__init__(name, source_code, start_line, end_line, file_path)
        self.variable_type = variable_type
        self.visibility = visibility
        self.is_constant = is_constant
        self.is_state_variable = is_state_variable
        self.default_value = default_value
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation
        
        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result.update({
            "variable_type": self.variable_type,
            "visibility": self.visibility,
            "is_constant": self.is_constant,
            "is_state_variable": self.is_state_variable,
            "default_value": self.default_value,
        })
        return result


class FunctionDef(CodeEntity):
    """Function definition"""
    
    def __init__(
        self,
        name: str,
        source_code: str,
        start_line: int,
        end_line: int,
        file_path: str,
        visibility: str = "public",
        is_constructor: bool = False,
        is_fallback: bool = False,
        is_receive: bool = False,
        is_modifier: bool = False,
        is_view: bool = False,
        is_pure: bool = False,
        is_payable: bool = False,
        parameters: List[Dict[str, str]] = None,
        return_type: Optional[str] = None,
        modifiers: List[str] = None,
        docstring: Optional[str] = None,
    ):
        """
        Initialize a function definition
        
        Args:
            name: Name of the function
            source_code: Source code of the function
            start_line: Starting line number
            end_line: Ending line number
            file_path: Path to the file containing the function
            visibility: Visibility of the function
            is_constructor: Whether the function is a constructor
            is_fallback: Whether the function is a fallback function
            is_receive: Whether the function is a receive function
            is_modifier: Whether the function is a modifier
            is_view: Whether the function is a view function
            is_pure: Whether the function is a pure function
            is_payable: Whether the function is payable
            parameters: List of parameter information
            return_type: Return type of the function
            modifiers: List of modifiers applied to the function
            docstring: Documentation string for the function
        """
        super().__init__(name, source_code, start_line, end_line, file_path)
        self.visibility = visibility
        self.is_constructor = is_constructor
        self.is_fallback = is_fallback
        self.is_receive = is_receive
        self.is_modifier = is_modifier
        self.is_view = is_view
        self.is_pure = is_pure
        self.is_payable = is_payable
        self.parameters = parameters or []
        self.return_type = return_type
        self.modifiers = modifiers or []
        self.docstring = docstring
        self.called_functions: List[FunctionDef] = []
        self.called_by: List[FunctionDef] = []
        self.variables_read: List[VariableDef] = []
        self.variables_written: List[VariableDef] = []
        self.business_flow: Optional[BusinessFlow] = None
        
    def add_function_call(self, function: 'FunctionDef') -> None:
        """
        Add a function call relationship
        
        Args:
            function: Function being called
        """
        if function not in self.called_functions:
            self.called_functions.append(function)
            if self not in function.called_by:
                function.called_by.append(self)
                
    def add_variable_read(self, variable: VariableDef) -> None:
        """
        Add a variable read relationship
        
        Args:
            variable: Variable being read
        """
        if variable not in self.variables_read:
            self.variables_read.append(variable)
            
    def add_variable_write(self, variable: VariableDef) -> None:
        """
        Add a variable write relationship
        
        Args:
            variable: Variable being written
        """
        if variable not in self.variables_written:
            self.variables_written.append(variable)
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation
        
        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result.update({
            "visibility": self.visibility,
            "is_constructor": self.is_constructor,
            "is_fallback": self.is_fallback,
            "is_receive": self.is_receive,
            "is_modifier": self.is_modifier,
            "is_view": self.is_view,
            "is_pure": self.is_pure,
            "is_payable": self.is_payable,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "modifiers": self.modifiers,
            "docstring": self.docstring,
            "called_functions": [func.name for func in self.called_functions],
            "called_by": [func.name for func in self.called_by],
            "variables_read": [var.name for var in self.variables_read],
            "variables_written": [var.name for var in self.variables_written],
            "has_business_flow": self.business_flow is not None,
        })
        return result


class ContractDef(CodeEntity):
    """Contract definition"""
    
    def __init__(
        self,
        name: str,
        source_code: str,
        start_line: int,
        end_line: int,
        file_path: str,
        contract_type: str = "contract",
        inheritance: List[str] = None,
        is_abstract: bool = False,
        docstring: Optional[str] = None,
    ):
        """
        Initialize a contract definition
        
        Args:
            name: Name of the contract
            source_code: Source code of the contract
            start_line: Starting line number
            end_line: Ending line number
            file_path: Path to the file containing the contract
            contract_type: Type of contract (contract, interface, library)
            inheritance: List of inherited contracts
            is_abstract: Whether the contract is abstract
            docstring: Documentation string for the contract
        """
        super().__init__(name, source_code, start_line, end_line, file_path)
        self.contract_type = contract_type
        self.inheritance = inheritance or []
        self.is_abstract = is_abstract
        self.docstring = docstring
        self.functions: Dict[str, FunctionDef] = {}
        self.variables: Dict[str, VariableDef] = {}
        self.business_flows: List[BusinessFlow] = []
        
    def add_function(self, function: FunctionDef) -> None:
        """
        Add a function to the contract
        
        Args:
            function: Function to add
        """
        self.functions[function.name] = function
        self.add_child(function)
        
    def add_variable(self, variable: VariableDef) -> None:
        """
        Add a variable to the contract
        
        Args:
            variable: Variable to add
        """
        self.variables[variable.name] = variable
        self.add_child(variable)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation
        
        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result.update({
            "contract_type": self.contract_type,
            "inheritance": self.inheritance,
            "is_abstract": self.is_abstract,
            "docstring": self.docstring,
            "functions": list(self.functions.keys()),
            "variables": list(self.variables.keys()),
            "business_flows_count": len(self.business_flows),
        })
        return result


class BusinessFlow:
    """Business flow extracted from code"""
    
    def __init__(
        self,
        name: str,
        flow_type: str,
        description: str,
        source_functions: List[FunctionDef],
        extracted_code: str,
        context: str,
        lines: List[int],
    ):
        """
        Initialize a business flow
        
        Args:
            name: Name of the business flow
            flow_type: Type of business flow
            description: Description of the business flow
            source_functions: Functions that are part of this flow
            extracted_code: Code extracted for this flow
            context: Context information for the flow
            lines: Line numbers involved in the flow
        """
        self.name = name
        self.flow_type = flow_type
        self.description = description
        self.source_functions = source_functions
        self.extracted_code = extracted_code
        self.context = context
        self.lines = lines
        self.contracts: Set[ContractDef] = set()
        self.sub_flows: List['BusinessFlow'] = []
        self.parent_flow: Optional['BusinessFlow'] = None
        
        # Link to source functions
        for function in source_functions:
            function.business_flow = self
            if function.parent and isinstance(function.parent, ContractDef):
                self.contracts.add(function.parent)
                
    def add_sub_flow(self, flow: 'BusinessFlow') -> None:
        """
        Add a sub-flow to this business flow
        
        Args:
            flow: Sub-flow to add
        """
        self.sub_flows.append(flow)
        flow.parent_flow = self
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "flow_type": self.flow_type,
            "description": self.description,
            "source_functions": [func.name for func in self.source_functions],
            "contracts": [contract.name for contract in self.contracts],
            "sub_flows": [flow.name for flow in self.sub_flows],
            "parent_flow": self.parent_flow.name if self.parent_flow else None,
            "lines_count": len(self.lines),
        }


class CodebaseContext:
    """
    Context for managing code entities, relationships, and business flows
    in a codebase
    """
    
    def __init__(self, base_path: str):
        """
        Initialize the codebase context
        
        Args:
            base_path: Base path of the codebase
        """
        self.base_path = Path(base_path)
        self.contracts: Dict[str, ContractDef] = {}
        self.functions: Dict[str, FunctionDef] = {}
        self.variables: Dict[str, VariableDef] = {}
        self.business_flows: Dict[str, BusinessFlow] = {}
        self.file_to_entities: Dict[str, List[CodeEntity]] = {}
        
    def add_contract(self, contract: ContractDef) -> None:
        """
        Add a contract to the context
        
        Args:
            contract: Contract to add
        """
        self.contracts[contract.name] = contract
        
        # Add to file mapping
        rel_path = os.path.relpath(contract.file_path, self.base_path)
        if rel_path not in self.file_to_entities:
            self.file_to_entities[rel_path] = []
        self.file_to_entities[rel_path].append(contract)
        
    def add_function(self, function: FunctionDef) -> None:
        """
        Add a function to the context
        
        Args:
            function: Function to add
        """
        self.functions[function.name] = function
        
        # Add to file mapping
        rel_path = os.path.relpath(function.file_path, self.base_path)
        if rel_path not in self.file_to_entities:
            self.file_to_entities[rel_path] = []
        self.file_to_entities[rel_path].append(function)
        
    def add_variable(self, variable: VariableDef) -> None:
        """
        Add a variable to the context
        
        Args:
            variable: Variable to add
        """
        self.variables[variable.name] = variable
        
        # Add to file mapping
        rel_path = os.path.relpath(variable.file_path, self.base_path)
        if rel_path not in self.file_to_entities:
            self.file_to_entities[rel_path] = []
        self.file_to_entities[rel_path].append(variable)
        
    def add_business_flow(self, flow: BusinessFlow) -> None:
        """
        Add a business flow to the context
        
        Args:
            flow: Business flow to add
        """
        self.business_flows[flow.name] = flow
        
        # Link to contracts
        for contract in flow.contracts:
            contract.business_flows.append(flow)
            
    def get_entities_in_file(self, file_path: str) -> List[CodeEntity]:
        """
        Get all entities in a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of entities in the file
        """
        rel_path = os.path.relpath(Path(file_path), self.base_path)
        return self.file_to_entities.get(rel_path, [])
        
    def get_contract_by_name(self, name: str) -> Optional[ContractDef]:
        """
        Get a contract by name
        
        Args:
            name: Name of the contract
            
        Returns:
            Contract or None if not found
        """
        return self.contracts.get(name)
        
    def get_function_by_name(self, name: str) -> Optional[FunctionDef]:
        """
        Get a function by name
        
        Args:
            name: Name of the function
            
        Returns:
            Function or None if not found
        """
        return self.functions.get(name)
        
    def get_variable_by_name(self, name: str) -> Optional[VariableDef]:
        """
        Get a variable by name
        
        Args:
            name: Name of the variable
            
        Returns:
            Variable or None if not found
        """
        return self.variables.get(name)
        
    def get_business_flow_by_name(self, name: str) -> Optional[BusinessFlow]:
        """
        Get a business flow by name
        
        Args:
            name: Name of the business flow
            
        Returns:
            Business flow or None if not found
        """
        return self.business_flows.get(name)
        
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Get a summary of the codebase
        
        Returns:
            Dictionary with codebase summary
        """
        return {
            "base_path": str(self.base_path),
            "contract_count": len(self.contracts),
            "function_count": len(self.functions),
            "variable_count": len(self.variables),
            "business_flow_count": len(self.business_flows),
            "file_count": len(self.file_to_entities),
        }