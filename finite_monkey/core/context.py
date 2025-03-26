import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

class Context:
    """
    Context object for storing and sharing data between pipeline stages and tools.
    
    The Context object serves as a central data storage and state tracking mechanism,
    allowing different components of the system to share information without tight coupling.
    """
    
    def __init__(self):
        """Initialize a new Context object with default attributes."""
        # Core data storage
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._created_at = datetime.now()
        self._modified_at = self._created_at
        
        # Analysis assets
        self.contracts: Dict[str, Dict[str, Any]] = {}  # Contract name -> contract data
        self.functions: List[Dict[str, Any]] = []  # List of functions across all contracts
        self.files: List[str] = []  # List of file paths
        self.file_contents: Dict[str, str] = {}  # File path -> content
        
        # Analysis results
        self.analysis_results: Dict[str, Any] = {}  # Tool name -> results
        self.errors: List[Dict[str, Any]] = []  # List of errors
        self.executed_tools: Set[str] = set()  # Set of executed tool names
        
        # Workflow tracking
        self.processed_items: int = 0
        self.total_items: int = 0
        self.stage_results: Dict[str, Any] = {}  # Stage name -> results
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context data.
        
        Args:
            key: The key to look up
            default: Default value to return if key is not found
            
        Returns:
            The value associated with the key or the default value
        """
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the context data.
        
        Args:
            key: The key to set
            value: The value to store
        """
        self._data[key] = value
        self._modified_at = datetime.now()
    
    def add_file(self, file_path: str, content: str) -> None:
        """
        Add a file to the context.
        
        Args:
            file_path: Path to the file
            content: Content of the file
        """
        if file_path not in self.files:
            self.files.append(file_path)
        self.file_contents[file_path] = content
    
    def add_contract(self, contract_name: str, contract_data: Dict[str, Any]) -> None:
        """
        Add a contract to the context.
        
        Args:
            contract_name: Name of the contract
            contract_data: Dictionary containing contract information
        """
        self.contracts[contract_name] = contract_data
        logger.debug(f"Added contract '{contract_name}' to context")
        
        # If contract contains functions, add them to the functions list
        if "functions" in contract_data:
            for function in contract_data["functions"]:
                # Add contract name to each function for reference
                function["contract_name"] = contract_name
                self.functions.append(function)
    
    def add_functions(self, functions: List[Dict[str, Any]], contract_name: Optional[str] = None) -> None:
        """
        Add functions to the context.
        
        Args:
            functions: List of function dictionaries
            contract_name: Optional contract name to associate with the functions
        """
        if contract_name:
            for function in functions:
                function["contract_name"] = contract_name
        
        self.functions.extend(functions)
    
    def add_error(self, source: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an error to the context.
        
        Args:
            source: Source of the error (e.g., stage name, tool name)
            message: Error message
            details: Optional dictionary with additional error details
        """
        error = {
            "source": source,
            "message": message,
            "timestamp": datetime.now(),
            "details": details or {}
        }
        self.errors.append(error)
        logger.error(f"Error in {source}: {message}")
    
    def add_analysis_result(self, tool_name: str, result: Any) -> None:
        """
        Add an analysis result to the context.
        
        Args:
            tool_name: Name of the tool that produced the result
            result: The analysis result
        """
        self.analysis_results[tool_name] = result
        self.executed_tools.add(tool_name)
        self._modified_at = datetime.now()
    
    def has_contract(self, contract_name: str) -> bool:
        """
        Check if a contract exists in the context.
        
        Args:
            contract_name: Name of the contract to check
            
        Returns:
            True if the contract exists, False otherwise
        """
        return contract_name in self.contracts
    
    def get_contract(self, contract_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a contract by name.
        
        Args:
            contract_name: Name of the contract to get
            
        Returns:
            The contract data dictionary or None if not found
        """
        return self.contracts.get(contract_name)
    
    def get_contracts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all contracts.
        
        Returns:
            Dictionary of all contracts
        """
        return self.contracts
    
    def get_functions(self, contract_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get functions, optionally filtered by contract name.
        
        Args:
            contract_name: Optional contract name to filter by
            
        Returns:
            List of function dictionaries
        """
        if contract_name:
            return [f for f in self.functions if f.get("contract_name") == contract_name]
        return self.functions
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get context metadata.
        
        Returns:
            Dictionary with metadata about the context
        """
        return {
            "created_at": self._created_at,
            "modified_at": self._modified_at,
            "file_count": len(self.files),
            "contract_count": len(self.contracts),
            "function_count": len(self.functions),
            "error_count": len(self.errors),
            "executed_tools": list(self.executed_tools),
            "processed_items": self.processed_items,
            "total_items": self.total_items,
            **self._metadata
        }
    
    def clear(self) -> None:
        """Clear all data in the context."""
        self._data.clear()
        self._metadata.clear()
        self.contracts.clear()
        self.functions.clear()
        self.files.clear()
        self.file_contents.clear()
        self.analysis_results.clear()
        self.errors.clear()
        self.executed_tools.clear()
        self.stage_results.clear()
        self.processed_items = 0
        self.total_items = 0
        self._modified_at = datetime.now()
    
    def __str__(self) -> str:
        """String representation of the context."""
        return (f"Context(files={len(self.files)}, contracts={len(self.contracts)}, "
                f"functions={len(self.functions)}, errors={len(self.errors)})")
