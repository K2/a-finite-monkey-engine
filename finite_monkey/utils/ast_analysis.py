"""
AST-based analysis utilities for Solidity smart contracts

This module provides utilities for analyzing Solidity code using Abstract Syntax Trees
to extract more accurate information about function calls and relationships.
"""

import os
from typing import Dict, List, Any, Optional, Set, Tuple
import json
import subprocess

class AstAnalyzer:
    """
    AST-based analyzer for Solidity code
    
    This class uses the Solidity compiler to generate AST information
    and extract accurate call relationships between functions.
    """
    
    def __init__(self, solc_path: Optional[str] = None):
        """
        Initialize the AST analyzer
        
        Args:
            solc_path: Optional path to the Solidity compiler executable
        """
        # Find solc in PATH if not specified
        self.solc_path = solc_path or self._find_solc()
        self.ast_cache = {}
    
    def _find_solc(self) -> str:
        """Find solc executable in PATH"""
        try:
            # Try to locate solc in the system path
            result = subprocess.run(
                ["which", "solc"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
            
        # Default path attempts
        for path in ["/usr/bin/solc", "/usr/local/bin/solc"]:
            if os.path.isfile(path):
                return path
                
        return "solc"  # Fall back to letting the system try to find it
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Solidity file using AST
        
        Args:
            file_path: Path to the Solidity file
            
        Returns:
            AST data for the file
        """
        # Check if we've already analyzed this file
        if file_path in self.ast_cache:
            return self.ast_cache[file_path]
            
        try:
            # Generate AST using solc
            result = subprocess.run(
                [self.solc_path, "--ast-json", file_path],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                print(f"Warning: solc failed to generate AST for {file_path}: {result.stderr}")
                return {}
                
            # Parse the AST output
            ast_output = result.stdout
            ast_data = self._parse_ast_output(ast_output)
            
            # Cache the result
            self.ast_cache[file_path] = ast_data
            return ast_data
            
        except Exception as e:
            print(f"Error analyzing {file_path} with AST: {e}")
            return {}
    
    def _parse_ast_output(self, ast_output: str) -> Dict[str, Any]:
        """
        Parse AST output from solc
        
        Args:
            ast_output: Output from solc --ast-json
            
        Returns:
            Parsed AST data
        """
        ast_data = {}
        
        try:
            # Extract JSON AST from solc output
            ast_json_sections = ast_output.split("======= ")
            
            for section in ast_json_sections:
                if not section.strip():
                    continue
                    
                # Extract file name and JSON AST
                try:
                    file_heading, json_str = section.split(" =======\n", 1)
                    file_name = file_heading.strip()
                    
                    # Find the AST JSON section
                    if "JSON AST" in json_str:
                        _, ast_json = json_str.split("JSON AST:", 1)
                        ast_data[file_name] = json.loads(ast_json.strip())
                except ValueError:
                    continue
            
            return ast_data
            
        except Exception as e:
            print(f"Error parsing AST output: {e}")
            return {}
    
    def extract_function_calls(self, ast_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract function call relationships from AST data
        
        Args:
            ast_data: AST data from analyze_file()
            
        Returns:
            Dictionary mapping function names to lists of called functions
        """
        function_calls = {}
        
        # Process AST data to extract function calls
        # This is a placeholder for the actual implementation
        # A full implementation would traverse the AST and identify function calls
        
        print("Note: AST-based function call analysis not fully implemented")
        print("Consider using a dedicated Solidity static analysis tool for production use")
        
        return function_calls

# Create a function that explains the limitations of basic call analysis
def explain_call_analysis_limitations() -> str:
    """
    Return a clear explanation of the limitations of basic call analysis
    
    Returns:
        A string explaining the limitations
    """
    return """
Function Call Analysis Limitations:
----------------------------------
The call graph analysis in this tool has the following limitations:

1. Basic Pattern Matching: The analysis uses regular expressions to identify potential 
   function calls, which may miss complex patterns or have false positives.

2. Limited Context: The analysis doesn't understand the full Solidity type system 
   or inheritance hierarchy.

3. Lack of Contract Resolution: The analysis cannot fully resolve contract types 
   for variables, so it may miss function calls through contract variables.

4. No Dynamic Call Analysis: Function calls through delegate calls, assembly, 
   or dynamic dispatch are not detected.

For more accurate analysis, consider using dedicated Solidity static analysis 
tools like Slither, Mythril, or solc's own analysis capabilities.
"""
