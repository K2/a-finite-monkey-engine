"""
Validator agent for code analysis validation

This agent cross-checks analysis results using tree-sitter static analysis
and LLM validation to verify initial findings.
"""

import os
import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from tree_sitter import Tree, Node, TreeCursor, QueryPredicate, Parser, Query, Language, Point
from tree_sitter_solidity import language
from finite_monkey.adapters import Ollama
from finite_monkey.models import CodeAnalysis, ValidationResult
from finite_monkey.models.analysis import ValidationIssue
from finite_monkey.utils.prompting import get_validation_prompt


class TreeSitterAnalyzer:
    """
    Tree-sitter based static analyzer for code
    
    This analyzer uses tree-sitter for static code analysis if available,
    otherwise falls back to regex-based analysis.
    """
    
    def __init__(self):
        """Initialize the tree-sitter analyzer"""
        self.tree_sitter_available = False
        self.solidity_language = None
        
        # Try to initialize tree-sitter
        try:
            #from tree_sitter import Language, Parser
            from tree_sitter import Tree, Node, TreeCursor, QueryPredicate, Parser, Query, Language, Point
            from tree_sitter_solidity import language
            
            # Define path to language libraries (would be configured properly in production)
            language_path = os.path.join(os.path.dirname(__file__), "../../tree_sitter_languages/libtree-sitter-solidity.so")
            
            # Check if the language file exists
            if os.path.exists(language_path):
                # Use the straightforward initialization that matches our tree-sitter version
                try:
                    # Create a specialized language module for Solidity
                    import tree_sitter
                    
                    # Create a buffer for tree-sitter
                    file_buffer = bytearray(1024*1024*2)  # 2MB buffer
                    
                    # Try multiple initialization patterns for different versions
                    try:
                        # First try: modern tree-sitter (0.20+)
                        self.solidity_language =Language(language())
                    except Exception as e1:
                        try:
                            # Second try: tree-sitter-solidity import
                            try:
                                from tree_sitter_solidity import language
                                self.solidity_language =  Language(language())
                            except ImportError:
                                # Third try: with language ID
                                self.solidity_language = Language(language_path, 'solidity')
                        except Exception as e2:
                            try:
                                # Fourth try: legacy initialization with 0
                                self.solidity_language = Language(language_path, 0)
                            except Exception as e3:
                                print(f"All tree-sitter initialization attempts failed:")
                                print(f"  First attempt: {e1}")
                                print(f"  Second attempt: {e2}")
                                print(f"  Third attempt: {e3}")
                                print("Falling back to regex analysis")
                                return
                    
                    self.parser = Parser()
                    self.tree_sitter_available = True
                    print("Tree-sitter initialized successfully for Solidity")
                except Exception as e:
                    print(f"Tree-sitter initialization failed: {e}")
                    print("Falling back to regex analysis")
                    return
            else:
                print(f"Solidity language file not found at {language_path}, falling back to regex analysis")
        except ImportError:
            print("Tree-sitter not available, falling back to regex analysis")
        except Exception as e:
            print(f"Error initializing tree-sitter: {e}, falling back to regex analysis")
    
    async def analyze_code(
        self,
        code: str,
        language: str = "solidity",
    ) -> Dict[str, Any]:
        """
        Analyze code using tree-sitter or regex fallback
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            Analysis results
        """
        results = {
            "patterns": {},
            "issues": [],
            "using_tree_sitter": self.tree_sitter_available
        }
        
        # Use tree-sitter if available
        if self.tree_sitter_available and language.lower() == "solidity":
            try:
                # Parse the code
                tree = self.parser.parse(bytes(code, "utf8"))
                
                # Query for patterns
                # This would be more sophisticated in a production environment
                # with proper query patterns for different vulnerability types
                
                # Example patterns (simplified for demonstration)
                patterns = {
                    "unchecked_return": "(call_expression function: (member_expression property: (identifier) @method) @call) @call_expr",
                    "reentrancy": "(member_expression property: (identifier) @prop (#match? @prop \"call\")) @call",
                    "tx_origin": "(member_expression object: (identifier) @obj (#eq? @obj \"tx\") property: (identifier) @prop (#eq? @prop \"origin\")) @tx_origin",
                    "timestamp_dependency": "(member_expression object: (identifier) @obj (#eq? @obj \"block\") property: (identifier) @prop (#eq? @prop \"timestamp\")) @timestamp",
                }
                
                # Run each query and collect results
                for name, query_str in patterns.items():
                    try:
                        query = self.solidity_language.query(query_str)
                        matches = query.captures(tree.root_node)
                        results["patterns"][name] = len(matches)
                        
                        # Add issues for matches (simplified for demonstration)
                        # In production, you'd analyze the context more thoroughly
                        self._add_issues_for_pattern(results, name, matches, code)
                    except Exception as e:
                        print(f"Error running tree-sitter query '{name}': {e}")
                
                return results
            
            except Exception as e:
                print(f"Tree-sitter analysis failed: {e}, falling back to regex")
                # Fall back to regex if tree-sitter fails
                pass
        
        # Fallback: Use regex pattern matching
        
        # Define some simple patterns to look for
        patterns = {
            "unchecked_return": r"\.transfer\(|\.send\(",
            "reentrancy": r"\.call\{value:",
            "tx_origin": r"tx\.origin",
            "timestamp_dependency": r"block\.timestamp",
        }
        
        # Check for patterns
        for name, pattern in patterns.items():
            matches = re.findall(pattern, code)
            results["patterns"][name] = len(matches)
            
            if matches:
                # Add as potential issue
                if name == "unchecked_return":
                    results["issues"].append({
                        "title": "Unchecked Return Value",
                        "description": "Return values from external calls are not checked",
                        "severity": "Medium",
                        "pattern": pattern,
                    })
                elif name == "reentrancy":
                    results["issues"].append({
                        "title": "Potential Reentrancy",
                        "description": "Low-level call with value transfer detected",
                        "severity": "High",
                        "pattern": pattern,
                    })
                elif name == "tx_origin":
                    results["issues"].append({
                        "title": "tx.origin Usage",
                        "description": "tx.origin used for authorization",
                        "severity": "Medium",
                        "pattern": pattern,
                    })
                elif name == "timestamp_dependency":
                    results["issues"].append({
                        "title": "Timestamp Dependency",
                        "description": "Contract relies on block.timestamp",
                        "severity": "Low",
                        "pattern": pattern,
                    })
        
        # Check for other security patterns
        if "suicide" in code or "selfdestruct" in code:
            results["issues"].append({
                "title": "Selfdestruct Usage",
                "description": "Contract can be destroyed",
                "severity": "Medium",
                "pattern": "selfdestruct|suicide",
            })
        
        # Return analysis results
        return results
        
    def _add_issues_for_pattern(self, results, pattern_name, matches, code):
        """
        Add issues based on tree-sitter matches
        
        Args:
            results: Results dictionary to update
            pattern_name: Name of the matched pattern
            matches: Tree-sitter matches
            code: Source code
            
        Returns:
            None (updates results in place)
        """
        if not matches:
            return
            
        # Create appropriate issues based on the pattern
        if pattern_name == "unchecked_return":
            results["issues"].append({
                "title": "Unchecked Return Value",
                "description": "Return values from external calls are not checked",
                "severity": "Medium",
                "locations": [self._get_node_location(match[0], code) for match in matches],
            })
        elif pattern_name == "reentrancy":
            results["issues"].append({
                "title": "Potential Reentrancy",
                "description": "Low-level call with value transfer detected",
                "severity": "High",
                "locations": [self._get_node_location(match[0], code) for match in matches],
            })
        elif pattern_name == "tx_origin":
            results["issues"].append({
                "title": "tx.origin Usage",
                "description": "tx.origin used for authorization",
                "severity": "Medium",
                "locations": [self._get_node_location(match[0], code) for match in matches],
            })
        elif pattern_name == "timestamp_dependency":
            results["issues"].append({
                "title": "Timestamp Dependency",
                "description": "Contract relies on block.timestamp",
                "severity": "Low",
                "locations": [self._get_node_location(match[0], code) for match in matches],
            })
    
    def _get_node_location(self, node, code):
        """
        Get location information for a tree-sitter node
        
        Args:
            node: Tree-sitter node
            code: Source code
            
        Returns:
            Location information dictionary
        """
        start_point = node.start_point
        end_point = node.end_point
        
        # Get line numbers (1-based)
        start_line = start_point[0] + 1
        end_line = end_point[0] + 1
        
        # Get line content
        lines = code.splitlines()
        if 0 <= start_point[0] < len(lines):
            line_content = lines[start_point[0]]
        else:
            line_content = ""
        
        return {
            "start_line": start_line,
            "end_line": end_line,
            "start_col": start_point[1],
            "end_col": end_point[1],
            "line_content": line_content
        }


class Validator:
    """
    Validator agent that cross-checks code analysis results
    
    This agent is responsible for:
    1. Validating analysis results using tree-sitter static analysis
    2. Cross-checking with LLM to assess the validity of findings
    3. Producing a final validation report
    """
    
    def __init__(
        self,
        tree_sitter_analyzer: Optional[TreeSitterAnalyzer] = None,
        llm_client: Optional[Ollama] = None,
        model_name: str = None,
    ):
        """
        Initialize the validator agent
        
        Args:
            tree_sitter_analyzer: Tree-sitter analyzer
            llm_client: Ollama client for validation
            model_name: Model to use for validation
        """
        # Get model name from config if not provided
        from ..nodes_config import nodes_config
        config = nodes_config()
        self.model_name = model_name or config.WORKFLOW_MODEL or "qwen2.5-coder:latest"
        
        self.tree_sitter_analyzer = tree_sitter_analyzer or TreeSitterAnalyzer()
        self.llm_client = llm_client or Ollama(model=self.model_name)
    
    async def validate_with_static_analysis_async(
        self,
        code_path: str,
    ) -> Dict[str, Any]:
        """
        Validate code using static analysis
        
        Args:
            code_path: Path to the code file
            
        Returns:
            Static analysis results
        """
        # Read the file
        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        # Get file extension to determine language
        _, ext = os.path.splitext(code_path)
        language = "solidity" if ext == ".sol" else "javascript"
        
        # Run tree-sitter analysis
        return await self.tree_sitter_analyzer.analyze_code(
            code=code,
            language=language,
        )
    
    async def validate_with_llm_async(
        self,
        code: str,
        analysis: CodeAnalysis,
        llm: Optional[Ollama] = None,
    ) -> Dict[str, Any]:
        """
        Validate analysis with LLM
        
        Args:
            code: Source code
            analysis: Code analysis to validate
            llm: Optional Ollama client
            
        Returns:
            LLM validation results
        """
        # Use provided LLM or default
        llm_client = llm or self.llm_client
        
        # Convert analysis to dict
        analysis_dict = analysis.model_dump()
        
        # Get issues to validate
        issues = analysis.findings
        
        # Build prompt for validation
        validation_prompt = get_validation_prompt(
            code=code,
            analysis=analysis_dict,
            issues=issues,
        )
        
        # Get LLM validation
        validation_text = await llm_client.acomplete(
            prompt=validation_prompt,
            model=self.model_name,
        )
        
        # Parse the validation results
        return self._parse_validation(validation_text, issues)
    
    def _parse_validation(
        self,
        validation_text: str,
        original_issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parse validation text into structured format
        
        Args:
            validation_text: Validation text from LLM
            original_issues: Original issues to validate
            
        Returns:
            Parsed validation results
        """
        # Extract summary (first paragraph)
        lines = validation_text.split("\n")
        summary_lines = []
        for line in lines:
            if line.strip():
                summary_lines.append(line.strip())
            elif summary_lines:
                break
        
        summary = " ".join(summary_lines) if summary_lines else "No validation summary available."
        
        # Parse validation for each issue
        validated_issues = []
        current_issue = None
        
        for line in lines:
            line = line.strip()
            
            # Check for new issue
            issue_match = re.match(r"^(\d+)\.\s+(.+)$", line)
            if issue_match:
                # Save previous issue
                if current_issue:
                    validated_issues.append(current_issue)
                
                # Start new issue
                issue_num = int(issue_match.group(1)) - 1
                issue_title = issue_match.group(2)
                
                # Find corresponding original issue
                original = {} if issue_num >= len(original_issues) else original_issues[issue_num]
                
                current_issue = {
                    "title": original.get("title", issue_title),
                    "description": "",
                    "severity": original.get("severity", "Medium"),
                    "confirmation_status": "Not Determined",
                    "confidence": 0.5,
                }
            
            # Check for confirmation status
            elif current_issue and "confirmation status:" in line.lower():
                status_parts = line.split(":", 1)
                if len(status_parts) > 1:
                    status = status_parts[1].strip().lower()
                    if "confirm" in status:
                        current_issue["confirmation_status"] = "Confirmed"
                        current_issue["confidence"] = 0.9
                    elif "false" in status:
                        current_issue["confirmation_status"] = "False Positive"
                        current_issue["confidence"] = 0.8
                    else:
                        current_issue["confirmation_status"] = "Needs More Context"
                        current_issue["confidence"] = 0.4
            
            # Check for severity assessment
            elif current_issue and "severity:" in line.lower():
                severity_parts = line.split(":", 1)
                if len(severity_parts) > 1:
                    severity = severity_parts[1].strip()
                    if severity in ["Critical", "High", "Medium", "Low", "Informational"]:
                        current_issue["severity"] = severity
            
            # Add to description
            elif current_issue:
                if current_issue["description"]:
                    current_issue["description"] += " " + line
                else:
                    current_issue["description"] = line
        
        # Add last issue
        if current_issue:
            validated_issues.append(current_issue)
        
        # Convert to ValidationIssue objects
        issues = []
        has_critical = False
        
        for issue in validated_issues:
            if issue["confirmation_status"] == "Confirmed":
                severity = issue["severity"]
                if severity == "Critical":
                    has_critical = True
                
                issues.append(ValidationIssue(
                    title=issue["title"],
                    description=issue["description"],
                    severity=severity,
                    confidence=issue["confidence"],
                ))
        
        # Create validation result
        validation_result = {
            "summary": summary,
            "issues": [issue.model_dump() for issue in issues],
            "has_critical_issues": has_critical,
            "validation_methods": ["LLM Validation"],
            "raw_validation": validation_text,
        }
        
        return validation_result
    
    async def validate_analysis(
        self,
        code: str,
        analysis: CodeAnalysis,
    ) -> ValidationResult:
        """
        Validate a code analysis using multiple methods
        
        Args:
            code: Source code
            analysis: Code analysis to validate
            
        Returns:
            Validation result
        """
        # Run both validation methods in parallel
        static_task = asyncio.create_task(self.tree_sitter_analyzer.analyze_code(code))
        llm_task = asyncio.create_task(self.validate_with_llm_async(code, analysis))
        
        # Wait for both tasks to complete
        static_results, llm_results = await asyncio.gather(static_task, llm_task)
        
        # Merge validation results
        summary = llm_results.get("summary", "")
        
        # Combine issues from both sources
        all_issues = []
        
        # Add LLM issues
        for issue in llm_results.get("issues", []):
            issue_obj = ValidationIssue(**issue)
            all_issues.append(issue_obj)
        
        # Add static analysis issues
        for issue in static_results.get("issues", []):
            # Check if already covered by LLM
            title = issue.get("title", "")
            if not any(i.title == title for i in all_issues):
                all_issues.append(ValidationIssue(
                    title=title,
                    description=issue.get("description", ""),
                    severity=issue.get("severity", "Medium"),
                    confidence=0.7,  # Lower confidence for static analysis
                ))
        
        # Check for critical issues
        has_critical = any(issue.severity == "Critical" for issue in all_issues)
        
        # Create validation methods list
        validation_methods = ["Tree-sitter Analysis", "LLM Validation"]
        
        # Create validation result
        validation_result = ValidationResult(
            source_code=code,
            summary=summary,
            issues=all_issues,
            has_critical_issues=has_critical,
            validation_methods=validation_methods,
            metadata={
                "static_analysis": static_results,
                "llm_validation": llm_results,
            },
        )
        
        return validation_result