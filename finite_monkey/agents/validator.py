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

from ..adapters import Ollama
from ..models import CodeAnalysis, ValidationResult
from ..models.analysis import ValidationIssue
from ..utils.prompting import get_validation_prompt


class TreeSitterAnalyzer:
    """
    Tree-sitter based static analyzer for code
    
    This class is a placeholder for the actual tree-sitter analyzer
    implementation. In a real implementation, this would use tree-sitter
    for static analysis of code.
    """
    
    def __init__(self):
        """Initialize the tree-sitter analyzer"""
        pass
    
    async def analyze_code(
        self,
        code: str,
        language: str = "solidity",
    ) -> Dict[str, Any]:
        """
        Analyze code using tree-sitter
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            Analysis results
        """
        # This is a stub implementation
        # In a real implementation, this would use tree-sitter
        
        # Simulate tree-sitter analysis with regex pattern matching
        results = {
            "patterns": {},
            "issues": [],
        }
        
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
        model_name: str = "llama3",
    ):
        """
        Initialize the validator agent
        
        Args:
            tree_sitter_analyzer: Tree-sitter analyzer
            llm_client: Ollama client for validation
            model_name: Model to use for validation
        """
        self.tree_sitter_analyzer = tree_sitter_analyzer or TreeSitterAnalyzer()
        self.llm_client = llm_client or Ollama(model=model_name)
        self.model_name = model_name
    
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