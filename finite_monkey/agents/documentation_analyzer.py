"""
Documentation Analyzer Agent for the Finite Monkey framework

This module implements a specialized agent for analyzing the relationship
between code and its documentation/comments, detecting inconsistencies
that might indicate security vulnerabilities.
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

from ..adapters import Ollama
from ..models import CodeAnalysis, InconsistencyReport


@dataclass
class CodeComment:
    """Represents a code comment and its context"""
    text: str
    line_number: int
    context_before: List[str]
    context_after: List[str]
    comment_type: str  # "inline", "block", "docstring", "natspec"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "line_number": self.line_number,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "comment_type": self.comment_type
        }


@dataclass
class DocumentationInconsistency:
    """Represents an inconsistency between documentation and code"""
    comment: CodeComment
    code_snippet: str
    inconsistency_type: str  # "functional_mismatch", "security_implication", "missing_check", etc.
    description: str
    severity: str  # "critical", "high", "medium", "low", "informational"
    confidence: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "comment": self.comment.to_dict(),
            "code_snippet": self.code_snippet,
            "inconsistency_type": self.inconsistency_type,
            "description": self.description,
            "severity": self.severity,
            "confidence": self.confidence
        }


class DocumentationAnalyzer:
    """
    Agent that analyzes documentation and comments for inconsistencies with code
    
    This agent specializes in finding mismatches between:
    1. What the code says it does (comments, documentation)
    2. What the code actually does
    
    Such inconsistencies often indicate misunderstandings by the developer
    which can lead to security vulnerabilities.
    """
    
    def __init__(
        self,
        llm_client: Optional[Ollama] = None,
        model_name: str = "qwen2.5-coder:7b",
    ):
        """
        Initialize the Documentation Analyzer agent
        
        Args:
            llm_client: LLM client for analysis
            model_name: Default model to use
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
    async def extract_comments(self, code: str) -> List[CodeComment]:
        """
        Extract all comments from code with their context
        
        Args:
            code: Source code to analyze
            
        Returns:
            List of CodeComment objects
        """
        comments = []
        lines = code.split('\n')
        
        # Different comment patterns based on language
        if code.strip().endswith('.sol') or '.sol' in code or 'contract ' in code:
            # Solidity
            inline_pattern = r'//(.+)$'
            block_start = r'/\*'
            block_end = r'\*/'
            natspec_pattern = r'///(.+)$'
            natspec_block_start = r'/\*\*'
        elif code.strip().endswith('.py') or '.py' in code or 'def ' in code or 'class ' in code:
            # Python
            inline_pattern = r'#(.+)$'
            block_start = r'"""'
            block_end = r'"""'
            natspec_pattern = None
            natspec_block_start = None
        else:
            # Default to C-style
            inline_pattern = r'//(.+)$'
            block_start = r'/\*'
            block_end = r'\*/'
            natspec_pattern = None
            natspec_block_start = None
        
        in_block_comment = False
        block_comment_start_line = 0
        block_comment_text = []
        
        # Process each line
        for i, line in enumerate(lines):
            # Check if we're in a block comment
            if in_block_comment:
                if re.search(block_end, line):
                    # End of block comment
                    in_block_comment = False
                    block_comment_text.append(line.split(block_end)[0])
                    
                    # Get context
                    context_before = lines[max(0, block_comment_start_line - 3):block_comment_start_line]
                    context_after = lines[i+1:min(len(lines), i+4)]
                    
                    comments.append(CodeComment(
                        text='\n'.join(block_comment_text),
                        line_number=block_comment_start_line,
                        context_before=context_before,
                        context_after=context_after,
                        comment_type='block' if not natspec_block_start or not line.strip().startswith(natspec_block_start) else 'natspec'
                    ))
                else:
                    block_comment_text.append(line)
                continue
            
            # Check for start of block comment
            if block_start and re.search(block_start, line) and not in_block_comment:
                in_block_comment = True
                block_comment_start_line = i
                block_comment_text = [line.split(block_start)[1]]
                if block_end in line:  # Single line block comment
                    in_block_comment = False
                    block_comment_text[0] = block_comment_text[0].split(block_end)[0]
                    
                    # Get context
                    context_before = lines[max(0, i-3):i]
                    context_after = lines[i+1:min(len(lines), i+4)]
                    
                    comments.append(CodeComment(
                        text=block_comment_text[0],
                        line_number=i,
                        context_before=context_before,
                        context_after=context_after,
                        comment_type='block'
                    ))
                continue
            
            # Check for inline comments
            inline_match = re.search(inline_pattern, line)
            if inline_match:
                comment_text = inline_match.group(1).strip()
                if comment_text:
                    # Get context
                    context_before = lines[max(0, i-3):i]
                    context_after = lines[i+1:min(len(lines), i+4)]
                    
                    comments.append(CodeComment(
                        text=comment_text,
                        line_number=i,
                        context_before=context_before,
                        context_after=context_after,
                        comment_type='inline'
                    ))
            
            # Check for NatSpec comments (Solidity)
            if natspec_pattern:
                natspec_match = re.search(natspec_pattern, line)
                if natspec_match:
                    comment_text = natspec_match.group(1).strip()
                    if comment_text:
                        # Get context
                        context_before = lines[max(0, i-3):i]
                        context_after = lines[i+1:min(len(lines), i+4)]
                        
                        comments.append(CodeComment(
                            text=comment_text,
                            line_number=i,
                            context_before=context_before,
                            context_after=context_after,
                            comment_type='natspec'
                        ))
        
        return comments
    
    async def analyze_inconsistencies(
        self, 
        code: str, 
        comments: List[CodeComment]
    ) -> List[DocumentationInconsistency]:
        """
        Analyze inconsistencies between code and comments
        
        Args:
            code: Source code
            comments: Extracted comments
            
        Returns:
            List of identified inconsistencies
        """
        inconsistencies = []
        
        # Skip if no comments to analyze
        if not comments:
            return inconsistencies
        
        # Group comments by proximity to analyze in context
        grouped_comments = self._group_comments_by_proximity(comments)
        
        # Analyze each group of comments
        for group in grouped_comments:
            # Find the relevant code section for this comment group
            min_line = min(comment.line_number for comment in group)
            max_line = max(comment.line_number for comment in group)
            
            # Extract code context (including a buffer of lines before/after)
            buffer = 10
            code_lines = code.split('\n')
            start_line = max(0, min_line - buffer)
            end_line = min(len(code_lines), max_line + buffer)
            code_section = '\n'.join(code_lines[start_line:end_line])
            
            # Format comments for analysis
            comment_texts = [
                f"Line {c.line_number + 1}: {c.text} ({c.comment_type})"
                for c in group
            ]
            
            # Create analysis prompt
            prompt = f"""
            You are an expert smart contract security auditor and code analyzer.
            
            I'm going to show you a section of code with comments. Your task is to identify any inconsistencies 
            between what the comments say the code does and what the code actually does.
            
            Pay special attention to:
            1. Security assumptions in comments that aren't enforced in code
            2. Documented behaviors that don't match implementation
            3. Security guarantees mentioned in comments that may be violated
            4. Missing input validation mentioned in comments but not implemented
            5. Incorrect descriptions of business logic
            
            Code section:
            ```
            {code_section}
            ```
            
            Comments in this section:
            {json.dumps(comment_texts, indent=2)}
            
            For each inconsistency you find, provide:
            1. The specific comment that's inconsistent
            2. The relevant code snippet
            3. Type of inconsistency (functional_mismatch, security_implication, missing_check, incorrect_description)
            4. A detailed description of the inconsistency
            5. Severity (critical, high, medium, low, informational)
            6. Confidence level (0.0-1.0)
            
            Format your answer as a JSON array where each object has the fields:
            "comment", "code_snippet", "inconsistency_type", "description", "severity", "confidence"
            
            Return just the JSON with no additional text.
            """
            
            try:
                # Get analysis from LLM
                response = await self.llm_client.acomplete(
                    prompt=prompt, 
                    model=self.model_name
                )
                
                # Parse the response
                # First, ensure we're only processing the JSON part
                json_str = self._extract_json(response)
                if not json_str:
                    continue
                
                result = json.loads(json_str)
                
                # Process each reported inconsistency
                for item in result:
                    # Find the original comment
                    comment_line_match = re.search(r'Line (\d+):', item['comment'])
                    if not comment_line_match:
                        continue
                        
                    line_num = int(comment_line_match.group(1)) - 1
                    matching_comment = None
                    for c in group:
                        if c.line_number == line_num or c.line_number == line_num - 1:
                            matching_comment = c
                            break
                    
                    if not matching_comment:
                        # If we can't find an exact match, use the first comment in the group
                        matching_comment = group[0]
                    
                    # Create inconsistency object
                    inconsistency = DocumentationInconsistency(
                        comment=matching_comment,
                        code_snippet=item['code_snippet'],
                        inconsistency_type=item['inconsistency_type'],
                        description=item['description'],
                        severity=item['severity'],
                        confidence=item['confidence']
                    )
                    
                    inconsistencies.append(inconsistency)
                
            except Exception as e:
                self.logger.error(f"Error analyzing comment group: {str(e)}")
        
        return inconsistencies
    
    def _group_comments_by_proximity(self, comments: List[CodeComment], proximity: int = 5) -> List[List[CodeComment]]:
        """
        Group comments that are close to each other
        
        Args:
            comments: List of comments to group
            proximity: Max line distance between comments in same group
            
        Returns:
            List of comment groups
        """
        if not comments:
            return []
            
        # Sort comments by line number
        sorted_comments = sorted(comments, key=lambda c: c.line_number)
        
        # Group comments
        groups = []
        current_group = [sorted_comments[0]]
        
        for i in range(1, len(sorted_comments)):
            current_comment = sorted_comments[i]
            prev_comment = sorted_comments[i-1]
            
            # If this comment is close to the previous one, add to current group
            if current_comment.line_number - prev_comment.line_number <= proximity:
                current_group.append(current_comment)
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [current_comment]
        
        # Add the last group
        if current_group:
            groups.append(current_group)
            
        return groups
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text response
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON string or empty string
        """
        # Look for JSON markers
        json_pattern = r'\[\s*\{.+\}\s*\]'
        json_match = re.search(json_pattern, text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            # Validate that it's parseable
            try:
                json.loads(json_str)
                return json_str
            except:
                pass
        
        # Try common patterns
        try:
            # Try to extract array from beginning to end
            if text.strip().startswith('[') and text.strip().endswith(']'):
                json_str = text.strip()
                json.loads(json_str)
                return json_str
                
            # Look for code blocks with json
            code_block_pattern = r'```(?:json)?\s*(\[[\s\S]+?\])\s*```'
            code_match = re.search(code_block_pattern, text, re.DOTALL)
            if code_match:
                json_str = code_match.group(1)
                json.loads(json_str)
                return json_str
        except:
            pass
            
        # If all else fails, return empty string
        return ""
    
    async def analyze_code(self, code: str) -> InconsistencyReport:
        """
        Analyze code for documentation inconsistencies
        
        Args:
            code: Source code to analyze
            
        Returns:
            Report of inconsistencies
        """
        # Extract comments
        comments = await self.extract_comments(code)
        
        # Find inconsistencies
        inconsistencies = await self.analyze_inconsistencies(code, comments)
        
        # Generate report
        report = InconsistencyReport(
            total_comments=len(comments),
            inconsistencies=[inc.to_dict() for inc in inconsistencies],
            code_language=self._detect_language(code),
            timestamp=None  # Will be set by the caller
        )
        
        return report
    
    def _detect_language(self, code: str) -> str:
        """
        Detect the programming language of the code
        
        Args:
            code: Source code
            
        Returns:
            Detected language
        """
        # Very basic detection
        if code.strip().endswith('.sol') or '.sol' in code or 'contract ' in code or 'pragma solidity' in code:
            return 'solidity'
        elif code.strip().endswith('.py') or '.py' in code or 'def ' in code or 'class ' in code or 'import ' in code:
            return 'python'
        elif 'function ' in code and '{' in code:
            return 'javascript'
        else:
            return 'unknown'
    
    async def generate_linguistic_heatmap(self, code: str, inconsistencies: List[DocumentationInconsistency]) -> Dict[str, Any]:
        """
        Generate a heatmap data structure highlighting areas of concern
        
        Args:
            code: Source code
            inconsistencies: List of detected inconsistencies
            
        Returns:
            Heatmap data structure for visualization
        """
        lines = code.split('\n')
        heatmap = {
            'code_lines': lines,
            'heat_levels': [0 for _ in lines],  # 0-10 scale
            'annotations': {}
        }
        
        # Add heat levels based on inconsistencies
        for inc in inconsistencies:
            line_num = inc.comment.line_number
            severity_score = {
                'critical': 10,
                'high': 8,
                'medium': 5,
                'low': 3,
                'informational': 1
            }.get(inc.severity.lower(), 1)
            
            # Weight by confidence
            heat_value = int(severity_score * inc.confidence)
            
            # Update heat at comment location
            if 0 <= line_num < len(heatmap['heat_levels']):
                heatmap['heat_levels'][line_num] = max(heatmap['heat_levels'][line_num], heat_value)
                
                # Add annotation
                if line_num not in heatmap['annotations']:
                    heatmap['annotations'][line_num] = []
                
                heatmap['annotations'][line_num].append({
                    'type': inc.inconsistency_type,
                    'description': inc.description,
                    'severity': inc.severity,
                    'confidence': inc.confidence
                })
                
            # Also add heat to the related code section
            code_lines = inc.code_snippet.split('\n')
            for i, line in enumerate(code_lines):
                # Find this line in the original code
                try:
                    idx = lines.index(line.rstrip())
                    if 0 <= idx < len(heatmap['heat_levels']):
                        # Lower heat for related code but still significant
                        heatmap['heat_levels'][idx] = max(heatmap['heat_levels'][idx], heat_value - 2)
                except ValueError:
                    # Line not found exactly, could do fuzzy matching but skip for now
                    pass
        
        return heatmap