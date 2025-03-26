"""
Processors module for Finite Monkey Engine pipeline

This module provides processors for the pipeline:
- LlamaIndexProcessor: Process code with LlamaIndex
- LLMProcessor: Process code with LLM
- TreeSitterProcessor: Process code with Tree-sitter
- ValidationProcessor: Validate analysis results
"""

import os
import asyncio
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from pathlib import Path
from loguru import logger
import json
import time
import re

from ..nodes_config import nodes_config
from .core import Context

class LlamaIndexProcessor:
    """Process code with LlamaIndex for semantic understanding"""
    
    @staticmethod
    async def process(
        context: Context, 
        data: Optional[str] = None,
        query: Optional[str] = None,
        project_id: Optional[str] = None,
        include_chunks: bool = True,
    ) -> Context:
        """
        Process code with LlamaIndex
        
        Args:
            context: Pipeline context
            data: Optional file ID to process
            query: Query to run against the index
            project_id: Project identifier
            include_chunks: Whether to include chunks in the index
            
        Returns:
            Updated context with LlamaIndex results
        """
        try:
            # Import LlamaIndex only when needed
            try:
                from ..llama_index import AsyncIndexProcessor
            except ImportError:
                logger.error("LlamaIndex not available, skipping LlamaIndex processing")
                context.add_error(
                    stage="llama_index_processor",
                    message="LlamaIndex not available",
                )
                return context
            
            # Get project ID from context if not provided
            if not project_id:
                project_id = context.state.get("project_id", "default_project")
            
            # Create index processor
            index_processor = AsyncIndexProcessor(project_id=project_id)
            
            # Get documents to index
            if data is not None and isinstance(data, str):
                # Process a specific file
                file_data = context.files.get(data)
                if not file_data:
                    logger.warning(f"File {data} not found in context")
                    return context
                
                documents = [{
                    "id": file_data["id"],
                    "text": file_data["content"],
                    "metadata": {
                        "path": file_data["path"],
                        "name": file_data["name"],
                    }
                }]
                
                # Add chunks if requested
                if include_chunks and "chunks" in file_data:
                    for chunk in file_data["chunks"]:
                        documents.append({
                            "id": chunk["id"],
                            "text": chunk["content"],
                            "metadata": {
                                "path": file_data["path"],
                                "name": file_data["name"],
                                "chunk_type": chunk["chunk_type"],
                                "start_line": chunk.get("start_line"),
                                "end_line": chunk.get("end_line"),
                            }
                        })
            else:
                # Process all files
                documents = []
                for file_id, file_data in context.files.items():
                    if not file_data.get("is_solidity", False):
                        continue
                        
                    documents.append({
                        "id": file_data["id"],
                        "text": file_data["content"],
                        "metadata": {
                            "path": file_data["path"],
                            "name": file_data["name"],
                        }
                    })
                    
                    # Add chunks if requested
                    if include_chunks and "chunks" in file_data:
                        for chunk in file_data["chunks"]:
                            documents.append({
                                "id": chunk["id"],
                                "text": chunk["content"],
                                "metadata": {
                                    "path": file_data["path"],
                                    "name": file_data["name"],
                                    "chunk_type": chunk["chunk_type"],
                                    "start_line": chunk.get("start_line"),
                                    "end_line": chunk.get("end_line"),
                                }
                            })
            
            # Index documents
            logger.info(f"Indexing {len(documents)} documents with LlamaIndex")
            await index_processor.add_documents(documents)
            
            # Run query if provided
            results = None
            if query:
                logger.info(f"Running query: '{query}'")
                results = await index_processor.search(query)
                
                # Store results in context
                context.state["llama_index_results"] = results
                
                # Parse results to extract potential findings
                for result in results:
                    # Check for common vulnerability patterns in the response
                    if any(key in result.lower() for key in [
                        "vulnerab", "issue", "bug", "exploit", "attack", 
                        "reentrancy", "overflow", "underflow"
                    ]):
                        # Create a finding from this result
                        finding = {
                            "title": "Potential Vulnerability Identified",
                            "description": result,
                            "severity": "Medium",  # Default severity
                            "source": "LlamaIndex"
                        }
                        
                        # Try to determine severity
                        if any(term in result.lower() for term in [
                            "critical", "severe", "high risk", "major"
                        ]):
                            finding["severity"] = "High"
                        elif any(term in result.lower() for term in [
                            "moderate", "medium"
                        ]):
                            finding["severity"] = "Medium"
                        elif any(term in result.lower() for term in [
                            "low", "minor", "minimal"
                        ]):
                            finding["severity"] = "Low"
                        
                        context.add_finding(finding)
            
            return context
            
        except Exception as e:
            context.add_error(
                stage="llama_index_processor",
                message="Failed to process with LlamaIndex",
                exception=e
            )
            return context


class LLMProcessor:
    """Process code with LLM"""
    
    @staticmethod
    async def process(
        context: Context,
        data: Optional[str] = None,
        prompt_template: str = "",
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        extract_json: bool = False,
        extract_findings: bool = True,
    ) -> Context:
        """
        Process code with LLM
        
        Args:
            context: Pipeline context
            data: Optional content to include in prompt
            prompt_template: Template for the prompt
            system_prompt: System prompt for the LLM
            model: Model to use (defaults to config)
            extract_json: Whether to try to extract JSON from the response
            extract_findings: Whether to extract findings from the response
            
        Returns:
            Updated context with LLM results
        """
        try:
            # Import Ollama only when needed
            try:
                from ..adapters import Ollama
            except ImportError:
                logger.error("Ollama not available, skipping LLM processing")
                context.add_error(
                    stage="llm_processor",
                    message="Ollama not available",
                )
                return context
            
            # Get model from context or config if not provided
            config = nodes_config()
            if not model:
                model = context.config.get("model") or config.SCAN_MODEL or "llama3:8b-instruct-q6_K"
            
            # Initialize Ollama client
            ollama = Ollama(model=model)
            
            # Build prompt from template
            prompt = prompt_template
            
            # If data is a file ID, use its content
            if isinstance(data, str) and data in context.files:
                file_data = context.files[data]
                file_content = file_data.get("content", "")
                prompt = prompt.replace("{file_content}", file_content)
                
            # Replace other placeholders in the prompt
            for key, value in context.state.items():
                if isinstance(value, str):
                    prompt = prompt.replace(f"{{{key}}}", value)
            
            # Add query if available
            if "query" in context.state and "{query}" in prompt:
                prompt = prompt.replace("{query}", context.state["query"])
            
            # Process with LLM
            logger.info(f"Processing with LLM ({model})")
            start_time = time.time()
            
            response = await ollama.acomplete(
                prompt=prompt,
                system=system_prompt
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Store results in context
            context.state["llm_response"] = response
            context.state["llm_duration"] = duration
            
            # Extract JSON if requested
            if extract_json:
                try:
                    # Look for JSON blocks in the response
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
                    if json_match:
                        json_str = json_match.group(1)
                        parsed_json = json.loads(json_str)
                        context.state["llm_json"] = parsed_json
                    else:
                        # Try extracting anything that looks like JSON
                        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response)
                        if json_match:
                            json_str = json_match.group(1)
                            parsed_json = json.loads(json_str)
                            context.state["llm_json"] = parsed_json
                except Exception as e:
                    logger.warning(f"Failed to extract JSON from LLM response: {e}")
            
            # Extract findings if requested
            if extract_findings:
                findings = LLMProcessor.extract_findings_from_text(response)
                for finding in findings:
                    context.add_finding(finding)
            
            return context
            
        except Exception as e:
            context.add_error(
                stage="llm_processor",
                message="Failed to process with LLM",
                exception=e
            )
            return context
    
    @staticmethod
    def extract_findings_from_text(text: str) -> List[Dict[str, Any]]:
        """
        Extract findings from LLM output text
        
        Args:
            text: LLM output text
            
        Returns:
            List of findings
        """
        findings = []
        
        # Look for patterns like "1. Finding name: description" or "## Finding name"
        finding_patterns = [
            # Section headers
            r'[#]+\s+(.+?)(?:\n|$)',
            # Numbered findings
            r'(?:\d+\.\s+)(.+?)(?:\n|:)',
            # Bullet points
            r'(?:[-*]\s+)(.+?)(?:\n|:)',
            # Vulnerability: description
            r'(?:[Vv]ulnerability|[Ii]ssue|[Ff]inding):\s+(.+?)(?:\n|$)',
        ]
        
        # Extract potential findings using patterns
        for pattern in finding_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Check if this looks like a vulnerability finding
                finding_text = match.group(1).strip()
                if any(term in finding_text.lower() for term in [
                    "vulnerability", "issue", "bug", "finding", "attack", 
                    "exploit", "reentrancy", "overflow", "underflow", "risk"
                ]):
                    # Extract description (look for text after the match)
                    description = ""
                    start_pos = match.end()
                    next_section = re.search(r'(?:\n\s*\n|\n[#]+\s+|\n\d+\.)', text[start_pos:])
                    if next_section:
                        description = text[start_pos:start_pos + next_section.start()].strip()
                    else:
                        description = text[start_pos:].strip()
                    
                    # Determine severity
                    severity = "Medium"  # Default
                    if re.search(r'\b(?:critical|severe|high)\b', finding_text.lower()):
                        severity = "High"
                    elif re.search(r'\b(?:medium|moderate)\b', finding_text.lower()):
                        severity = "Medium"
                    elif re.search(r'\b(?:low|minor|info|informational)\b', finding_text.lower()):
                        severity = "Low"
                    
                    # Create finding
                    finding = {
                        "title": finding_text,
                        "description": description,
                        "severity": severity,
                        "source": "LLM"
                    }
                    findings.append(finding)
        
        return findings


class TreeSitterProcessor:
    """Process code with Tree-sitter for advanced analysis"""
    
    @staticmethod
    async def process(
        context: Context,
        data: Optional[str] = None,
        detect_vulnerabilities: bool = True
    ) -> Context:
        """
        Process code with Tree-sitter
        
        Args:
            context: Pipeline context
            data: Optional file ID to process (if None, process all files)
            detect_vulnerabilities: Whether to detect vulnerabilities
            
        Returns:
            Updated context with Tree-sitter results
        """
        try:
            # Import TreeSitterAnalyzer only when needed
            try:
                from ..sitter.analyzer import TreeSitterAnalyzer
                analyzer = TreeSitterAnalyzer()
            except ImportError:
                logger.error("Tree-sitter not available, skipping Tree-sitter processing")
                context.add_error(
                    stage="tree_sitter_processor",
                    message="Tree-sitter not available",
                )
                return context
            
            # Determine files to process
            if data is not None and isinstance(data, str):
                files_to_process = [data] if data in context.files else []
            else:
                files_to_process = [
                    file_id for file_id, file_data in context.files.items()
                    if file_data.get("is_solidity", False)
                ]
            
            # Process each file
            for file_id in files_to_process:
                file_data = context.files[file_id]
                content = file_data["content"]
                file_path = file_data["path"]
                
                # Analyze with Tree-sitter
                logger.info(f"Analyzing {file_path} with Tree-sitter")
                result = analyzer.analyze_code(content)
                
                # Store analysis in context
                file_data["tree_sitter_analysis"] = result
                
                # Extract findings if requested
                if detect_vulnerabilities:
                    for issue in result.get("potential_issues", []):
                        finding = {
                            "title": issue.get("title", "Unknown Issue"),
                            "description": issue.get("description", ""),
                            "severity": issue.get("severity", "Medium"),
                            "location": issue.get("location", ""),
                            "source": "TreeSitter",
                            "file_id": file_id,
                            "file_path": file_path
                        }
                        context.add_finding(finding)
            
            return context
            
        except Exception as e:
            context.add_error(
                stage="tree_sitter_processor",
                message="Failed to process with Tree-sitter",
                exception=e
            )
            return context


class ValidationProcessor:
    """Validate analysis results with another model"""
    
    @staticmethod
    async def process(
        context: Context,
        data: Optional[str] = None,
        validation_model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Context:
        """
        Validate analysis results with another model
        
        Args:
            context: Pipeline context
            data: Optional analysis results to validate (if None, use context)
            validation_model: Model to use for validation
            system_prompt: System prompt for the validation model
            
        Returns:
            Updated context with validation results
        """
        try:
            # Import Ollama only when needed
            try:
                from ..adapters import Ollama
            except ImportError:
                logger.error("Ollama not available, skipping validation")
                context.add_error(
                    stage="validation_processor",
                    message="Ollama not available",
                )
                return context
            
            # Get validation model from context or config
            config = nodes_config()
            if not validation_model:
                validation_model = (
                    context.config.get("validation_model") or
                    config.CONFIRMATION_MODEL or
                    context.config.get("model") or
                    config.SCAN_MODEL or
                    "llama3:8b-instruct-q6_K"
                )
            
            # Set up default system prompt if not provided
            if not system_prompt:
                system_prompt = (
                    "You are an expert smart contract security auditor. "
                    "Your task is to carefully validate the security analysis provided. "
                    "For each issue reported, determine if it is valid, a false positive, "
                    "or requires more context. Provide a clear explanation for each determination. "
                    "Be critical and thorough in your assessment."
                )
            
            # Get or prepare analysis to validate
            if isinstance(data, str):
                analysis_text = data
            else:
                # Compile findings from context
                findings = context.findings
                if findings:
                    analysis_text = "# Security Analysis Findings\n\n"
                    for i, finding in enumerate(findings, 1):
                        analysis_text += f"## {i}. {finding.get('title', 'Unnamed Issue')}\n"
                        analysis_text += f"**Severity:** {finding.get('severity', 'Medium')}\n\n"
                        analysis_text += f"{finding.get('description', 'No description provided.')}\n\n"
                        if finding.get('location'):
                            analysis_text += f"**Location:** {finding.get('location')}\n\n"
                else:
                    # Nothing to validate
                    logger.warning("No findings to validate")
                    return context
            
            # Prepare file content for context
            file_content = ""
            for file_id, file_data in context.files.items():
                if file_data.get("is_solidity", False):
                    file_name = file_data.get("name", "")
                    file_content += f"\n// File: {file_name}\n```solidity\n{file_data['content']}\n```\n"
            
            # Build validation prompt
            validation_prompt = (
                f"Please validate the following security analysis findings:\n\n"
                f"{analysis_text}\n\n"
                f"The code being analyzed is:\n\n{file_content}\n\n"
                f"For each finding, determine if it is Valid, a False Positive, or Needs More Context.\n"
                f"Provide your reasoning and any corrections or additional insights."
            )
            
            # Process with validation model
            logger.info(f"Validating analysis with {validation_model}")
            ollama = Ollama(model=validation_model)
            
            start_time = time.time()
            validation_response = await ollama.acomplete(
                prompt=validation_prompt,
                system=system_prompt
            )
            end_time = time.time()
            
            # Store validation results in context
            context.state["validation_response"] = validation_response
            context.state["validation_duration"] = end_time - start_time
            
            # Update findings with validation results
            ValidationProcessor.update_findings_with_validation(context, validation_response)
            
            return context
            
        except Exception as e:
            context.add_error(
                stage="validation_processor",
                message="Failed to validate analysis",
                exception=e
            )
            return context
    
    @staticmethod
    def update_findings_with_validation(context: Context, validation_text: str) -> None:
        """
        Update findings with validation results
        
        Args:
            context: Pipeline context
            validation_text: Validation response text
        """
        # Look for validation status for each finding
        for finding in context.findings:
            # Extract the title to look for
            title = finding.get('title', '')
            if not title:
                continue
                
            # Look for this finding in the validation text
            title_pattern = re.compile(
                rf'(?:^|\n)(?:\d+\.|\*|-|#{1,3})\s+{re.escape(title)}.*?'
                rf'(?:confirmation|status|assessment|determination):\s*([^\n.]+)',
                re.IGNORECASE | re.MULTILINE
            )
            
            status_match = title_pattern.search(validation_text)
            if status_match:
                status = status_match.group(1).strip().lower()
                
                # Update finding based on validation
                if "valid" in status or "confirmed" in status or "true" in status:
                    finding["validation_status"] = "Confirmed"
                elif "false" in status or "invalid" in status or "not an issue" in status:
                    finding["validation_status"] = "False Positive"
                else:
                    finding["validation_status"] = "Needs More Context"
                
                # Try to extract validation notes
                notes_pattern = re.compile(
                    rf'(?:^|\n)(?:\d+\.|\*|-|#{1,3})\s+{re.escape(title)}.*?'
                    rf'(?:confirmation|status|assessment|determination):[^\n]+\n+\s*([\s\S]+?)(?=\n\s*(?:\d+\.|\*|-|#{1,3})|$)',
                    re.IGNORECASE | re.MULTILINE
                )
                
                notes_match = notes_pattern.search(validation_text)
                if notes_match:
                    finding["validation_notes"] = notes_match.group(1).strip()