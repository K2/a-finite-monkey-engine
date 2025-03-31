"""
ThreatDetector pipeline component that uses vector search to detect potential security threats
by identifying similar code patterns or vulnerabilities in the vector database.
Integrates with TreeSitter for accurate code parsing.
"""
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from loguru import logger

from finite_monkey.core.pipeline import PipelineComponent, PipelineContext
from tools.vector_store_util import SimpleVectorStore

class ThreatDetectorVDB(PipelineComponent):
    """
    ThreatDetector leveraging vector database similarity search to identify potential security threats.
    
    This component performs the following:
    1. Uses TreeSitter to parse and extract code blocks from input source
    2. Uses vector similarity search to find similar patterns in the threat database
    3. Retrieves associated prompt templates for similar threats
    4. Generates threat assessment based on these templates
    """
    
    def __init__(
        self,
        vector_store_dir: str = None,
        collection_name: str = "threats",
        embedding_model: str = "local",
        embedding_device: str = "auto",
        similarity_threshold: float = 0.6,
        max_results: int = 20,
        threat_confidence_threshold: float = 0.7,
    ):
        """
        Initialize the ThreatDetector component with vector store settings.
        
        Args:
            vector_store_dir: Directory containing vector stores
            collection_name: Name of the threat collection to use
            embedding_model: Embedding model to use for similarity search
            embedding_device: Device to run embeddings on ("cpu", "xpu", "auto")
            similarity_threshold: Minimum similarity score to consider as a potential threat
            max_results: Maximum number of similar items to retrieve
            threat_confidence_threshold: Minimum confidence to report as a threat
        """
        super().__init__()
        
        # Store configuration
        self.vector_store_dir = vector_store_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.threat_confidence_threshold = threat_confidence_threshold
        
        # Initialize vector store
        self._vector_store = None
        self._treesitter_parser = None
        self._prompt_templates = {}
        self._initialization_complete = False
        
    async def initialize(self):
        """Initialize the vector store and load necessary resources."""
        try:
            logger.info(f"Initializing ThreatDetector with collection: {self.collection_name}")
            
            # Initialize vector store
            self._vector_store = SimpleVectorStore(
                storage_dir=self.vector_store_dir,
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
                embedding_device=self.embedding_device
            )
            
            # Initialize TreeSitter parser
            try:
                # Import the existing TreeSitter implementation
                from finite_monkey.parsers.sitter import CodeParser
                self._treesitter_parser = CodeParser()
                logger.info("TreeSitter parser initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import TreeSitter parser: {e}")
                logger.warning("TreeSitter integration disabled, falling back to basic code extraction")
                self._treesitter_parser = None
            
            # Load threat assessment prompt templates
            await self._load_prompt_templates()
            
            # Mark initialization as complete
            self._initialization_complete = True
            logger.info("ThreatDetector initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing ThreatDetector: {e}")
            self._initialization_complete = False
    
    async def _load_prompt_templates(self):
        """Load prompt templates for threat assessment."""
        try:
            templates_path = Path(__file__).parent / "templates" / "threat_prompts.json"
            if not templates_path.exists():
                logger.warning(f"Threat prompt templates not found at {templates_path}, using defaults")
                # Fallback to default templates
                self._prompt_templates = {
                    "default": "Analyze this code for security vulnerabilities. Identify specific threat patterns "
                               "related to: SQL injection, XSS, command injection, path traversal, insecure deserialization, "
                               "and authentication issues. Code to analyze:\n\n{code}"
                }
                return
            
            with open(templates_path, 'r') as f:
                self._prompt_templates = json.load(f)
            
            logger.info(f"Loaded {len(self._prompt_templates)} threat prompt templates")
        except Exception as e:
            logger.error(f"Error loading threat prompt templates: {e}")
            # Fallback to basic template
            self._prompt_templates = {
                "default": "Identify security vulnerabilities in this code: {code}"
            }
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process the input context and detect potential security threats.
        
        Args:
            context: PipelineContext containing code to analyze
            
        Returns:
            Updated PipelineContext with threat assessment
        """
        if not self._initialization_complete:
            await self.initialize()
            if not self._initialization_complete:
                logger.error("ThreatDetector initialization failed, skipping processing")
                return context
        
        try:
            # Extract code from context using TreeSitter if available
            code_blocks = await self._extract_code_blocks_from_context(context)
            if not code_blocks:
                logger.warning("No code blocks found in context, skipping threat detection")
                return context
            
            threats = []
            # Process each code block
            for code_block in code_blocks:
                # Detect threats in this code block
                block_threats = await self._detect_threats(code_block)
                if block_threats:
                    threats.extend(block_threats)
            
            # Add results to context
            context.set('threats', threats)
            
            # Log summary
            if threats:
                high_severity = sum(1 for threat in threats if threat.get('severity', '').lower() == 'high')
                medium_severity = sum(1 for threat in threats if threat.get('severity', '').lower() == 'medium')
                low_severity = sum(1 for threat in threats if threat.get('severity', '').lower() == 'low')
                
                logger.info(f"Detected {len(threats)} potential threats: "
                            f"{high_severity} high, {medium_severity} medium, {low_severity} low severity")
            else:
                logger.info("No threats detected")
            
            return context
        
        except Exception as e:
            logger.error(f"Error in ThreatDetector processing: {e}")
            context.set('error', str(e))
            return context
    
    async def _extract_code_blocks_from_context(self, context: PipelineContext) -> List[Dict[str, Any]]:
        """
        Extract code blocks from the PipelineContext for analysis using TreeSitter.
        
        Args:
            context: PipelineContext containing code in various formats
            
        Returns:
            List of code blocks as dictionaries with 'text' and 'metadata'
        """
        code_blocks = []
        
        # Use TreeSitter to extract code blocks if available
        if self._treesitter_parser:
            # Handle direct code input
            if context.has('code'):
                code = context.get('code')
                if isinstance(code, str):
                    # Let TreeSitter parse the code
                    try:
                        language = self._guess_language(code)
                        parsed_blocks = self._treesitter_parser.parse_code(code, language)
                        
                        # Convert TreeSitter output to the expected format
                        for i, block in enumerate(parsed_blocks):
                            code_blocks.append({
                                'text': block.get('code', ''),
                                'metadata': {
                                    'source': 'direct_input',
                                    'language': language,
                                    'type': block.get('type', 'unknown'),
                                    'name': block.get('name', f'block_{i}')
                                }
                            })
                    except Exception as e:
                        logger.error(f"Error parsing code with TreeSitter: {e}")
                        # Fallback: add the entire code as one block
                        code_blocks.append({
                            'text': code,
                            'metadata': {'source': 'direct_input', 'language': self._guess_language(code)}
                        })
            
            # Handle file paths
            if context.has('file_paths'):
                file_paths = context.get('file_paths')
                if isinstance(file_paths, str):
                    file_paths = [file_paths]
                
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        try:
                            # Let TreeSitter parse the file
                            language = self._get_language_from_file(file_path)
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                file_content = f.read()
                            
                            parsed_blocks = self._treesitter_parser.parse_code(file_content, language)
                            
                            # Convert TreeSitter output to the expected format
                            for i, block in enumerate(parsed_blocks):
                                code_blocks.append({
                                    'text': block.get('code', ''),
                                    'metadata': {
                                        'file_path': file_path,
                                        'language': language,
                                        'type': block.get('type', 'unknown'),
                                        'name': block.get('name', f'block_{i}')
                                    }
                                })
                        except Exception as e:
                            logger.error(f"Error parsing file with TreeSitter: {e}")
                            # Fallback: read the file directly
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                    file_content = f.read()
                                code_blocks.append({
                                    'text': file_content,
                                    'metadata': {'file_path': file_path, 'language': self._get_language_from_file(file_path)}
                                })
                            except Exception as read_err:
                                logger.error(f"Error reading file: {read_err}")
            
            # Handle AST or parse tree
            if context.has('ast'):
                ast_data = context.get('ast')
                
                # If TreeSitter parser is available, use it to handle the AST
                try:
                    code_text = self._treesitter_parser.extract_code_from_ast(ast_data)
                    if code_text:
                        code_blocks.append({
                            'text': code_text,
                            'metadata': {'source': 'ast', 'language': 'python'}
                        })
                except Exception as e:
                    logger.error(f"Error extracting code from AST with TreeSitter: {e}")
        else:
            # TreeSitter not available, use basic extraction
            # Handle direct code input
            if context.has('code'):
                code = context.get('code')
                if isinstance(code, str):
                    code_blocks.append({
                        'text': code,
                        'metadata': {'source': 'direct_input', 'language': self._guess_language(code)}
                    })
                elif isinstance(code, list):
                    for i, block in enumerate(code):
                        if isinstance(block, str):
                            code_blocks.append({
                                'text': block,
                                'metadata': {'source': 'direct_input', 'block_index': i, 
                                            'language': self._guess_language(block)}
                            })
                        elif isinstance(block, dict) and 'text' in block:
                            # Already in the right format
                            if 'metadata' not in block:
                                block['metadata'] = {'source': 'direct_input', 'block_index': i}
                            code_blocks.append(block)
            
            # Handle file paths
            if context.has('file_paths'):
                file_paths = context.get('file_paths')
                if isinstance(file_paths, str):
                    file_paths = [file_paths]
                
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                file_content = f.read()
                            code_blocks.append({
                                'text': file_content,
                                'metadata': {'file_path': file_path, 'language': self._get_language_from_file(file_path)}
                            })
                        except Exception as e:
                            logger.error(f"Error reading file: {e}")
        
        return code_blocks
    
    def _get_language_from_file(self, file_path: str) -> str:
        """Determine the language from a file path."""
        ext = os.path.splitext(file_path)[1].lower()
        
        # Map extensions to languages
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.sh': 'bash',
            '.md': 'markdown',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.sql': 'sql',
        }
        
        return ext_map.get(ext, 'unknown')
    
    def _guess_language(self, code: str) -> str:
        """
        Guess the programming language of a code snippet.
        
        Args:
            code: Code snippet
            
        Returns:
            Language identifier
        """
        # Simple language detection based on keywords and syntax
        keywords = {
            'python': ['def ', 'import ', 'class ', 'if __name__ == "__main__":', 'async def'],
            'javascript': ['function ', 'const ', 'let ', 'var ', '=>', 'export default'],
            'java': ['public class ', 'public static void main', 'import java.', '@Override'],
            'typescript': ['interface ', 'type ', 'export class ', 'implements ', '<T>'],
            'ruby': ['def ', 'require ', 'module ', 'class ', 'end'],
            'go': ['func ', 'package ', 'import (', 'type ', 'struct {'],
            'rust': ['fn ', 'impl ', 'use ', 'struct ', 'pub fn'],
            'c': ['#include <', 'int main(', 'void ', 'struct ', 'printf('],
            'cpp': ['#include <', 'namespace ', 'template<', 'std::', 'class '],
            'csharp': ['using ', 'namespace ', 'public class ', 'void ', 'static '],
            'php': ['<?php', 'function ', '$', '->', 'echo '],
        }
        
        # Count occurrences of each language's keywords
        scores = {lang: 0 for lang in keywords}
        for lang, patterns in keywords.items():
            for pattern in patterns:
                if pattern in code:
                    scores[lang] += 1
        
        # Return the language with the highest score
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        # Default to unknown if no clear match
        return 'unknown'
    
    async def _detect_threats(self, code_block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect potential security threats in a code block using vector search.
        
        Args:
            code_block: Code block with text and metadata
            
        Returns:
            List of detected threats
        """
        try:
            if not self._vector_store:
                logger.error("Vector store not initialized")
                return []
            
            code_text = code_block['text']
            language = code_block.get('metadata', {}).get('language', 'unknown')
            
            # Use adaptive search to find similar threat patterns
            results = await self._vector_store.query_with_adaptive_prompts(
                query_text=code_text,
                min_k=3,
                max_k=self.max_results,
                similarity_threshold=self.similarity_threshold,
                drop_off_factor=0.3
            )
            
            # Get top results and their prompts
            search_results = results.get('results', [])
            result_prompts = results.get('prompts', {})
            combined_prompt = result_prompts.get('combined', '')
            
            # If no results found, return empty list
            if not search_results:
                logger.debug("No similar patterns found in threat database")
                return []
            
            # Extract threats from the results
            threats = []
            
            # Get language-specific template if available
            threat_template = self._prompt_templates.get(language, self._prompt_templates.get('default'))
            
            for result in search_results:
                similarity_score = result.get('score', 0)
                if similarity_score < self.similarity_threshold:
                    continue
                
                # Get threat metadata from the result
                result_metadata = result.get('metadata', {})
                threat_type = result_metadata.get('threat_type', 'Unknown')
                severity = result_metadata.get('severity', 'medium')
                
                # Create a threat entry
                threat = {
                    'type': threat_type,
                    'severity': severity,
                    'confidence': similarity_score,
                    'matched_pattern': result.get('text', ''),
                    'code_snippet': code_text,
                    'description': result_metadata.get('description', f"Potential {threat_type} vulnerability detected")
                }
                
                # Only include threats with sufficient confidence
                if similarity_score >= self.threat_confidence_threshold:
                    threats.append(threat)
            
            # If we found threats and have prompt templates, we can use LLM to get more details
            if threats and combined_prompt:
                threats = await self._enhance_threat_details(threats, code_text, combined_prompt)
            
            return threats
            
        except Exception as e:
            logger.error(f"Error detecting threats: {e}")
            return []
    
    async def _enhance_threat_details(self, threats: List[Dict[str, Any]], 
                                     code_text: str, prompt_template: str) -> List[Dict[str, Any]]:
        """
        Enhance threat details using LLM and prompt templates.
        
        Args:
            threats: List of detected threats
            code_text: Code snippet to analyze
            prompt_template: Template for threat analysis
            
        Returns:
            Enhanced threat list
        """
        try:
            # Use the prompt generator if available
            if hasattr(self._vector_store, 'prompt_generator') and self._vector_store.prompt_generator:
                # Create a prompt using the template and code
                analysis_prompt = prompt_template.format(code=code_text)
                
                # Use the prompt generator to get enhanced threat details
                analysis_result = await self._vector_store.prompt_generator.generate_prompt({
                    'text': analysis_prompt,
                    'metadata': {
                        'purpose': 'threat_analysis',
                        'threats': [t['type'] for t in threats]
                    }
                })
                
                # Parse the analysis result and update threats
                if analysis_result:
                    # In a real implementation, we would parse the structured output
                    # Here we'll just append the analysis to each threat
                    for threat in threats:
                        threat['analysis'] = analysis_result
            
            return threats
            
        except Exception as e:
            logger.error(f"Error enhancing threat details: {e}")
            return threats
