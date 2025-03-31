"""
Prompt generator for vector store documents to enhance retrieval, analysis, and LLM interactions.
This module generates structured prompts for different document types and applications.
"""
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from loguru import logger
import httpx

class PromptGenerator:
    """
    Generates prompts for vector store documents to enhance retrieval and analysis.
    
    This class creates specialized prompts for different document types and applications,
    supporting both single model and multi-LLM approaches.
    """
    
    def __init__(
        self,
        generate_prompts: bool = True,
        use_ollama_for_prompts: bool = True,
        prompt_model: str = "gemma:2b",
        ollama_url: str = "http://localhost:11434",
        multi_llm_prompts: bool = False,
        ollama_timeout: float = 900.0,
        timeout: float = None,
        temperature: float = 0.2,  # Add configurability for temperature
        max_tokens: int = 4096,    # Allow longer responses
        provider_type: str = "ollama"  # Support for different providers
    ):
        """
        Initialize the PromptGenerator.
        
        Args:
            generate_prompts: Whether to generate prompts
            use_ollama_for_prompts: Whether to use Ollama for prompt generation
            prompt_model: Model to use for prompt generation
            ollama_url: URL for Ollama API
            multi_llm_prompts: Whether to generate prompts for multiple LLMs
            ollama_timeout: Timeout in seconds for Ollama API requests
            timeout: General timeout for all operations (defaults to ollama_timeout if not set)
            temperature: Temperature setting for the LLM (0.0-1.0)
            max_tokens: Maximum number of tokens in generated responses
            provider_type: Type of LLM provider ("ollama", "openai", etc.)
        """
        self.generate_prompts = generate_prompts
        self.use_ollama_for_prompts = use_ollama_for_prompts
        self.prompt_model = prompt_model
        self.ollama_url = ollama_url
        self.multi_llm_prompts = multi_llm_prompts
        self.ollama_timeout = ollama_timeout
        self.timeout = timeout if timeout is not None else ollama_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_type = provider_type
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load prompt templates for different document types."""
        try:
            # Define directory for templates
            templates_dir = os.path.join(os.path.dirname(__file__), "templates")
            os.makedirs(templates_dir, exist_ok=True)
            
            # Default templates
            self.general_template = """
            Based on the following document, generate a detailed prompt that can be used
            to guide an LLM to understand and respond about this content:
            
            {text}
            
            Focus on the key concepts, patterns, and insights from this document.
            The prompt should help an LLM develop a thorough understanding of this content.
            """
            
            self.code_template = """
            Analyze this code and generate a comprehensive prompt for an LLM to understand it:
            
            ```
            {text}
            ```
            
            Your prompt should:
            1. Identify the language, libraries, and frameworks used
            2. Explain the core functionality and purpose
            3. Highlight key algorithmic patterns and design choices
            4. Note any potential issues, optimizations, or security concerns
            """
            
            self.security_template = """
            Analyze this security-related content and create a detailed prompt to guide an LLM:
            
            {text}
            
            Your prompt should:
            1. Identify the type of security issue or vulnerability
            2. Explain the technical details of how it works
            3. Describe potential impact and attack vectors
            4. Note detection and mitigation strategies
            """
            
            # Load custom templates if available
            general_path = os.path.join(templates_dir, "general_prompt_template.txt")
            if os.path.exists(general_path):
                with open(general_path, 'r') as f:
                    self.general_template = f.read()
            
            code_path = os.path.join(templates_dir, "code_prompt_template.txt")
            if os.path.exists(code_path):
                with open(code_path, 'r') as f:
                    self.code_template = f.read()
            
            security_path = os.path.join(templates_dir, "security_prompt_template.txt")
            if os.path.exists(security_path):
                with open(security_path, 'r') as f:
                    self.security_template = f.read()
            
            # Multi-LLM templates for different models
            self.multi_llm_templates = {
                "openai": "Create a prompt for GPT models to fully understand: {text}",
                "gemma": "Generate a Gemma-optimized prompt to comprehend: {text}",
                "llama": "Design a Llama-specific prompt for understanding: {text}",
                "mistral": "Craft a Mistral-targeted prompt about: {text}"
            }
            
            # Load multi-LLM templates if available
            multi_llm_path = os.path.join(templates_dir, "multi_llm_templates.json")
            if os.path.exists(multi_llm_path):
                with open(multi_llm_path, 'r') as f:
                    self.multi_llm_templates = json.load(f)
            
            # Save default templates for future reference if they don't exist
            if not os.path.exists(general_path):
                try:
                    with open(general_path, 'w') as f:
                        f.write(self.general_template)
                except Exception as e:
                    logger.warning(f"Couldn't save general template: {e}")
            
            if not os.path.exists(code_path):
                try:
                    with open(code_path, 'w') as f:
                        f.write(self.code_template)
                except Exception as e:
                    logger.warning(f"Couldn't save code template: {e}")
            
            if not os.path.exists(security_path):
                try:
                    with open(security_path, 'w') as f:
                        f.write(self.security_template)
                except Exception as e:
                    logger.warning(f"Couldn't save security template: {e}")
            
            if not os.path.exists(multi_llm_path):
                try:
                    with open(multi_llm_path, 'w') as f:
                        json.dump(self.multi_llm_templates, f, indent=2)
                except Exception as e:
                    logger.warning(f"Couldn't save multi-LLM templates: {e}")
            
            logger.info("Loaded prompt generation templates")
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")
    
    async def generate_prompt(self, document: Dict[str, Any]) -> str:
        """
        Generate a prompt for a document.
        
        Args:
            document: Document to generate prompt for
            
        Returns:
            Generated prompt or empty string on failure
        """
        if not self.generate_prompts:
            return ""
        
        try:
            # Select appropriate template based on document metadata
            template = self._select_template(document)
            
            # Fill in template with document text
            text = document.get('text', '')
            template_values = {'text': text}
            prompt_template = template.format(**template_values)
            
            # Generate prompt using LLM
            if self.use_ollama_for_prompts:
                prompt = await self._query_ollama(prompt_template)
            else:
                prompt = self._generate_fallback(document)
            
            return prompt
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return ""
    
    async def generate_multi_llm_prompts(self, document: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate prompts for multiple LLMs.
        
        Args:
            document: Document to generate prompts for
            
        Returns:
            Dictionary mapping LLM names to prompts
        """
        if not self.generate_prompts or not self.multi_llm_prompts:
            return {}
        
        try:
            results = {}
            text = document.get('text', '')
            
            # Generate prompts for each LLM type
            for llm_name, template in self.multi_llm_templates.items():
                filled_template = template.format(text=text)
                
                if self.use_ollama_for_prompts:
                    prompt = await self._query_ollama(filled_template)
                else:
                    # Simple fallback for testing
                    prompt = f"Prompt for {llm_name}: Analyze the following content - {text[:100]}..."
                
                if prompt:
                    results[llm_name] = prompt
            
            return results
        except Exception as e:
            logger.error(f"Error generating multi-LLM prompts: {e}")
            return {}
    
    async def generate_invariant_analysis(self, document: Dict[str, Any], base_prompt: str = None) -> str:
        """
        Generate an invariant analysis for security vulnerabilities.
        
        Args:
            document: Document to analyze
            base_prompt: Optional existing prompt to build upon
            
        Returns:
            Invariant analysis prompt
        """
        if not self.generate_prompts:
            return ""
        
        try:
            # Check if this is a security-related document
            metadata = document.get('metadata', {})
            is_security = (
                'security' in metadata or 
                'vulnerability' in metadata or 
                'security_flaw' in metadata or
                'bug' in metadata
            )
            
            if not is_security:
                return ""
            
            # Create invariant analysis prompt
            text = document.get('text', '')
            invariant_template = """
            Perform an invariant analysis on the following security vulnerability:
            
            CODE OR DESCRIPTION:
            {text}
            
            EXISTING ANALYSIS:
            {base_prompt}
            
            Identify:
            1. What invariant properties are violated in this vulnerability?
            2. What conditions must always hold true to prevent this issue?
            3. What are the underlying security principles involved?
            4. What formal verification checks would detect this issue?
            """
            
            filled_template = invariant_template.format(
                text=text,
                base_prompt=base_prompt or "None provided"
            )
            
            # Generate invariant analysis
            if self.use_ollama_for_prompts:
                analysis = await self._query_ollama(filled_template)
            else:
                analysis = self._generate_fallback(document, "invariant")
            
            return analysis
        except Exception as e:
            logger.error(f"Error generating invariant analysis: {e}")
            return ""
    
    async def generate_general_pattern(self, document: Dict[str, Any], base_prompt: str = None) -> str:
        """
        Generate a general vulnerability pattern description.
        
        Args:
            document: Document to analyze
            base_prompt: Optional existing prompt to build upon
            
        Returns:
            General pattern description
        """
        if not self.generate_prompts:
            return ""
        
        try:
            # Check if this is a security-related document
            metadata = document.get('metadata', {})
            is_security = (
                'security' in metadata or 
                'vulnerability' in metadata or 
                'security_flaw' in metadata
            )
            
            if not is_security:
                return ""
            
            # Create general pattern prompt
            text = document.get('text', '')
            pattern_template = """
            Generate a generalized pattern for the following security vulnerability:
            
            CODE OR DESCRIPTION:
            {text}
            
            EXISTING ANALYSIS:
            {base_prompt}
            
            Provide:
            1. The general vulnerability class this belongs to
            2. The abstract pattern that makes this a vulnerability
            3. Common variations of this vulnerability pattern
            4. A general detection rule for this class of vulnerabilities
            """
            
            filled_template = pattern_template.format(
                text=text,
                base_prompt=base_prompt or "None provided"
            )
            
            # Generate general pattern
            if self.use_ollama_for_prompts:
                pattern = await self._query_ollama(filled_template)
            else:
                pattern = self._generate_fallback(document, "pattern")
            
            return pattern
        except Exception as e:
            logger.error(f"Error generating general pattern: {e}")
            return ""
    
    async def extract_quick_checks(self, document: Dict[str, Any], base_prompt: str = None) -> List[str]:
        """
        Extract quick check patterns for a security vulnerability.
        
        Args:
            document: Document to analyze
            base_prompt: Optional existing prompt to build upon
            
        Returns:
            List of quick check patterns
        """
        if not self.generate_prompts:
            return []
        
        try:
            # Check if this is a security-related document
            metadata = document.get('metadata', {})
            is_security = (
                'security' in metadata or 
                'vulnerability' in metadata or 
                'security_flaw' in metadata
            )
            
            if not is_security:
                return []
            
            # Create quick checks prompt
            text = document.get('text', '')
            checks_template = """
            Generate a list of quick checks for detecting this security vulnerability:
            
            CODE OR DESCRIPTION:
            {text}
            
            EXISTING ANALYSIS:
            {base_prompt}
            
            List simple patterns that could identify similar vulnerabilities:
            1. String patterns to search for in code
            2. AST patterns that indicate this vulnerability
            3. Function calls or API usages that are often misused
            4. Order of operation patterns that lead to this vulnerability
            
            Format as a JSON array of string patterns.
            """
            
            filled_template = checks_template.format(
                text=text,
                base_prompt=base_prompt or "None provided"
            )
            
            # Generate quick checks
            if self.use_ollama_for_prompts:
                response = await self._query_ollama(filled_template)
                
                # Try to parse as JSON
                try:
                    checks = json.loads(response)
                    if isinstance(checks, list):
                        return checks
                except json.JSONDecodeError:
                    # Extract lines that look like patterns
                    checks = []
                    for line in response.split('\n'):
                        line = line.strip()
                        if line.startswith('- ') or line.startswith('* '):
                            checks.append(line[2:])
                        elif line.startswith('"') and line.endswith('"'):
                            checks.append(line.strip('"'))
                    return checks
            else:
                return self._generate_fallback_checks(document)
            
            return []
        except Exception as e:
            logger.error(f"Error extracting quick checks: {e}")
            return []
    
    async def extract_api_interactions(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract API interaction patterns from code.
        
        Args:
            document: Document to analyze
            
        Returns:
            Dictionary of API names to interaction patterns
        """
        if not self.generate_prompts:
            return {}
        
        try:
            # Create API interactions prompt
            text = document.get('text', '')
            api_template = """
            Analyze this code for API interactions and potential security issues:
            
            {text}
            
            Identify:
            1. Which APIs or libraries are being used
            2. Common misuses or vulnerabilities associated with these APIs
            3. Security best practices for these APIs
            
            Format your response as a JSON object with API names as keys and analysis as values.
            """
            
            filled_template = api_template.format(text=text)
            
            # Generate API interactions
            if self.use_ollama_for_prompts:
                response = await self._query_ollama(filled_template)
                
                # Try to parse as JSON
                try:
                    interactions = json.loads(response)
                    if isinstance(interactions, dict):
                        return interactions
                except json.JSONDecodeError:
                    # Extract API names and descriptions
                    interactions = {}
                    current_api = None
                    current_desc = []
                    
                    for line in response.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check if this is an API heading
                        if line.isupper() or (len(line) < 50 and (': ' not in line or line.index(': ') > 10)):
                            # Save previous API if any
                            if current_api and current_desc:
                                interactions[current_api] = '\n'.join(current_desc)
                            
                            current_api = line
                            current_desc = []
                        elif current_api:
                            current_desc.append(line)
                    
                    # Save the last API
                    if current_api and current_desc:
                        interactions[current_api] = '\n'.join(current_desc)
                    
                    return interactions
            else:
                return self._generate_fallback_api_interactions(document)
            
            return {}
        except Exception as e:
            logger.error(f"Error extracting API interactions: {e}")
            return {}
    
    async def extract_call_flow(self, document: Dict[str, Any]) -> List[List[str]]:
        """
        Extract call flow from code or vulnerability description.
        
        Args:
            document: Document containing code or vulnerability description
            
        Returns:
            List of call flows, where each flow is a list of function calls
        """
        if not self.generate_prompts:
            return []
        
        try:
            # Create call flow prompt
            text = document.get('text', '')
            flow_template = """
            Extract the execution flow from this code or vulnerability description:
            
            {text}
            
            Identify the sequence of function calls or execution steps.
            Format your response as a list of execution paths, where each path is a 
            sequence of function calls separated by arrows (->).
            If multiple paths are possible, list each path on a separate line.
            """
            
            filled_template = flow_template.format(text=text)
            
            # Generate call flow
            if self.use_ollama_for_prompts:
                response = await self._query_ollama(filled_template)
                return self._parse_call_flow_response(response)
            else:
                return self._generate_fallback_call_flow(document)
            
            return []
        except Exception as e:
            logger.error(f"Error extracting call flow: {e}")
            return []
    
    async def identify_vulnerable_paths(self, document: Dict[str, Any], call_flow: List[List[str]]) -> List[List[str]]:
        """
        Identify vulnerable execution paths within a call flow.
        
        Args:
            document: Document containing code or vulnerability description
            call_flow: Call flow extracted from the document
            
        Returns:
            List of vulnerable paths, where each path is a list of function calls
        """
        if not self.generate_prompts or not call_flow:
            return []
        
        try:
            # Convert call flow to string representation
            flow_str = ""
            for i, path in enumerate(call_flow):
                flow_str += f"Path {i+1}: {' -> '.join(path)}\n"
            
            # Create vulnerable paths prompt
            text = document.get('text', '')
            paths_template = """
            Identify vulnerable execution paths in this call flow:
            
            CALL FLOW:
            {flow}
            
            CODE OR DESCRIPTION:
            {text}
            
            Which specific sequences of function calls could lead to security vulnerabilities?
            Format your response as a list of vulnerable paths, each represented as a 
            sequence of function calls separated by arrows (->).
            """
            
            filled_template = paths_template.format(flow=flow_str, text=text)
            
            # Generate vulnerable paths
            if self.use_ollama_for_prompts:
                response = await self._query_ollama(filled_template)
                return self._parse_call_flow_response(response)
            else:
                # Simple fallback - just return the original paths
                return call_flow
            
            return []
        except Exception as e:
            logger.error(f"Error identifying vulnerable paths: {e}")
            return []
    
    def _select_template(self, document: Dict[str, Any]) -> str:
        """
        Select the appropriate template based on document metadata.
        
        Args:
            document: Document to select template for
            
        Returns:
            Selected template
        """
        # Check document metadata for type
        metadata = document.get('metadata', {})
        text = document.get('text', '')
        
        # Check if this is code
        is_code = (
            'language' in metadata or
            'file_path' in metadata or
            text.strip().startswith('def ') or
            text.strip().startswith('class ') or
            text.strip().startswith('function ') or
            text.strip().startswith('import ') or
            text.strip().startswith('#include') or
            '```' in text
        )
        
        # Check if this is security-related
        is_security = (
            'security' in metadata or 
            'vulnerability' in metadata or 
            'security_flaw' in metadata or
            'bug' in metadata
        )
        
        # Select template
        if is_security:
            return self.security_template
        elif is_code:
            return self.code_template
        else:
            return self.general_template
    
    async def _query_ollama(self, prompt: str) -> str:
        """
        Query Ollama with the given prompt.
        
        Args:
            prompt: Prompt to send to Ollama
            
        Returns:
            Ollama response
        """
        try:
            # Select appropriate API endpoint based on provider type
            if self.provider_type == "ollama":
                url = f"{self.ollama_url.rstrip('/')}/api/generate"
                
                payload = {
                    "model": self.prompt_model,
                    "prompt": prompt,  # Don't truncate the prompt
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,  # Use configurable temperature
                        "num_predict": self.max_tokens    # Allow for longer responses
                    }
                }
                
                # Add timeout handling and retries
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Use the configured timeout
                        async with httpx.AsyncClient(timeout=self.ollama_timeout) as client:
                            logger.debug(f"Sending Ollama request with timeout={self.ollama_timeout}s, temp={self.temperature}")
                            response = await client.post(url, json=payload)
                            response.raise_for_status()
                            
                            result = response.json()
                            return result.get("response", "")
                    except httpx.TimeoutException:
                        if retry < max_retries - 1:
                            logger.warning(f"Ollama request timed out, retrying ({retry+1}/{max_retries})...")
                            await asyncio.sleep(1)  # Brief pause before retry
                        else:
                            raise TimeoutError(f"Ollama request timed out after {max_retries} attempts")
                    except httpx.HTTPStatusError as e:
                        logger.error(f"HTTP error from Ollama: {e.response.status_code} - {e.response.text}")
                        break  # Don't retry HTTP errors
            elif self.provider_type == "openai":
                # Placeholder for OpenAI API integration
                try:
                    # This section would be implemented for OpenAI
                    logger.warning("OpenAI provider not yet fully implemented")
                    return self._generate_fallback({"text": prompt}, "provider_not_implemented")
                except Exception as e:
                    logger.error(f"Error with OpenAI provider: {e}")
            elif self.provider_type == "hosted":
                # Placeholder for generic hosted provider support
                logger.warning("Generic hosted provider not yet implemented")
                return self._generate_fallback({"text": prompt}, "provider_not_implemented")
            else:
                logger.error(f"Unknown provider type: {self.provider_type}")
                return self._generate_fallback({"text": prompt}, "unknown_provider")
        except Exception as e:
            logger.error(f"Error generating with provider {self.provider_type}: {e}")
            
        # If we reach here, we had an error - use fallback
        logger.warning(f"Using fallback generator due to {self.provider_type} error")
        return self._generate_fallback({"text": prompt}, f"{self.provider_type}_error")
    
    def _generate_fallback(self, document: Dict[str, Any], fallback_type: str = "general") -> str:
        """
        Generate a fallback prompt when LLM generation fails.
        
        Args:
            document: Document to generate prompt for
            fallback_type: Type of fallback prompt to generate
            
        Returns:
            Fallback prompt
        """
        text = document.get('text', '')
        metadata = document.get('metadata', {})
        
        # Don't truncate text to ensure comprehensive prompts
        # Just guard against really extreme cases
        if len(text) > 10000:
            # Only truncate in extreme cases, but still keep a substantial portion
            truncated_text = text[:10000] + "\n...[content truncated for length]..."
        else:
            truncated_text = text
        
        # Generate different fallbacks based on type
        if fallback_type == "invariant":
            return f"""
            # Invariant Analysis
            
            Analyze the following for invariant properties that might be violated:
            
            ```
            {truncated_text}
            ```
            
            Consider what properties must always hold true in this code/system:
            - What assumptions must remain valid?
            - Which conditions must never be violated?
            - What security invariants must be maintained?
            """
        elif fallback_type == "pattern":
            return f"""
            # Vulnerability Pattern Analysis
            
            Identify the general pattern of vulnerability in:
            
            ```
            {truncated_text}
            ```
            
            Consider:
            - What category of vulnerability is this?
            - What are the common characteristics?
            - How does this pattern manifest in different contexts?
            - What are the key indicators to look for?
            """
        elif fallback_type.endswith("_error"):
            return f"""
            # Analysis Required
            
            The LLM service encountered an error. Please analyze this content:
            
            ```
            {truncated_text}
            ```
            
            Focus on:
            - Key concepts and functionality
            - Potential security issues
            - Notable patterns or anti-patterns
            - Important relationships or dependencies
            """
        else:
            # Extract key metadata for a more informed prompt
            metakeys = ", ".join([f"{k}: {v}" for k, v in metadata.items() if k in ["language", "type", "source", "category", "severity"]])
            
            if metakeys:
                return f"""
                # Content Analysis ({metakeys})
                
                Analyze this content with attention to its metadata characteristics:
                
                ```
                {truncated_text}
                ```
                
                Provide a detailed analysis covering:
                - Purpose and functionality
                - Key components or structures
                - Potential issues or vulnerabilities
                - Best practices that apply to this content
                """
            else:
                return f"""
                # Content Analysis
                
                Analyze this content:
                
                ```
                {truncated_text}
                ```
                
                Provide a thorough analysis including:
                - Main purpose and functionality
                - Key technical aspects
                - Potential areas of concern
                - Important patterns or structures
                """
    
    def _generate_fallback_checks(self, document: Dict[str, Any]) -> List[str]:
        """Generate fallback quick checks when LLM generation fails."""
        text = document.get('text', '').lower()
        checks = []
        
        # Look for common vulnerability patterns
        if "sql" in text or "query" in text:
            checks.append("Check for SQL string concatenation with user input")
        if "exec" in text or "system" in text or "subprocess" in text:
            checks.append("Check for command execution with user input")
        if "open" in text or "file" in text:
            checks.append("Check for unsanitized file paths")
        if "password" in text or "auth" in text:
            checks.append("Check for hardcoded credentials")
        if "html" in text or "innerHtml" in text:
            checks.append("Check for unsanitized HTML output")
        
        return checks
    
    def _generate_fallback_api_interactions(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback API interactions when LLM generation fails."""
        text = document.get('text', '').lower()
        interactions = {}
        
        # Look for common APIs
        if "sql" in text or "query" in text or "cursor" in text:
            interactions["SQL/Database API"] = "Check for proper parameterization of queries"
        if "http" in text or "request" in text or "fetch" in text:
            interactions["HTTP Client"] = "Verify proper input validation and output sanitization"
        if "file" in text or "open" in text or "read" in text:
            interactions["File System"] = "Ensure proper path validation and access controls"
        if "json" in text or "parse" in text:
            interactions["JSON Parsing"] = "Check for proper error handling and validation"
        
        return interactions
    
    def _generate_fallback_call_flow(self, document: Dict[str, Any]) -> List[List[str]]:
        """Generate fallback call flow when LLM generation fails."""
        import re
        text = document.get('text', '')
        
        # Extract function calls using regex
        calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text)
        
        # Return as a single flow
        if calls:
            return [calls]
        return []
    
    def _parse_call_flow_response(self, response: str) -> List[List[str]]:
        """
        Parse call flow from LLM response.
        
        Args:
            response: LLM response to the call flow prompt
            
        Returns:
            List of call flows, where each flow is a list of function calls
        """
        flows = []
        
        # Split by lines and look for paths
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Remove "Path X:" prefix if present
            if line.startswith('Path ') and ':' in line:
                line = line.split(':', 1)[1].strip()
            
            # Check for "->" or "→" separators
            if '->' in line or '→' in line:
                # Normalize arrows
                line = line.replace('->', '→')
                
                # Split by arrow and clean function names
                functions = [f.strip() for f in line.split('→')]
                
                # Only add non-empty paths
                if functions and all(functions):
                    flows.append(functions)
        
        # If no flows found, try to extract function calls from the text
        if not flows:
            import re
            calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', response)
            if calls:
                flows.append(calls)
        
        return flows

# Example usage
async def test_prompt_generator():
    """Test the prompt generator functionality."""
    generator = PromptGenerator()
    
    # Example document
    document = {
        "text": """
        def authenticate(username, password):
            query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
            cursor.execute(query)
            user = cursor.fetchone()
            if user:
                return True
            return False
        """,
        "metadata": {
            "language": "python",
            "security_flaw": "sql_injection",
            "description": "SQL injection vulnerability in authentication function"
        }
    }
    
    # Generate a prompt
    prompt = await generator.generate_prompt(document)
    print(f"Generated prompt: {prompt}\n")
    
    # Generate an invariant analysis
    invariant = await generator.generate_invariant_analysis(document, prompt)
    print(f"Invariant analysis: {invariant}\n")
    
    # Extract quick checks
    checks = await generator.extract_quick_checks(document, prompt)
    print(f"Quick checks: {checks}\n")
    
    # Extract API interactions
    interactions = await generator.extract_api_interactions(document)
    print(f"API interactions: {interactions}\n")
    
    # Extract call flow
    flow = await generator.extract_call_flow(document)
    print(f"Call flow: {flow}\n")
    
    # Identify vulnerable paths
    if flow:
        paths = await generator.identify_vulnerable_paths(document, flow)
        print(f"Vulnerable paths: {paths}")

if __name__ == "__main__":
    asyncio.run(test_prompt_generator())