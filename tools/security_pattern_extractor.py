#!/usr/bin/env python3
"""
Security Pattern Extractor for generating enhanced security analysis prompts and extracting
common vulnerability patterns, API interactions, and quick check rules.
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class SecurityPatternExtractor:
    """
    Extracts security patterns and generates enhanced prompts for security vulnerabilities.
    
    This class provides methods to:
    1. Generate invariant analysis prompts for deeper understanding of security flaws
    2. Create generalized patterns for categories of security vulnerabilities
    3. Extract quick check patterns for simple vulnerability detection
    4. Identify API interaction patterns that might lead to vulnerabilities
    5. Detect cognitive biases and counterfactual scenarios in code
    """
    
    def __init__(
        self,
        llm_model: str = "gemma:2b",
        ollama_url: str = "http://localhost:11434",
        use_ollama: bool = True,
    ):
        """
        Initialize the SecurityPatternExtractor.
        
        Args:
            llm_model: LLM to use for pattern extraction and prompt generation
            ollama_url: URL for Ollama API if using Ollama
            use_ollama: Whether to use Ollama for LLM queries
        """
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        self.use_ollama = use_ollama
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load prompt templates for different types of analysis."""
        try:
            templates_dir = Path(__file__).parent / "templates"
            os.makedirs(templates_dir, exist_ok=True)
            
            # Invariant analysis template
            inv_template_path = templates_dir / "invariant_analysis_template.txt"
            self.invariant_template = self._load_template_file(
                inv_template_path,
                """
                Perform an invariant analysis on the following security vulnerability:
                
                CODE:
                {code}
                
                VULNERABILITY DESCRIPTION:
                {description}
                
                Identify the following:
                1. What invariant properties are violated in this code?
                2. What conditions must always hold true to prevent this vulnerability?
                3. What are the general principles that apply to similar vulnerabilities?
                4. How would you formulate a formal check to detect this issue?
                """
            )
            
            # General flaw pattern template
            pattern_template_path = templates_dir / "general_pattern_template.txt"
            self.pattern_template = self._load_template_file(
                pattern_template_path,
                """
                Given the following specific vulnerability:
                
                CODE:
                {code}
                
                VULNERABILITY DESCRIPTION:
                {description}
                
                Generate a generalized pattern that describes this type of security flaw:
                1. Summarize the general vulnerability class
                2. Identify the key elements that make this a vulnerability
                3. Describe similar patterns that would lead to the same vulnerability
                4. Create a general rule that could detect this class of vulnerabilities
                """
            )
            
            # Quick check template
            quick_check_template_path = templates_dir / "quick_check_template.txt"
            self.quick_check_template = self._load_template_file(
                quick_check_template_path,
                """
                Given the following vulnerability:
                
                CODE:
                {code}
                
                VULNERABILITY DESCRIPTION:
                {description}
                
                Generate a list of quick checks that could identify this issue:
                1. Simple string patterns to search for
                2. AST patterns that indicate this vulnerability
                3. Common function calls or API usages that are often misused
                4. Order of operation patterns that lead to this vulnerability
                
                Format your response as a JSON list of quick check rules.
                """
            )
            
            # Call flow extraction template
            flow_template_path = templates_dir / "call_flow_template.txt"
            self.flow_template = self._load_template_file(
                flow_template_path,
                """
                Extract the call flow from the following code or vulnerability description:
                
                {text}
                
                Represent the execution flow as a series of function/method calls in sequence.
                Format your response as a list of function names representing the execution path.
                If multiple paths are possible, list each distinct path separately.
                """
            )
            
            logger.info("Loaded security analysis templates")
            
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            # Set minimal default templates
            self.invariant_template = "Analyze invariant properties violated in this code: {code}\n\nVulnerability: {description}"
            self.pattern_template = "Generate a general pattern for this vulnerability: {code}\n\nVulnerability: {description}"
            self.quick_check_template = "Suggest quick checks for this vulnerability: {code}\n\nVulnerability: {description}"
            self.flow_template = "Extract call flows from this code or description: {text}"
    
    def _load_template_file(self, file_path: Path, default_template: str) -> str:
        """Load a template file or use default if not found."""
        if file_path.exists():
            with open(file_path, 'r') as f:
                return f.read()
        
        # Save default template for future use
        try:
            with open(file_path, 'w') as f:
                f.write(default_template)
        except Exception as e:
            logger.warning(f"Could not save default template to {file_path}: {e}")
            
        return default_template
    
    async def generate_invariant_analysis(self, document: Dict[str, Any], vulnerability_prompt: str = None) -> str:
        """
        Generate an invariant analysis for a security vulnerability.
        
        Args:
            document: Document containing code and metadata
            vulnerability_prompt: Optional existing vulnerability prompt
            
        Returns:
            Invariant analysis prompt
        """
        try:
            code = document.get('text', '')
            metadata = document.get('metadata', {})
            
            # Extract vulnerability description from metadata or prompt
            description = metadata.get('description', '')
            if not description and vulnerability_prompt:
                description = vulnerability_prompt
            
            # Fill template
            prompt = self.invariant_template.format(
                code=code,
                description=description
            )
            
            # Get response from LLM
            invariant_analysis = await self._query_llm(prompt)
            return invariant_analysis
        except Exception as e:
            logger.error(f"Error generating invariant analysis: {e}")
            return ""
    
    async def generate_general_pattern(self, document: Dict[str, Any], vulnerability_prompt: str = None) -> str:
        """
        Generate a general pattern for a security vulnerability.
        
        Args:
            document: Document containing code and metadata
            vulnerability_prompt: Optional existing vulnerability prompt
            
        Returns:
            General flaw pattern description
        """
        try:
            code = document.get('text', '')
            metadata = document.get('metadata', {})
            
            # Extract vulnerability description from metadata or prompt
            description = metadata.get('description', '')
            if not description and vulnerability_prompt:
                description = vulnerability_prompt
            
            # Fill template
            prompt = self.pattern_template.format(
                code=code,
                description=description
            )
            
            # Get response from LLM
            general_pattern = await self._query_llm(prompt)
            return general_pattern
        except Exception as e:
            logger.error(f"Error generating general pattern: {e}")
            return ""
    
    async def extract_quick_checks(self, document: Dict[str, Any], vulnerability_prompt: str = None) -> List[str]:
        """
        Extract quick check patterns for a security vulnerability.
        
        Args:
            document: Document containing code and metadata
            vulnerability_prompt: Optional existing vulnerability prompt
            
        Returns:
            List of quick check patterns
        """
        try:
            code = document.get('text', '')
            metadata = document.get('metadata', {})
            
            # Extract vulnerability description from metadata or prompt
            description = metadata.get('description', '')
            if not description and vulnerability_prompt:
                description = vulnerability_prompt
            
            # Fill template
            prompt = self.quick_check_template.format(
                code=code,
                description=description
            )
            
            # Get response from LLM
            response = await self._query_llm(prompt)
            
            # Parse response to get the list of quick checks
            quick_checks = self._extract_checks_from_response(response)
            return quick_checks
        except Exception as e:
            logger.error(f"Error extracting quick checks: {e}")
            return []
    
    async def extract_call_flow(self, document: Dict[str, Any]) -> List[List[str]]:
        """
        Extract call flow from code or vulnerability description.
        
        Args:
            document: Document containing code or vulnerability description
            
        Returns:
            List of call flows, where each flow is a list of function calls
        """
        try:
            text = document.get('text', '')
            
            # Fill template
            prompt = self.flow_template.format(text=text)
            
            # Get response from LLM
            response = await self._query_llm(prompt)
            
            # Parse response to get call flows
            call_flows = self._extract_flows_from_response(response)
            return call_flows
        except Exception as e:
            logger.error(f"Error extracting call flow: {e}")
            return []
    
    async def _query_llm(self, prompt: str) -> str:
        """
        Query the LLM with the given prompt.
        
        Args:
            prompt: Prompt to send to the LLM
            
        Returns:
            LLM response
        """
        try:
            if self.use_ollama:
                return await self._query_ollama(prompt)
            else:
                return await self._query_alternative_llm(prompt)
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return ""
    
    async def _query_ollama(self, prompt: str) -> str:
        """
        Query Ollama with the given prompt.
        
        Args:
            prompt: Prompt to send to Ollama
            
        Returns:
            Ollama response
        """
        try:
            import httpx
            
            url = f"{self.ollama_url.rstrip('/')}/api/generate"
            
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for more deterministic outputs
                    "num_predict": 4096  # Increase max tokens to handle longer responses
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            # Fallback to local extraction
            return self._fallback_extraction(prompt)
    
    async def _query_alternative_llm(self, prompt: str) -> str:
        """
        Query an alternative LLM implementation.
        
        Args:
            prompt: Prompt to send to the LLM
            
        Returns:
            LLM response
        """
        # Here you could implement other LLM backends
        # For now, just return a fallback extraction
        return self._fallback_extraction(prompt)
    
    def _fallback_extraction(self, prompt: str) -> str:
        """
        Fallback extraction using simple pattern matching when LLM is unavailable.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Extracted information using regex patterns
        """
        # This is a very basic fallback that only works for simple cases
        try:
            import re
            
            if "invariant analysis" in prompt.lower():
                return "Unable to perform detailed invariant analysis without LLM access."
            
            if "general pattern" in prompt.lower():
                return "Unable to generate general pattern without LLM access."
            
            if "quick checks" in prompt.lower():
                # Try to extract patterns from the code
                code_match = re.search(r'CODE:\s*(.*?)(?:VULNERABILITY DESCRIPTION|\Z)', prompt, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip()
                    
                    # Common check patterns
                    checks = []
                    
                    # SQL injection
                    if re.search(r"SELECT.*FROM.*WHERE.*=\s*['\"]\s*\+", code) or \
                       re.search(r"cursor\.execute\(.*\+", code) or \
                       re.search(r"f[\"']SELECT.*\{", code):
                        checks.append("Check for SQL string concatenation with user input")
                    
                    # Command injection
                    if re.search(r"(?:system|exec|execvp|popen|subprocess\.call)\(.*\+", code) or \
                       re.search(r"(?:system|exec|execvp|popen|subprocess\.call)\(.*\{", code):
                        checks.append("Check for command execution with user input")
                    
                    # Return any found checks or a generic message
                    return json.dumps(checks) if checks else "[]"
                return "[]"
            
            if "call flow" in prompt.lower():
                # Try to extract function calls
                text_match = re.search(r'\{text\}\s*(.*?)(?:\Z)', prompt, re.DOTALL)
                if text_match:
                    text = text_match.group(1).strip()
                    
                    # Extract function calls using regex
                    calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text)
                    
                    if calls:
                        # Return as a simple flow
                        return json.dumps([calls])
                return "[]"
            
            # Default fallback
            return "Unable to extract information without LLM access."
        except Exception as e:
            logger.error(f"Error in fallback extraction: {e}")
            return "Error in fallback extraction"
    
    def _extract_checks_from_response(self, response: str) -> List[str]:
        """
        Extract quick check patterns from LLM response.
        
        Args:
            response: LLM response to the quick checks prompt
            
        Returns:
            List of quick check patterns
        """
        try:
            import re
            
            # First check if response is already in JSON format
            try:
                checks = json.loads(response)
                if isinstance(checks, list):
                    return checks
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON array from the response
            json_match = re.search(r'\[\s*".*"\s*\]', response.replace('\n', ' '), re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    checks = json.loads(json_str)
                    if isinstance(checks, list):
                        return checks
                except json.JSONDecodeError:
                    pass
            
            # Fall back to extracting lines prefixed with numbers or bullet points
            checks = []
            for line in response.split('\n'):
                line = line.strip()
                # Match numbered lists (1. Check for X)
                match = re.match(r'^\d+\.\s+(.+)$', line)
                if match:
                    check = match.group(1).strip()
                    if check:
                        checks.append(check)
                    continue
                
                # Match bullet points (• Check for X, - Check for X, * Check for X)
                match = re.match(r'^[\•\-\*]\s+(.+)$', line)
                if match:
                    check = match.group(1).strip()
                    if check:
                        checks.append(check)
            
            return checks
        except Exception as e:
            logger.error(f"Error extracting checks from response: {e}")
            return []
    
    def _extract_flows_from_response(self, response: str) -> List[List[str]]:
        """
        Extract call flows from LLM response.
        
        Args:
            response: LLM response to the call flow prompt
            
        Returns:
            List of call flows, where each flow is a list of function calls
        """
        try:
            import re
            
            # First check if response is already in JSON format
            try:
                flows = json.loads(response)
                if isinstance(flows, list) and all(isinstance(flow, list) for flow in flows):
                    return flows
            except json.JSONDecodeError:
                pass
            
            # Try to extract structured paths with "→" or "->" separators
            flows = []
            
            # Split by lines and look for flows
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check for flows separated by arrows
                if '→' in line or '->' in line:
                    # Normalize to use the same arrow type
                    line = line.replace('->', '→')
                    
                    # Remove any "Path X:" prefix
                    if re.match(r'^Path\s+\d+:', line):
                        line = re.sub(r'^Path\s+\d+:\s*', '', line)
                    
                    # Split by arrow and clean up each function name
                    flow = [func.strip() for func in line.split('→')]
                    flows.append(flow)
            
            # If no flows with arrows found, try to parse as a list of functions
            if not flows:
                # Extract function calls from the text
                calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', response)
                if calls:
                    flows.append(calls)
            
            return flows
        except Exception as e:
            logger.error(f"Error extracting flows from response: {e}")
            return []

# Basic usage example
async def process_example():
    """Example usage of the SecurityPatternExtractor"""
    extractor = SecurityPatternExtractor()
    
    # Example document
    document = {
        "text": """
        def authenticate(request):
            username = request.POST.get('username')
            password = request.POST.get('password')
            query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
            cursor.execute(query)
            user = cursor.fetchone()
            if user:
                return True
            return False
        """,
        "metadata": {
            "description": "SQL Injection vulnerability in authentication function"
        }
    }
    
    # Generate invariant analysis
    invariant = await extractor.generate_invariant_analysis(document)
    print("INVARIANT ANALYSIS:")
    print(invariant)
    print("\n" + "-"*50 + "\n")
    
    # Extract quick checks
    checks = await extractor.extract_quick_checks(document)
    print("QUICK CHECKS:")
    for check in checks:
        print(f"- {check}")
    print("\n" + "-"*50 + "\n")
    
    # Extract call flow
    flows = await extractor.extract_call_flow(document)
    print("CALL FLOWS:")
    for i, flow in enumerate(flows):
        print(f"Flow {i+1}: {' → '.join(flow)}")

if __name__ == "__main__":
    asyncio.run(process_example())
