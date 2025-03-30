"""
Documentation analyzer for smart contracts.

This module analyzes the quality, completeness, and accuracy of
smart contract documentation and comments.
"""

import re
import json
import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger

from llama_index.core.settings import Settings
from ..pipeline.core import Context
from ..models.documentation import DocumentationQuality
from ..nodes_config import config
from ..utils.llm_method_resolver import call_llm_async, extract_content_from_response

class DocumentationAnalyzer:
    """
    Analyzer for smart contract documentation quality.
    
    Evaluates code comments, NatSpec documentation, and general documentation
    quality for smart contracts.
    """
    
    def __init__(self, llm_adapter=None):
        """
        Initialize the documentation analyzer
        
        Args:
            llm_adapter: LlamaIndex adapter for LLM access
        """
        if llm_adapter is None:
            try:
                from ..llm.llama_index_adapter import LlamaIndexAdapter
                
                # Use configuration explicitly to ensure all parameters are set
                self.llm_adapter = LlamaIndexAdapter(
                    model_name=config.DOCUMENTATION_MODEL,
                    provider=config.DOCUMENTATION_MODEL_PROVIDER,
                    base_url=config.DOCUMENTATION_MODEL_BASE_URL
                )
                logger.info(f"Created documentation analysis LLM adapter with model: {config.DOCUMENTATION_MODEL}")
                logger.info(f"Provider: {config.DOCUMENTATION_MODEL_PROVIDER}, Base URL: {config.DOCUMENTATION_MODEL_BASE_URL}")
            except Exception as e:
                logger.error(f"Failed to create documentation LLM adapter: {e}")
                self.llm_adapter = None
        else:
            self.llm_adapter = llm_adapter
        
        # NatSpec comment patterns
        self.natspec_patterns = {
            "title": r"///\s*@title\s+(.*?)(?:\n|$)",
            "author": r"///\s*@author\s+(.*?)(?:\n|$)",
            "notice": r"///\s*@notice\s+(.*?)(?:\n|$)",
            "dev": r"///\s*@dev\s+(.*?)(?:\n|$)",
            "param": r"///\s*@param\s+(\w+)\s+(.*?)(?:\n|$)",
            "return": r"///\s*@return\s+(.*?)(?:\n|$)"
        }
    
    async def process(self, context: Context) -> Context:
        """
        Process the context to analyze documentation quality
        
        Args:
            context: Processing context with contract files
            
        Returns:
            Updated context with documentation analysis
        """
        logger.info("Starting documentation analysis")
        
        # Initialize documentation findings in context
        if not hasattr(context, "documentation_quality"):
            context.documentation_quality = {}
        
        # Get list of solidity files to analyze
        solidity_files = [(file_id, file_data) for file_id, file_data in context.files.items() 
                          if file_data.get("is_solidity", False)]
        
        logger.info(f"Analyzing documentation in {len(solidity_files)} Solidity files")
        
        # Process files in chunks to manage resources
        chunk_size = 5
        for i in range(0, len(solidity_files), chunk_size):
            chunk = solidity_files[i:i+chunk_size]
            
            # Process this chunk of files concurrently
            tasks = [self._analyze_file(context, file_id, file_data) for file_id, file_data in chunk]
            await asyncio.gather(*tasks)
            
            # Small delay to prevent resource exhaustion
            await asyncio.sleep(0.1)
        
        # Calculate project-wide documentation quality
        await self._calculate_project_quality(context)
        
        logger.info("Documentation analysis complete")
        return context
    
    async def _analyze_file(self, context: Context, file_id: str, file_data: Dict[str, Any]):
        """
        Analyze documentation quality in a single file
        
        Args:
            context: Processing context
            file_id: File ID
            file_data: File data
        """
        try:
            # Extract basic metrics
            content = file_data["content"]
            contract_name = file_data.get("name", file_id)
            
            # Calculate basic documentation metrics
            metrics = self._calculate_metrics(content)
            
            # Extract NatSpec documentation
            natspec = self._extract_natspec(content)
            
            # Evaluate documentation quality using LLM
            quality_assessment = await self._assess_quality(content, file_data.get("type", "solidity"))
            
            # Create documentation quality result
            quality = DocumentationQuality(
                contract_name=contract_name,
                file_path=file_data.get("path", file_id),
                metrics=metrics,
                natspec=natspec,
                quality_assessment=quality_assessment,
                recommendations=quality_assessment.get("recommendations", [])
            )
            
            # Store in context
            context.documentation_quality[file_id] = quality
            
            logger.info(f"Completed documentation analysis for {contract_name}")
            
        except Exception as e:
            logger.error(f"Error analyzing documentation in {file_id}: {str(e)}")
            context.add_error(
                stage="documentation_analysis",
                message=f"Failed to analyze file: {file_id}",
                exception=e
            )
    
    def _calculate_metrics(self, content: str) -> Dict[str, Any]:
        """
        Calculate basic documentation metrics
        
        Args:
            content: Contract source code
            
        Returns:
            Dictionary of documentation metrics
        """
        # Count lines and functions
        lines = content.split("\n")
        total_lines = len(lines)
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith("//") and not line.strip().startswith("/*"))
        comment_lines = sum(1 for line in lines if line.strip() and (line.strip().startswith("//") or line.strip().startswith("/*")))
        
        # Count functions
        function_matches = re.finditer(r"function\s+(\w+)", content)
        functions = [match.group(1) for match in function_matches]
        function_count = len(functions)
        
        # Count documented functions (with NatSpec comments)
        documented_function_count = 0
        for func in functions:
            pattern = r"///[^{]*?function\s+" + re.escape(func)
            if re.search(pattern, content, re.DOTALL):
                documented_function_count += 1
        
        # Calculate documentation ratio
        doc_ratio = comment_lines / total_lines if total_lines > 0 else 0
        function_doc_ratio = documented_function_count / function_count if function_count > 0 else 0
        
        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "documentation_ratio": doc_ratio,
            "function_count": function_count,
            "documented_functions": documented_function_count,
            "function_documentation_ratio": function_doc_ratio
        }
    
    def _extract_natspec(self, content: str) -> Dict[str, Any]:
        """
        Extract NatSpec documentation from contract code
        
        Args:
            content: Contract source code
            
        Returns:
            Dictionary of NatSpec documentation
        """
        natspec = {
            "contract": {
                "title": None,
                "author": None,
                "notice": None,
                "dev": None
            },
            "functions": {}
        }
        
        # Extract contract-level NatSpec
        for field, pattern in self.natspec_patterns.items():
            if field not in ["param", "return"]:
                match = re.search(pattern, content)
                if match:
                    natspec["contract"][field] = match.group(1).strip()
        
        # Extract function-level NatSpec
        function_pattern = r"(///[^{]*?)function\s+(\w+)\s*\(([^)]*)\)"
        function_matches = re.finditer(function_pattern, content, re.DOTALL)
        
        for match in function_matches:
            comments = match.group(1)
            function_name = match.group(2)
            parameters = match.group(3)
            
            func_natspec = {
                "notice": None,
                "dev": None,
                "params": {},
                "return": None
            }
            
            # Extract notice and dev comments
            notice_match = re.search(r"///\s*@notice\s+(.*?)(?=///\s*@|$)", comments, re.DOTALL)
            if notice_match:
                func_natspec["notice"] = notice_match.group(1).strip()
            
            dev_match = re.search(r"///\s*@dev\s+(.*?)(?=///\s*@|$)", comments, re.DOTALL)
            if dev_match:
                func_natspec["dev"] = dev_match.group(1).strip()
            
            # Extract param comments
            param_matches = re.finditer(r"///\s*@param\s+(\w+)\s+(.*?)(?=///\s*@|$)", comments, re.DOTALL)
            for param_match in param_matches:
                param_name = param_match.group(1)
                param_desc = param_match.group(2).strip()
                func_natspec["params"][param_name] = param_desc
            
            # Extract return comment
            return_match = re.search(r"///\s*@return\s+(.*?)(?=///\s*@|$)", comments, re.DOTALL)
            if return_match:
                func_natspec["return"] = return_match.group(1).strip()
            
            natspec["functions"][function_name] = func_natspec
        
        return natspec
    
    async def _assess_quality(self, content: str, file_type: str) -> Dict[str, Any]:
        """
        Assess the quality of documentation in the given content.
        
        Args:
            content: Content to analyze
            file_type: Type of file being analyzed
            
        Returns:
            Dictionary with quality assessment
        """
        try:
            # Create a prompt for the LLM
            prompt = self._create_quality_assessment_prompt(content, file_type)
            
            # Use the resolver to call the LLM
            response = await call_llm_async(
                llm=self.llm_adapter,
                input_text=prompt,
                as_chat=True
            )
            
            # Extract and parse the response
            response_text = extract_content_from_response(response)
            
            try:
                # Try to parse as JSON
                import json
                result_dict = json.loads(response_text)
                return result_dict
            except json.JSONDecodeError:
                # If not valid JSON, extract structured data using regex
                import re
                result_dict = {"quality_score": 0}
                
                # Extract quality score
                score_match = re.search(r'quality_score["\s:]+(\d+)', response_text)
                if score_match:
                    try:
                        result_dict["quality_score"] = int(score_match.group(1))
                    except ValueError:
                        pass
                
                # Extract issues
                result_dict["issues"] = []
                issue_matches = re.finditer(r'issue["\s:]+([^,}]+)', response_text)
                for match in issue_matches:
                    result_dict["issues"].append(match.group(1).strip(' "\''))
                
                return result_dict
                
        except Exception as e:
            logger.error(f"Error in documentation quality assessment: {e}")
            return {"error": str(e), "quality_score": 0}

    def _create_quality_assessment_prompt(self, content: str, file_type: str) -> str:
        """
        Create a prompt for assessing documentation quality.
        
        Args:
            content: Content to analyze
            file_type: Type of file being analyzed
            
        Returns:
            Prompt for the LLM
        """
        # Determine documentation standards based on file type
        if file_type.lower() == 'sol' or file_type.lower() == 'solidity':
            doc_standards = (
                "- NatSpec format (/// for documentation comments)\n"
                "- @title, @author, @notice, @dev tags for contracts\n"
                "- @param, @return, @notice for functions\n"
                "- Documentation for state variables\n"
                "- Explanations for complex logic"
            )
        else:
            doc_standards = (
                "- Clear file-level documentation explaining purpose\n"
                "- Function/method documentation with parameters and return values\n"
                "- Comments for complex sections of code\n"
                "- Examples where appropriate"
            )
        
        # Create the prompt
        prompt = f"""
Analyze the documentation quality in this {file_type} file.

Documentation standards to check:
{doc_standards}

Rate the overall documentation quality on a scale of 0-10, where:
- 0-2: Missing or severely inadequate documentation
- 3-5: Basic documentation but significant gaps
- 6-8: Good documentation with minor improvements needed
- 9-10: Excellent, comprehensive documentation

File content:
```
{content}
```

Provide a JSON response with the following fields:
- quality_score: Numeric score (0-10)
- issues: List of identified issues
- recommendations: List of recommendations for improvement
"""
        return prompt

    async def _calculate_project_quality(self, context: Context):
        """
        Calculate project-wide documentation quality
        
        Args:
            context: Processing context
        """
        if not context.documentation_quality:
            return
        
        # Calculate average scores
        scores = {
            "overall_score": 0,
            "completeness": 0,
            "clarity": 0,
            "accuracy": 0,
            "natspec_compliance": 0
        }
        
        file_count = len(context.documentation_quality)
        
        for file_id, quality in context.documentation_quality.items():
            assessment = quality.quality_assessment
            for metric in scores:
                if metric in assessment:
                    scores[metric] += assessment[metric] / file_count
        
        # Generate project-wide recommendations
        recommendations = []
        for file_id, quality in context.documentation_quality.items():
            if "recommendations" in quality.quality_assessment:
                for rec in quality.quality_assessment["recommendations"][:2]:  # Limit to top 2 per file
                    recommendations.append(rec)
        
        # Remove duplicates and limit to top 5
        unique_recommendations = list(set(recommendations))[:5]
        
        # Store project quality
        context.project_documentation_quality = {
            "scores": scores,
            "recommendations": unique_recommendations,
            "files_analyzed": file_count,
            "documentation_level": self._get_quality_level(scores["overall_score"])
        }
    
    def _get_quality_level(self, score: float) -> str:
        """
        Convert numeric score to quality level
        
        Args:
            score: Numeric score (0-10)
            
        Returns:
            Quality level description
        """
        if score >= 9:
            return "Excellent"
        elif score >= 7:
            return "Good"
        elif score >= 4:
            return "Adequate"
        else:
            return "Poor"
