#!/usr/bin/env python3
"""
Simple end-to-end analyzer for smart contracts using the existing components.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components from previous project
try:
    # Try to import from the previous project
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "previous", "finite-monkey-engine", "src"))
    
    try:
        from scripts.sitter import tsg
        from scripts.sitterQL import sitterQL
        logger.info("Successfully imported sitter modules from previous project")
        import_error = None
    except ImportError as e:
        # If importing directly from scripts fails, try with the full path
        sys.path.pop()  # Remove the previous path
        
        # Add the scripts directory directly
        scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "previous", "finite-monkey-engine", "src", "scripts")
        sys.path.append(scripts_dir)
        
        try:
            from sitter import tsg
            from sitterQL import sitterQL
            logger.info("Successfully imported sitter modules from scripts directory")
            import_error = None
        except ImportError as e2:
            import_error = str(e2)
            logger.warning(f"Failed to import from scripts directory: {e2}")
            logger.warning("Will use simplified analysis without Tree-Sitter")
            
except Exception as e:
    import_error = str(e)
    logger.warning(f"Error setting up import paths: {e}")
    logger.warning("Will use simplified analysis without Tree-Sitter")

# Import components from current project
from finite_monkey.adapters.ollama import Ollama
from finite_monkey.db.manager import DatabaseManager
from finite_monkey.nodes_config import nodes_config

class SimpleAnalyzer:
    """
    A simple end-to-end analyzer for smart contracts.
    
    This class uses the existing sitter.py from the previous project
    to analyze Solidity contracts, and then uses the LLM components
    from the current project to generate a security report.
    """
    
    def __init__(self, db_url=None, primary_model=None, secondary_model=None):
        """Initialize the analyzer."""
        # Load config
        self.config = nodes_config()
        
        # Set model names
        self.primary_model = primary_model or self.config.SCAN_MODEL 
        self.secondary_model = secondary_model or self.config.CONFIRMATION_MODEL or "llama3:70b"
        
        # Set up LLM clients
        self.primary_llm = Ollama(model=self.primary_model, 
                                base_url=self.config.OPENAI_API_BASE or "http://localhost:11434")
        self.secondary_llm = Ollama(model=self.secondary_model, 
                                  base_url=self.config.OPENAI_API_BASE or "http://localhost:11434")
        
        # Set up database manager
        self.db_url = db_url or self.config.DATABASE_URL
        if self.db_url and "postgresql" in self.db_url:
            # Convert the standard PostgreSQL URL to an async one
            if "postgresql:" in self.db_url and "postgresql+asyncpg:" not in self.db_url:
                self.db_url = self.db_url.replace("postgresql:", "postgresql+asyncpg:")
            
            self.db_manager = DatabaseManager(db_url=self.db_url)
            logger.info(f"Initialized PostgreSQL database manager with {self.db_url}")
        else:
            self.db_manager = None
            logger.warning("No PostgreSQL database URL found. Analysis results won't be persisted.")
        
        # Initialize Tree-Sitter if possible
        self.tree_sitter_available = import_error is None
        
        # Check if the tree-sitter-solidity.so file exists
        tree_sitter_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree_sitter_languages", "solidity.so")
        if not os.path.exists(tree_sitter_lib_path):
            logger.warning(f"Tree-sitter library not found at {tree_sitter_lib_path}")
            tree_sitter_lib_path = "libtree-sitter-solidity.so"  # Look in system path
        
        if self.tree_sitter_available:
            try:
                # Monkey patch the language function if needed
                if 'tree_sitter_solidity' not in sys.modules:
                    import types
                    
                    # Create a dummy language module
                    tree_sitter_solidity = types.ModuleType('tree_sitter_solidity')
                    
                    # Create a language function that returns the path
                    def language_func():
                        return tree_sitter_lib_path
                    
                    # Assign the function to the module
                    tree_sitter_solidity.language = language_func
                    
                    # Add the module to sys.modules
                    sys.modules['tree_sitter_solidity'] = tree_sitter_solidity
                
                # Initialize the Tree-Sitter parser
                self.ts = tsg()
                logger.info("Tree-Sitter initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Tree-Sitter: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.tree_sitter_available = False
        
        if not self.tree_sitter_available:
            logger.warning("Tree-Sitter not available. Will use basic regex-based analysis.")
    
    async def create_tables(self):
        """Create necessary database tables."""
        if self.db_manager:
            try:
                await self.db_manager.create_tables()
                logger.info("Database tables created")
            except Exception as e:
                logger.error(f"Error creating tables: {e}")
    
    async def analyze_contract(self, file_path: str, project_id: str, query: str) -> Dict[str, Any]:
        """
        Analyze a Solidity contract.
        
        Args:
            file_path: Path to the Solidity file
            project_id: Project ID
            query: Analysis query
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing contract: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # Parse contract using Tree-Sitter
            ast_info = {}
            flows = []
            dot_graph = None
            
            if self.tree_sitter_available:
                # Use Tree-Sitter for parsing
                logger.info("Using Tree-Sitter for parsing")
                
                # Parse the contract
                for node in self.ts.parse_sol([file_path], 
                                             treeSitterScm=sitterQL.flowTrack(), 
                                             GenerateDOTTy=True):
                    # Extract AST information for interesting nodes
                    if hasattr(node, 'type'):
                        if node.type in ['contract_declaration', 'function_definition', 'assignment_expression']:
                            node_text = tsg.read_node(self.ts.lineArr, node)
                            if node_text:
                                if node.type not in ast_info:
                                    ast_info[node.type] = []
                                ast_info[node.type].append({
                                    'start': node.start_point, 
                                    'end': node.end_point,
                                    'text': node_text
                                })
                
                # Extract flow information
                for key, nodes in ast_info.items():
                    if key == 'assignment_expression':
                        for node in nodes:
                            flows.append({
                                'type': 'state_change',
                                'text': node['text'],
                                'location': f"Line {node['start'][0]+1}"
                            })
                    elif key == 'function_definition':
                        for node in nodes:
                            # Check for external calls
                            if any(pattern in node['text'] for pattern in ['.transfer(', '.send(', '.call{']):
                                flows.append({
                                    'type': 'external_call',
                                    'text': node['text'],
                                    'location': f"Line {node['start'][0]+1}"
                                })
                
                # Get DOT graph
                dot_graph = self.ts.dotGraphBuffer.getvalue()
            
            # Read contract source
            with open(file_path, 'r', encoding='utf-8') as f:
                contract_source = f.read()
            
            # Create prompt for primary analysis
            prompt = f"""Analyze the following Solidity smart contract for security vulnerabilities.

Contract:
```solidity
{contract_source}
```

Focus on these specific aspects:
{query}
"""
            
            # Add flow information to prompt if available
            if flows:
                prompt += "\nFlow Analysis Results:\n"
                for i, flow in enumerate(flows):
                    prompt += f"\n{i+1}. {flow['type']} at {flow['location']}: {flow['text'][:100]}..."
            
            prompt += """

Provide a detailed analysis of the contract. For each vulnerability you find, include:
1. Title and severity (Critical, High, Medium, Low)
2. Detailed description of the vulnerability
3. The exact location in the code (line numbers if possible)
4. How the vulnerability could be exploited
5. Recommended fixes

Format your response as a valid JSON with the following structure:
{
  "summary": "Overall assessment of the contract security",
  "findings": [
    {
      "title": "Title of the vulnerability",
      "severity": "Critical/High/Medium/Low",
      "description": "Detailed description",
      "location": "Line numbers or function name",
      "exploitation": "How it could be exploited",
      "recommendation": "Suggested fix"
    }
  ]
}

IMPORTANT: Your response must be valid JSON. Do not include any other text or explanations outside the JSON.
"""
            
            # Run analysis with LLM
            logger.info("Running LLM analysis")
            response = await self.primary_llm.query(prompt=prompt, temperature=0.1, json=True)
            
            # Parse LLM response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON")
                # Try to extract JSON from response
                import re
                json_pattern = r'```json(.*?)```|```(.*?)```|({.*})'
                match = re.search(json_pattern, response, re.DOTALL)
                if match:
                    json_str = match.group(1) or match.group(2) or match.group(3)
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        result = {
                            "summary": "Failed to parse analysis results",
                            "findings": [],
                            "error": "LLM response was not valid JSON"
                        }
                else:
                    result = {
                        "summary": "Failed to parse analysis results",
                        "findings": [],
                        "error": "LLM response was not valid JSON"
                    }
            
            # Generate report
            report_path = self._generate_report(file_path, result, dot_graph)
            
            # Store results in database if available
            if self.db_manager:
                await self._store_results(file_path, project_id, result)
            
            # Add report path to result
            result["report_path"] = report_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing contract: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _generate_report(self, file_path: str, result: Dict[str, Any], dot_graph: str = None) -> str:
        """Generate a report from analysis results."""
        # Create timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(file_path).split(".")[0]
        
        # Create report paths
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = os.path.join(reports_dir, f"{filename}_report_{timestamp}.md")
        results_path = os.path.join(reports_dir, f"{filename}_results_{timestamp}.json")
        
        # Create graph path if dot_graph is available
        graph_path = None
        if dot_graph:
            graph_path = os.path.join(reports_dir, f"{filename}_graph_{timestamp}.html")
            
            # Create HTML visualization of the DOT graph
            with open(graph_path, "w") as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Contract Graph: {filename}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/@hpcc-js/wasm@0.3.14/dist/index.min.js"></script>
    <script src="https://unpkg.com/d3-graphviz@4.0.0/build/d3-graphviz.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #333; }}
        #graph {{ width: 100%; height: 800px; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>Contract Structure: {filename}</h1>
    <div id="graph"></div>
    <script>
        const dotSource = `{dot_graph}`;
        d3.select("#graph").graphviz()
            .fade(false)
            .fit(true)
            .renderDot(dotSource);
    </script>
</body>
</html>""")
        
        # Write markdown report
        with open(report_path, "w") as f:
            f.write(f"# Security Analysis Report: {filename}\n\n")
            f.write(f"## Summary\n\n{result.get('summary', 'No summary provided.')}\n\n")
            
            # Add link to graph if available
            if graph_path:
                rel_graph_path = os.path.relpath(graph_path, os.path.dirname(report_path))
                f.write(f"[View Contract Structure Graph]({rel_graph_path})\n\n")
            
            f.write(f"## Findings\n\n")
            
            for i, finding in enumerate(result.get("findings", []), 1):
                severity = finding.get("severity", "Unknown")
                severity_emoji = {
                    "Critical": "🔴", 
                    "High": "🟠", 
                    "Medium": "🟡", 
                    "Low": "🟢",
                    "Unknown": "⚪"
                }.get(severity, "⚪")
                
                f.write(f"### {i}. {severity_emoji} {finding.get('title')} ({severity})\n\n")
                f.write(f"**Location:** {finding.get('location', 'Unknown')}\n\n")
                f.write(f"**Description:**\n{finding.get('description', 'No description provided.')}\n\n")
                
                if finding.get("exploitation"):
                    f.write(f"**Exploitation:**\n{finding.get('exploitation')}\n\n")
                
                if finding.get("recommendation"):
                    f.write(f"**Recommendation:**\n{finding.get('recommendation')}\n\n")
                
                f.write("---\n\n")
            
            f.write(f"\n\n*Report generated by Finite Monkey Engine at {timestamp}*\n")
        
        # Write JSON results
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Report generated: {report_path}")
        if graph_path:
            logger.info(f"Graph generated: {graph_path}")
        
        return report_path
    
    async def _store_results(self, file_path: str, project_id: str, result: Dict[str, Any]) -> None:
        """Store analysis results in the database."""
        if not self.db_manager:
            return
        
        try:
            # Get or create project
            project = await self.db_manager.get_project(project_id)
            if not project:
                project = await self.db_manager.create_project(
                    project_id=project_id,
                    name=os.path.basename(os.path.dirname(file_path)),
                    description=f"Project containing {os.path.basename(file_path)}"
                )
            
            # Add file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            file = await self.db_manager.add_file(
                project_id=project_id,
                file_path=os.path.basename(file_path),
                content=content
            )
            
            # Create audit
            audit = await self.db_manager.create_audit(
                project_id=project_id,
                file_path=os.path.basename(file_path),
                query="Security analysis",
                model_name=self.primary_model
            )
            
            # Save analysis results
            await self.db_manager.save_analysis_results(
                audit_id=audit.id,
                summary=result.get("summary", ""),
                findings=result.get("findings", []),
                results=result
            )
            
            # Update audit status
            await self.db_manager.update_audit_status(
                audit_id=audit.id,
                status="completed",
                completed=True
            )
            
            logger.info(f"Analysis results stored in database for {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error storing results in database: {e}")
            import traceback
            logger.error(traceback.format_exc())

async def main():
    """Main function to run the analyzer."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the simple analyzer on a Solidity contract")
    parser.add_argument("file", help="Path to the Solidity file or directory to analyze")
    parser.add_argument("--project", help="Project ID", default="default")
    parser.add_argument("--query", help="Analysis query", default="Analyze for common vulnerabilities")
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SimpleAnalyzer()
    
    # Create tables
    await analyzer.create_tables()
    
    # Analyze file or directory
    file_path = args.file
    if os.path.isdir(file_path):
        logger.info(f"Analyzing directory: {file_path}")
        
        # Find all Solidity files (limited to main contracts for efficiency)
        sol_files = []
        
        # Check if there's a main contract file (like LiFiDiamond.sol)
        main_contract = None
        for root, _, files in os.walk(file_path):
            for file in files:
                if file.endswith(".sol") and not file.startswith("I"):
                    # Skip interface files for efficiency
                    if "Diamond" in file or "Main" in file or "Core" in file:
                        main_contract = os.path.join(root, file)
                        break
            if main_contract:
                break
        
        if main_contract:
            logger.info(f"Found main contract: {main_contract}")
            sol_files.append(main_contract)
        
        # Add any non-interface implementation files
        for root, _, files in os.walk(file_path):
            for file in files:
                if file.endswith(".sol") and not file.startswith("I") and "Facet" in file:
                    file_path = os.path.join(root, file)
                    if file_path != main_contract:  # Avoid duplicate
                        sol_files.append(file_path)
        
        # If we have too many files, limit to a reasonable number
        if len(sol_files) > 10:
            logger.info(f"Found {len(sol_files)} Solidity files, limiting analysis to first 10 files")
            sol_files = sol_files[:10]
        else:
            logger.info(f"Found {len(sol_files)} Solidity files")
        
        # Analyze each file
        for file in sol_files:
            result = await analyzer.analyze_contract(file, args.project, args.query)
            
            if "error" in result:
                logger.error(f"Error analyzing {file}: {result['error']}")
            else:
                logger.info(f"Analysis completed for {file}")
                logger.info(f"Report: {result.get('report_path')}")
                
                # Print findings
                findings = result.get("findings", [])
                if findings:
                    logger.info(f"Found {len(findings)} issues:")
                    for i, finding in enumerate(findings, 1):
                        logger.info(f"{i}. {finding.get('title')} (Severity: {finding.get('severity')})")
                else:
                    logger.info("No issues found")
    else:
        logger.info(f"Analyzing file: {file_path}")
        
        result = await analyzer.analyze_contract(file_path, args.project, args.query)
        
        if "error" in result:
            logger.error(f"Error analyzing {file_path}: {result['error']}")
        else:
            logger.info(f"Analysis completed for {file_path}")
            logger.info(f"Report: {result.get('report_path')}")
            
            # Print findings
            findings = result.get("findings", [])
            if findings:
                logger.info(f"Found {len(findings)} issues:")
                for i, finding in enumerate(findings, 1):
                    logger.info(f"{i}. {finding.get('title')} (Severity: {finding.get('severity')})")
            else:
                logger.info("No issues found")

if __name__ == "__main__":
    asyncio.run(main())