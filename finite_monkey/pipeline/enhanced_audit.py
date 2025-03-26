"""
Enhanced audit pipeline for Solidity smart contracts

This module provides a comprehensive audit pipeline that includes:
1. Project preparation and parsing
2. Chunking and analysis with LLMs
3. Validation of findings
4. Visualization of contract relationships
5. Report generation
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from finite_monkey.nodes_config import config
from ..utils.logger import logger
from ..visualizer.graph_factory import GraphFactory
from ..utils.chunking import AsyncContractChunker
from ..core_async_analyzer import AsyncAnalyzer

async def prepare_project(project_path: str) -> Dict[str, Any]:
    """
    Prepare project for analysis
    
    Args:
        project_path: Path to project directory
        
    Returns:
        Project metadata and file inventory
    """
    # No more need to call config() as it's now directly imported
    output_dir = os.path.join(project_path, config.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find Solidity files
    solidity_files = []
    for root, _, files in os.walk(project_path):
        # Skip ignored folders
        if any(ignored in root for ignored in config.IGNORE_FOLDERS.split(',')):
            continue
            
        for file in files:
            if file.endswith('.sol'):
                solidity_files.append(os.path.join(root, file))
    
    # Create project metadata
    project_name = os.path.basename(os.path.normpath(project_path))
    project_id = config.id or project_name
    
    project_metadata = {
        "project_id": project_id,
        "project_name": project_name,
        "file_count": len(solidity_files),
        "files": [{"path": f, "name": os.path.basename(f)} for f in solidity_files],
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save metadata
    with open(os.path.join(output_dir, "project_metadata.json"), "w") as f:
        json.dump(project_metadata, f, indent=2)
    
    logger.info(f"Found {len(solidity_files)} Solidity files in project {project_name}")
    return project_metadata

async def analyze_with_chunking(
    project_metadata: Dict[str, Any], 
    analyzer: AsyncAnalyzer,
    max_concurrent: int = 4
) -> Dict[str, Any]:
    """
    Analyze project with chunking strategy
    
    Args:
        project_metadata: Project metadata from prepare_project()
        analyzer: AsyncAnalyzer instance
        max_concurrent: Maximum number of concurrent analyses
        
    Returns:
        Analysis results
    """
    solidity_files = [f["path"] for f in project_metadata["files"]]
    project_id = project_metadata["project_id"]
    output_dir = project_metadata["output_dir"]
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Initialize chunker
    chunker = AsyncContractChunker(
        max_chunk_size=8000,  # Default size that works well with most LLMs
        overlap_size=500,
        chunk_by_contract=True,
        include_call_graph=True
    )
    
    # Initialize call graph
    chunker.initialize_call_graph(os.path.dirname(solidity_files[0]))
    
    async def analyze_file_with_semaphore(file_path: str):
        async with semaphore:
            return await analyzer.analyze_file_with_chunking(
                file_path=file_path,
                query=config.USER_QUERY,
                project_id=project_id,
                max_chunk_size=chunker.max_chunk_size,
                include_call_graph=True
            )
    
    # Analyze each file with controlled concurrency
    tasks = []
    for file_path in solidity_files:
        tasks.append(analyze_file_with_semaphore(file_path))
    
    results = await asyncio.gather(*tasks)
    
    # Combine results
    combined_results = {
        "project_id": project_id,
        "file_count": len(solidity_files),
        "files": {os.path.basename(solidity_files[i]): result for i, result in enumerate(results)},
        "timestamp": datetime.now().isoformat()
    }
    
    # Extract and aggregate findings
    findings = []
    for file_result in results:
        if "final_report" in file_result and "findings" in file_result["final_report"]:
            for finding in file_result["final_report"]["findings"]:
                finding["file"] = file_result.get("file_id", "unknown")
                findings.append(finding)
    
    combined_results["findings"] = findings
    combined_results["finding_count"] = len(findings)
    
    # Group findings by severity
    severity_counts = {}
    for finding in findings:
        severity = finding.get("severity", "Unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    combined_results["severity_summary"] = severity_counts
    
    # Save results
    with open(os.path.join(output_dir, "analysis_results.json"), "w") as f:
        json.dump(combined_results, f, indent=2)
    
    return combined_results

async def validate_findings(
    analysis_results: Dict[str, Any],
    analyzer: AsyncAnalyzer
) -> Dict[str, Any]:
    """
    Validate findings with secondary LLM
    
    Args:
        analysis_results: Results from analyze_with_chunking()
        analyzer: AsyncAnalyzer instance
        
    Returns:
        Validation results
    """
    findings = analysis_results.get("findings", [])
    output_dir = analysis_results.get("output_dir", ".")
    
    if not findings:
        logger.warning("No findings to validate")
        return {"validated_findings": [], "confirmation_stats": {}}
    
    # Group findings by file for efficient validation
    findings_by_file = {}
    for finding in findings:
        file_id = finding.get("file", "unknown")
        if file_id not in findings_by_file:
            findings_by_file[file_id] = []
        findings_by_file[file_id].append(finding)
    
    # Validate findings for each file
    validated_findings = []
    validation_stats = {
        "confirmed": 0,
        "false_positive": 0,
        "needs_info": 0,
        "total": 0
    }
    
    for file_id, file_findings in findings_by_file.items():
        # Get file data from analysis results
        file_data = analysis_results["files"].get(file_id, {})
        contract_data = file_data.get("contract_data", {})
        
        if not contract_data:
            logger.warning(f"Missing contract data for file {file_id}, skipping validation")
            continue
        
        # Prepare findings for validation
        file_validation = await analyzer._run_secondary_validation(
            contract_data=contract_data,
            primary_analysis={"findings": file_findings},
            expressions=[],  # No test expressions needed for validation
            flow_data=file_data.get("flow_data", {})
        )
        
        # Add validated findings to overall results
        for validation in file_validation.get("validations", []):
            validation["file"] = file_id
            validated_findings.append(validation)
        
        # Update validation stats
        stats = file_validation.get("confirmation_stats", {})
        validation_stats["confirmed"] += stats.get("confirmed", 0)
        validation_stats["false_positive"] += stats.get("false_positive", 0)
        validation_stats["needs_info"] += stats.get("needs_info", 0)
        validation_stats["total"] += stats.get("total", 0)
    
    validation_results = {
        "validated_findings": validated_findings,
        "confirmation_stats": validation_stats
    }
    
    # Save validation results
    with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
        json.dump(validation_results, f, indent=2)
    
    return validation_results

async def generate_visualizations(
    project_metadata: Dict[str, Any],
    analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate visualizations of contract relationships
    
    Args:
        project_metadata: Project metadata
        analysis_results: Analysis results
        
    Returns:
        Paths to generated visualizations
    """
    output_dir = project_metadata["output_dir"]
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Create contract graph factory
    graph_factory = GraphFactory()
    
    # Generate project graph
    project_graph = graph_factory.analyze_solidity_directory(os.path.dirname(project_metadata["files"][0]["path"]))
    project_html = os.path.join(visualization_dir, "project_graph.html")
    project_graph.export_html(project_html)
    
    # Generate individual contract graphs
    contract_htmls = []
    for file_info in project_metadata["files"]:
        file_path = file_info["path"]
        graph = graph_factory.analyze_solidity_file(file_path)
        
        if graph and graph.nodes:  # Only generate visualization if there are nodes
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            html_path = os.path.join(visualization_dir, f"{file_name}_graph.html")
            graph.export_html(html_path)
            contract_htmls.append(html_path)
    
    visualization_results = {
        "project_graph": project_html,
        "contract_graphs": contract_htmls
    }
    
    return visualization_results

async def generate_reports(
    project_metadata: Dict[str, Any],
    analysis_results: Dict[str, Any],
    validation_results: Dict[str, Any],
    visualization_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate final reports
    
    Args:
        project_metadata: Project metadata
        analysis_results: Analysis results
        validation_results: Validation results
        visualization_results: Visualization results
        
    Returns:
        Report paths
    """
    output_dir = project_metadata["output_dir"]
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate HTML report
    html_report = os.path.join(reports_dir, "audit_report.html")
    
    # Generate the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Contract Audit Report: {project_metadata["project_name"]}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            .critical {{ background-color: #ffebee; }}
            .high {{ background-color: #fff8e1; }}
            .medium {{ background-color: #e8f5e9; }}
            .low {{ background-color: #e3f2fd; }}
            .finding {{ margin-bottom: 20px; padding: 10px; border-radius: 5px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-box {{ background-color: #e0e0e0; padding: 15px; border-radius: 5px; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Smart Contract Audit Report</h1>
        <h2>{project_metadata["project_name"]}</h2>
        <p>Date: {datetime.now().strftime("%Y-%m-%d")}</p>
        
        <div class="summary">
            <h3>Summary</h3>
            <p>Files analyzed: {project_metadata["file_count"]}</p>
            <p>Total findings: {analysis_results["finding_count"]}</p>
            
            <div class="stats">
                <div class="stat-box">
                    <h4>Critical</h4>
                    <p>{analysis_results["severity_summary"].get("Critical", 0)}</p>
                </div>
                <div class="stat-box">
                    <h4>High</h4>
                    <p>{analysis_results["severity_summary"].get("High", 0)}</p>
                </div>
                <div class="stat-box">
                    <h4>Medium</h4>
                    <p>{analysis_results["severity_summary"].get("Medium", 0)}</p>
                </div>
                <div class="stat-box">
                    <h4>Low</h4>
                    <p>{analysis_results["severity_summary"].get("Low", 0)}</p>
                </div>
                <div class="stat-box">
                    <h4>Informational</h4>
                    <p>{analysis_results["severity_summary"].get("Informational", 0)}</p>
                </div>
            </div>
            
            <h3>Validation Results</h3>
            <p>Confirmed findings: {validation_results["confirmation_stats"].get("confirmed", 0)}</p>
            <p>False positives: {validation_results["confirmation_stats"].get("false_positive", 0)}</p>
            <p>Needs more information: {validation_results["confirmation_stats"].get("needs_info", 0)}</p>
        </div>
        
        <h3>Findings</h3>
        <table>
            <tr>
                <th>ID</th>
                <th>Title</th>
                <th>Severity</th>
                <th>File</th>
                <th>Status</th>
            </tr>
    """
    
    # Add findings to the table
    validated_findings_dict = {
        f.get("original_finding_id", i): f 
        for i, f in enumerate(validation_results.get("validated_findings", []))
    }
    
    for i, finding in enumerate(analysis_results.get("findings", [])):
        finding_id = i + 1
        title = finding.get("title", "Untitled")
        severity = finding.get("severity", "Unknown")
        file = finding.get("file", "Unknown")
        
        # Get validation status
        validated_finding = validated_findings_dict.get(finding_id)
        status = "Not Validated"
        if validated_finding:
            status = validated_finding.get("confirmation_status", "Not Validated")
            
        status_color = {
            "Confirmed": "#e8f5e9",
            "False Positive": "#ffebee",
            "Needs More Information": "#fff8e1",
            "Not Validated": "#f5f5f5"
        }.get(status, "#f5f5f5")
        
        html_content += f"""
            <tr style="background-color: {status_color}">
                <td>FIND-{finding_id}</td>
                <td>{title}</td>
                <td>{severity}</td>
                <td>{file}</td>
                <td>{status}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h3>Detailed Findings</h3>
    """
    
    # Add detailed findings
    for i, finding in enumerate(analysis_results.get("findings", [])):
        finding_id = i + 1
        title = finding.get("title", "Untitled")
        severity = finding.get("severity", "Unknown")
        description = finding.get("description", "No description")
        location = finding.get("location", "Unknown location")
        impact = finding.get("impact", "No impact information")
        recommendation = finding.get("recommendation", "No recommendation")
        file = finding.get("file", "Unknown")
        
        # Get severity class
        severity_class = severity.lower() if severity.lower() in ["critical", "high", "medium", "low"] else "low"
        
        # Get validation info
        validated_finding = validated_findings_dict.get(finding_id)
        validation_note = ""
        if validated_finding:
            status = validated_finding.get("confirmation_status", "Not Validated")
            reasoning = validated_finding.get("reasoning", "")
            validation_note = f"""
                <div>
                    <h4>Validation: {status}</h4>
                    <p>{reasoning}</p>
                </div>
            """
        
        html_content += f"""
            <div class="finding {severity_class}">
                <h3>FIND-{finding_id}: {title}</h3>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>File:</strong> {file}</p>
                <p><strong>Location:</strong> {location}</p>
                <h4>Description</h4>
                <p>{description}</p>
                <h4>Impact</h4>
                <p>{impact}</p>
                <h4>Recommendation</h4>
                <p>{recommendation}</p>
                {validation_note}
            </div>
        """
    
    # Add visualizations
    html_content += """
        <h3>Visualizations</h3>
        <p>The following visualizations are available:</p>
        <ul>
    """
    
    html_content += f'<li><a href="../visualizations/project_graph.html" target="_blank">Project Overview Graph</a></li>'
    
    for graph_path in visualization_results.get("contract_graphs", []):
        graph_name = os.path.basename(graph_path)
        html_content += f'<li><a href="../visualizations/{graph_name}" target="_blank">{graph_name}</a></li>'
    
    html_content += """
        </ul>
        
        <h3>Conclusion</h3>
        <p>This report was generated using the Finite Monkey Engine enhanced audit pipeline.</p>
        <p>Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body>
    </html>
    """
    
    # Write the HTML report
    with open(html_report, "w") as f:
        f.write(html_content)
    
    # Also generate a markdown report
    md_report = os.path.join(reports_dir, "audit_report.md")
    md_content = f"""# Smart Contract Audit Report: {project_metadata["project_name"]}

Date: {datetime.now().strftime("%Y-%m-%d")}

## Summary

- Files analyzed: {project_metadata["file_count"]}
- Total findings: {analysis_results["finding_count"]}
- Critical: {analysis_results["severity_summary"].get("Critical", 0)}
- High: {analysis_results["severity_summary"].get("High", 0)}
- Medium: {analysis_results["severity_summary"].get("Medium", 0)}
- Low: {analysis_results["severity_summary"].get("Low", 0)}
- Informational: {analysis_results["severity_summary"].get("Informational", 0)}

### Validation Results

- Confirmed findings: {validation_results["confirmation_stats"].get("confirmed", 0)}
- False positives: {validation_results["confirmation_stats"].get("false_positive", 0)}
- Needs more information: {validation_results["confirmation_stats"].get("needs_info", 0)}

## Findings Summary

| ID | Title | Severity | File | Status |
|----|-------|----------|------|--------|
"""
    
    # Add findings to the table
    for i, finding in enumerate(analysis_results.get("findings", [])):
        finding_id = i + 1
        title = finding.get("title", "Untitled")
        severity = finding.get("severity", "Unknown")
        file = finding.get("file", "Unknown")
        
        # Get validation status
        validated_finding = validated_findings_dict.get(finding_id)
        status = "Not Validated"
        if validated_finding:
            status = validated_finding.get("confirmation_status", "Not Validated")
            
        md_content += f"| FIND-{finding_id} | {title} | {severity} | {file} | {status} |\n"
    
    # Generate report summary
    report_results = {
        "html_report": html_report,
        "markdown_report": md_report,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save markdown report
    with open(md_report, "w") as f:
        f.write(md_content)
    
    return report_results

async def create_enhanced_audit_pipeline(project_path: str) -> Dict[str, Any]:
    """
    Create and execute the enhanced audit pipeline
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Pipeline results
    """
    # Initialize analyzer
    from ..adapters.ollama import AsyncOllamaClient as Ollama
    from ..db.manager import DatabaseManager
    
    # Create DB manager if configured
    db_manager = None
    if config.DATABASE_URL:
        try:
            db_url = config.DATABASE_URL
            if "postgresql:" in db_url and "postgresql+asyncpg:" not in db_url:
                db_url = db_url.replace("postgresql:", "postgresql+asyncpg:")
            db_manager = DatabaseManager(db_url=db_url)
        except Exception as e:
            logger.warning(f"Failed to initialize database: {e}")
    
    # Initialize analyzer with appropriate LLM clients
    try:
        primary_llm = Ollama(model=config.SCAN_MODEL)
        secondary_llm = Ollama(model=config.CONFIRMATION_MODEL)
        
        analyzer = AsyncAnalyzer(
            primary_llm_client=primary_llm,
            secondary_llm_client=secondary_llm,
            db_manager=db_manager,
            primary_model_name=config.SCAN_MODEL,
            secondary_model_name=config.CONFIRMATION_MODEL
        )
    except Exception as e:        
        logger.error(f"Failed to initialize analyzer: {e}")
        
    
    # Step 1: Project preparation
    logger.info("Step 1: Project preparation")
    project_metadata = await prepare_project(project_path)
    
    # Step 2: Analyze with chunking
    logger.info("Step 2: Analyzing with chunking strategy")
    analysis_results = await analyze_with_chunking(
        project_metadata=project_metadata,
        analyzer=analyzer,
        max_concurrent=config.MAX_THREADS_OF_SCAN
    )
    
    # Step 3: Validate findings
    logger.info("Step 3: Validating findings")
    validation_results = await validate_findings(
        analysis_results=analysis_results,
        analyzer=analyzer
    )
    
    # Step 4: Generate visualizations
    logger.info("Step 4: Generating visualizations")
    visualization_results = await generate_visualizations(
        project_metadata=project_metadata,
        analysis_results=analysis_results
    )
    
    # Step 5: Generate reports
    logger.info("Step 5: Generating reports")
    report_results = await generate_reports(
        project_metadata=project_metadata,
        analysis_results=analysis_results,
        validation_results=validation_results,
        visualization_results=visualization_results
    )
    
    # Combine all results
    pipeline_results = {
        "project_metadata": project_metadata,
        "analysis_results": analysis_results,
        "validation_results": validation_results,
        "visualization_results": visualization_results,
        "report_results": report_results,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info("Enhanced audit pipeline completed successfully")
    return pipeline_results
