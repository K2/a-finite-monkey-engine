"""
API routes for the Finite Monkey framework web interface
"""

from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os
import tempfile
import json

from ...agents import WorkflowOrchestrator
from ...db.manager import TaskManager
from ...nodes_config import nodes_config
from .github import GitHubClient, GitHubIssue, GitHubRepoInfo

# Create a router
router = APIRouter(prefix="/api", tags=["api"])

# Global references to components
_orchestrator = None
_task_manager = None

# Models for API requests and responses
class AuditRequest(BaseModel):
    files: List[str]
    query: str
    project_name: Optional[str] = None
    wait_for_completion: bool = False
    
class ConfigUpdateRequest(BaseModel):
    settings: Dict[str, Any]
    
class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    file: Optional[str] = None
    type: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
class TelemetryResponse(BaseModel):
    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    audit_start_time: Optional[str] = None
    audit_end_time: Optional[str] = None
    active_tasks: Dict[str, Dict[str, Any]] = {}
    error: Optional[str] = None
    
# GitHub Integration Models
class GitHubFetchRequest(BaseModel):
    repo_url: str
    
class GitHubCloneRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    commit_sha: Optional[str] = None
    
class GitHubIssueRequest(BaseModel):
    owner: str
    repo: str
    title: str
    body: str
    labels: List[str] = ["security", "vulnerability"]
    
class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    details: Optional[Any] = None

# Dependency to get orchestrator instance
async def get_orchestrator() -> WorkflowOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        config = nodes_config()
        task_manager = await get_task_manager()
        _orchestrator = WorkflowOrchestrator(
            task_manager=task_manager,
            model_name=config.WORKFLOW_MODEL,
            base_dir=config.base_dir,
            db_url=config.ASYNC_DB_URL,
        )
    return _orchestrator

# Dependency to get task manager instance
async def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        config = nodes_config()
        _task_manager = TaskManager(db_url=config.ASYNC_DB_URL)
        await _task_manager.start()
    return _task_manager

# Routes
@router.get("/config", response_model=Dict[str, Any])
async def get_config():
    """Get the current configuration"""
    config = nodes_config()
    # Convert to dict and filter out sensitive values
    config_dict = config.model_dump()
    sensitive_keys = ["API_KEY", "PASSWORD", "SECRET", "TOKEN"]
    
    # Mask sensitive values
    for key in config_dict:
        if any(sensitive in key for sensitive in sensitive_keys):
            if config_dict[key]:  # Only mask non-empty values
                config_dict[key] = "********"
    
    return config_dict

@router.post("/config", response_model=Dict[str, Any])
async def update_config(request: ConfigUpdateRequest):
    """Update the configuration"""
    # This would update the configuration
    # For now, just return the request
    return {"message": "Configuration updated", "updated_settings": request.settings}

@router.post("/audit", response_model=Dict[str, Any])
async def start_audit(
    request: AuditRequest, 
    background_tasks: BackgroundTasks,
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Start a new audit"""
    # If not waiting for completion, run in background task
    if not request.wait_for_completion:
        background_tasks.add_task(
            orchestrator.run_audit_workflow,
            solidity_paths=request.files,
            query=request.query,
            project_name=request.project_name,
            wait_for_completion=True  # We always wait in the background task
        )
        return {
            "message": "Audit started in background",
            "files": request.files,
            "project_name": request.project_name
        }
    
    # Otherwise run and wait for completion
    try:
        result = await orchestrator.run_audit_workflow(
            solidity_paths=request.files,
            query=request.query,
            project_name=request.project_name,
            wait_for_completion=True
        )
        
        # Convert report to dict
        if hasattr(result, "model_dump"):
            return result.model_dump()
        elif hasattr(result, "to_dict"):
            return result.to_dict()
        else:
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks", response_model=List[TaskStatusResponse])
async def get_tasks(
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get all tasks status"""
    # Use metrics from orchestrator
    tasks = []
    for task_id, task_data in orchestrator.metrics.get("active_tasks", {}).items():
        tasks.append(TaskStatusResponse(
            task_id=task_id,
            **task_data
        ))
    return tasks

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Get task status by ID"""
    try:
        task_status = await task_manager.get_task_status(task_id)
        
        # Enrich with any additional info from orchestrator metrics
        orchestrator = await get_orchestrator()
        task_metrics = orchestrator.metrics.get("active_tasks", {}).get(task_id, {})
        
        # Combine status and metrics
        result = {
            "task_id": task_id,
            "status": task_status.get("status", "unknown"),
        }
        
        # Add other fields if available
        for field in ["file", "type", "created_at", "started_at", "completed_at", "error"]:
            if field in task_metrics:
                result[field] = task_metrics[field]
            elif field in task_status:
                result[field] = task_status[field]
                
        return TaskStatusResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Task not found: {str(e)}")

@router.get("/telemetry", response_model=TelemetryResponse)
async def get_telemetry(
    orchestrator: WorkflowOrchestrator = Depends(get_orchestrator)
):
    """Get telemetry data"""
    return TelemetryResponse(**orchestrator.metrics)

# GitHub API Routes
@router.post("/github/fetch", response_model=Union[GitHubRepoInfo, ErrorResponse])
async def fetch_github_repo(request: GitHubFetchRequest):
    """
    Fetch information about a GitHub repository
    """
    try:
        config = nodes_config()
        client = GitHubClient(token=config.GITHUB_TOKEN)
        repo_info = await client.fetch_repo_info(request.repo_url)
        return repo_info
    except Exception as e:
        return ErrorResponse(message=f"Failed to fetch repository: {str(e)}")

@router.post("/github/clone", response_model=Union[Dict[str, Any], ErrorResponse])
async def clone_github_repo(request: GitHubCloneRequest):
    """
    Clone a GitHub repository for analysis
    """
    try:
        config = nodes_config()
        client = GitHubClient(token=config.GITHUB_TOKEN)
        
        # Clone the repository
        repo_dir = await client.clone_repository(
            request.repo_url,
            branch=request.branch,
            commit_sha=request.commit_sha
        )
        
        # Return path info
        return {
            "status": "success",
            "repo_dir": repo_dir,
            "message": f"Repository cloned to {repo_dir}"
        }
    except Exception as e:
        return ErrorResponse(message=f"Failed to clone repository: {str(e)}")

@router.post("/github/issue", response_model=Union[Dict[str, Any], ErrorResponse])
async def create_github_issue(request: GitHubIssueRequest):
    """
    Create a new issue in a GitHub repository
    """
    try:
        config = nodes_config()
        if not config.GITHUB_TOKEN:
            return ErrorResponse(message="GitHub token not configured. Set GITHUB_TOKEN in configuration.")
            
        client = GitHubClient(token=config.GITHUB_TOKEN)
        
        # Create the issue
        issue = GitHubIssue(
            title=request.title,
            body=request.body,
            labels=request.labels
        )
        
        result = await client.create_issue(request.owner, request.repo, issue)
        
        return {
            "status": "success",
            "issue_url": result.get("html_url"),
            "issue_number": result.get("number"),
            "message": f"Issue created: {result.get('html_url')}"
        }
    except Exception as e:
        return ErrorResponse(message=f"Failed to create issue: {str(e)}")

# Reports API
@router.get("/reports", response_model=List[Dict[str, Any]])
async def get_reports():
    """
    Get a list of all analysis reports
    """
    try:
        # Get reports directory
        reports_dir = os.path.join(os.getcwd(), "reports")
        if not os.path.exists(reports_dir):
            return []
            
        # Find all files in the reports directory
        reports = []
        
        # Process JSON result files first for detailed metadata
        json_files = [f for f in os.listdir(reports_dir) if f.endswith("_results.json")]
        for file in json_files:
            try:
                # Extract report ID from filename
                report_id = file.replace("_results.json", "")
                
                # Read report data
                with open(os.path.join(reports_dir, file), "r") as f:
                    report_data = json.load(f)
                    
                # Add report ID and file path
                report_data["id"] = report_id
                report_data["file_path"] = os.path.join(reports_dir, file)
                report_data["file_name"] = file
                report_data["format"] = "json"
                
                # Check for corresponding Markdown or HTML report
                md_path = os.path.join(reports_dir, f"{report_id}_report.md")
                html_path = os.path.join(reports_dir, f"{report_id}_graph.html")
                
                if os.path.exists(md_path):
                    report_data["has_markdown"] = True
                    report_data["markdown_path"] = md_path
                else:
                    report_data["has_markdown"] = False
                
                if os.path.exists(html_path):
                    report_data["has_graph"] = True
                    report_data["graph_path"] = html_path
                else:
                    report_data["has_graph"] = False
                
                reports.append(report_data)
            except Exception as e:
                print(f"Error reading report {file}: {e}")
                
        # Check for markdown files without corresponding JSON
        md_files = [f for f in os.listdir(reports_dir) if f.endswith(".md") and not f.endswith("_report.md")]
        for file in md_files:
            try:
                # Generate a simple report entry for this file
                report_id = os.path.splitext(file)[0]
                file_path = os.path.join(reports_dir, file)
                
                # Get file stats for timestamp
                stats = os.stat(file_path)
                created_time = stats.st_mtime
                
                # Read first few lines to extract project name or title
                with open(file_path, "r") as f:
                    first_line = f.readline().strip()
                    project_name = first_line.replace("#", "").strip()
                
                # Create a report entry
                report_data = {
                    "id": report_id,
                    "file_path": file_path,
                    "file_name": file,
                    "format": "markdown",
                    "project_name": project_name,
                    "timestamp": created_time,
                    "has_markdown": True,
                    "markdown_path": file_path,
                    "has_graph": False
                }
                
                reports.append(report_data)
            except Exception as e:
                print(f"Error processing markdown file {file}: {e}")
        
        # Sort by timestamp (newest first)
        reports.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return reports
    except Exception as e:
        print(f"Error loading reports: {e}")
        return []

@router.get("/reports/{report_id}", response_model=Dict[str, Any])
async def get_report(report_id: str):
    """
    Get a specific analysis report by ID
    """
    try:
        # Find the report file
        report_path = os.path.join(os.getcwd(), "reports", f"{report_id}_results.json")
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
            
        # Read report data
        with open(report_path, "r") as f:
            report_data = json.load(f)
            
        # Add report ID
        report_data["id"] = report_id
        
        return report_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading report: {str(e)}")

@router.get("/reports/{report_id}/view", response_class=HTMLResponse)
async def view_report(report_id: str):
    """
    View a report in HTML format
    """
    try:
        # Find the report files
        reports_dir = os.path.join(os.getcwd(), "reports")
        
        # Check if the requested report is a specific file or a report ID
        if report_id.endswith(".md"):
            # Direct file access
            md_path = os.path.join(reports_dir, report_id)
            report_title = os.path.splitext(report_id)[0]
        else:
            # Report ID - check for report files
            md_path = os.path.join(reports_dir, f"{report_id}_report.md")
            report_title = report_id
        
        json_path = os.path.join(reports_dir, f"{report_id}_results.json")
        graph_path = os.path.join(reports_dir, f"{report_id}_graph.html")
        
        # Initialize variables
        content = ""
        report_data = {}
        has_graph = os.path.exists(graph_path)
        has_json = os.path.exists(json_path)
        
        # Try to load report data
        if has_json:
            try:
                with open(json_path, "r") as f:
                    report_data = json.load(f)
            except:
                pass
        
        # Check if we have any report files
        if not (os.path.exists(md_path) or has_json):
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
            
        # Read the markdown content if available
        if os.path.exists(md_path):
            try:
                with open(md_path, "r") as f:
                    content = f.read()
                    
                # More sophisticated markdown to HTML conversion
                # Use Python's markdown module if available, otherwise fallback to basic conversion
                try:
                    import markdown
                    md = markdown.Markdown(extensions=['tables', 'fenced_code'])
                    content = md.convert(content)
                except ImportError:
                    # Basic markdown to HTML conversion
                    content = content.replace("\n", "<br>")
                    content = content.replace("# ", "<h1>").replace("<br>", "</h1>", 1)
                    content = content.replace("## ", "<h2>").replace("<br>", "</h2>", 1)
                    content = content.replace("### ", "<h3>").replace("<br>", "</h3>", 1)
                    content = content.replace("**", "<strong>").replace("**", "</strong>")
                    content = content.replace("*", "<em>").replace("*", "</em>")
                    
                    # Code blocks
                    content = content.replace("```", "<pre><code>").replace("```", "</code></pre>")
                    
                    # Lists
                    content = content.replace("- ", "• ")
                    
                    # Tables (very basic)
                    content = content.replace("| ", "<td>").replace(" |", "</td>")
                    content = content.replace("<td></td>", "<tr>").replace("</td><td>", "</td><tr><td>")
            except Exception as e:
                content = f"<p>Error parsing markdown: {str(e)}</p><pre>{content}</pre>"
        elif has_json:
            # Create a summary view of the JSON data
            content = "<h2>Report Summary</h2>"
            
            if "project_name" in report_data:
                content += f"<p><strong>Project:</strong> {report_data['project_name']}</p>"
                
            if "timestamp" in report_data:
                try:
                    from datetime import datetime
                    ts = report_data['timestamp']
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts)
                        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        formatted_date = ts
                    content += f"<p><strong>Date:</strong> {formatted_date}</p>"
                except:
                    content += f"<p><strong>Date:</strong> {report_data['timestamp']}</p>"
            
            if "files_analyzed" in report_data:
                files = report_data['files_analyzed']
                if isinstance(files, list):
                    content += "<p><strong>Files Analyzed:</strong></p><ul>"
                    for file in files:
                        content += f"<li>{file}</li>"
                    content += "</ul>"
                else:
                    content += f"<p><strong>Files Analyzed:</strong> {files}</p>"
            
            if "issues" in report_data and report_data["issues"]:
                content += f"<h2>Issues Found ({len(report_data['issues'])})</h2>"
                
                for i, issue in enumerate(report_data["issues"], 1):
                    severity_class = "info"
                    if "severity" in issue:
                        severity = issue["severity"].lower()
                        if severity in ["high", "critical"]:
                            severity_class = "high"
                        elif severity in ["medium"]:
                            severity_class = "medium"
                        elif severity in ["low"]:
                            severity_class = "low"
                    
                    content += f"""
                    <div class="issue issue-{severity_class}">
                        <h3>{i}. {issue.get('title', 'Unnamed Issue')}</h3>
                        <p><strong>Severity:</strong> {issue.get('severity', 'Unknown')}</p>
                    """
                    
                    if "description" in issue:
                        content += f"<p>{issue['description']}</p>"
                        
                    if "location" in issue:
                        content += f"<p><strong>Location:</strong> {issue['location']}</p>"
                        
                    if "recommendation" in issue:
                        content += f"<p><strong>Recommendation:</strong> {issue['recommendation']}</p>"
                        
                    content += "</div>"
            else:
                content += "<p>No issues found in this report.</p>"
        else:
            content = "<p>No report content available.</p>"
            
        # Determine title
        title = report_title
        if "project_name" in report_data:
            title = report_data["project_name"]
            
        # Build the HTML page with the content
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Report: {title} - Finite Monkey Engine</title>
                <link rel="stylesheet" href="/static/css/main.css">
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .report-header {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 20px;
                        border-bottom: 1px solid #eee;
                        padding-bottom: 10px;
                    }}
                    .report-nav {{
                        display: flex;
                        gap: 10px;
                    }}
                    .report-nav a {{
                        display: inline-block;
                        padding: 5px 10px;
                        background-color: #f5f5f5;
                        text-decoration: none;
                        color: #333;
                        border-radius: 3px;
                    }}
                    .report-nav a:hover {{
                        background-color: #e0e0e0;
                    }}
                    .report-content {{
                        background-color: #fff;
                        padding: 20px;
                        border-radius: 5px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    pre {{
                        background-color: #f8f8f8;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                    code {{
                        font-family: 'Courier New', monospace;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 15px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    .issue {{
                        margin-bottom: 20px;
                        padding: 15px;
                        border-radius: 5px;
                    }}
                    .issue-high {{
                        background-color: #ffeeee;
                        border-left: 5px solid #dc3545;
                    }}
                    .issue-medium {{
                        background-color: #fff8ee;
                        border-left: 5px solid #fd7e14;
                    }}
                    .issue-low {{
                        background-color: #fffbee;
                        border-left: 5px solid #ffc107;
                    }}
                    .issue-info {{
                        background-color: #eef8ff;
                        border-left: 5px solid #17a2b8;
                    }}
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>{title}</h1>
                    <div class="report-nav">
                        <a href="/reports">Back to Reports</a>
                        {f'<a href="/reports/{report_id}/download">Download JSON</a>' if has_json else ''}
                        {f'<a href="/reports/{report_id}_graph.html" target="_blank">View Graph</a>' if has_graph else ''}
                    </div>
                </div>
                <div class="report-content">
                    {content}
                </div>
            </body>
        </html>
        """)
    except HTTPException:
        raise
    except Exception as e:
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Error - Finite Monkey Engine</title>
                <link rel="stylesheet" href="/static/css/main.css">
            </head>
            <body>
                <h1>Error Loading Report</h1>
                <p>{str(e)}</p>
                <p><a href="/reports">Back to Reports</a></p>
            </body>
        </html>
        """)

@router.get("/reports/{report_id}/download")
async def download_report(report_id: str):
    """
    Download a report in JSON format
    """
    try:
        # Find the report file
        report_path = os.path.join(os.getcwd(), "reports", f"{report_id}_results.json")
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
            
        # Read report data
        with open(report_path, "r") as f:
            report_data = json.load(f)
            
        # Return as downloadable JSON
        return JSONResponse(
            content=report_data,
            headers={
                "Content-Disposition": f"attachment; filename={report_id}_results.json"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading report: {str(e)}")

# Files API for GitHub integration
@router.get("/files", response_model=Dict[str, Any])
async def get_files(path: str):
    """
    Get file and directory listing for a path
    """
    try:
        if not os.path.exists(path):
            return {
                "status": "error",
                "message": f"Path {path} does not exist"
            }
            
        # Build file tree recursively
        tree = {}
        
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            
            if os.path.isdir(item_path):
                # For directories, recursively scan (but limit depth to avoid huge responses)
                if item not in [".git", "node_modules", ".github"]:
                    tree[item] = get_directory_structure(item_path, max_depth=3)
            else:
                # For files, just add as leaf nodes
                tree[item] = True
        
        return {
            "status": "success",
            "files": tree
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading directory: {str(e)}"
        }

def get_directory_structure(path, max_depth=3, current_depth=0):
    """
    Recursively build directory structure
    """
    if current_depth >= max_depth:
        return {"...": True}  # Indicate truncated structure
        
    result = {}
    
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            
            if os.path.isdir(item_path):
                if item not in [".git", "node_modules", ".github"]:
                    result[item] = get_directory_structure(
                        item_path, 
                        max_depth=max_depth, 
                        current_depth=current_depth + 1
                    )
            else:
                result[item] = True
    except Exception:
        pass
        
    return result

# Error tracking API
@router.post("/errors", response_model=Dict[str, Any])
async def log_error(request: Request):
    """
    Log a client-side error for debugging
    """
    try:
        # Parse the error data from the request
        error_data = await request.json()
        
        # Create a log file if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", "client_errors.json")
        
        # Read existing errors
        existing_errors = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    existing_errors = json.load(f)
            except:
                pass
            
        # Add the new error
        existing_errors.append({
            "timestamp": error_data.get("timestamp", "unknown"),
            "message": error_data.get("message", "unknown"),
            "stack": error_data.get("stack", ""),
            "user_agent": error_data.get("user_agent", ""),
            "url": error_data.get("url", "")
        })
        
        # Write the updated error log
        with open(log_path, "w") as f:
            json.dump(existing_errors, f, indent=2)
            
        return {
            "status": "success",
            "message": "Error logged successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to log error: {str(e)}"
        }