"""
Main web application for the Finite Monkey framework.

This module provides a FastAPI-based web interface for monitoring, configuration,
and debugging of the Finite Monkey framework.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from ..agents import WorkflowOrchestrator
from ..db.manager import TaskManager
from ..nodes_config import nodes_config
from .api.routes import router as api_router, get_orchestrator
from .ui.ipython_terminal import AsyncIPythonBridge


# Create FastAPI application
app = FastAPI(
    title="Finite Monkey Web Interface",
    description="Web interface for the Finite Monkey framework",
    version="0.1.0",
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templating
from . import TEMPLATE_DIR, STATIC_DIR

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include API routes
app.include_router(api_router)

# Terminal connections
terminals: Dict[str, AsyncIPythonBridge] = {}
terminal_connections: Dict[str, List[WebSocket]] = {}

async def get_or_create_terminal(terminal_id: str, orchestrator: WorkflowOrchestrator = None) -> AsyncIPythonBridge:
    """Get or create a terminal with the given ID."""
    global terminals
    
    if terminal_id not in terminals:
        # Create namespace with important components
        namespace = {}
        
        if orchestrator is None:
            orchestrator = await get_orchestrator()
        
        # Add components to namespace
        namespace.update({
            "orchestrator": orchestrator,
            "task_manager": orchestrator.task_manager,
            "researcher": orchestrator.researcher,
            "validator": orchestrator.validator,
            "documentor": orchestrator.documentor,
            "config": nodes_config,
        })
        
        # Create terminal
        terminals[terminal_id] = AsyncIPythonBridge(namespace)
        
        # Start output polling
        await terminals[terminal_id].start_polling()
        
        # Set output callback to broadcast to all connections
        async def broadcast_output(output: str):
            if terminal_id in terminal_connections:
                for connection in terminal_connections[terminal_id]:
                    try:
                        await connection.send_text(output)
                    except Exception:
                        pass
        
        terminals[terminal_id].set_output_callback(broadcast_output)
    
    return terminals[terminal_id]

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint that returns the HTML for the web interface."""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        # Fallback to inline HTML if template not found
        print(f"Error loading template: {e}")
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Finite Monkey Engine</title>
                <link rel="stylesheet" href="/static/css/main.css">
            </head>
            <body>
                <h1>Finite Monkey Engine</h1>
                <p>Web interface for the Smart Contract Security Analysis Framework</p>
                <ul>
                    <li><a href="/config">Configuration</a></li>
                    <li><a href="/telemetry">Telemetry</a></li>
                    <li><a href="/terminal/main">IPython Terminal</a></li>
                    <li><a href="/github">GitHub Repository Analysis</a></li>
                    <li><a href="/reports">Analysis Reports</a></li>
                </ul>
            </body>
        </html>
        """)

@app.get("/config", response_class=HTMLResponse)
async def config_page():
    """Config page that renders a form for updating configuration."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Configuration - Finite Monkey Engine</title>
            <link rel="stylesheet" href="/static/css/main.css">
            <script>
                async function loadConfig() {
                    const response = await fetch('/api/config');
                    const config = await response.json();
                    
                    const configContainer = document.getElementById('config-container');
                    configContainer.innerHTML = '';
                    
                    for (const [key, value] of Object.entries(config)) {
                        const group = document.createElement('div');
                        group.className = 'config-group';
                        
                        const label = document.createElement('label');
                        label.textContent = key;
                        
                        const input = document.createElement('input');
                        input.type = 'text';
                        input.name = key;
                        input.value = value;
                        
                        group.appendChild(label);
                        group.appendChild(input);
                        configContainer.appendChild(group);
                    }
                }
                
                async function saveConfig() {
                    const form = document.getElementById('config-form');
                    const formData = new FormData(form);
                    
                    const settings = {};
                    for (const [key, value] of formData.entries()) {
                        settings[key] = value;
                    }
                    
                    const response = await fetch('/api/config', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ settings })
                    });
                    
                    const result = await response.json();
                    alert('Configuration saved');
                }
                
                window.addEventListener('load', loadConfig);
            </script>
        </head>
        <body>
            <h1>Configuration</h1>
            <form id="config-form">
                <div id="config-container"></div>
                <button type="button" onclick="saveConfig()">Save</button>
            </form>
        </body>
    </html>
    """)

@app.get("/telemetry", response_class=HTMLResponse)
async def telemetry_page():
    """Telemetry page that displays metrics."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Telemetry - Finite Monkey Engine</title>
            <link rel="stylesheet" href="/static/css/main.css">
            <script>
                async function loadTelemetry() {
                    const response = await fetch('/api/telemetry');
                    const telemetry = await response.json();
                    
                    document.getElementById('tasks-created').textContent = telemetry.tasks_created;
                    document.getElementById('tasks-completed').textContent = telemetry.tasks_completed;
                    document.getElementById('tasks-failed').textContent = telemetry.tasks_failed;
                    
                    if (telemetry.audit_start_time) {
                        document.getElementById('audit-start-time').textContent = new Date(telemetry.audit_start_time).toLocaleString();
                    }
                    
                    if (telemetry.audit_end_time) {
                        document.getElementById('audit-end-time').textContent = new Date(telemetry.audit_end_time).toLocaleString();
                    }
                    
                    const tasksContainer = document.getElementById('tasks-container');
                    tasksContainer.innerHTML = '';
                    
                    for (const [taskId, taskData] of Object.entries(telemetry.active_tasks)) {
                        const taskElem = document.createElement('div');
                        taskElem.className = `task task-${taskData.status}`;
                        
                        taskElem.innerHTML = `
                            <div class="task-header">
                                <span class="task-id">${taskId}</span>
                                <span class="task-status">${taskData.status}</span>
                            </div>
                            <div class="task-details">
                                <div class="task-file">${taskData.file || ''}</div>
                                <div class="task-type">${taskData.type || ''}</div>
                                <div class="task-created">${taskData.created_at ? new Date(taskData.created_at).toLocaleString() : ''}</div>
                                ${taskData.completed_at ? `<div class="task-completed">Completed: ${new Date(taskData.completed_at).toLocaleString()}</div>` : ''}
                                ${taskData.failed_at ? `<div class="task-failed">Failed: ${new Date(taskData.failed_at).toLocaleString()}</div>` : ''}
                                ${taskData.error ? `<div class="task-error">Error: ${taskData.error}</div>` : ''}
                            </div>
                        `;
                        
                        tasksContainer.appendChild(taskElem);
                    }
                    
                    // Schedule next update
                    setTimeout(loadTelemetry, 2000);
                }
                
                window.addEventListener('load', loadTelemetry);
            </script>
        </head>
        <body>
            <h1>Telemetry</h1>
            <div class="metrics">
                <div class="metric">
                    <label>Tasks Created:</label>
                    <span id="tasks-created">0</span>
                </div>
                <div class="metric">
                    <label>Tasks Completed:</label>
                    <span id="tasks-completed">0</span>
                </div>
                <div class="metric">
                    <label>Tasks Failed:</label>
                    <span id="tasks-failed">0</span>
                </div>
                <div class="metric">
                    <label>Audit Start Time:</label>
                    <span id="audit-start-time">-</span>
                </div>
                <div class="metric">
                    <label>Audit End Time:</label>
                    <span id="audit-end-time">-</span>
                </div>
            </div>
            <h2>Tasks</h2>
            <div id="tasks-container"></div>
        </body>
    </html>
    """)

@app.get("/github", response_class=HTMLResponse)
async def github_page():
    """GitHub integration page that allows importing repositories."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>GitHub Repository Analysis - Finite Monkey Engine</title>
            <link rel="stylesheet" href="/static/css/main.css">
            <style>
                .repo-form {
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f5f5f5;
                    border-radius: 5px;
                }
                
                .form-group {
                    margin-bottom: 15px;
                }
                
                .form-group label {
                    display: block;
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                
                .form-group input, .form-group select {
                    width: 100%;
                    padding: 8px;
                    box-sizing: border-box;
                }
                
                .repo-info {
                    display: none;
                    margin-top: 20px;
                }
                
                .branch-list, .commit-list {
                    margin-top: 15px;
                }
                
                .branch-item, .commit-item {
                    padding: 10px;
                    margin-bottom: 5px;
                    background-color: #f0f0f0;
                    border-radius: 3px;
                    cursor: pointer;
                }
                
                .branch-item:hover, .commit-item:hover {
                    background-color: #e0e0e0;
                }
                
                .branch-item.selected, .commit-item.selected {
                    background-color: #d4edda;
                    border-left: 4px solid #28a745;
                }
                
                .actions {
                    margin-top: 20px;
                    display: flex;
                    gap: 10px;
                }
                
                button {
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    background-color: #007bff;
                    color: white;
                    cursor: pointer;
                }
                
                button:hover {
                    background-color: #0069d9;
                }
                
                button:disabled {
                    background-color: #cccccc;
                    cursor: not-allowed;
                }
                
                .error-message {
                    color: #dc3545;
                    margin-top: 5px;
                    display: none;
                }
                
                #loading-indicator {
                    display: none;
                    margin-top: 10px;
                }
                
                .file-selector {
                    display: none;
                    margin-top: 20px;
                }
                
                .file-tree {
                    max-height: 300px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    padding: 10px;
                    margin-top: 10px;
                }
                
                .file-item {
                    padding: 5px;
                    cursor: pointer;
                }
                
                .file-item:hover {
                    background-color: #f5f5f5;
                }
                
                .file-item.selected {
                    background-color: #d4edda;
                }
                
                .file-item.directory {
                    font-weight: bold;
                }
                
                .progress-bar-container {
                    width: 100%;
                    height: 20px;
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    margin-top: 10px;
                    overflow: hidden;
                }
                
                .progress-bar {
                    height: 100%;
                    background-color: #4CAF50;
                    width: 0%;
                    transition: width 0.3s;
                }
                
                .selected-files {
                    margin-top: 10px;
                }
                
                .toast {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 15px;
                    background-color: #333;
                    color: white;
                    border-radius: 5px;
                    z-index: 1000;
                    display: none;
                    max-width: 300px;
                }
                
                .toast.error {
                    background-color: #dc3545;
                }
                
                .toast.success {
                    background-color: #28a745;
                }
            </style>
        </head>
        <body>
            <h1>GitHub Repository Analysis</h1>
            <p>Import and analyze code from GitHub repositories</p>
            
            <div class="repo-form">
                <h2>Step 1: Import Repository</h2>
                <div class="form-group">
                    <label for="repo-url">GitHub Repository URL</label>
                    <input type="text" id="repo-url" placeholder="https://github.com/owner/repo">
                    <div class="error-message" id="repo-url-error"></div>
                </div>
                
                <button id="fetch-repo-btn">Fetch Repository Info</button>
                <div id="loading-indicator">Loading...</div>
            </div>
            
            <div class="repo-info" id="repo-info">
                <h2>Step 2: Select Branch and Commit</h2>
                
                <div class="form-group">
                    <label for="branch-select">Branch</label>
                    <select id="branch-select"></select>
                </div>
                
                <div class="commit-list" id="commit-list">
                    <h3>Recent Commits</h3>
                    <!-- Commit items will be added here dynamically -->
                </div>
                
                <button id="clone-repo-btn">Clone Repository</button>
            </div>
            
            <div class="file-selector" id="file-selector">
                <h2>Step 3: Select Files for Analysis</h2>
                
                <p>Select the files you want to include in the security analysis:</p>
                
                <div class="file-tree" id="file-tree">
                    <!-- File tree will be populated dynamically -->
                </div>
                
                <div class="selected-files">
                    <h3>Selected Files</h3>
                    <ul id="selected-files-list">
                        <!-- Selected files will be displayed here -->
                        <li>No files selected</li>
                    </ul>
                </div>
                
                <div class="actions">
                    <button id="start-analysis-btn" disabled>Start Analysis</button>
                    <button id="clear-selection-btn">Clear Selection</button>
                </div>
            </div>
            
            <div id="analysis-status" style="margin-top: 30px; display: none;">
                <h2>Analysis Status</h2>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="analysis-progress"></div>
                </div>
                <p id="status-message">Initializing analysis...</p>
            </div>
            
            <div id="toast" class="toast"></div>
            
            <script>
                // Global state
                const state = {
                    repoInfo: null,
                    selectedBranch: null,
                    selectedCommit: null,
                    repoPath: null,
                    selectedFiles: [],
                    fileTree: {}
                };
                
                // DOM Elements
                const repoUrlInput = document.getElementById('repo-url');
                const fetchRepoBtn = document.getElementById('fetch-repo-btn');
                const repoUrlError = document.getElementById('repo-url-error');
                const loadingIndicator = document.getElementById('loading-indicator');
                const repoInfoDiv = document.getElementById('repo-info');
                const branchSelect = document.getElementById('branch-select');
                const commitList = document.getElementById('commit-list');
                const cloneRepoBtn = document.getElementById('clone-repo-btn');
                const fileSelector = document.getElementById('file-selector');
                const fileTree = document.getElementById('file-tree');
                const selectedFilesList = document.getElementById('selected-files-list');
                const startAnalysisBtn = document.getElementById('start-analysis-btn');
                const clearSelectionBtn = document.getElementById('clear-selection-btn');
                const analysisStatus = document.getElementById('analysis-status');
                const analysisProgress = document.getElementById('analysis-progress');
                const statusMessage = document.getElementById('status-message');
                const toast = document.getElementById('toast');
                
                // Helper functions
                function showToast(message, type = 'info') {
                    toast.textContent = message;
                    toast.className = `toast ${type}`;
                    toast.style.display = 'block';
                    
                    setTimeout(() => {
                        toast.style.display = 'none';
                    }, 5000);
                }
                
                function showError(element, message) {
                    element.textContent = message;
                    element.style.display = 'block';
                }
                
                function hideError(element) {
                    element.style.display = 'none';
                }
                
                // Event listeners
                fetchRepoBtn.addEventListener('click', async () => {
                    const repoUrl = repoUrlInput.value.trim();
                    
                    if (!repoUrl) {
                        showError(repoUrlError, 'Please enter a GitHub repository URL');
                        return;
                    }
                    
                    // URL format validation
                    if (!repoUrl.startsWith('https://github.com/')) {
                        showError(repoUrlError, 'Please enter a valid GitHub repository URL (https://github.com/owner/repo)');
                        return;
                    }
                    
                    hideError(repoUrlError);
                    loadingIndicator.style.display = 'block';
                    fetchRepoBtn.disabled = true;
                    
                    try {
                        const response = await fetch('/api/github/fetch', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ repo_url: repoUrl })
                        });
                        
                        const data = await response.json();
                        
                        if (data.status === 'error') {
                            showError(repoUrlError, data.message);
                            loadingIndicator.style.display = 'none';
                            fetchRepoBtn.disabled = false;
                            return;
                        }
                        
                        // Store repo info
                        state.repoInfo = data;
                        
                        // Update UI
                        populateBranchSelect(data.branches);
                        populateCommitList(data.commits);
                        
                        // Show repo info section
                        repoInfoDiv.style.display = 'block';
                        
                    } catch (error) {
                        showError(repoUrlError, `Error: ${error.message}`);
                    } finally {
                        loadingIndicator.style.display = 'none';
                        fetchRepoBtn.disabled = false;
                    }
                });
                
                function populateBranchSelect(branches) {
                    branchSelect.innerHTML = '';
                    
                    branches.forEach(branch => {
                        const option = document.createElement('option');
                        option.value = branch.name;
                        option.textContent = branch.name;
                        
                        if (branch.name === state.repoInfo.default_branch) {
                            option.selected = true;
                            state.selectedBranch = branch.name;
                        }
                        
                        branchSelect.appendChild(option);
                    });
                }
                
                function populateCommitList(commits) {
                    commitList.innerHTML = '<h3>Recent Commits</h3>';
                    
                    commits.forEach(commit => {
                        const commitItem = document.createElement('div');
                        commitItem.className = 'commit-item';
                        commitItem.dataset.sha = commit.sha;
                        
                        // Format date
                        const date = new Date(commit.date);
                        const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
                        
                        commitItem.innerHTML = `
                            <div><strong>${commit.message.split('\\n')[0]}</strong></div>
                            <div>Author: ${commit.author}</div>
                            <div>Date: ${formattedDate}</div>
                            <div class="commit-sha">SHA: ${commit.sha.substring(0, 7)}</div>
                        `;
                        
                        commitItem.addEventListener('click', () => {
                            // Remove selection from all commit items
                            document.querySelectorAll('.commit-item').forEach(item => {
                                item.classList.remove('selected');
                            });
                            
                            // Add selection to clicked item
                            commitItem.classList.add('selected');
                            
                            // Update state
                            state.selectedCommit = commit.sha;
                        });
                        
                        commitList.appendChild(commitItem);
                        
                        // Select the first commit by default
                        if (!state.selectedCommit && commitList.children.length === 2) { // 1 for header + 1 for first commit
                            commitItem.click();
                        }
                    });
                }
                
                branchSelect.addEventListener('change', () => {
                    state.selectedBranch = branchSelect.value;
                    
                    // Clear commit selection when branch changes
                    state.selectedCommit = null;
                    document.querySelectorAll('.commit-item').forEach(item => {
                        item.classList.remove('selected');
                    });
                });
                
                cloneRepoBtn.addEventListener('click', async () => {
                    if (!state.repoInfo) {
                        showToast('Please fetch repository information first', 'error');
                        return;
                    }
                    
                    const branch = state.selectedBranch || state.repoInfo.default_branch;
                    
                    loadingIndicator.style.display = 'block';
                    cloneRepoBtn.disabled = true;
                    
                    try {
                        const response = await fetch('/api/github/clone', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                repo_url: repoUrlInput.value.trim(),
                                branch: branch,
                                commit_sha: state.selectedCommit
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (data.status === 'error') {
                            showToast(data.message, 'error');
                            loadingIndicator.style.display = 'none';
                            cloneRepoBtn.disabled = false;
                            return;
                        }
                        
                        // Store repo path
                        state.repoPath = data.repo_dir;
                        
                        // Load file tree
                        await loadFileTree(data.repo_dir);
                        
                        // Show file selector
                        fileSelector.style.display = 'block';
                        
                        showToast('Repository cloned successfully', 'success');
                        
                    } catch (error) {
                        showToast(`Error: ${error.message}`, 'error');
                    } finally {
                        loadingIndicator.style.display = 'none';
                        cloneRepoBtn.disabled = false;
                    }
                });
                
                async function loadFileTree(repoPath) {
                    try {
                        // Fetch file list from backend
                        const response = await fetch(`/api/files?path=${encodeURIComponent(repoPath)}`);
                        const data = await response.json();
                        
                        if (data.status === 'error') {
                            showToast(data.message, 'error');
                            return;
                        }
                        
                        state.fileTree = data.files;
                        
                        // Render file tree
                        renderFileTree(data.files, fileTree);
                        
                    } catch (error) {
                        showToast(`Error loading file tree: ${error.message}`, 'error');
                    }
                }
                
                function renderFileTree(files, container, basePath = '') {
                    container.innerHTML = '';
                    
                    // Sort: directories first, then files
                    const sortedFiles = Object.entries(files).sort((a, b) => {
                        const aIsDir = typeof a[1] === 'object';
                        const bIsDir = typeof b[1] === 'object';
                        
                        if (aIsDir && !bIsDir) return -1;
                        if (!aIsDir && bIsDir) return 1;
                        return a[0].localeCompare(b[0]);
                    });
                    
                    for (const [name, value] of sortedFiles) {
                        const isDirectory = typeof value === 'object';
                        const fullPath = basePath ? `${basePath}/${name}` : name;
                        
                        // Skip node_modules and hidden files/directories
                        if (name === 'node_modules' || name.startsWith('.')) {
                            continue;
                        }
                        
                        const fileItem = document.createElement('div');
                        fileItem.className = `file-item ${isDirectory ? 'directory' : ''}`;
                        fileItem.dataset.path = fullPath;
                        fileItem.textContent = name;
                        
                        if (isDirectory) {
                            // For directories, create expandable structure
                            const directoryContainer = document.createElement('div');
                            directoryContainer.className = 'directory-container';
                            directoryContainer.style.display = 'none';
                            directoryContainer.style.paddingLeft = '20px';
                            
                            fileItem.addEventListener('click', (e) => {
                                e.stopPropagation();
                                directoryContainer.style.display = directoryContainer.style.display === 'none' ? 'block' : 'none';
                            });
                            
                            renderFileTree(value, directoryContainer, fullPath);
                            
                            const directoryWrapper = document.createElement('div');
                            directoryWrapper.appendChild(fileItem);
                            directoryWrapper.appendChild(directoryContainer);
                            container.appendChild(directoryWrapper);
                        } else {
                            // For files, add selection capability
                            // Only select contract files by default
                            if (name.endsWith('.sol')) {
                                fileItem.addEventListener('click', () => {
                                    fileItem.classList.toggle('selected');
                                    
                                    if (fileItem.classList.contains('selected')) {
                                        // Add to selected files
                                        if (!state.selectedFiles.includes(fullPath)) {
                                            state.selectedFiles.push(fullPath);
                                        }
                                    } else {
                                        // Remove from selected files
                                        state.selectedFiles = state.selectedFiles.filter(file => file !== fullPath);
                                    }
                                    
                                    updateSelectedFilesList();
                                });
                                
                                container.appendChild(fileItem);
                            }
                        }
                    }
                }
                
                function updateSelectedFilesList() {
                    if (state.selectedFiles.length === 0) {
                        selectedFilesList.innerHTML = '<li>No files selected</li>';
                        startAnalysisBtn.disabled = true;
                    } else {
                        selectedFilesList.innerHTML = '';
                        state.selectedFiles.forEach(file => {
                            const listItem = document.createElement('li');
                            listItem.textContent = file;
                            selectedFilesList.appendChild(listItem);
                        });
                        startAnalysisBtn.disabled = false;
                    }
                }
                
                clearSelectionBtn.addEventListener('click', () => {
                    state.selectedFiles = [];
                    document.querySelectorAll('.file-item.selected').forEach(item => {
                        item.classList.remove('selected');
                    });
                    updateSelectedFilesList();
                });
                
                startAnalysisBtn.addEventListener('click', async () => {
                    if (state.selectedFiles.length === 0) {
                        showToast('Please select at least one file for analysis', 'error');
                        return;
                    }
                    
                    // Prepare file paths (convert from relative to absolute)
                    const filePaths = state.selectedFiles.map(file => `${state.repoPath}/${file}`);
                    
                    analysisStatus.style.display = 'block';
                    startAnalysisBtn.disabled = true;
                    statusMessage.textContent = 'Initializing analysis...';
                    analysisProgress.style.width = '5%';
                    
                    try {
                        // Start audit in background
                        const response = await fetch('/api/audit', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                files: filePaths,
                                query: "Perform a security audit of these smart contracts",
                                project_name: state.repoInfo ? `${state.repoInfo.owner}/${state.repoInfo.repo}` : 'github-import',
                                wait_for_completion: false
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (!response.ok) {
                            throw new Error(data.detail || 'Failed to start analysis');
                        }
                        
                        showToast('Analysis started successfully', 'success');
                        
                        // Start progress polling
                        pollAnalysisProgress();
                        
                    } catch (error) {
                        showToast(`Error: ${error.message}`, 'error');
                        startAnalysisBtn.disabled = false;
                    }
                });
                
                let progressInterval;
                
                function pollAnalysisProgress() {
                    let progress = 5;
                    
                    progressInterval = setInterval(async () => {
                        try {
                            const response = await fetch('/api/telemetry');
                            const data = await response.json();
                            
                            // Calculate progress based on tasks
                            if (data.tasks_created > 0) {
                                const completedPercentage = (data.tasks_completed / data.tasks_created) * 100;
                                progress = Math.max(5, Math.min(95, completedPercentage));
                            }
                            
                            // Update UI
                            analysisProgress.style.width = `${progress}%`;
                            
                            // Update status message
                            if (data.tasks_failed > 0) {
                                statusMessage.textContent = `Analysis in progress... ${data.tasks_completed} completed, ${data.tasks_failed} failed`;
                            } else {
                                statusMessage.textContent = `Analysis in progress... ${data.tasks_completed} of ${data.tasks_created} tasks completed`;
                            }
                            
                            // Check if completed
                            if (data.audit_end_time && progress >= 95) {
                                clearInterval(progressInterval);
                                analysisProgress.style.width = '100%';
                                statusMessage.textContent = 'Analysis completed! View results in the Reports section.';
                                startAnalysisBtn.disabled = false;
                            }
                            
                        } catch (error) {
                            console.error('Error polling progress:', error);
                        }
                    }, 2000);
                }
                
                // Error handling
                window.addEventListener('error', function(event) {
                    // Log client-side errors to the server
                    fetch('/api/errors', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            timestamp: new Date().toISOString(),
                            message: event.message,
                            stack: event.error ? event.error.stack : '',
                            url: event.filename,
                            user_agent: navigator.userAgent
                        })
                    }).catch(console.error);
                    
                    // Show toast to user
                    showToast(`An error occurred: ${event.message}`, 'error');
                });
            </script>
        </body>
    </html>
    """)

@app.get("/reports", response_class=HTMLResponse)
async def reports_page():
    """Reports page that lists all available reports and provides viewing options."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Analysis Reports - Finite Monkey Engine</title>
            <link rel="stylesheet" href="/static/css/main.css">
            <style>
                .reports-list {
                    margin-top: 20px;
                }
                
                .report-item {
                    background-color: #f5f5f5;
                    border-left: 4px solid #007bff;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 4px;
                }
                
                .report-header {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                }
                
                .report-title {
                    font-size: 1.2em;
                    font-weight: bold;
                }
                
                .report-date {
                    color: #666;
                }
                
                .report-actions {
                    margin-top: 10px;
                }
                
                .report-actions a {
                    display: inline-block;
                    padding: 5px 10px;
                    margin-right: 10px;
                    background-color: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 3px;
                }
                
                .report-actions a:hover {
                    background-color: #0069d9;
                }
                
                .report-issues {
                    margin-top: 10px;
                }
                
                .issue-item {
                    padding: 8px;
                    border-bottom: 1px solid #eee;
                }
                
                .issue-severity-high {
                    border-left: 4px solid #dc3545;
                    padding-left: 10px;
                }
                
                .issue-severity-medium {
                    border-left: 4px solid #fd7e14;
                    padding-left: 10px;
                }
                
                .issue-severity-low {
                    border-left: 4px solid #ffc107;
                    padding-left: 10px;
                }
                
                .issue-severity-info {
                    border-left: 4px solid #17a2b8;
                    padding-left: 10px;
                }
                
                .no-reports {
                    padding: 20px;
                    background-color: #f5f5f5;
                    border-radius: 4px;
                    text-align: center;
                }
                
                .github-issue-modal {
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0, 0, 0, 0.5);
                    z-index: 1000;
                }
                
                .modal-content {
                    background-color: white;
                    margin: 10% auto;
                    padding: 20px;
                    width: 80%;
                    max-width: 600px;
                    border-radius: 5px;
                }
                
                .close-button {
                    float: right;
                    cursor: pointer;
                    font-size: 24px;
                }
                
                .github-issue-form {
                    margin-top: 20px;
                }
                
                .github-issue-form input, .github-issue-form textarea {
                    width: 100%;
                    padding: 8px;
                    margin-bottom: 10px;
                }
                
                .github-issue-form textarea {
                    height: 200px;
                }
            </style>
        </head>
        <body>
            <h1>Analysis Reports</h1>
            <p>View and manage security analysis results</p>
            
            <div id="reports-container">
                <div class="loading">Loading reports...</div>
            </div>
            
            <div id="github-issue-modal" class="github-issue-modal">
                <div class="modal-content">
                    <span class="close-button" id="close-modal">&times;</span>
                    <h2>Create GitHub Issue</h2>
                    
                    <div class="github-issue-form">
                        <div class="form-group">
                            <label for="issue-title">Issue Title</label>
                            <input type="text" id="issue-title">
                        </div>
                        
                        <div class="form-group">
                            <label for="issue-body">Issue Description</label>
                            <textarea id="issue-body"></textarea>
                        </div>
                        
                        <div class="form-group">
                            <label for="repo-owner">Repository Owner</label>
                            <input type="text" id="repo-owner">
                        </div>
                        
                        <div class="form-group">
                            <label for="repo-name">Repository Name</label>
                            <input type="text" id="repo-name">
                        </div>
                        
                        <button id="create-issue-btn">Create Issue</button>
                        <div id="issue-creation-result"></div>
                    </div>
                </div>
            </div>
            
            <script>
                // Load reports on page load
                window.addEventListener('load', loadReports);
                
                async function loadReports() {
                    try {
                        const response = await fetch('/api/reports');
                        const data = await response.json();
                        
                        const reportsContainer = document.getElementById('reports-container');
                        
                        if (data.length === 0) {
                            reportsContainer.innerHTML = `
                                <div class="no-reports">
                                    <p>No reports found. Run an analysis to generate reports.</p>
                                    <a href="/github">Import from GitHub</a>
                                </div>
                            `;
                            return;
                        }
                        
                        // Render reports
                        reportsContainer.innerHTML = '<div class="reports-list"></div>';
                        const reportsList = reportsContainer.querySelector('.reports-list');
                        
                        data.forEach(report => {
                            const reportItem = document.createElement('div');
                            reportItem.className = 'report-item';
                            
                            // Format date
                            const date = new Date(report.timestamp);
                            const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
                            
                            // Build issues summary
                            let issuesHtml = '';
                            let issueCount = 0;
                            
                            if (report.issues && report.issues.length > 0) {
                                issuesHtml = '<div class="report-issues">';
                                report.issues.forEach(issue => {
                                    issueCount++;
                                    issuesHtml += `
                                        <div class="issue-item issue-severity-${issue.severity.toLowerCase()}">
                                            <div><strong>${issue.title}</strong></div>
                                            <div>${issue.description.substring(0, 100)}${issue.description.length > 100 ? '...' : ''}</div>
                                        </div>
                                    `;
                                });
                                issuesHtml += '</div>';
                            }
                            
                            reportItem.innerHTML = `
                                <div class="report-header">
                                    <div class="report-title">${report.project_name || report.id || 'Unnamed Project'}</div>
                                    <div class="report-date">${formattedDate}</div>
                                </div>
                                <div>Format: ${report.format || 'JSON'}</div>
                                ${report.files_analyzed ? `<div>Files analyzed: ${Array.isArray(report.files_analyzed) ? report.files_analyzed.length : report.files_analyzed}</div>` : ''}
                                ${report.issues ? `<div>Issues found: ${issueCount}</div>` : ''}
                                ${issuesHtml}
                                <div class="report-actions">
                                    <a href="/reports/${report.id}/view" target="_blank">View Report</a>
                                    ${report.has_markdown ? `<a href="/reports/${report.id}_report.md" target="_blank">View Markdown</a>` : ''}
                                    ${report.has_graph ? `<a href="/reports/${report.id}_graph.html" target="_blank">View Graph</a>` : ''}
                                    ${report.format === 'json' ? `<a href="/reports/${report.id}/download">Download JSON</a>` : ''}
                                    <a href="#" class="create-issue-btn" data-report-id="${report.id}">Create GitHub Issue</a>
                                </div>
                            `;
                            
                            reportsList.appendChild(reportItem);
                        });
                        
                        // Add event listeners for GitHub issue creation
                        document.querySelectorAll('.create-issue-btn').forEach(btn => {
                            btn.addEventListener('click', (e) => {
                                e.preventDefault();
                                openGitHubIssueModal(btn.dataset.reportId);
                            });
                        });
                        
                    } catch (error) {
                        document.getElementById('reports-container').innerHTML = `
                            <div class="error">
                                <p>Error loading reports: ${error.message}</p>
                            </div>
                        `;
                    }
                }
                
                // GitHub Issue Modal
                const modal = document.getElementById('github-issue-modal');
                const closeModal = document.getElementById('close-modal');
                
                closeModal.addEventListener('click', () => {
                    modal.style.display = 'none';
                });
                
                window.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        modal.style.display = 'none';
                    }
                });
                
                async function openGitHubIssueModal(reportId) {
                    try {
                        // Fetch report details
                        const response = await fetch(`/api/reports/${reportId}`);
                        const report = await response.json();
                        
                        // Prepare issue title and body with report details
                        const issueTitle = `Security Analysis: ${report.project_name || 'Smart Contract Audit'}`;
                        let issueBody = `# Security Analysis Report\n\n`;
                        
                        if (report.issues && report.issues.length > 0) {
                            issueBody += `## Issues Found\n\n`;
                            
                            report.issues.forEach(issue => {
                                issueBody += `### ${issue.title} (${issue.severity})\n\n`;
                                issueBody += `${issue.description}\n\n`;
                                
                                if (issue.location) {
                                    issueBody += `**Location:** ${issue.location}\n\n`;
                                }
                                
                                if (issue.recommendation) {
                                    issueBody += `**Recommendation:** ${issue.recommendation}\n\n`;
                                }
                                
                                issueBody += `---\n\n`;
                            });
                        } else {
                            issueBody += `No issues found in the analysis.\n\n`;
                        }
                        
                        issueBody += `## Analysis Details\n\n`;
                        issueBody += `- Date: ${new Date(report.timestamp).toLocaleString()}\n`;
                        issueBody += `- Files Analyzed: ${report.files_analyzed || 'N/A'}\n`;
                        issueBody += `- Analysis Engine: Finite Monkey Engine\n\n`;
                        
                        // Set form values
                        document.getElementById('issue-title').value = issueTitle;
                        document.getElementById('issue-body').value = issueBody;
                        
                        // Try to extract repo owner and name from project_name or files
                        if (report.project_name && report.project_name.includes('/')) {
                            const parts = report.project_name.split('/');
                            document.getElementById('repo-owner').value = parts[0];
                            document.getElementById('repo-name').value = parts[1];
                        }
                        
                        // Show modal
                        modal.style.display = 'block';
                        
                    } catch (error) {
                        alert(`Error loading report: ${error.message}`);
                    }
                }
                
                // GitHub Issue Creation
                document.getElementById('create-issue-btn').addEventListener('click', async () => {
                    const title = document.getElementById('issue-title').value;
                    const body = document.getElementById('issue-body').value;
                    const owner = document.getElementById('repo-owner').value;
                    const repo = document.getElementById('repo-name').value;
                    const resultDiv = document.getElementById('issue-creation-result');
                    
                    if (!title || !body || !owner || !repo) {
                        resultDiv.textContent = 'Please fill in all fields';
                        resultDiv.style.color = 'red';
                        return;
                    }
                    
                    try {
                        resultDiv.textContent = 'Creating issue...';
                        resultDiv.style.color = 'black';
                        
                        const response = await fetch('/api/github/issue', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                owner,
                                repo,
                                title,
                                body,
                                labels: ['security', 'vulnerability', 'audit']
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (data.status === 'error') {
                            throw new Error(data.message);
                        }
                        
                        resultDiv.textContent = `Issue created successfully! View it at: ${data.issue_url}`;
                        resultDiv.style.color = 'green';
                        
                        // Add link to issue
                        const link = document.createElement('a');
                        link.href = data.issue_url;
                        link.textContent = 'View Issue on GitHub';
                        link.target = '_blank';
                        resultDiv.appendChild(document.createElement('br'));
                        resultDiv.appendChild(link);
                        
                    } catch (error) {
                        resultDiv.textContent = `Error creating issue: ${error.message}`;
                        resultDiv.style.color = 'red';
                    }
                });
            </script>
        </body>
    </html>
    """)

@app.get("/terminal/{terminal_id}", response_class=HTMLResponse)
async def terminal_page(terminal_id: str):
    """Terminal page that provides an IPython terminal."""
    css_styles = """
        .terminal-container {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 200px);
            min-height: 500px;
            border: 1px solid #444;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .terminal-controls {
            display: flex;
            background-color: #333;
            padding: 5px;
            border-bottom: 1px solid #444;
            align-items: center;
        }
        
        .terminal-controls button {
            margin-right: 5px;
            padding: 5px 10px;
            background-color: #555;
            color: #fff;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        
        .terminal-controls button:hover {
            background-color: #666;
        }
        
        .kill-button {
            background-color: #c9302c !important;
        }
        
        .kill-button:hover {
            background-color: #d9534f !important;
        }
        
        #terminal-status {
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 12px;
            color: #fff;
            background-color: #555;
        }
        
        #terminal-status.connected {
            background-color: #28a745; /* Green for connected */
        }
        
        #terminal-status.connecting {
            background-color: #ffc107; /* Yellow for connecting */
        }
        
        #terminal-status.disconnected {
            background-color: #dc3545; /* Red for disconnected */
        }
        
        .terminal-wrapper {
            display: flex;
            flex-grow: 1;
            overflow: hidden;
            min-height: 400px;
            height: 500px;
            margin-bottom: 10px; /* Provide space between terminal and input */
        }
        
        .terminal {
            background-color: #272822;
            color: #f8f8f2;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            line-height: 1.5;
            padding: 10px;
            flex: 3;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            height: 100%;
        }
        
        .scope-inspector {
            width: 300px;
            min-width: 250px;
            flex: 1;
            max-width: 350px;
            background-color: #333;
            color: #f8f8f2;
            border-left: 1px solid #444;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: 100%;
        }
        
        .scope-inspector h3 {
            margin: 0;
            padding: 10px;
            background-color: #444;
            color: #fff;
            font-size: 14px;
        }
        
        .scope-content {
            flex-grow: 1;
            padding: 10px;
            overflow-y: auto;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            height: calc(100% - 70px); /* Make sure content area doesn't overflow */
        }
        
        .scope-item {
            margin-bottom: 8px;
            padding: 4px;
            border-radius: 3px;
            cursor: pointer;
            user-select: none;
        }
        
        .scope-item:hover {
            background-color: #444;
        }
        
        .scope-item-key {
            color: #66d9ef;
            display: block;
            font-weight: bold;
            margin-bottom: 3px;
            word-break: break-word;
        }
        
        .scope-item-type {
            color: #a6e22e;
            font-size: 11px;
            margin-left: 4px;
        }
        
        .scope-item-value {
            color: #e6db74;
            margin-left: 15px;
            display: none;
            overflow-wrap: break-word;
            word-break: break-all;
            max-width: calc(100% - 20px);
            padding: 4px;
            background-color: #2a2a2a;
            border-radius: 3px;
        }
        
        .scope-item.expanded .scope-item-value {
            display: block;
        }
        
        .scope-loader {
            display: inline-block;
            color: #888;
            font-style: italic;
            font-size: 90%;
        }
        
        .refresh-button {
            margin: 10px;
            padding: 5px;
            background-color: #555;
            color: #fff;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        
        /* ANSI color codes for IPython */
        .ansi-black { color: #3E3D32; }
        .ansi-red { color: #F92672; }
        .ansi-green { color: #A6E22E; }
        .ansi-yellow { color: #FD971F; }
        .ansi-blue { color: #66D9EF; }
        .ansi-magenta { color: #AE81FF; }
        .ansi-cyan { color: #A1EFE4; }
        .ansi-white { color: #F8F8F2; }
        
        .ansi-bright-black { color: #75715E; }
        .ansi-bright-red { color: #F92672; }
        .ansi-bright-green { color: #A6E22E; }
        .ansi-bright-yellow { color: #E6DB74; }
        .ansi-bright-blue { color: #66D9EF; }
        .ansi-bright-magenta { color: #AE81FF; }
        .ansi-bright-cyan { color: #A1EFE4; }
        .ansi-bright-white { color: #F9F8F5; }
        
        .ansi-bg-black { background-color: #272822; }
        .ansi-bg-red { background-color: #F92672; }
        .ansi-bg-green { background-color: #A6E22E; }
        .ansi-bg-yellow { background-color: #FD971F; }
        .ansi-bg-blue { background-color: #66D9EF; }
        .ansi-bg-magenta { background-color: #AE81FF; }
        .ansi-bg-cyan { background-color: #A1EFE4; }
        .ansi-bg-white { background-color: #F8F8F2; }
        
        .terminal-input {
            display: flex;
            margin-top: 10px;
        }
        
        .terminal-input input {
            flex: 1;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            padding: 8px;
            background-color: #272822;
            color: #f8f8f2;
            border: 1px solid #75715E;
        }
        
        .terminal-input button {
            background-color: #66D9EF;
            color: #272822;
            border: none;
            padding: 8px 15px;
            margin-left: 5px;
            font-weight: bold;
            cursor: pointer;
        }
    """
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Terminal - Finite Monkey Engine</title>
            <link rel="stylesheet" href="/static/css/main.css">
            <style>
                {css_styles}
                
                .terminal-input {{
                    display: flex;
                    margin-top: 10px;
                }}
                
                .terminal-input input {{
                    flex: 1;
                    font-family: monospace;
                    padding: 5px;
                }}
                
                .terminal-input button {{
                    padding: 5px 10px;
                    margin-left: 5px;
                }}
            </style>
            <script>
                let ws;
                
                function initWebSocket() {{
                    ws = new WebSocket(`${{window.location.protocol === 'https:' ? 'wss' : 'ws'}}://${{window.location.host}}/ws/terminal/{terminal_id}`);
                    
                    const statusEl = document.getElementById('terminal-status');
                    statusEl.textContent = 'Connecting...';
                    statusEl.className = 'connecting';
                    
                    ws.onopen = function(e) {{
                        console.log('Connected to terminal');
                        statusEl.textContent = 'Connected';
                        statusEl.className = 'connected';
                        
                        // Initial scope refresh when connection is established
                        setTimeout(refreshScope, 1000);
                    }};
                    
                    ws.onmessage = function(e) {{
                        // Check if this is a scope inspection result
                        if (e.data.startsWith('__scope_result__:')) {{
                            // Extract the HTML content
                            const htmlContent = e.data.substring('__scope_result__:'.length);
                            
                            // Update the scope inspector
                            const scopeContent = document.getElementById('scope-content');
                            scopeContent.innerHTML = htmlContent;
                            
                            // Clear the request pending flag since we got a response
                            if (window.scopeRequestPending) {{
                                window.scopeRequestPending = false;
                                clearTimeout(window.scopeTimeoutTimer);
                                console.log("Scope data received and updated");
                            }}
                            
                            // Don't display this message in the terminal
                            return;
                        }}
                        
                        // Skip JSON output for scope inspection and other system messages
                        if (e.data.includes('__scope_inspect__:')) {{
                            return;
                        }}
                        
                        // Filter out JSON responses that are likely from periodic I/O
                        let shouldSkip = false;
                        const dataStr = e.data.toString();
                        const trimmedData = dataStr.trim();
                        
                        // Improved JSON filtering approach
                        // First check if it looks like JSON
                        if (trimmedData.startsWith('{{') && trimmedData.includes('}}')) {{
                            // Then check for specific patterns that indicate system messages
                            if (
                                dataStr.includes('terminal_ready') || 
                                dataStr.includes('json.dumps') ||
                                dataStr.includes('__scope_inspect__') ||
                                dataStr.includes('__pycache__')
                            ) {{
                                shouldSkip = true;
                            }}
                            
                            // Also skip if it's definitely valid JSON and not meant for user display
                            try {{
                                const jsonObj = JSON.parse(trimmedData);
                                // If it has certain properties that indicate system messages
                                if (jsonObj.type === 'system' || jsonObj.internal === true) {{
                                    shouldSkip = true;
                                }}
                            }} catch (e) {{
                                // Not valid JSON or couldn't parse - show it anyway
                            }}
                        }}
                        
                        if (shouldSkip) {{
                            // Skip this output as it's likely system-generated JSON
                            console.log("Filtering system JSON message");
                            return;
                        }}
                        
                        const terminal = document.getElementById('terminal');
                        
                        // Process ANSI color codes
                        const data = processAnsiCodes(e.data);
                        
                        // Append the processed HTML instead of text
                        terminal.innerHTML += data;
                        terminal.scrollTop = terminal.scrollHeight;
                    }};
                    
                    // Function to process ANSI color codes
                    function processAnsiCodes(text) {{
                        // Replace ANSI color/style codes with HTML spans
                        // This handles both the \x1b escape character and the literal ESC character (char code 27)
                        // which might be represented differently in the string
                        const ansiPattern = /(?:\\x1b|\\u001b|\u001b)\\[(\\d+)(;\\d+)*m/g;
                        
                        // Current style classes
                        let currentClasses = [];
                        
                        // Replace backspace character sequences that might erase characters
                        text = text.replace(/.\x08/g, '');
                        
                        // Split by ANSI codes while keeping the codes
                        const parts = text.split(/((?:\\x1b|\\u001b|\u001b)\\[\\d+(?:;\\d+)*m)/);
                        let result = '';
                        
                        for (const part of parts) {{
                            // Check if this part is an ANSI code
                            if (part && part.match(ansiPattern)) {{
                                // Extract the code numbers
                                const codes = part.match(/\\d+/g);
                                
                                if (codes) {{
                                    for (const code of codes) {{
                                        switch (code) {{
                                            case '0': // Reset
                                                currentClasses = [];
                                                break;
                                            case '30': currentClasses.push('ansi-black'); break;
                                            case '31': currentClasses.push('ansi-red'); break;
                                            case '32': currentClasses.push('ansi-green'); break;
                                            case '33': currentClasses.push('ansi-yellow'); break;
                                            case '34': currentClasses.push('ansi-blue'); break;
                                            case '35': currentClasses.push('ansi-magenta'); break;
                                            case '36': currentClasses.push('ansi-cyan'); break;
                                            case '37': currentClasses.push('ansi-white'); break;
                                            
                                            case '90': currentClasses.push('ansi-bright-black'); break;
                                            case '91': currentClasses.push('ansi-bright-red'); break;
                                            case '92': currentClasses.push('ansi-bright-green'); break;
                                            case '93': currentClasses.push('ansi-bright-yellow'); break;
                                            case '94': currentClasses.push('ansi-bright-blue'); break;
                                            case '95': currentClasses.push('ansi-bright-magenta'); break;
                                            case '96': currentClasses.push('ansi-bright-cyan'); break;
                                            case '97': currentClasses.push('ansi-bright-white'); break;
                                            
                                            case '40': currentClasses.push('ansi-bg-black'); break;
                                            case '41': currentClasses.push('ansi-bg-red'); break;
                                            case '42': currentClasses.push('ansi-bg-green'); break;
                                            case '43': currentClasses.push('ansi-bg-yellow'); break;
                                            case '44': currentClasses.push('ansi-bg-blue'); break;
                                            case '45': currentClasses.push('ansi-bg-magenta'); break;
                                            case '46': currentClasses.push('ansi-bg-cyan'); break;
                                            case '47': currentClasses.push('ansi-bg-white'); break;
                                        }}
                                    }}
                                }}
                            }} else if (part) {{
                                // Regular text content
                                if (currentClasses.length > 0) {{
                                    // Apply current styles
                                    const classAttr = currentClasses.join(' ');
                                    
                                    // Escape HTML characters
                                    const safeText = part
                                        .replace(/&/g, '&amp;')
                                        .replace(/</g, '&lt;')
                                        .replace(/>/g, '&gt;')
                                        .replace(/"/g, '&quot;')
                                        .replace(/'/g, '&#039;');
                                    
                                    result += `<span class="${{classAttr}}">${{safeText}}</span>`;
                                }} else {{
                                    // No styles
                                    // Escape HTML characters
                                    const safeText = part
                                        .replace(/&/g, '&amp;')
                                        .replace(/</g, '&lt;')
                                        .replace(/>/g, '&gt;')
                                        .replace(/"/g, '&quot;')
                                        .replace(/'/g, '&#039;');
                                    
                                    result += safeText;
                                }}
                            }}
                        }}
                        
                        // Remove cursor position requests and other terminal control sequences
                        result = result.replace(/(?:\\x1b|\\u001b|\u001b)\\[\\d*[A-Za-z]/g, '');
                        result = result.replace(/(?:\\x1b|\\u001b|\u001b)\\][0-9].*?(?:\\x07|\\u0007|\u0007)/g, '');
                        
                        // Replace newlines with breaks for proper display
                        result = result.replace(/\\n/g, '<br>');
                        
                        return result;
                    }};
                    
                    ws.onclose = function(e) {{
                        console.log('Disconnected from terminal');
                        const statusEl = document.getElementById('terminal-status');
                        statusEl.textContent = 'Disconnected';
                        statusEl.className = 'disconnected';
                        
                        // Try to reconnect
                        setTimeout(initWebSocket, 1000);
                    }};
                    
                    ws.onerror = function(e) {{
                        console.error('WebSocket error:', e);
                    }};
                }}
                
                function sendCommand() {{
                    const input = document.getElementById('terminal-input');
                    const command = input.value;
                    
                    if (command) {{
                        try {{
                            // First show the command in the terminal directly for immediate feedback
                            const terminal = document.getElementById('terminal');
                            terminal.innerHTML += `<span class="ansi-bright-green">&gt;&gt;&gt; ${{command}}</span><br>`;
                            terminal.scrollTop = terminal.scrollHeight;
                            
                            // Then send to the server
                            if (ws && ws.readyState === WebSocket.OPEN) {{
                                ws.send(command);
                                input.value = '';
                                
                                // If we're running a command that might change scope, refresh the scope
                                // after a short delay to allow the command to complete
                                setTimeout(refreshScope, 1000);
                            }} else {{
                                terminal.innerHTML += `<span class="ansi-red">Error: Connection lost. Trying to reconnect...</span><br>`;
                                terminal.scrollTop = terminal.scrollHeight;
                                setTimeout(initWebSocket, 1000);
                            }}
                        }} catch (error) {{
                            console.error("Error sending command:", error);
                            const terminal = document.getElementById('terminal');
                            terminal.innerHTML += `<span class="ansi-red">Error sending command: ${{error.message}}</span><br>`;
                            terminal.scrollTop = terminal.scrollHeight;
                        }}
                    }}
                }}
                
                function sendInterrupt() {{
                    // Send Ctrl+C signal to interrupt running commands
                    if (ws && ws.readyState === WebSocket.OPEN) {{
                        ws.send("\x03");  // ASCII code for Ctrl+C
                        
                        // Add visual feedback
                        const terminal = document.getElementById('terminal');
                        terminal.innerHTML += '<span class="ansi-red">^C</span><br>';
                        terminal.scrollTop = terminal.scrollHeight;
                    }}
                }}
                
                function clearTerminal() {{
                    // Clear the terminal display
                    const terminal = document.getElementById('terminal');
                    terminal.innerHTML = '';
                }}
                
                async function refreshScope() {{
                    const scopeContent = document.getElementById('scope-content');
                    
                    try {{
                        // Only proceed if connected
                        if (ws && ws.readyState === WebSocket.OPEN) {{
                            // Display loading message if empty
                            if (!scopeContent.innerHTML || scopeContent.innerHTML.includes('Waiting')) {{
                                scopeContent.innerHTML = '<p>Loading scope data...</p>';
                            }}
                            
                            // Use a timeout to detect if scope refresh doesn't respond within a reasonable time
                            window.scopeRequestPending = true;
                            
                            // Set a timeout to clear the pending flag if no response comes back
                            if (window.scopeTimeoutTimer) {{
                                clearTimeout(window.scopeTimeoutTimer);
                            }}
                            
                            window.scopeTimeoutTimer = setTimeout(() => {{
                                window.scopeRequestPending = false;
                                console.log("Scope refresh timeout - no response received");
                            }}, 5000); // 5 second timeout for scope refresh response
                            
                            try {{
                                // Execute dir() command to get available objects directly
                                ws.send("__scope_inspect__:dir()");
                                console.log("Sent scope inspection request");
                            }} catch (sendError) {{
                                console.warn("Unable to send scope inspect request:", sendError);
                                window.scopeRequestPending = false;
                                clearTimeout(window.scopeTimeoutTimer);
                                // Don't set up refresh interval if we can't send
                                return;
                            }}
                            
                            // Cancel existing refresh interval if connection changed
                            if (window.scopeRefreshInterval) {{
                                clearInterval(window.scopeRefreshInterval);
                                window.scopeRefreshInterval = null;
                            }}
                            
                            // Set up a new refresh interval - every 15 seconds which is reasonable
                            // for updating variables without too many requests
                            window.scopeRefreshInterval = setInterval(() => {{
                                try {{
                                    // Only send a new request if we're connected and not waiting for a previous response
                                    if (ws && ws.readyState === WebSocket.OPEN && !window.scopeRequestPending) {{
                                        // Set flag to indicate a request is in progress
                                        window.scopeRequestPending = true;
                                        
                                        // Set a timeout for response
                                        if (window.scopeTimeoutTimer) {{
                                            clearTimeout(window.scopeTimeoutTimer);
                                        }}
                                        
                                        window.scopeTimeoutTimer = setTimeout(() => {{
                                            window.scopeRequestPending = false;
                                        }}, 5000);
                                        
                                        // Refresh scope data
                                        ws.send("__scope_inspect__:dir()");
                                        console.log("Auto-refreshing scope data");
                                    }} else if (!ws || ws.readyState !== WebSocket.OPEN) {{
                                        // Clear interval if disconnected
                                        console.log("Clearing scope interval - connection lost");
                                        clearInterval(window.scopeRefreshInterval);
                                        window.scopeRefreshInterval = null;
                                        window.scopeRequestPending = false;
                                    }}
                                }} catch (e) {{
                                    console.warn("Error in scope refresh:", e);
                                    window.scopeRequestPending = false;
                                    clearInterval(window.scopeRefreshInterval);
                                    window.scopeRefreshInterval = null;
                                }}
                            }}, 15000); // Refresh every 15 seconds
                        }} else {{
                            scopeContent.innerHTML = '<p>Waiting for terminal connection...</p>';
                            // Try again after a delay
                            setTimeout(refreshScope, 3000);
                        }}
                    }} catch (error) {{
                        console.error('Scope refresh error:', error);
                        scopeContent.innerHTML = '<p class="ansi-red">Error loading scope: ' + error.message + '</p>';
                        window.scopeRequestPending = false;
                    }}
                }}
                
                // Handle scope item clicks
                document.addEventListener('click', function(event) {{
                    if (event.target.classList.contains('scope-item') || 
                        event.target.parentElement.classList.contains('scope-item')) {{
                        
                        const item = event.target.classList.contains('scope-item') ? 
                            event.target : event.target.parentElement;
                        
                        // Toggle expanded state
                        item.classList.toggle('expanded');
                        
                        // If expanding and no content yet, load the details
                        const valueElement = item.querySelector('.scope-item-value');
                        const objName = item.getAttribute('data-name');
                        
                        if (item.classList.contains('expanded') && valueElement && 
                            (!valueElement.textContent || valueElement.textContent === 'Loading...')) {{
                            
                            valueElement.textContent = 'Loading...';
                            
                            // Request object details
                            if (ws && ws.readyState === WebSocket.OPEN) {{
                                ws.send(`__scope_inspect__:vars(${{objName}})`);
                            }}
                        }}
                    }}
                }});
                
                // Setup event listeners
                window.addEventListener('load', function() {{
                    initWebSocket();
                    
                    // Add event listener for Enter key in input field
                    document.getElementById('terminal-input').addEventListener('keydown', function(e) {{
                        if (e.key === 'Enter') {{
                            e.preventDefault();
                            sendCommand();
                        }}
                    }});
                    
                    // Refresh scope on load
                    setTimeout(refreshScope, 1000);
                }});
            </script>
        </head>
        <body>
            <h1>IPython Terminal</h1>
            <p>Status: <span id="terminal-status">Connecting...</span></p>
            <p>
                This terminal provides access to the following objects:
                <ul>
                    <li><code>orchestrator</code> - The WorkflowOrchestrator instance</li>
                    <li><code>task_manager</code> - The TaskManager instance</li>
                    <li><code>researcher</code> - The Researcher agent</li>
                    <li><code>validator</code> - The Validator agent</li>
                    <li><code>documentor</code> - The Documentor agent</li>
                    <li><code>config</code> - The configuration object</li>
                </ul>
            </p>
            <div class="terminal-container">
                <div class="terminal-controls">
                    <button onclick="sendInterrupt()" class="kill-button" title="Send Ctrl+C to interrupt">Kill (Ctrl+C)</button>
                    <button onclick="clearTerminal()" title="Clear terminal output">Clear</button>
                    <span style="flex-grow: 1"></span>
                    <span id="terminal-status">Connecting...</span>
                </div>
                <div class="terminal-wrapper">
                    <div class="terminal" id="terminal"></div>
                    <div class="scope-inspector" id="scope-inspector">
                        <h3>Scope Inspector</h3>
                        <div class="scope-content" id="scope-content">
                            <p>Loading scope data...</p>
                        </div>
                        <button onclick="refreshScope()" class="refresh-button">Refresh Scope</button>
                    </div>
                </div>
                <div class="terminal-input">
                    <input type="text" id="terminal-input" placeholder="Enter command...">
                    <button onclick="sendCommand()">Send</button>
                </div>
            </div>
        </body>
    </html>
    """)

@app.websocket("/ws/terminal/{terminal_id}")
async def terminal_websocket(websocket: WebSocket, terminal_id: str):
    """WebSocket handler for IPython terminal."""
    await websocket.accept()
    
    # Get terminal
    terminal = await get_or_create_terminal(terminal_id)
    
    # Add to connections
    if terminal_id not in terminal_connections:
        terminal_connections[terminal_id] = []
    
    terminal_connections[terminal_id].append(websocket)
    
    # Set up a flag to track connection state
    is_connected = True
    
    try:
        # Poll for initial output which may include the banner
        await asyncio.sleep(0.5)  # Wait a bit for startup
        initial_output = ""
        if hasattr(terminal, 'get_output'):
            initial_output = terminal.get_output()
        elif hasattr(terminal.terminal, 'get_output'):
            initial_output = terminal.terminal.get_output()
            
        if initial_output:
            if is_connected:
                await websocket.send_text(initial_output)
        else:
            # Send fallback welcome message
            if is_connected:
                await websocket.send_text("Welcome to the Finite Monkey IPython Terminal!\n")
                await websocket.send_text("Type Python commands to interact with the framework.\n\n")
        
        # Process incoming messages
        while is_connected:
            data = await websocket.receive_text()
            
            # Skip command echo since client now does this
            # We don't send ">>> {data}" here anymore
            
            # Check for special commands
            if data == "\x03":  # Ctrl+C
                try:
                    # Send interrupt to terminal
                    if hasattr(terminal, 'send_input'):
                        terminal.send_input("\x03")
                    elif hasattr(terminal.terminal, 'send_input'):
                        terminal.terminal.send_input("\x03")
                    
                    # No response expected for interrupt
                except Exception as e:
                    if is_connected:
                        await websocket.send_text(f"\nError sending interrupt: {str(e)}\n")
            
            elif data.startswith("__scope_inspect__:"):
                try:
                    # This is a special command for scope inspection
                    scope_command = data.split(":", 1)[1].strip()
                    
                    # Execute the scope inspection command
                    result = await terminal.run_code(f"import json; print(json.dumps({scope_command}))", highlight=False)
                    
                    # Parse the JSON result
                    if result:
                        # If it's a dir() result, format it as a list of items
                        if scope_command == "dir()":
                            try:
                                items = json.loads(result)
                                html = ""
                                
                                # Filter internal/private items
                                filtered_items = [item for item in items if not (item.startswith('_') and item != '__main__')]
                                
                                # Categorize items
                                categories = {
                                    "Agents": [item for item in filtered_items if item in ['researcher', 'validator', 'documentor', 'orchestrator']],
                                    "Data": [item for item in filtered_items if item in ['config', 'task_manager']],
                                    "Other": [item for item in filtered_items if item not in ['researcher', 'validator', 'documentor', 'orchestrator', 'config', 'task_manager']]
                                }
                                
                                # Generate HTML for categories
                                for category, cat_items in categories.items():
                                    if cat_items:
                                        html += f"<h4>{category}</h4>"
                                        for item in sorted(cat_items):
                                            html += f"""
                                            <div class="scope-item" data-name="{item}">
                                                <span class="scope-item-key">{item}</span>
                                                <div class="scope-item-value">
                                                    <span class="scope-loader">Loading...</span>
                                                </div>
                                            </div>
                                            """
                                
                                # Send the HTML
                                if is_connected:
                                    await websocket.send_text("__scope_result__:" + html)
                            except json.JSONDecodeError:
                                if is_connected:
                                    await websocket.send_text("__scope_result__:<p class='ansi-red'>Error parsing scope data</p>")
                        else:
                            # This is a vars() result for a specific object
                            try:
                                obj_data = json.loads(result)
                                html = ""
                                
                                if isinstance(obj_data, dict):
                                    for key, value in sorted(obj_data.items()):
                                        if not key.startswith('_'):  # Skip internal/private attributes
                                            # Get type safely without calling type() directly on value
                                            type_name = "object"
                                            if isinstance(value, int):
                                                type_name = "int"
                                            elif isinstance(value, float):
                                                type_name = "float"
                                            elif isinstance(value, str):
                                                type_name = "str"
                                            elif isinstance(value, bool):
                                                type_name = "bool"
                                            elif isinstance(value, list):
                                                type_name = "list"
                                            elif isinstance(value, dict):
                                                type_name = "dict"
                                            
                                            value_str = str(value)
                                            if len(value_str) > 100:
                                                value_str = value_str[:100] + "..."
                                                
                                            html += """
                                            <div>
                                                <span class="scope-item-key">{}</span>
                                                <span class="scope-item-type">{}</span>
                                                <div>{}</div>
                                            </div>
                                            """.format(key, type_name, value_str)
                                else:
                                    html = "<div>{}</div>".format(str(obj_data))
                            
                                # Send the HTML
                                await websocket.send_text("__scope_result__:" + html)
                            except json.JSONDecodeError:
                                await websocket.send_text("__scope_result__:<p class='ansi-red'>Error parsing object data</p>")
                    else:
                        await websocket.send_text("__scope_result__:<p class='ansi-red'>No data returned</p>")
                
                except Exception as e:
                    await websocket.send_text("__scope_result__:<p class='ansi-red'>Error inspecting scope: {}</p>".format(str(e)))
            else:
                try:
                    # Regular command execution
                    if hasattr(terminal, 'run_code'):
                        result = await terminal.run_code(data)
                    else:
                        # For AsyncIPythonBridge
                        result = await terminal.run_code(data, highlight=False)
                    
                    # Send the output, or a placeholder if empty
                    if result:
                        await websocket.send_text(result)
                    else:
                        await websocket.send_text("Command executed (no output)\n")
                except Exception as e:
                    error_message = f"Error executing command: {str(e)}"
                    print(f"Terminal error: {error_message}")
                    await websocket.send_text(f"\n{error_message}\n")
    
    except WebSocketDisconnect:
        # Remove from connections
        if terminal_id in terminal_connections:
            if websocket in terminal_connections[terminal_id]:
                terminal_connections[terminal_id].remove(websocket)
    
    except Exception as e:
        # Log error
        print(f"Error in terminal websocket: {e}")
        
        # Try to send error message
        try:
            await websocket.send_text(f"Error: {str(e)}\n")
        except:
            pass

# Application lifecycle
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    # Create static and template directories if they don't exist
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    
    # Create CSS directory
    css_dir = STATIC_DIR / "css"
    os.makedirs(css_dir, exist_ok=True)
    
    # Create basic CSS file if it doesn't exist
    css_file = css_dir / "main.css"
    if not css_file.exists():
        with open(css_file, "w") as f:
            f.write("""
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            
            h1, h2, h3 {
                color: #333;
            }
            
            .config-group {
                margin-bottom: 10px;
            }
            
            .config-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            
            .config-group input {
                width: 100%;
                padding: 5px;
                box-sizing: border-box;
            }
            
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .metric {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
            }
            
            .metric label {
                display: block;
                font-weight: bold;
                margin-bottom: 5px;
            }
            
            .task {
                margin-bottom: 10px;
                padding: 10px;
                border-radius: 5px;
            }
            
            .task-pending {
                background-color: #f5f5f5;
            }
            
            .task-running {
                background-color: #e6f7ff;
            }
            
            .task-completed {
                background-color: #f6ffed;
            }
            
            .task-failed {
                background-color: #fff2f0;
            }
            
            .task-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            }
            
            .task-id {
                font-family: monospace;
                font-weight: bold;
            }
            
            .task-status {
                font-weight: bold;
            }
            
            .task-details {
                font-size: 0.9em;
            }
            
            .task-error {
                color: #f5222d;
                margin-top: 5px;
            }
            """)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    # Stop all terminals
    for terminal_id, terminal in terminals.items():
        await terminal.stop_polling()


def run_server():
    """
    Function to run the web server as an entry point.
    This is used when the package is installed and the command-line script is invoked.
    """
    import os
    import sys
    import argparse
    import uvicorn
    from pathlib import Path
    
    from finite_monkey.nodes_config import nodes_config
    
    # Load configuration
    config = nodes_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Finite Monkey - Web Interface",
    )
    
    parser.add_argument(
        "--host",
        default=config.WEB_HOST or "0.0.0.0",
        help=f"Host to bind to (default: {config.WEB_HOST or '0.0.0.0'})",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=config.WEB_PORT or 8000,
        help=f"Port to bind to (default: {config.WEB_PORT or 8000})",
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    parser.add_argument(
        "--enable-ipython",
        action="store_true",
        help="Enable IPython terminal functionality",
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("=" * 60)
    print("Finite Monkey Engine - Web Interface")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    print(f"IPython: {'Enabled' if args.enable_ipython else 'Disabled'}")
    print("=" * 60)
    
    # Set up database directory
    db_dir = Path("db")
    db_dir.mkdir(exist_ok=True)
    
    # Set up outputs directory
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    # Create config
    config = uvicorn.Config(
        "finite_monkey.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload or args.debug,
        log_level="debug" if args.debug else "info",
    )

    # Run the server - handle both sync and async contexts
    try:
        # Check if we're already in an event loop
        asyncio.get_running_loop()
        # If we are, we need to use Server directly
        server = uvicorn.Server(config)
        loop = asyncio.get_running_loop()
        loop.run_until_complete(server.serve())
    except RuntimeError:
        # No event loop running, we can use the simpler API
        uvicorn.run(
            "finite_monkey.web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload or args.debug,
            log_level="debug" if args.debug else "info",
        )
    
    return 0