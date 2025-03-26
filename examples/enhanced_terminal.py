#!/usr/bin/env python3
"""
Enhanced Jupyter Terminal with Dashboard Features
for the Finite Monkey Engine.

This script creates a comprehensive web interface that includes:
1. Jupyter terminal for code execution
2. Agent status dashboard
3. Project configuration panel
4. Visualization tools
5. Report viewing
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, Request, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("enhanced-terminal")

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TERM'] = 'xterm-256color'
os.environ['FORCE_COLOR'] = '1'

# Add project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Create FastAPI app
app = FastAPI(title="Enhanced Finite Monkey Terminal",
              description="A comprehensive UI for the Finite Monkey Engine")

# Create templates directory
templates_dir = Path(project_dir) / "templates"
if not templates_dir.exists():
    templates_dir.mkdir(parents=True)
    
# Create a basic HTML template if it doesn't exist
template_file = templates_dir / "terminal.html"
if not template_file.exists():
    # Create a simple template
    with open(template_file, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Finite Monkey Terminal</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css">
    <style>
        :root {
            --primary-color: #5D3FD3;
            --secondary-color: #9D4EDD;
            --background-color: #121212;
            --terminal-bg: #1E1E1E;
            --terminal-text: #F8F8F2;
            --scope-bg: #252526;
            --border-color: #333;
            --accent-color: #66D9EF;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            line-height: 1.5;
            color: var(--terminal-text);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }
        
        .sidebar {
            background-color: #1A1A1A;
            color: #ccc;
            padding: 1rem;
            height: 100vh;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
        }
        
        .sidebar-nav {
            list-style: none;
            padding: 0;
        }
        
        .sidebar-nav a {
            color: #ccc;
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            display: block;
            margin-bottom: 0.5rem;
        }
        
        .sidebar-nav a:hover,
        .sidebar-nav a.active {
            background-color: var(--primary-color);
            color: white;
        }
        
        .sidebar-nav i {
            margin-right: 0.5rem;
        }
        
        .terminal-container {
            background-color: var(--terminal-bg);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 1.5rem;
            height: 400px;
            display: flex;
            flex-direction: column;
        }
        
        .terminal-header {
            background-color: var(--primary-color);
            color: white;
            padding: 0.5rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .terminal-output {
            flex-grow: 1;
            padding: 1rem;
            font-family: 'Fira Code', monospace;
            overflow-y: auto;
            color: var(--terminal-text);
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .terminal-input-area {
            padding: 0.5rem 1rem;
            display: flex;
            border-top: 1px solid var(--border-color);
        }
        
        .terminal-input {
            flex-grow: 1;
            background-color: rgba(255, 255, 255, 0.1);
            border: none;
            color: var(--terminal-text);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-family: 'Fira Code', monospace;
        }
        
        .terminal-input:focus {
            outline: none;
            box-shadow: 0 0 0 2px var(--accent-color);
        }
        
        .terminal-submit {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin-left: 0.5rem;
        }
        
        .dashboard-card {
            background-color: var(--scope-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .dashboard-card h3 {
            margin-top: 0;
            color: var(--accent-color);
            margin-bottom: 1rem;
        }
        
        .welcome-message {
            margin-bottom: 1rem;
            padding: 0.75rem;
            border-left: 4px solid var(--accent-color);
            background-color: rgba(102, 217, 239, 0.1);
        }
        
        .agent-status {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
        }
        
        .agent-active {
            background-color: #4CAF50;
            color: white;
        }
        
        .agent-inactive {
            background-color: #F44336;
            color: white;
        }
        
        .agent-pending {
            background-color: #FFC107;
            color: black;
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
        
        .file-card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .file-card:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        
        .file-card-name {
            font-weight: bold;
            color: var(--accent-color);
        }
        
        .file-card-info {
            font-size: 0.8rem;
            color: #AAA;
            margin-top: 0.25rem;
        }
        
        .config-item {
            margin-bottom: 1rem;
        }
        
        .config-item label {
            display: block;
            margin-bottom: 0.25rem;
            color: #AAA;
        }
        
        .config-item input,
        .config-item select,
        .config-item textarea {
            width: 100%;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--terminal-text);
        }
        
        .config-item input:focus,
        .config-item select:focus,
        .config-item textarea:focus {
            outline: none;
            border-color: var(--accent-color);
        }
        
        .visualizations-tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }
        
        .visualizations-tab {
            padding: 0.5rem 1rem;
            cursor: pointer;
            border-bottom: 2px solid transparent;
        }
        
        .visualizations-tab.active {
            border-bottom-color: var(--accent-color);
            color: var(--accent-color);
        }
        
        pre {
            background-color: #333;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 p-0 sidebar">
                <div class="text-center mb-4">
                    <h3>Finite Monkey</h3>
                </div>
                <ul class="sidebar-nav">
                    <li><a href="#" class="active" data-section="terminal"><i class="bi bi-terminal"></i> Terminal</a></li>
                    <li><a href="#" data-section="agents"><i class="bi bi-robot"></i> Agents</a></li>
                    <li><a href="#" data-section="files"><i class="bi bi-file-text"></i> Files</a></li>
                    <li><a href="#" data-section="reports"><i class="bi bi-file-earmark-bar-graph"></i> Reports</a></li>
                    <li><a href="#" data-section="visualizations"><i class="bi bi-graph-up"></i> Visualizations</a></li>
                    <li><a href="#" data-section="config"><i class="bi bi-gear"></i> Configuration</a></li>
                </ul>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 p-4">
                <!-- Terminal Section -->
                <div id="terminal-section" class="content-section">
                    <h2 class="mb-3">IPython Terminal</h2>
                    <p>Interactive Python terminal for the Finite Monkey Engine</p>
                    
                    <div class="welcome-message">
                        <strong>Welcome to the Finite Monkey Terminal!</strong><br>
                        Type Python commands to interact with the framework.<br>
                        <br>
                        Run these commands to initialize the framework objects:<br>
                        <pre>
import asyncio

# Ensure we have a proper event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from finite_monkey.agents.orchestrator import WorkflowOrchestrator
from finite_monkey.agents.researcher import Researcher
from finite_monkey.agents.validator import Validator

# Initialize the agents
orchestrator = WorkflowOrchestrator()
researcher = Researcher()
validator = Validator()

print("✅ Framework objects initialized successfully!")
                        </pre>
                    </div>
                    
                    <div class="terminal-container">
                        <div class="terminal-header">
                            <span>Python</span>
                            <button class="btn btn-sm" onclick="clearTerminal()">
                                <i class="bi bi-trash"></i> Clear
                            </button>
                        </div>
                        <div id="terminal-output" class="terminal-output"></div>
                        <div class="terminal-input-area">
                            <input type="text" id="terminal-input" class="terminal-input" placeholder="Enter Python command...">
                            <button id="terminal-submit" class="terminal-submit">Run</button>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="dashboard-card">
                                <h3>Quick Actions</h3>
                                <div class="d-grid gap-2">
                                    <button class="btn btn-primary mb-2" onclick="runCommand('await orchestrator.run_analysis(\"examples/Vault.sol\")')">
                                        <i class="bi bi-play"></i> Run Analysis on Vault.sol
                                    </button>
                                    <button class="btn btn-secondary mb-2" onclick="runCommand('await researcher.analyze_code_async(\"contract Vault { /* code */ }\")')">
                                        <i class="bi bi-search"></i> Test Researcher Agent
                                    </button>
                                    <button class="btn btn-info" onclick="runCommand('print(validator)')">
                                        <i class="bi bi-info-circle"></i> Print Validator Info
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="dashboard-card">
                                <h3>Recent Commands</h3>
                                <ul id="recent-commands" class="list-group list-group-flush bg-transparent">
                                    <!-- Recent commands will be added here -->
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Agents Section -->
                <div id="agents-section" class="content-section" style="display: none;">
                    <h2 class="mb-3">Agent Management</h2>
                    <p>Monitor and control the framework agents</p>
                    
                    <div class="row">
                        <div class="col-md-4 mb-4">
                            <div class="dashboard-card">
                                <h3>Orchestrator</h3>
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>Status: <span class="agent-status agent-inactive" id="orchestrator-status">Inactive</span></div>
                                    <button class="btn btn-sm btn-primary" onclick="runCommand('orchestrator = WorkflowOrchestrator(); print(\"✅ Orchestrator initialized\")')">Initialize</button>
                                </div>
                                <p>The main workflow coordinator for the framework.</p>
                                <div class="mt-3">
                                    <button class="btn btn-sm btn-secondary mb-2 w-100" onclick="runCommand('await orchestrator.run_analysis(\"examples/Vault.sol\")')">
                                        Run Analysis
                                    </button>
                                    <button class="btn btn-sm btn-secondary w-100" onclick="runCommand('print(orchestrator)')">
                                        Show Details
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-4 mb-4">
                            <div class="dashboard-card">
                                <h3>Researcher</h3>
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>Status: <span class="agent-status agent-inactive" id="researcher-status">Inactive</span></div>
                                    <button class="btn btn-sm btn-primary" onclick="runCommand('researcher = Researcher(); print(\"✅ Researcher initialized\")')">Initialize</button>
                                </div>
                                <p>Analyzes code and generates initial findings.</p>
                                <div class="mt-3">
                                    <button class="btn btn-sm btn-secondary mb-2 w-100" onclick="runCommand('await researcher.analyze_code_async(\"contract Example { uint value; function set(uint v) public { value = v; } }\")')">
                                        Test Analysis
                                    </button>
                                    <button class="btn btn-sm btn-secondary w-100" onclick="runCommand('print(researcher)')">
                                        Show Details
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-4 mb-4">
                            <div class="dashboard-card">
                                <h3>Validator</h3>
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>Status: <span class="agent-status agent-inactive" id="validator-status">Inactive</span></div>
                                    <button class="btn btn-sm btn-primary" onclick="runCommand('validator = Validator(); print(\"✅ Validator initialized\")')">Initialize</button>
                                </div>
                                <p>Validates analysis results and provides feedback.</p>
                                <div class="mt-3">
                                    <button class="btn btn-sm btn-secondary mb-2 w-100" onclick="runCommand('await validator.validate_result(\"This contract has a reentrancy vulnerability\")')">
                                        Test Validation
                                    </button>
                                    <button class="btn btn-sm btn-secondary w-100" onclick="runCommand('print(validator)')">
                                        Show Details
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Files Section -->
                <div id="files-section" class="content-section" style="display: none;">
                    <h2 class="mb-3">File Management</h2>
                    <p>Browse and manage files for analysis</p>
                    
                    <div class="row mb-4">
                        <div class="col">
                            <div class="dashboard-card">
                                <h3>Upload File</h3>
                                <div class="mb-3">
                                    <input class="form-control bg-dark text-light" type="file" id="file-upload">
                                </div>
                                <button class="btn btn-primary" onclick="uploadFile()">Upload</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="dashboard-card">
                                <h3>Example Files</h3>
                                <div class="file-card" onclick="loadFile('examples/Vault.sol')">
                                    <div class="file-card-name">Vault.sol</div>
                                    <div class="file-card-info">Example Vault contract with reentrancy vulnerability</div>
                                </div>
                                <div class="file-card" onclick="loadFile('examples/Token.sol')">
                                    <div class="file-card-name">Token.sol</div>
                                    <div class="file-card-info">Example ERC20 token contract</div>
                                </div>
                                <div class="file-card" onclick="loadFile('examples/StakingPool.sol')">
                                    <div class="file-card-name">StakingPool.sol</div>
                                    <div class="file-card-info">Example staking pool contract</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="col-md-6">
                            <div class="dashboard-card">
                                <h3>Recently Uploaded</h3>
                                <div id="user-files">
                                    <!-- User files will be listed here -->
                                    <p class="text-muted">No files uploaded yet.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Reports Section -->
                <div id="reports-section" class="content-section" style="display: none;">
                    <h2 class="mb-3">Analysis Reports</h2>
                    <p>View and analyze security reports</p>
                    
                    <div class="row">
                        <div class="col">
                            <div class="dashboard-card">
                                <h3>Generated Reports</h3>
                                <div id="reports-list">
                                    <!-- Reports will be listed here -->
                                    <div class="d-flex justify-content-between align-items-center mb-3 p-3 bg-dark rounded">
                                        <div>
                                            <h5>Vault.sol Security Audit</h5>
                                            <div class="text-muted small">Generated on 2025-03-04</div>
                                        </div>
                                        <div>
                                            <button class="btn btn-sm btn-primary" onclick="viewReport('Vault_report_20250304_075110')">View</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Visualizations Section -->
                <div id="visualizations-section" class="content-section" style="display: none;">
                    <h2 class="mb-3">Visualizations</h2>
                    <p>Interactive visualizations of smart contracts</p>
                    
                    <div class="visualizations-tabs">
                        <div class="visualizations-tab active" data-tab="contract-graph">Contract Graph</div>
                        <div class="visualizations-tab" data-tab="call-tree">Call Tree</div>
                        <div class="visualizations-tab" data-tab="vulnerabilities">Vulnerabilities</div>
                    </div>
                    
                    <div class="dashboard-card">
                        <div id="visualization-container" style="height: 500px; background-color: #1A1A1A;">
                            <div class="d-flex justify-content-center align-items-center h-100">
                                <p>Select a contract to visualize</p>
                            </div>
                        </div>
                        
                        <div class="mt-3">
                            <div class="btn-group" role="group">
                                <button class="btn btn-secondary" onclick="generateVisualization('Vault.sol')">Vault.sol</button>
                                <button class="btn btn-secondary" onclick="generateVisualization('Token.sol')">Token.sol</button>
                                <button class="btn btn-secondary" onclick="generateVisualization('StakingPool.sol')">StakingPool.sol</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Configuration Section -->
                <div id="config-section" class="content-section" style="display: none;">
                    <h2 class="mb-3">Framework Configuration</h2>
                    <p>Configure the Finite Monkey Engine</p>
                    
                    <div class="dashboard-card">
                        <h3>LLM Configuration</h3>
                        <div class="row">
                            <div class="col-md-6">
                                <div class="config-item">
                                    <label for="workflow-model">Workflow Model</label>
                                    <select id="workflow-model" class="form-control">
                                        <option value="llama3:8b-instruct">llama3:8b-instruct</option>
                                        <option value="CLAUDE-API">Claude API</option>
                                        <option value="GPT4">GPT-4</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="config-item">
                                    <label for="api-base">API Base URL</label>
                                    <input type="text" id="api-base" class="form-control" value="http://127.0.0.1:11434/v1">
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <div class="config-item">
                                    <label for="api-key">API Key</label>
                                    <input type="password" id="api-key" class="form-control" placeholder="Enter API key">
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="config-item">
                                    <label for="embedding-model">Embedding Model</label>
                                    <input type="text" id="embedding-model" class="form-control" value="BAAI/bge-small-en-v1.5">
                                </div>
                            </div>
                        </div>
                        <button class="btn btn-primary mt-3" onclick="saveConfig()">Save Configuration</button>
                    </div>
                    
                    <div class="dashboard-card mt-4">
                        <h3>Storage Settings</h3>
                        <div class="row">
                            <div class="col-md-6">
                                <div class="config-item">
                                    <label for="vector-store">Vector Store Path</label>
                                    <input type="text" id="vector-store" class="form-control" value="lancedb">
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="config-item">
                                    <label for="db-dir">Database Directory</label>
                                    <input type="text" id="db-dir" class="form-control" value="db">
                                </div>
                            </div>
                        </div>
                        <button class="btn btn-primary mt-3" onclick="saveStorageConfig()">Save Storage Settings</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let socket;
        const terminalOutput = document.getElementById('terminal-output');
        const terminalInput = document.getElementById('terminal-input');
        const recentCommands = document.getElementById('recent-commands');
        let commandHistory = [];
        let commandIndex = -1;
        
        // Connect to WebSocket
        function connectWebSocket() {
            socket = new WebSocket(`ws://${window.location.host}/ws/terminal`);
            
            socket.onopen = function(e) {
                console.log('WebSocket connection established');
                appendToTerminal('<span style="color: #8a8">Connected to terminal.</span><br>');
            };
            
            socket.onmessage = function(event) {
                appendToTerminal(event.data);
                terminalOutput.scrollTop = terminalOutput.scrollHeight;
                
                // Check for agent status updates
                checkAgentStatus();
            };
            
            socket.onclose = function(event) {
                if (event.wasClean) {
                    appendToTerminal(`<span style="color: #f33">Connection closed.</span><br>`);
                } else {
                    appendToTerminal(`<span style="color: #f33">Connection lost. Please refresh the page.</span><br>`);
                }
                
                // Try to reconnect after a delay
                setTimeout(connectWebSocket, 3000);
            };
            
            socket.onerror = function(error) {
                appendToTerminal(`<span style="color: #f33">WebSocket error: ${error.message}</span><br>`);
            };
        }
        
        // Initialize the terminal
        function initTerminal() {
            connectWebSocket();
            
            // Add event listeners
            document.getElementById('terminal-submit').addEventListener('click', sendCommand);
            
            terminalInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    sendCommand();
                } else if (e.key === 'ArrowUp') {
                    navigateHistory(-1);
                    e.preventDefault();
                } else if (e.key === 'ArrowDown') {
                    navigateHistory(1);
                    e.preventDefault();
                }
            });
            
            // Initialize sidebar navigation
            const navLinks = document.querySelectorAll('.sidebar-nav a');
            navLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    // Remove active class from all links
                    navLinks.forEach(l => l.classList.remove('active'));
                    
                    // Add active class to clicked link
                    this.classList.add('active');
                    
                    // Show the corresponding section
                    const sectionId = this.getAttribute('data-section');
                    showSection(sectionId);
                });
            });
            
            // Initialize visualization tabs
            const vizTabs = document.querySelectorAll('.visualizations-tab');
            vizTabs.forEach(tab => {
                tab.addEventListener('click', function() {
                    vizTabs.forEach(t => t.classList.remove('active'));
                    this.classList.add('active');
                    
                    // Show the corresponding visualization
                    const tabId = this.getAttribute('data-tab');
                    // You would normally switch visualizations here
                });
            });
            
            // Focus the input field
            terminalInput.focus();
        }
        
        // Append text to the terminal
        function appendToTerminal(text) {
            terminalOutput.innerHTML += text;
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
        }
        
        // Send a command to the server
        function sendCommand() {
            const command = terminalInput.value.trim();
            if (!command) return;
            
            if (socket.readyState === WebSocket.OPEN) {
                socket.send(command);
                terminalInput.value = '';
                
                // Add to command history
                addToCommandHistory(command);
            } else {
                appendToTerminal('<span style="color: #f33">Not connected to server. Please refresh the page.</span><br>');
            }
        }
        
        // Run a pre-defined command
        function runCommand(command) {
            if (socket.readyState === WebSocket.OPEN) {
                // Set the command in the input field
                terminalInput.value = command;
                
                // Send it
                socket.send(command);
                terminalInput.value = '';
                
                // Add to command history
                addToCommandHistory(command);
            } else {
                appendToTerminal('<span style="color: #f33">Not connected to server. Please refresh the page.</span><br>');
            }
        }
        
        // Add a command to the history
        function addToCommandHistory(command) {
            // Add to internal history array
            commandHistory.unshift(command);
            if (commandHistory.length > 20) {
                commandHistory.pop();
            }
            
            // Reset index
            commandIndex = -1;
            
            // Update the UI
            updateRecentCommands();
        }
        
        // Update the recent commands list
        function updateRecentCommands() {
            if (!recentCommands) return;
            
            recentCommands.innerHTML = '';
            
            // Add the commands to the list
            commandHistory.slice(0, 5).forEach(cmd => {
                const li = document.createElement('li');
                li.className = 'list-group-item bg-transparent text-light border-0 py-1 px-0';
                li.innerHTML = `<button class="btn btn-sm btn-link text-light p-0" onclick="runCommand('${cmd.replace(/'/g, "\\'")}')">${cmd}</button>`;
                recentCommands.appendChild(li);
            });
        }
        
        // Navigate through command history
        function navigateHistory(direction) {
            if (commandHistory.length === 0) return;
            
            commandIndex += direction;
            
            if (commandIndex < -1) {
                commandIndex = commandHistory.length - 1;
            } else if (commandIndex >= commandHistory.length) {
                commandIndex = -1;
            }
            
            if (commandIndex === -1) {
                terminalInput.value = '';
            } else {
                terminalInput.value = commandHistory[commandIndex];
            }
        }
        
        // Clear the terminal
        function clearTerminal() {
            terminalOutput.innerHTML = '';
        }
        
        // Show a specific section
        function showSection(sectionId) {
            // Hide all sections
            const sections = document.querySelectorAll('.content-section');
            sections.forEach(section => {
                section.style.display = 'none';
            });
            
            // Show the requested section
            const section = document.getElementById(`${sectionId}-section`);
            if (section) {
                section.style.display = 'block';
            }
        }
        
        // Check the status of agents
        function checkAgentStatus() {
            // This is a simple implementation that just checks for keywords in the terminal output
            const output = terminalOutput.innerHTML;
            
            // Check for orchestrator initialization
            if (output.includes('Orchestrator initialized') || 
                output.includes('WorkflowOrchestrator()') || 
                output.includes('orchestrator = WorkflowOrchestrator()')) {
                document.getElementById('orchestrator-status').textContent = 'Active';
                document.getElementById('orchestrator-status').className = 'agent-status agent-active';
            }
            
            // Check for researcher initialization
            if (output.includes('Researcher initialized') || 
                output.includes('Researcher()') || 
                output.includes('researcher = Researcher()')) {
                document.getElementById('researcher-status').textContent = 'Active';
                document.getElementById('researcher-status').className = 'agent-status agent-active';
            }
            
            // Check for validator initialization
            if (output.includes('Validator initialized') || 
                output.includes('Validator()') || 
                output.includes('validator = Validator()')) {
                document.getElementById('validator-status').textContent = 'Active';
                document.getElementById('validator-status').className = 'agent-status agent-active';
            }
        }
        
        // Load a file
        function loadFile(path) {
            runCommand(`with open("${path}", "r") as f: print(f.read())`);
        }
        
        // Upload a file
        function uploadFile() {
            const fileInput = document.getElementById('file-upload');
            if (!fileInput.files.length) {
                alert('Please select a file to upload.');
                return;
            }
            
            const file = fileInput.files[0];
            
            // Create a FormData object
            const formData = new FormData();
            formData.append('file', file);
            
            // Use fetch to upload the file
            fetch('/upload-file', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`File "${file.name}" uploaded successfully.`);
                    // Update the user files list
                    updateUserFiles();
                } else {
                    alert(`Error uploading file: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while uploading the file.');
            });
        }
        
        // Update the list of user files
        function updateUserFiles() {
            fetch('/user-files')
            .then(response => response.json())
            .then(data => {
                const userFilesContainer = document.getElementById('user-files');
                if (!data.files || data.files.length === 0) {
                    userFilesContainer.innerHTML = '<p class="text-muted">No files uploaded yet.</p>';
                    return;
                }
                
                userFilesContainer.innerHTML = '';
                data.files.forEach(file => {
                    const fileCard = document.createElement('div');
                    fileCard.className = 'file-card';
                    fileCard.onclick = () => loadFile(file.path);
                    
                    fileCard.innerHTML = `
                        <div class="file-card-name">${file.name}</div>
                        <div class="file-card-info">Uploaded on ${file.uploaded}</div>
                    `;
                    
                    userFilesContainer.appendChild(fileCard);
                });
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('user-files').innerHTML = 
                    '<p class="text-danger">Error loading file list.</p>';
            });
        }
        
        // View a report
        function viewReport(reportId) {
            fetch(`/reports/${reportId}`)
            .then(response => response.text())
            .then(html => {
                // Create a new window or tab and write the HTML content
                const reportWindow = window.open('', '_blank');
                reportWindow.document.write(html);
                reportWindow.document.close();
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error loading report.');
            });
        }
        
        // Generate a visualization
        function generateVisualization(contractName) {
            const vizContainer = document.getElementById('visualization-container');
            
            // Show loading indicator
            vizContainer.innerHTML = '<div class="d-flex justify-content-center align-items-center h-100"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div></div>';
            
            // Execute the command to generate the visualization
            runCommand(`from finite_monkey.visualization import GraphFactory; graph = GraphFactory.analyze_solidity_file("examples/${contractName}"); print("Visualization generated")`);
            
            // For demo, just show a placeholder visualization after a delay
            setTimeout(() => {
                vizContainer.innerHTML = `<div class="d-flex justify-content-center align-items-center h-100"><img src="/static/placeholder-graph.png" alt="Contract visualization" style="max-width: 100%; max-height: 100%"></div>`;
            }, 2000);
        }
        
        // Save configuration
        function saveConfig() {
            const config = {
                workflow_model: document.getElementById('workflow-model').value,
                api_base: document.getElementById('api-base').value,
                api_key: document.getElementById('api-key').value,
                embedding_model: document.getElementById('embedding-model').value
            };
            
            fetch('/save-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Configuration saved successfully.');
                } else {
                    alert(`Error saving configuration: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while saving the configuration.');
            });
        }
        
        // Save storage configuration
        function saveStorageConfig() {
            const config = {
                vector_store: document.getElementById('vector-store').value,
                db_dir: document.getElementById('db-dir').value
            };
            
            fetch('/save-storage-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Storage settings saved successfully.');
                } else {
                    alert(`Error saving storage settings: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while saving the storage settings.');
            });
        }
        
        // Initialize the terminal when the page loads
        window.addEventListener('DOMContentLoaded', initTerminal);
    </script>
</body>
</html>
        """)

# Create templates object
templates = Jinja2Templates(directory=str(templates_dir))

# Create a directory for static files
static_dir = Path(project_dir) / "static"
static_dir.mkdir(exist_ok=True)

# Create a placeholder graph image if it doesn't exist
placeholder_graph = static_dir / "placeholder-graph.png"
if not placeholder_graph.exists():
    # Import or create a placeholder image
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        
        # Create a simple placeholder image
        img = Image.new('RGB', (800, 600), color=(26, 26, 26))
        draw = ImageDraw.Draw(img)
        
        # Draw some circles and lines to simulate a graph
        for i in range(10):
            x = np.random.randint(100, 700)
            y = np.random.randint(100, 500)
            size = np.random.randint(30, 60)
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            draw.ellipse((x-size//2, y-size//2, x+size//2, y+size//2), outline=color, width=3)
            
        # Draw some connecting lines
        for i in range(15):
            x1 = np.random.randint(100, 700)
            y1 = np.random.randint(100, 500)
            x2 = np.random.randint(100, 700)
            y2 = np.random.randint(100, 500)
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            draw.line((x1, y1, x2, y2), fill=color, width=2)
            
        # Add some labels
        for i in range(5):
            x = np.random.randint(100, 700)
            y = np.random.randint(100, 500)
            draw.rectangle((x-40, y-10, x+40, y+10), fill=(40, 40, 40))
            
        # Save the image
        img.save(placeholder_graph)
    except ImportError:
        # If PIL is not available, create an empty file
        with open(placeholder_graph, 'wb') as f:
            f.write(b'')

# Serve static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Import Jupyter kernel functionality
try:
    from jupyter_client import KernelManager
    has_jupyter = True
except ImportError:
    has_jupyter = False
    logger.warning("jupyter_client not found. Will run with simulated output.")

# Terminal sessions store
terminal_sessions = {}

# Jupyter kernel output queue
output_queue = asyncio.Queue()

class JupyterKernel:
    """Simple wrapper for Jupyter kernel interactions."""
    
    def __init__(self):
        """Initialize the kernel."""
        self.km = None
        self.kc = None
        self.initialized = False
        
    async def start(self):
        """Start the Jupyter kernel."""
        if not has_jupyter:
            logger.warning("Jupyter client not available. Using simulated output.")
            self.initialized = True
            return True
            
        try:
            # Start the kernel
            self.km = KernelManager()
            self.km.start_kernel()
            
            # Get the kernel client
            self.kc = self.km.client()
            self.kc.start_channels()
            
            # Wait for the kernel to be ready
            self.kc.wait_for_ready(timeout=300)
            
            # Mark as initialized
            self.initialized = True
            
            # Start the output polling task
            asyncio.create_task(self._poll_output())
            
            logger.info("Jupyter kernel started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting Jupyter kernel: {e}")
            return False
    
    async def execute(self, code):
        """Execute code in the kernel."""
        if not self.initialized:
            return f"<span style='color: #f55;'>Kernel not initialized</span>"
            
        if not has_jupyter:
            # Simulate output
            return f"<span style='color: #5f5;'>&gt;&gt;&gt; {code}</span><br>Simulated output (Jupyter client not available)"
        
        try:
            # Execute the code
            msg_id = self.kc.execute(code)
            
            # Return the message ID for tracking
            return msg_id
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return f"<span style='color: #f55;'>Error executing code: {e}</span>"
    
    async def _poll_output(self):
        """Poll for output from the kernel."""
        if not has_jupyter or not self.initialized:
            return
            
        try:
            while True:
                # Get messages from the kernel
                try:
                    msg = self.kc.get_iopub_msg(timeout=0.1)
                    
                    # Process the message
                    await self._process_message(msg)
                except:
                    # No message available
                    pass
                
                # Sleep to avoid CPU hogging
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in output polling: {e}")
    
    async def _process_message(self, msg):
        """Process a message from the kernel."""
        # Check the message type
        msg_type = msg['header']['msg_type']
        content = msg['content']
        
        if msg_type == 'stream':
            # Text output
            stream_name = content['name']  # 'stdout' or 'stderr'
            text = content['text']
            
            # Format the output
            if stream_name == 'stderr':
                formatted = f"<span class='ansi-red'>{text}</span>"
            else:
                formatted = text
                
            # Add to the queue
            await output_queue.put(formatted)
            
        elif msg_type == 'execute_result':
            # Execution result
            data = content['data']
            
            # Check for different representations
            if 'text/html' in data:
                html = data['text/html']
                await output_queue.put(html)
            elif 'text/plain' in data:
                text = data['text/plain']
                await output_queue.put(text)
                
        elif msg_type == 'display_data':
            # Display data
            data = content['data']
            
            # Check for different representations
            if 'text/html' in data:
                html = data['text/html']
                await output_queue.put(html)
            elif 'text/plain' in data:
                text = data['text/plain']
                await output_queue.put(text)
                
        elif msg_type == 'error':
            # Error message
            ename = content['ename']
            evalue = content['evalue']
            traceback = content.get('traceback', [])
            
            # Format the error
            error_msg = f"<span class='ansi-red'>{ename}: {evalue}</span><br>"
            for tb_line in traceback:
                # Remove ANSI escape sequences
                tb_line = tb_line.replace('<', '&lt;').replace('>', '&gt;')
                error_msg += f"{tb_line}<br>"
                
            # Add to the queue
            await output_queue.put(error_msg)
    
    def shutdown(self):
        """Shutdown the kernel."""
        if not has_jupyter or not self.initialized:
            return
            
        try:
            if self.kc:
                self.kc.stop_channels()
            if self.km:
                self.km.shutdown_kernel()
                
            logger.info("Jupyter kernel shut down")
        except Exception as e:
            logger.error(f"Error shutting down kernel: {e}")

# Initialize the kernel
kernel = JupyterKernel()

@app.on_event("startup")
async def startup_event():
    """Start the kernel on application startup."""
    await kernel.start()

@app.on_event("shutdown")
def shutdown_event():
    """Shutdown the kernel on application shutdown."""
    kernel.shutdown()

@app.get("/", response_class=HTMLResponse)
async def get_terminal(request: Request):
    """Return the terminal HTML page."""
    return templates.TemplateResponse("terminal.html", {"request": request})

@app.websocket("/ws/terminal")
async def websocket_terminal(websocket: WebSocket):
    """WebSocket endpoint for terminal communication."""
    await websocket.accept()
    
    # Generate a unique session ID
    session_id = str(id(websocket))
    terminal_sessions[session_id] = websocket
    
    # Send welcome message
    await websocket.send_text(f"<span style='color: #5D3FD3;'>Terminal session started. Type Python commands to interact with the framework.</span><br>")
    
    # Start output forwarding task
    output_task = asyncio.create_task(forward_output(websocket))
    
    try:
        # Handle incoming messages
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            # Echo the command
            await websocket.send_text(f"<span class='ansi-bright-green'>&gt;&gt;&gt; {message}</span><br>")
            
            # Execute the command
            await kernel.execute(message)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up
        output_task.cancel()
        terminal_sessions.pop(session_id, None)

async def forward_output(websocket: WebSocket):
    """Forward kernel output to the WebSocket."""
    try:
        while True:
            # Get output from the queue
            try:
                # Use a short timeout to avoid blocking
                if not output_queue.empty():
                    output = await output_queue.get()
                    if output:
                        await websocket.send_text(output)
            except Exception as e:
                logger.error(f"Error getting output: {e}")
            
            # Sleep before checking again
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # Task was cancelled, clean up
        pass
    except Exception as e:
        logger.error(f"Error forwarding output: {e}")

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the server."""
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path(project_dir) / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        # Save the file
        file_path = uploads_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        return {"success": True, "path": str(file_path)}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {"success": False, "error": str(e)}

@app.get("/user-files")
async def get_user_files():
    """Get a list of user-uploaded files."""
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path(project_dir) / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        # Get all files in the directory
        files = []
        for file_path in uploads_dir.glob("*"):
            if file_path.is_file():
                # Get file stats
                stats = file_path.stat()
                
                # Add to the list
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": stats.st_size,
                    "uploaded": stats.st_mtime
                })
                
        return {"files": files}
    except Exception as e:
        logger.error(f"Error getting user files: {e}")
        return {"error": str(e)}

@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Get a report by ID."""
    try:
        # Reports directory path
        reports_dir = Path(project_dir) / "reports"
        
        # Check for report file
        report_file = reports_dir / f"{report_id}.md"
        if report_file.exists():
            # Create a simple HTML page from the markdown
            with open(report_file, "r") as f:
                content = f.read()
                
            # Convert markdown to HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Report: {report_id}</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 900px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    
                    h1, h2, h3, h4, h5, h6 {{
                        color: #5D3FD3;
                    }}
                    
                    pre {{
                        background-color: #f5f5f5;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                    
                    code {{
                        font-family: 'Courier New', monospace;
                    }}
                    
                    blockquote {{
                        border-left: 4px solid #5D3FD3;
                        padding-left: 15px;
                        color: #666;
                    }}
                    
                    img {{
                        max-width: 100%;
                    }}
                    
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                    }}
                    
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    
                    th {{
                        background-color: #f2f2f2;
                    }}
                </style>
            </head>
            <body>
                {markdown.markdown(content, extensions=['extra', 'codehilite'])}
            </body>
            </html>
            """
            
            return HTMLResponse(content=html_content)
        else:
            # Report not found
            return HTMLResponse(content="<h1>Report not found</h1>", status_code=404)
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.post("/save-config")
async def save_config(request: Request):
    """Save configuration settings."""
    try:
        # Get the configuration from the request
        config = await request.json()
        
        # Create config directory if it doesn't exist
        config_dir = Path(project_dir) / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Save the configuration to a file
        with open(config_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        return {"success": True}
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return {"success": False, "error": str(e)}

@app.post("/save-storage-config")
async def save_storage_config(request: Request):
    """Save storage configuration settings."""
    try:
        # Get the configuration from the request
        config = await request.json()
        
        # Create config directory if it doesn't exist
        config_dir = Path(project_dir) / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Save the configuration to a file
        with open(config_dir / "storage.json", "w") as f:
            json.dump(config, f, indent=4)
            
        return {"success": True}
    except Exception as e:
        logger.error(f"Error saving storage configuration: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Run the application."""
    port = 8888  # Use a different port to avoid conflicts
    
    # Create logs directory
    logs_dir = Path(project_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(logs_dir / "enhanced_terminal.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log the startup
    logger.info(f"Starting Enhanced Terminal on port {port}")
    logger.info(f"Open http://localhost:{port} in your browser")
    
    # Open browser automatically
    import webbrowser
    webbrowser.open(f"http://localhost:{port}")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()