"""
FastHTML-based web interface for the Finite Monkey Engine.
This replaces the previous web interface with a more maintainable implementation.
"""

from fasthtml.common import *
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import asyncio
import uuid
import json
import os
import sys
import re
import markdown
import glob
from pathlib import Path

# Import terminal integration
from finite_monkey.web.ui.jupyter_terminal import JupyterTerminal
from finite_monkey.agents.orchestrator import WorkflowOrchestrator
from finite_monkey.agents.researcher import Researcher
from finite_monkey.agents.validator import Validator

# Import visualizations module
from finite_monkey.web import visualizations

# Import code editor component
from finite_monkey.web.components import CodeEditor

# Configure SQLAlchemy
Base = declarative_base()
DB_URI = os.environ.get('FINITE_MONKEY_DB_URI', 'sqlite:///finite_monkey.db')
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)

# Terminal sessions storage
terminal_connections = {}

# Define SQLAlchemy models
class TerminalSession(Base):
    __tablename__ = 'terminal_sessions'
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    commands = relationship("TerminalCommand", back_populates="session", cascade="all, delete-orphan")

class TerminalCommand(Base):
    __tablename__ = 'terminal_commands'
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey('terminal_sessions.id'), nullable=False)
    command = Column(Text, nullable=False)
    output = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session = relationship("TerminalSession", back_populates="commands")

# Create database tables if they don't exist
Base.metadata.create_all(engine)

# CSS for the application
terminal_css = """
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

.app-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1rem;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}

.header h1 {
    font-size: 1.8rem;
    margin: 0;
    color: var(--accent-color);
    font-weight: 700;
}

.header-controls {
    display: flex;
    gap: 1rem;
}

.header-controls a, .header-controls button {
    background-color: var(--primary-color);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    text-decoration: none;
    border: none;
    cursor: pointer;
    font-size: 0.9rem;
    transition: background-color 0.2s;
}

.header-controls a:hover, .header-controls button:hover {
    background-color: var(--secondary-color);
}

.terminal-container {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 200px);
    min-height: 500px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}

.terminal-header {
    background-color: var(--primary-color);
    padding: 0.5rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.terminal-header h2 {
    margin: 0;
    font-size: 1rem;
    color: white;
}

.terminal-header-controls {
    display: flex;
    gap: 0.5rem;
}

.terminal-header-controls button {
    background: none;
    border: none;
    color: white;
    cursor: pointer;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.25rem;
}

.terminal-header-controls button:hover {
    text-decoration: underline;
}

.terminal-wrapper {
    display: flex;
    flex-grow: 1;
    overflow: hidden;
}

.terminal {
    background-color: var(--terminal-bg);
    color: var(--terminal-text);
    font-family: 'Fira Code', 'Menlo', 'Monaco', 'Courier New', monospace;
    line-height: 1.5;
    padding: 1rem;
    flex: 3;
    overflow-y: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    height: 100%;
    box-sizing: border-box;
}

.scope-inspector {
    width: 300px;
    min-width: 250px;
    flex: 1;
    max-width: 350px;
    background-color: var(--scope-bg);
    color: var(--terminal-text);
    border-left: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    height: 100%;
}

.scope-inspector h3 {
    margin: 0;
    padding: 10px;
    background-color: var(--primary-color);
    color: white;
    font-size: 14px;
    font-weight: 500;
}

.scope-content {
    flex-grow: 1;
    padding: 10px;
    overflow-y: auto;
    font-family: 'Fira Code', 'Menlo', 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    height: calc(100% - 70px);
}

.scope-item {
    margin-bottom: 8px;
    padding: 4px;
    border-radius: 3px;
    cursor: pointer;
    user-select: none;
    transition: background-color 0.2s;
}

.scope-item:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

.scope-item-key {
    color: var(--accent-color);
    display: block;
    font-weight: bold;
    margin-bottom: 3px;
    word-break: break-word;
}

.scope-item-type {
    color: #A6E22E;
    font-size: 11px;
    margin-left: 4px;
}

.scope-item-value {
    color: #E6DB74;
    margin-left: 15px;
    display: none;
    overflow-wrap: break-word;
    word-break: break-all;
    max-width: calc(100% - 20px);
    padding: 4px;
    background-color: rgba(0, 0, 0, 0.2);
    border-radius: 3px;
    margin-top: 5px;
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
    padding: 5px 10px;
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-size: 0.8rem;
    transition: background-color 0.2s;
}

.refresh-button:hover {
    background-color: var(--secondary-color);
}

.terminal-input-area {
    display: flex;
    padding: 0.5rem;
    background-color: var(--terminal-bg);
    border-top: 1px solid var(--border-color);
}

.terminal-input {
    flex: 1;
    font-family: 'Fira Code', 'Menlo', 'Monaco', 'Courier New', monospace;
    padding: 0.5rem;
    background-color: rgba(255, 255, 255, 0.05);
    color: var(--terminal-text);
    border: 1px solid var(--border-color);
    border-radius: 4px;
}

.terminal-submit {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    margin-left: 0.5rem;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.terminal-submit:hover {
    background-color: var(--secondary-color);
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

.welcome-message {
    margin-bottom: 1rem;
    padding: 0.75rem;
    border-left: 4px solid var(--accent-color);
    background-color: rgba(102, 217, 239, 0.1);
}

.dashboard {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.dashboard-card {
    background-color: var(--terminal-bg);
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}

.dashboard-card h3 {
    margin-top: 0;
    font-size: 1.2rem;
    color: var(--accent-color);
    margin-bottom: 1rem;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
}

.metric-item {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 1rem;
    border-radius: 6px;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: bold;
    color: var(--accent-color);
    margin-bottom: 0.25rem;
}

.metric-label {
    font-size: 0.85rem;
    color: #AAA;
}

.chart-container {
    height: 200px;
    margin-top: 1rem;
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

.code-editor-wrapper {
    height: 300px;
    background-color: var(--terminal-bg);
    border-radius: 8px;
    overflow: hidden;
}

.output-preview {
    height: 150px;
    background-color: var(--scope-bg);
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
    overflow: auto;
    font-family: 'Fira Code', 'Menlo', 'Monaco', 'Courier New', monospace;
    font-size: 0.9rem;
}

.editor-controls {
    display: flex;
    justify-content: flex-end;
    margin-top: 1rem;
    gap: 0.5rem;
}

/* Reports styles */
.reports-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
}

.report-card {
    background-color: var(--terminal-bg);
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: pointer;
}

.report-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
}

.report-card h3 {
    color: var(--accent-color);
    margin-top: 0;
    margin-bottom: 10px;
    font-size: 1.2rem;
}

.report-card-meta {
    color: #aaa;
    font-size: 0.85rem;
    margin-bottom: 10px;
}

.report-card-description {
    font-size: 0.9rem;
    line-height: 1.4;
}

.report-detail {
    background-color: var(--terminal-bg);
    border-radius: 8px;
    padding: 2rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    margin: 1rem 0;
    max-width: 100%;
    overflow-x: auto;
}

.report-content {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    line-height: 1.6;
}

.report-content h1 {
    font-size: 2rem;
    color: var(--accent-color);
    margin-top: 0;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}

.report-content h2 {
    font-size: 1.5rem;
    color: var(--accent-color);
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.report-content h3 {
    font-size: 1.25rem;
    color: var(--accent-color);
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

.report-content h4 {
    font-size: 1.1rem;
    color: var(--accent-color);
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
}

.report-content p {
    margin-bottom: 1rem;
}

.report-content ul, .report-content ol {
    margin-bottom: 1rem;
    padding-left: 2rem;
}

.report-content li {
    margin-bottom: 0.5rem;
}

.report-content code {
    font-family: 'Fira Code', monospace;
    background-color: rgba(0, 0, 0, 0.2);
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-size: 0.9em;
}

.report-content pre {
    background-color: rgba(0, 0, 0, 0.2);
    padding: 1rem;
    border-radius: 5px;
    overflow-x: auto;
    margin-bottom: 1rem;
}

.report-content pre code {
    background-color: transparent;
    padding: 0;
    border-radius: 0;
    font-size: 0.9rem;
    display: block;
}

.report-content blockquote {
    border-left: 4px solid var(--primary-color);
    padding-left: 1rem;
    margin-left: 0;
    color: #aaa;
}

.report-content table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1rem;
}

.report-content th, .report-content td {
    border: 1px solid var(--border-color);
    padding: 0.5rem;
}

.report-content th {
    background-color: rgba(0, 0, 0, 0.2);
}

.report-content img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1rem auto;
    border-radius: 5px;
}

.report-content hr {
    border: none;
    border-top: 1px solid var(--border-color);
    margin: 2rem 0;
}

.report-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    margin-top: 2rem;
}

.severity {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.85rem;
    font-weight: bold;
    margin-right: 0.5rem;
}

.severity-critical {
    background-color: #FF5252;
    color: white;
}

.severity-high {
    background-color: #FF9100;
    color: white;
}

.severity-medium {
    background-color: #FFEB3B;
    color: black;
}

.severity-low {
    background-color: #4CAF50;
    color: white;
}

.severity-info {
    background-color: #2196F3;
    color: white;
}

.pagination {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin: 2rem 0;
}

.pagination button {
    background-color: var(--terminal-bg);
    border: 1px solid var(--border-color);
    color: var(--terminal-text);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
}

.pagination button.active {
    background-color: var(--primary-color);
    border-color: var(--primary-color);
}

.pagination button:hover:not(.active) {
    background-color: rgba(255, 255, 255, 0.1);
}

.search-tools {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.search-tools input {
    flex-grow: 1;
    background-color: var(--terminal-bg);
    border: 1px solid var(--border-color);
    color: var(--terminal-text);
    padding: 0.5rem 1rem;
    border-radius: 4px;
}

.filter-dropdown {
    background-color: var(--terminal-bg);
    border: 1px solid var(--border-color);
    color: var(--terminal-text);
    padding: 0.5rem 1rem;
    border-radius: 4px;
}

@media (max-width: 768px) {
    .terminal-wrapper {
        flex-direction: column;
    }
    
    .scope-inspector {
        max-width: 100%;
        height: 200px;
        border-left: none;
        border-top: 1px solid var(--border-color);
    }
    
    .dashboard {
        grid-template-columns: 1fr;
    }
    
    .reports-list {
        grid-template-columns: 1fr;
    }
    
    .search-tools {
        flex-direction: column;
    }
}
"""

# Terminal-specific JavaScript
terminal_js = """
document.addEventListener('DOMContentLoaded', function() {
    // Toggle scope item expansion when clicked
    document.addEventListener('click', function(event) {
        if (event.target.classList.contains('scope-item-key') || 
            event.target.classList.contains('scope-item')) {
            const item = event.target.closest('.scope-item');
            if (item) {
                item.classList.toggle('expanded');
                
                // If expanded and not loaded yet, load the details
                if (item.classList.contains('expanded')) {
                    const loader = item.querySelector('.scope-loader');
                    if (loader) {
                        const name = item.getAttribute('data-name');
                        
                        // Send WebSocket message to get object details
                        if (window.ws && window.ws.readyState === WebSocket.OPEN) {
                            window.ws.send(`__scope_inspect__:vars().get('${name}', {})`);
                        }
                    }
                }
            }
        }
    });
    
    // Allow Ctrl+Enter in input field to submit the form
    const terminalInput = document.getElementById('terminal-input');
    if (terminalInput) {
        terminalInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.getElementById('terminal-form').requestSubmit();
            }
        });
    }
    
    // Auto-scroll terminal to bottom when new content is added
    const terminal = document.getElementById('terminal');
    if (terminal) {
        const observer = new MutationObserver(function() {
            terminal.scrollTop = terminal.scrollHeight;
            
            // If we have an output preview, update it with the last terminal output
            const outputPreview = document.getElementById('output-preview');
            if (outputPreview && outputPreview.querySelector('.running-indicator')) {
                // Get the last few lines added to the terminal
                const terminalContent = terminal.innerHTML;
                const lastContent = terminal.lastChild;
                
                if (lastContent && lastContent.textContent && !lastContent.textContent.includes('Running code from editor')) {
                    // Format the content for the preview
                    outputPreview.innerHTML = lastContent.outerHTML;
                }
            }
        });
        
        observer.observe(terminal, { childList: true, subtree: true });
    }

    // Clear terminal button
    const clearBtn = document.getElementById('clear-terminal');
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            const terminal = document.getElementById('terminal');
            if (terminal) {
                // Keep only the welcome message if it exists
                const welcome = terminal.querySelector('.welcome-message');
                terminal.innerHTML = '';
                if (welcome) {
                    terminal.appendChild(welcome);
                }
            }
        });
    }
    
    // Make WebSocket available globally for the code editor
    document.addEventListener('htmx:wsOpen', function(event) {
        // Store the websocket globally for access from the code editor
        window.ws = event.detail.socketWrapper;
        console.log('WebSocket opened and stored globally');
    });
});

// Process ANSI color codes
function processAnsiCodes(text) {
    // ANSI code pattern
    const pattern = /\\u001b\\[(\\d+)m([^\\u001b]*)/g;
    let result = '';
    let lastIndex = 0;
    let match;
    
    // Color mapping
    const colorMap = {
        '30': 'ansi-black', '31': 'ansi-red', '32': 'ansi-green',
        '33': 'ansi-yellow', '34': 'ansi-blue', '35': 'ansi-magenta',
        '36': 'ansi-cyan', '37': 'ansi-white',
        '90': 'ansi-bright-black', '91': 'ansi-bright-red', '92': 'ansi-bright-green',
        '93': 'ansi-bright-yellow', '94': 'ansi-bright-blue', '95': 'ansi-bright-magenta',
        '96': 'ansi-bright-cyan', '97': 'ansi-bright-white'
    };
    
    while ((match = pattern.exec(text)) !== null) {
        // Add text before this match
        result += text.substring(lastIndex, match.index);
        
        // Extract code and content
        const code = match[1];
        const content = match[2];
        
        // Get the appropriate CSS class
        const cssClass = colorMap[code] || '';
        
        // Add styled content
        if (cssClass) {
            result += `<span class="${cssClass}">${content}</span>`;
        } else {
            result += content;
        }
        
        lastIndex = pattern.lastIndex;
    }
    
    // Add remaining text
    result += text.substring(lastIndex);
    
    // Replace newlines with breaks
    return result.replace(/\\n/g, '<br>');
}
"""

# Initialize FastHTML app with custom headers
app = FastHTML(
    exts='ws',  # Enable WebSocket extension
    hdrs=(
        Style(terminal_css),
        Script(terminal_js),
        # Font for code and terminal
        Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Inter:wght@400;500;700&display=swap"),
        # Icon library
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"),
    )
)
rt = app.route

# Helper functions
async def get_or_create_terminal(terminal_id):
    """Get or create an IPython terminal instance."""
    if terminal_id in terminal_connections:
        return terminal_connections[terminal_id]
    
    try:
        # Create namespace with available agents
        namespace = {
            'orchestrator': WorkflowOrchestrator(),
            'researcher': Researcher(),
            'validator': Validator(),
            # Add other relevant objects to the namespace
            'sys': sys,
            'os': os,
            'json': json,
            're': re,
        }
        
        # Create terminal
        terminal = AsyncIPythonBridge(namespace)
        await terminal.start_polling()
        
        # Store terminal connection
        terminal_connections[terminal_id] = terminal
        
        # Create database record
        session = Session()
        try:
            terminal_session = TerminalSession(id=terminal_id, user_id="default_user")
            session.add(terminal_session)
            session.commit()
        except Exception as e:
            print(f"Error storing terminal session: {e}")
            session.rollback()
        finally:
            session.close()
        
        return terminal
    except Exception as e:
        print(f"Error creating terminal: {e}")
        raise

def process_ansi_codes(text):
    """Convert ANSI color codes to HTML spans with appropriate classes."""
    # Mapping of ANSI color codes to CSS classes
    color_map = {
        '30': 'ansi-black', '31': 'ansi-red', '32': 'ansi-green',
        '33': 'ansi-yellow', '34': 'ansi-blue', '35': 'ansi-magenta',
        '36': 'ansi-cyan', '37': 'ansi-white', '0': 'ansi-reset',
        # Add bright variants
        '90': 'ansi-bright-black', '91': 'ansi-bright-red', '92': 'ansi-bright-green',
        '93': 'ansi-bright-yellow', '94': 'ansi-bright-blue', '95': 'ansi-bright-magenta',
        '96': 'ansi-bright-cyan', '97': 'ansi-bright-white'
    }
    
    # Implementation with regex pattern matching
    pattern = re.compile(r'\x1b\[(\d+)m(.*?)(?=\x1b|\Z)')
    result = ""
    last_end = 0
    
    for match in pattern.finditer(text):
        # Add text before the color code
        result += text[last_end:match.start()]
        
        # Extract code and content
        code = match.group(1)
        content = match.group(2)
        
        # Add colored content
        css_class = color_map.get(code, '')
        if css_class:
            result += f'<span class="{css_class}">{content}</span>'
        else:
            result += content
            
        last_end = match.end()
    
    # Add any remaining text
    result += text[last_end:]
    
    # Replace newlines with line breaks for proper display
    result = result.replace('\n', '<br>')
    
    return result

def is_system_json(text):
    """Check if this is system JSON output that should be filtered."""
    if not text or not text.strip().startswith('{'):
        return False
        
    # Check for known system JSON patterns
    patterns = ['terminal_ready', 'json.dumps', '__scope_inspect__', '__pycache__']
    return any(pattern in text for pattern in patterns)

async def process_scope_command(terminal, command):
    """Process a scope inspection command and return formatted HTML."""
    # Execute the command to get data
    result = await terminal.run_code(f"import json; print(json.dumps({command}))", highlight=False)
    
    # Parse the results
    try:
        data = json.loads(result)
        
        if command == "dir()":
            # Format directory listing as HTML
            html = "<div class='scope-list'>"
            
            # Filter and categorize items
            filtered_items = [item for item in data if not (item.startswith('_') and item != '__main__')]
            
            # Categorize items
            categories = {
                "Agents": [item for item in filtered_items if item in ['orchestrator', 'researcher', 'validator', 'documentor']],
                "Data": [item for item in filtered_items if item in ['config', 'task_manager']],
                "Utilities": [item for item in filtered_items if item in ['os', 'sys', 're', 'json']],
                "Other": [item for item in filtered_items if item not in [
                    'orchestrator', 'researcher', 'validator', 'documentor', 
                    'config', 'task_manager', 'os', 'sys', 're', 'json']]
            }
            
            # Generate HTML
            for category, items in categories.items():
                if items:
                    html += f"<h4>{category}</h4>"
                    for item in sorted(items):
                        html += f"""
                        <div class="scope-item" data-name="{item}">
                            <span class="scope-item-key">{item}</span>
                            <div class="scope-item-value">
                                <span class="scope-loader">Loading...</span>
                            </div>
                        </div>
                        """
            
            html += "</div>"
            return html
        else:
            # Get type of the object
            try:
                obj_type = "object"
                if isinstance(data, dict):
                    html = "<div class='object-details'>"
                    for key, value in sorted(data.items()):
                        if not key.startswith('_'):  # Skip private attributes
                            # Get a simplified type name
                            type_name = type(value).__name__ if isinstance(value, (int, float, str, bool, list, dict)) else "object"
                            
                            # Truncate long values
                            value_str = str(value)
                            if len(value_str) > 100:
                                value_str = value_str[:100] + "..."
                                
                            html += f"""
                            <div class="attribute">
                                <span class="attr-name">{key}</span>
                                <span class="attr-type">{type_name}</span>
                                <span class="attr-value">{value_str}</span>
                            </div>
                            """
                    html += "</div>"
                else:
                    # Format simple object or array as JSON
                    html = f"<pre>{json.dumps(data, indent=2)}</pre>"
                
                return html
            except Exception as e:
                return f"<p class='error'>Error processing object: {str(e)}</p>"
            
    except json.JSONDecodeError:
        if "Error" in result:
            return f"<p class='error'>{result}</p>"
        return "<p class='error'>Error parsing scope data</p>"

async def on_connect(send, terminal_id):
    """Handle WebSocket connection initiation."""
    # Get terminal for this session
    terminal = await get_or_create_terminal(terminal_id)
    
    # Send welcome message
    welcome_html = """
    <div class="welcome-message">
        <strong>Welcome to the Finite Monkey IPython Terminal!</strong><br>
        Type Python commands to interact with the framework.<br>
        Available objects: orchestrator, researcher, validator
    </div>
    """
    await send(Div(NotStr(welcome_html), id="terminal", hx_swap_oob="beforeend"))
    
    # Initialize scope inspector
    asyncio.create_task(refresh_scope(terminal_id, terminal, send))

async def refresh_scope(terminal_id, terminal, send):
    """Periodically refresh the scope inspector."""
    while terminal_id in terminal_connections:
        try:
            result = await process_scope_command(terminal, "dir()")
            await send(Div(NotStr(result), id="scope-content"))
        except Exception as e:
            print(f"Error refreshing scope: {e}")
        
        # Wait before next refresh
        await asyncio.sleep(15)  # Refresh every 15 seconds

# Define WebSocket route
async def on_disconnect(terminal_id):
    """Handle WebSocket disconnection."""
    print(f"Terminal disconnected: {terminal_id}")

@app.ws('/ws/terminal/{terminal_id}')
async def terminal_ws(terminal_id: str, msg: str = None, send = None):
    """WebSocket handler for terminal communication."""
    # Handle connection event (no message)
    if msg is None:
        # This is a connection event
        await on_connect(send, terminal_id)
        return

    terminal = await get_or_create_terminal(terminal_id)
    
    # Process different message types
    if msg.startswith("__scope_inspect__:"):
        # Handle scope inspection request
        scope_command = msg.split(":", 1)[1].strip()
        result = await process_scope_command(terminal, scope_command)
        return Div(NotStr(result), id="scope-content")
    
    elif msg == "\x03":  # Ctrl+C
        # Handle interrupt
        if hasattr(terminal, 'send_input'):
            terminal.send_input("\x03")
        elif hasattr(terminal.terminal, 'send_input'):
            terminal.terminal.send_input("\x03")
        return Div("^C", cls="ansi-red", id="terminal", hx_swap_oob="beforeend")
    
    else:
        # Regular command
        # Echo command to terminal
        command_echo = Div(NotStr(f'<span class="ansi-bright-green">&gt;&gt;&gt; {msg}</span><br>'), 
                         id="terminal", hx_swap_oob="beforeend")
        
        try:
            # Execute command
            result = await terminal.run_code(msg)
            
            # Store command in database
            session = Session()
            try:
                cmd = TerminalCommand(session_id=terminal_id, command=msg, output=result)
                session.add(cmd)
                session.commit()
            except Exception as e:
                print(f"Error storing command: {e}")
                session.rollback()
            finally:
                session.close()
            
            # Process ANSI colors for output
            processed_result = process_ansi_codes(result)
            
            # Filter out system JSON responses
            if is_system_json(processed_result):
                return command_echo  # Just return the echo without the filtered output
            
            # Return command output
            return command_echo, Div(NotStr(processed_result), id="terminal", hx_swap_oob="beforeend")
        except Exception as e:
            # Handle errors
            error_message = f"Error executing command: {str(e)}"
            print(error_message)
            return command_echo, Div(NotStr(f'<span class="ansi-red">{error_message}</span><br>'), 
                                    id="terminal", hx_swap_oob="beforeend")

# Report handling functions
def get_all_reports():
    """Get a list of all security audit reports in the reports directory."""
    reports_dir = os.path.join(os.getcwd(), "reports")
    report_files = glob.glob(os.path.join(reports_dir, "*.md"))
    
    reports = []
    for report_file in report_files:
        filename = os.path.basename(report_file)
        
        # Extract project name and date from filename
        parts = filename.split('_')
        if len(parts) >= 3:
            # Handle files like Vault_report_20250304_075110.md
            project_name = parts[0]
            report_type = parts[1] if parts[1] != "report" else "Security Audit"
            
            # Parse timestamp
            try:
                timestamp_parts = '_'.join(parts[2:]).replace('.md', '')
                if len(timestamp_parts) >= 15:  # Format: 20250304_075110
                    date = f"{timestamp_parts[4:6]}/{timestamp_parts[6:8]}/{timestamp_parts[0:4]}"
                    time = f"{timestamp_parts[9:11]}:{timestamp_parts[11:13]}:{timestamp_parts[13:15]}"
                    formatted_date = f"{date} {time}"
                else:
                    formatted_date = timestamp_parts
            except:
                formatted_date = timestamp_parts
                
        else:
            # Handle simpler filenames like default_report_20250310_193858.md
            project_name = "Unknown"
            if "report" in filename:
                report_type = "Security Audit"
            else:
                report_type = "Analysis"
            formatted_date = filename.replace('.md', '')
            
        # Read first few lines to get a summary
        summary = ""
        try:
            with open(report_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    # Extract title from first heading
                    title = lines[0].strip('#').strip()
                    
                    # Look for an executive summary
                    for i, line in enumerate(lines):
                        if "Executive Summary" in line or "Summary" in line:
                            if i+1 < len(lines):
                                summary = lines[i+1].strip()
                                break
                                
                    # If no summary found, use first paragraph after title
                    if not summary:
                        for line in lines[1:]:
                            if line.strip() and not line.startswith('#'):
                                summary = line.strip()
                                break
                                
                    # Limit summary length
                    if len(summary) > 150:
                        summary = summary[:150] + "..."
        except Exception as e:
            print(f"Error parsing report {filename}: {e}")
            title = filename
            summary = "Error parsing report content"
            
        # Construct report data
        reports.append({
            "id": filename.replace('.md', ''),
            "filename": filename,
            "path": report_file,
            "title": title,
            "project": project_name,
            "type": report_type,
            "date": formatted_date,
            "summary": summary
        })
    
    # Sort by date (most recent first)
    reports.sort(key=lambda x: x["filename"], reverse=True)
    return reports

def get_report_by_id(report_id):
    """Get a specific report by its ID."""
    reports = get_all_reports()
    for report in reports:
        if report["id"] == report_id:
            # Read full content
            with open(report["path"], 'r') as f:
                content = f.read()
                
            # Parse content as markdown to HTML
            html_content = markdown.markdown(content, extensions=['extra', 'fenced_code'])
            
            # Add the HTML content
            report["content"] = html_content
            return report
    
    return None

def get_report_json_by_id(report_id):
    """Get the associated JSON data for a report if available."""
    json_path = os.path.join(os.getcwd(), "reports", f"{report_id}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON for report {report_id}: {e}")
    return None

def get_report_graph_by_id(report_id):
    """Get the associated HTML graph for a report if available."""
    graph_path = os.path.join(os.getcwd(), "reports", f"{report_id.replace('report', 'graph')}.html")
    if os.path.exists(graph_path):
        try:
            with open(graph_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading graph for report {report_id}: {e}")
    return None

# Common header component
def common_header():
    """Render the common header with navigation."""
    return Div(
        H1("Finite Monkey Engine"),
        Div(
            A("Dashboard", href="/"),
            A("Terminal", href="/terminal"),
            A("Jupyter Terminal", href="/jupyter_terminal_page"),
            A("Code Editor", href="/editor"),
            A("Reports", href="/reports"),
            A("Visualizations", href="/visualizations"),
            A("Documentation", href="https://github.com/yourusername/a-finite-monkey-engine"),
            cls="header-controls"
        ),
        cls="header"
    )

# Main application routes
@rt('/')
def get():
    """Main dashboard view."""
    terminal_id = str(uuid.uuid4())
    
    # Get recent reports
    reports = get_all_reports()[:3]  # Get 3 most recent reports
    
    return Titled("Finite Monkey Engine", 
        Div(
            # Header
            common_header(),
            
            # Dashboard content
            H2("Dashboard"),
            P("Welcome to the Finite Monkey Engine, a framework for smart contract security analysis."),
            
            # Dashboard cards
            Div(
                # Recent activity card
                Div(
                    H3("Recent Activity"),
                    Ul(
                        Li("Terminal session started - 10 minutes ago"),
                        Li("Security analysis completed - 1 hour ago"),
                        Li("New project added - 3 hours ago")
                    ),
                    cls="dashboard-card"
                ),
                
                # Key metrics card
                Div(
                    H3("Key Metrics"),
                    Div(
                        Div(
                            Div("12", cls="metric-value"),
                            Div("Projects", cls="metric-label"),
                            cls="metric-item"
                        ),
                        Div(
                            Div("36", cls="metric-value"),
                            Div("Analyses", cls="metric-label"),
                            cls="metric-item"
                        ),
                        Div(
                            Div("85%", cls="metric-value"),
                            Div("Accuracy", cls="metric-label"),
                            cls="metric-item"
                        ),
                        Div(
                            Div("143", cls="metric-value"),
                            Div("Issues Found", cls="metric-label"),
                            cls="metric-item"
                        ),
                        cls="metrics-grid"
                    ),
                    cls="dashboard-card"
                ),
                
                # Quick access card
                Div(
                    H3("Quick Access"),
                    P("Start a new analysis or continue where you left off."),
                    A("IPython Terminal", href="/terminal", cls="terminal-submit"),
                    A("Jupyter Terminal", href="/jupyter_terminal_page", cls="terminal-submit", style="margin-left: 10px;"),
                    A("Code Editor", href="/editor", cls="terminal-submit", style="margin-left: 10px;"),
                    A("View Reports", href="/reports", cls="terminal-submit", style="margin-left: 10px;"),
                    cls="dashboard-card"
                ),
                
                cls="dashboard"
            ),
            
            # Recent reports section
            H2("Recent Reports"),
            Div(
                *[
                    A(
                        Div(
                            H3(report["title"]),
                            Div(f"Project: {report['project']} | {report['date']}", cls="report-card-meta"),
                            Div(report["summary"], cls="report-card-description"),
                            cls="report-card"
                        ),
                        href=f"/reports/{report['id']}",
                        style="text-decoration: none; color: inherit;"
                    ) for report in reports
                ],
                cls="reports-list"
            ) if reports else Div("No reports available yet.", style="margin: 2rem 0;"),
            
            cls="app-container"
        )
    )

@rt('/terminal')
def get():
    """Terminal view."""
    terminal_id = str(uuid.uuid4())
    
    return Titled("Terminal - Finite Monkey Engine",
        Div(
            # Header
            common_header(),
            
            # Terminal container
            Div(
                # Terminal header
                Div(
                    H2("IPython Terminal"),
                    Div(
                        Button(
                            NotStr('<i class="fas fa-broom"></i> Clear'),
                            id="clear-terminal"
                        ),
                        Button(
                            NotStr('<i class="fas fa-sync-alt"></i> Refresh Scope'),
                            hx_post=f"/refresh-scope/{terminal_id}",
                            hx_target="#scope-content"
                        ),
                        cls="terminal-header-controls"
                    ),
                    cls="terminal-header"
                ),
                
                # Terminal and scope inspector
                Div(
                    # Terminal display with HTMX WebSocket connection
                    Div(
                        id="terminal", 
                        cls="terminal",
                        hx_ext="ws",
                        ws_connect=f"/ws/terminal/{terminal_id}"
                    ),
                    
                    # Scope inspector
                    Div(
                        H3("Scope Inspector"),
                        Div(id="scope-content", cls="scope-content"),
                        cls="scope-inspector"
                    ),
                    
                    cls="terminal-wrapper"
                ),
                
                # Input area
                Form(
                    Input(id="terminal-input", name="command", placeholder="Enter Python command...", cls="terminal-input"),
                    Button("Run", type="submit", cls="terminal-submit"),
                    id="terminal-form",
                    hx_post=f"/send-command/{terminal_id}",
                    hx_swap="none"
                ),
                
                cls="terminal-container"
            ),
            
            cls="app-container"
        )
    )

@rt('/send-command/{terminal_id}')
def post(terminal_id: str, command: str):
    """Handle command submission via form POST."""
    # Send command to WebSocket
    return Script(f"""
    if (window.ws && window.ws.readyState === WebSocket.OPEN) {{
        window.ws.send({json.dumps(command)});
    }}
    document.getElementById('terminal-input').value = '';
    """)

@rt('/refresh-scope/{terminal_id}')
def post(terminal_id: str):
    """Manual scope refresh."""
    return Script(f"""
    if (window.ws && window.ws.readyState === WebSocket.OPEN) {{
        window.ws.send("__scope_inspect__:dir()");
    }}
    """)

@rt('/api/commands')
def get():
    """API endpoint to get recent commands."""
    session = Session()
    try:
        commands = session.query(TerminalCommand).order_by(TerminalCommand.timestamp.desc()).limit(20).all()
        result = [{"id": cmd.id, "command": cmd.command, "timestamp": cmd.timestamp.isoformat()} for cmd in commands]
        return JSONResponse(result)
    finally:
        session.close()

@rt('/visualizations')
def get():
    """Visualizations page with various charts."""
    # Generate sample data
    sample_data = visualizations.generate_sample_data()
    
    # Create charts and get their HTML representations
    histogram_html = visualizations.create_chart_html(
        visualizations.histogram_chart(
            sample_data["histogram"]["data"],
            title=sample_data["histogram"]["title"],
            xlabel=sample_data["histogram"]["xlabel"],
            ylabel=sample_data["histogram"]["ylabel"],
            bins=sample_data["histogram"]["bins"]
        )
    )
    
    bar_chart_html = visualizations.create_chart_html(
        visualizations.bar_chart(
            sample_data["bar_chart"]["categories"],
            sample_data["bar_chart"]["values"],
            title=sample_data["bar_chart"]["title"],
            xlabel=sample_data["bar_chart"]["xlabel"],
            ylabel=sample_data["bar_chart"]["ylabel"]
        )
    )
    
    line_chart_html = visualizations.create_chart_html(
        visualizations.line_chart(
            sample_data["line_chart"]["x_data"],
            sample_data["line_chart"]["y_data"],
            title=sample_data["line_chart"]["title"],
            xlabel=sample_data["line_chart"]["xlabel"],
            ylabel=sample_data["line_chart"]["ylabel"]
        )
    )
    
    # Create a radar chart with vulnerability categories
    vuln_categories = ["Access Control", "Arithmetic", "Reentrancy", "Front-running", "Logic Error", "Validation"]
    vuln_values = [85, 70, 90, 65, 75, 80]
    
    radar_chart_html = visualizations.create_chart_html(
        visualizations.radar_chart(vuln_categories, vuln_values, "Smart Contract Security Coverage")
    )
    
    # Create a pie chart with issue types
    issue_types = ["Critical", "High", "Medium", "Low", "Informational"]
    issue_counts = [5, 12, 28, 45, 53]
    
    pie_chart_html = visualizations.create_chart_html(
        visualizations.pie_chart(issue_types, issue_counts, "Issues by Severity")
    )
    
    # Return the visualizations page
    return Titled("Visualizations - Finite Monkey Engine",
        Div(
            # Header
            common_header(),
            
            # Visualizations content
            H2("Visualizations"),
            P("Interactive charts and visualizations for smart contract security analysis."),
            
            # Visualization grid
            Div(
                # First row
                Div(
                    Div(
                        H3("Smart Contract Security Coverage"),
                        Div(NotStr(radar_chart_html), cls="chart-container"),
                        cls="dashboard-card"
                    ),
                    Div(
                        H3("Issues by Severity"),
                        Div(NotStr(pie_chart_html), cls="chart-container"),
                        cls="dashboard-card"
                    ),
                    cls="dashboard"
                ),
                
                # Second row
                Div(
                    Div(
                        H3("Normal Distribution"),
                        Div(NotStr(histogram_html), cls="chart-container"),
                        cls="dashboard-card"
                    ),
                    cls="dashboard"
                ),
                
                # Third row
                Div(
                    Div(
                        H3("Security Issues by Category"),
                        Div(NotStr(bar_chart_html), cls="chart-container"),
                        cls="dashboard-card"
                    ),
                    Div(
                        H3("Trend Analysis"),
                        Div(NotStr(line_chart_html), cls="chart-container"),
                        cls="dashboard-card"
                    ),
                    cls="dashboard"
                ),
            ),
            
            cls="app-container"
        )
    )

# Reports routes
@rt('/reports')
def get(page: int = 1, search: str = "", type: str = ""):
    """Reports listing page."""
    # Get all reports
    all_reports = get_all_reports()
    
    # Filter reports if search or type filters provided
    if search:
        search = search.lower()
        all_reports = [r for r in all_reports if (
            search in r["title"].lower() or 
            search in r["project"].lower() or
            search in r["summary"].lower()
        )]
    
    if type:
        all_reports = [r for r in all_reports if r["type"].lower() == type.lower()]
    
    # Calculate pagination
    per_page = 12
    total_reports = len(all_reports)
    total_pages = max(1, (total_reports + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    
    # Get reports for current page
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_reports)
    current_reports = all_reports[start_idx:end_idx]
    
    return Titled("Security Reports - Finite Monkey Engine",
        Div(
            # Header
            common_header(),
            
            # Reports content
            H2("Security Reports"),
            P("Browse security audit reports generated by the Finite Monkey Engine."),
            
            # Search and filter
            Form(
                Div(
                    Input(name="search", placeholder="Search reports...", value=search, cls="search-input"),
                    Select(
                        Option("All Types", value="", selected=type == ""),
                        Option("Security Audit", value="Security Audit", selected=type == "Security Audit"),
                        Option("Analysis", value="Analysis", selected=type == "Analysis"),
                        name="type",
                        cls="filter-dropdown"
                    ),
                    Button("Search", type="submit", cls="terminal-submit"),
                    cls="search-tools"
                ),
                action="/reports",
                method="get"
            ),
            
            # Reports grid
            Div(
                *[
                    A(
                        Div(
                            H3(report["title"]),
                            Div(f"Project: {report['project']} | {report['date']}", cls="report-card-meta"),
                            Div(report["summary"], cls="report-card-description"),
                            cls="report-card"
                        ),
                        href=f"/reports/{report['id']}",
                        style="text-decoration: none; color: inherit;"
                    ) for report in current_reports
                ],
                cls="reports-list"
            ) if current_reports else Div("No reports found matching your criteria.", style="margin: 2rem 0;"),
            
            # Pagination
            Div(
                *[
                    # First page button
                    Button("«", 
                           onclick=f"window.location='/reports?page=1&search={search}&type={type}'",
                           cls="page-button",
                           disabled=page == 1),
                    # Previous page button  
                    Button("‹",
                           onclick=f"window.location='/reports?page={page-1}&search={search}&type={type}'",
                           cls="page-button",
                           disabled=page == 1),
                    # Page indicators
                    *[
                        Button(str(p), 
                             cls=f"page-button {'active' if p == page else ''}",
                             onclick=f"window.location='/reports?page={p}&search={search}&type={type}'"
                             )
                        for p in range(max(1, page-2), min(total_pages+1, page+3))
                    ],
                    # Next page button
                    Button("›",
                           onclick=f"window.location='/reports?page={page+1}&search={search}&type={type}'",
                           cls="page-button",
                           disabled=page == total_pages),
                    # Last page button
                    Button("»", 
                           onclick=f"window.location='/reports?page={total_pages}&search={search}&type={type}'",
                           cls="page-button",
                           disabled=page == total_pages)
                ],
                cls="pagination"
            ) if total_pages > 1 else "",
            
            cls="app-container"
        )
    )

@rt('/reports/{report_id}')
def get(report_id: str):
    """Report detail view."""
    # Get the report by ID
    report = get_report_by_id(report_id)
    
    if not report:
        return Titled("Report Not Found - Finite Monkey Engine",
            Div(
                common_header(),
                H2("Report Not Found"),
                P("The requested report does not exist."),
                A("Back to Reports", href="/reports", cls="terminal-submit"),
                cls="app-container"
            )
        )
    
    # Check if graph is available
    graph_html = get_report_graph_by_id(report_id)
    
    # Check if raw data is available
    json_data = get_report_json_by_id(report_id)
    
    return Titled(f"{report['title']} - Finite Monkey Engine",
        Div(
            # Header
            common_header(),
            
            # Report header
            H2(report["title"]),
            Div(
                f"Project: {report['project']} | Date: {report['date']} | Type: {report['type']}",
                cls="report-card-meta",
                style="margin-bottom: 2rem;"
            ),
            
            # Report content
            Div(
                Div(
                    NotStr(report["content"]),
                    cls="report-content"
                ),
                cls="report-detail"
            ),
            
            # Visualization section if available
            (
                Div(
                    H3("Visualization"),
                    Div(
                        NotStr(graph_html),
                        style="width: 100%; height: 600px; overflow: auto; border: 1px solid var(--border-color); border-radius: 8px;"
                    ),
                    style="margin-top: 2rem;"
                )
            ) if graph_html else "",
            
            # Raw data section if available
            (
                Div(
                    H3("Raw Data"),
                    A("Download JSON", 
                      href=f"/reports/{report_id}/download", 
                      cls="terminal-submit",
                      style="display: inline-block; margin-bottom: 1rem;"),
                    P("This data can be used for further analysis or integration with other tools."),
                    style="margin-top: 2rem;"
                )
            ) if json_data else "",
            
            # Actions
            Div(
                A("Back to Reports", href="/reports", cls="terminal-submit"),
                A("Generate Similar Report", href="/terminal", cls="terminal-submit"),
                A("Download PDF", href=f"/reports/{report_id}/pdf", cls="terminal-submit"),
                cls="report-actions"
            ),
            
            cls="app-container"
        )
    )

@rt('/reports/{report_id}/download')
def get(report_id: str):
    """Download report raw data as JSON."""
    json_data = get_report_json_by_id(report_id)
    
    if not json_data:
        return Titled("Data Not Found - Finite Monkey Engine",
            Div(
                common_header(),
                H2("Report Data Not Found"),
                P("The requested report data does not exist."),
                A("Back to Reports", href="/reports", cls="terminal-submit"),
                cls="app-container"
            )
        )
    
    # Return JSON response
    return JSONResponse(json_data)

@rt('/reports/{report_id}/pdf')
def get(report_id: str):
    """PDF generation is not implemented in this example."""
    return Titled("Feature Not Available - Finite Monkey Engine",
        Div(
            common_header(),
            H2("PDF Generation Not Available"),
            P("PDF generation is not implemented in this version."),
            A("Back to Report", href=f"/reports/{report_id}", cls="terminal-submit"),
            cls="app-container"
        )
    )

@rt('/editor')
def get():
    """Code editor page."""
    # Generate a terminal_id that will be used for WebSocket connection
    terminal_id = str(uuid.uuid4())
    
    return Titled("Code Editor - Finite Monkey Engine",
        Div(
            # Header
            common_header(),
            
            # Editor content
            H2("Code Editor"),
            P("Create, edit, and run Python code with our interactive code editor."),
            
            # Code editor and terminal layout
            Div(
                # Main layout container
                Div(
                    # Left side: Code editor
                    Div(
                        CodeEditor(),
                        cls="w-full lg:w-3/5 p-4"
                    ),
                    
                    # Right side: Terminal for output
                    Div(
                        Div(
                            # Terminal title with connection status
                            Div(
                                H3("Terminal Output"),
                                Div(
                                    NotStr('<span class="connection-status">Connecting...</span>'),
                                    id="connection-status"
                                ),
                                Script("""
                                    document.addEventListener('htmx:wsOpen', function() {
                                        document.getElementById('connection-status').innerHTML = 
                                            '<span class="connection-status connected">Connected</span>';
                                    });
                                    document.addEventListener('htmx:wsClose', function() {
                                        document.getElementById('connection-status').innerHTML = 
                                            '<span class="connection-status disconnected">Disconnected</span>';
                                    });
                                """),
                                cls="flex justify-between items-center mb-4"
                            ),
                            
                            # Terminal display with WebSocket connection
                            Div(
                                Div(
                                    NotStr("""
                                    <div class="welcome-message">
                                        <strong>Welcome to the Finite Monkey IPython Terminal!</strong><br>
                                        Type Python commands to interact with the framework.<br>
                                        Available objects: orchestrator, researcher, validator<br>
                                        <em>Connecting to IPython kernel...</em>
                                    </div>
                                    """)
                                ),
                                id="terminal", 
                                cls="terminal h-96 overflow-y-auto",
                                hx_ext="ws",
                                ws_connect=f"/ws/terminal/{terminal_id}"
                            ),
                            
                            # Terminal input
                            Form(
                                Input(id="terminal-input", name="command", placeholder="Enter Python command...", cls="terminal-input"),
                                Button("Run", type="submit", cls="terminal-submit"),
                                id="terminal-form",
                                hx_post=f"/send-command/{terminal_id}",
                                hx_swap="none"
                            ),
                            
                            cls="terminal-container"
                        ),
                        cls="w-full lg:w-2/5 p-4"
                    ),
                    
                    cls="flex flex-wrap"
                ),
                
                # Add some styles for the connection status
                Style("""
                    .connection-status {
                        font-size: 0.8rem;
                        padding: 0.25rem 0.5rem;
                        border-radius: 0.25rem;
                    }
                    .connection-status.connected {
                        background-color: rgba(76, 175, 80, 0.2);
                        color: #4CAF50;
                    }
                    .connection-status.disconnected {
                        background-color: rgba(244, 67, 54, 0.2);
                        color: #F44336;
                    }
                """),
                
                cls="mb-8"
            ),
            
            # Documentation
            Div(
                H3("Using the Code Editor", cls="text-xl text-accent-color mb-4"),
                Div(
                    H4("Key Features:", cls="font-bold mb-2"),
                    Ul(
                        Li("Integrated terminal output"),
                        Li("Syntax highlighting and autocompletion"),
                        Li("Run code with Ctrl+Enter"),
                        Li("Solidity syntax support"),
                        Li("Direct connection to IPython environment"),
                        cls="list-disc ml-6 mb-4"
                    ),
                    
                    H4("Available Actions:", cls="font-bold mb-2"),
                    Ul(
                        Li("Change language using the dropdown"),
                        Li("Run code with the Run button or Ctrl+Enter"),
                        Li("Save code snippets for later use"),
                        Li("View output directly in the terminal"),
                        cls="list-disc ml-6"
                    ),
                    
                    cls="bg-terminal-bg p-6 rounded-lg mt-4"
                ),
                cls="mt-8"
            ),
            
            cls="app-container"
        )
    )

# Route to save code to file
@rt('/save-code')
async def post(request: Request):
    """Save code to a file."""
    # Get the code and filename from request
    data = await request.json()
    code = data.get('code', '')
    filename = data.get('filename', 'code.txt')
    
    # Ensure we're saving to a safe directory
    save_dir = os.path.join(os.getcwd(), 'saved_code')
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Clean the filename (basic security)
    clean_filename = os.path.basename(filename)
    
    # Construct the file path
    file_path = os.path.join(save_dir, clean_filename)
    
    try:
        # Save the file
        with open(file_path, 'w') as f:
            f.write(code)
        
        # Return success
        return JSONResponse({
            'success': True,
            'path': file_path
        })
    except Exception as e:
        # Return error
        return JSONResponse({
            'success': False,
            'error': str(e)
        })

# Global objects for Jupyter terminal
jupyter_terminal = None

# Helper function to initialize Jupyter terminal
async def init_jupyter_terminal():
    """Initialize the Jupyter terminal with the namespace."""
    global jupyter_terminal
    
    if jupyter_terminal is not None:
        return jupyter_terminal
        
    # Create the namespace - but don't initialize objects that need asyncio
    namespace = {
        # We'll add these objects later in the terminal itself
        'sys': sys,
        'os': os,
        'json': json,
        're': re,
    }
    
    try:
        # Create Jupyter terminal
        jupyter_terminal = JupyterTerminal(namespace=namespace, app=app, port=8000)
        
        # Start the terminal
        await jupyter_terminal.start()
    except RuntimeError as e:
        # Handle case where there's no running event loop
        print(f"Note: Event loop issue detected during init: {e}")
        print("Terminal will be initialized on first connection instead.")
        # Still create the terminal object but don't start it yet
        jupyter_terminal = JupyterTerminal(namespace=namespace, app=app, port=8000)
    except Exception as e:
        print(f"Error initializing Jupyter terminal: {e}")
        # Create a minimal terminal instance that will be properly initialized on first connection
        jupyter_terminal = JupyterTerminal(namespace=namespace, app=app, port=8000)
    
    return jupyter_terminal

# Initialize the terminal at app startup
@app.on_event("startup")
async def startup_event():
    """Initialize components at app startup."""
    # Initialize the terminal in a background task to avoid blocking startup
    asyncio.create_task(_initialize_terminal_background())

async def _initialize_terminal_background():
    """Initialize terminal in background to avoid blocking startup."""
    try:
        await init_jupyter_terminal()
        print("Jupyter terminal initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Jupyter terminal at startup: {e}")
        print("Terminal will be initialized on first connection instead")

# Jupyter Terminal route
@rt('/jupyter_terminal_page')
def get():
    """Jupyter terminal page."""
    # No need to check for jupyter_terminal - we'll use JupyUviAsync for iframe integration
    
    return Titled("Jupyter Terminal - Finite Monkey Engine",
        Div(
            # Header
            common_header(),
            
            # Jupyter Terminal content
            H2("Jupyter Terminal"),
            P("A fully-featured IPython terminal with direct access to the Finite Monkey framework."),
            
            # Container for the terminal
            Div(
                # Terminal explanation
                Div(
                    H3("Available Objects"),
                    P("The following objects are available in the terminal:"),
                    Ul(
                        Li(B("orchestrator"), ": WorkflowOrchestrator for running full analysis workflows"),
                        Li(B("researcher"), ": Researcher agent for code analysis"),
                        Li(B("validator"), ": Validator agent for validating results"),
                        cls="list-disc ml-6 mb-4"
                    ),
                    
                    H3("Setup Commands"),
                    P("First, run these commands to initialize the framework objects:"),
                    Pre("""import asyncio

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
""", cls="bg-gray-800 p-4 rounded my-4 text-sm"),
                    
                    H3("Example Commands"),
                    P("Then try these commands:"),
                    Pre("""# Run a full analysis
await orchestrator.run_analysis("examples/Vault.sol")

# Directly use the researcher agent
result = await researcher.analyze_code_async("contract Vault { ... }")

# Print information about an agent
print(validator)
""", cls="bg-gray-800 p-4 rounded my-4 text-sm"),
                    cls="mb-6"
                ),
                
                # Jupyter terminal iframe - embed the simple terminal
                Div(
                    NotStr("""
                    <div style="margin-top: 10px; margin-bottom: 5px; display: flex; gap: 15px;">
                        <a href="http://localhost:8899" target="_blank" style="color: #5D3FD3; text-decoration: underline;">
                            Open terminal in new tab
                        </a>
                        <span style="color: #999; font-style: italic;">
                            (Make sure you've started the terminal with ./run_simple_terminal.sh)
                        </span>
                    </div>
                    <iframe 
                        src="http://localhost:8899" 
                        style="width: 100%; height: 500px; border: none; border-radius: 5px; background-color: #252526;" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; fullscreen">
                    </iframe>
                    """),
                    cls="w-full rounded overflow-hidden mt-4"
                ),
                
                cls="mt-6"
            ),
            
            cls="app-container"
        )
    )

# Define the direct Jupyter terminal route
@app.get('/jupyter_terminal')
def jupyter_terminal_view():
    # This is the page that will be loaded in the iframe
    # It will contain the Jupyter terminal powered by FastHTML's Jupyter integration
    return Div(
        # Header
        H3("IPython Terminal", style="color: #5D3FD3; margin-bottom: 10px;"),
        P("Type Python commands to interact with the framework.", style="margin-bottom: 15px;"),
        
        # Welcome message with instructions
        Div(
            NotStr("""
            <div class="welcome-message">
                <strong>Welcome to the Finite Monkey IPython Terminal!</strong><br>
                Type Python commands to interact with the framework.<br>
                <br>
                Run these commands to initialize the framework objects:<br>
                <pre style="background-color: #333; padding: 10px; margin: 5px 0; border-radius: 5px;">
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
            """),
            cls="mb-4"
        ),
        
        # The actual Jupyter terminal will be inserted here by JupyUviAsync
        Div(
            id="jupyter-term",
            style="min-height: 400px; background-color: #1e1e1e; color: #f0f0f0; padding: 10px; border-radius: 5px; font-family: monospace; overflow-y: auto;"
        ),
        
        # Styles for the terminal
        Style("""
        .welcome-message {
            margin-bottom: 15px;
            padding: 10px;
            background-color: rgba(93, 63, 211, 0.1);
            border-left: 4px solid #5D3FD3;
            border-radius: 5px;
        }
        
        body {
            background-color: #252526;
            color: #f0f0f0;
            font-family: 'Segoe UI', 'Helvetica', sans-serif;
            padding: 15px;
        }
        
        pre {
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }
        
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
        """
        )
    )

# Start the application
if __name__ == "__main__":
    serve()
    
# ASGI app - for compatibility with Uvicorn when imported elsewhere
asgi_app = app