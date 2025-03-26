from fasthtml.common import *

editor_script = Script("""
let editor;
let lastCursorPosition = {row: 0, column: 0};
let terminalSocket;

function initEditor() {
    editor = ace.edit("code-editor");
    editor.setTheme("ace/theme/monokai");
    editor.session.setMode("ace/mode/python");
    editor.setOptions({
        fontSize: "14px",
        showPrintMargin: false,
        showGutter: true,
        highlightActiveLine: true,
        wrap: true,
        enableBasicAutocompletion: true,
        enableLiveAutocompletion: true,
        enableSnippets: true
    });
    
    // Default code based on selected language
    const defaultCode = {
        'python': `# A simple Python analysis example

from finite_monkey.agents.orchestrator import WorkflowOrchestrator
from finite_monkey.agents.researcher import Researcher

# Create an orchestrator instance
orchestrator = WorkflowOrchestrator()

# Analyze the example Vault contract
file_path = "examples/Vault.sol"

# Run a simple analysis
async def analyze_contract():
    print(f"Analyzing {file_path}...")
    results = await orchestrator.run_analysis(file_path)
    return results
    
# To run the analysis, execute this in the terminal
print("To run the analysis, execute:")
print("results = await analyze_contract()")
`,
        'solidity': `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title Vault
 * @dev A simple vault contract for storing and managing ETH
 * @notice This is a sample contract with security vulnerabilities
 */
contract Vault {
    mapping(address => uint256) public balances;
    bool private locked;
    
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    
    // Simple reentrancy guard modifier
    modifier nonReentrant() {
        require(!locked, "Reentrant call");
        locked = true;
        _;
        locked = false;
    }
    
    /**
     * @dev Deposit ETH into the vault
     */
    function deposit() external payable {
        require(msg.value > 0, "Deposit amount must be greater than 0");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }
    
    /**
     * @dev Withdraw all ETH from the vault
     * @notice Has a reentrancy vulnerability
     */
    function withdraw() external {
        uint256 balance = balances[msg.sender];
        require(balance > 0, "No funds to withdraw");
        
        // Vulnerability: State changes after external call
        (bool success, ) = msg.sender.call{value: balance}("");
        require(success, "Withdrawal failed");
        
        // State changes after external call (reentrancy vulnerability)
        balances[msg.sender] = 0;
        
        emit Withdrawal(msg.sender, balance);
    }
}`
    };
    
    // Set initial code
    editor.setValue(defaultCode['python']);
    
    // Update the output preview with the current filename and language
    function updateOutputPreview(filename, language) {
        const outputPreview = document.getElementById('output-preview');
        if (outputPreview) {
            outputPreview.innerHTML = `
                <div class="file-info">
                    <strong>File:</strong> ${filename || 'Unsaved'} 
                    <span class="language-badge">${language}</span>
                </div>
            `;
        }
    }
    
    // Initial update
    updateOutputPreview('Example.py', 'python');
    
    // Update code when language changes
    document.getElementById('editor-language').addEventListener('change', function(e) {
        // Get the selected language
        const language = e.target.value;
        
        // Update the editor mode
        let mode = "ace/mode/" + language;
        editor.session.setMode(mode);
        
        // Update code if we have a sample for this language
        if (defaultCode[language]) {
            if (editor.getValue().trim() === '' || 
                confirm('Replace current code with example ' + language + ' code?')) {
                editor.setValue(defaultCode[language]);
                editor.clearSelection();
                
                // Update the output preview
                const extensions = {
                    'python': '.py',
                    'javascript': '.js',
                    'solidity': '.sol',
                    'html': '.html',
                    'css': '.css'
                };
                updateOutputPreview('Example' + (extensions[language] || '.txt'), language);
            }
        }
    });
    
    editor.clearSelection();
    
    window.addEventListener('resize', function() {
        editor.resize();
    });

    document.getElementById('editor-language').addEventListener('change', function(e) {
        let mode = "ace/mode/" + e.target.value;
        editor.session.setMode(mode);
    });

    // Save cursor position on change
    editor.session.selection.on('changeCursor', function() {
        lastCursorPosition = editor.getCursorPosition();
    });

    // Connect the Run button to the terminal
    document.getElementById('run-code').addEventListener('click', function() {
        sendCodeToTerminal();
    });
    
    // Connect the Save button to save functionality
    document.getElementById('save-code').addEventListener('click', function() {
        saveCodeToFile();
    });

    // Add keyboard shortcut for running (Ctrl+Enter)
    editor.commands.addCommand({
        name: 'runCode',
        bindKey: {win: 'Ctrl-Enter', mac: 'Command-Enter'},
        exec: function(editor) {
            sendCodeToTerminal();
        }
    });

    // Try to find the terminal WebSocket
    const checkTerminalConnection = setInterval(() => {
        if (window.ws && window.ws.readyState === WebSocket.OPEN) {
            terminalSocket = window.ws;
            clearInterval(checkTerminalConnection);
            console.log("Terminal WebSocket connected");
        }
    }, 1000);
}

function sendCodeToTerminal() {
    const code = editor.getValue();
    
    // Set output preview to running state
    document.getElementById('output-preview').innerHTML = '<div class="running-indicator">Running code...</div>';
    
    // Find the terminal form and submit the code through it
    const terminalForm = document.getElementById('terminal-form');
    const terminalInput = document.getElementById('terminal-input');
    
    if (terminalForm && terminalInput) {
        // Break the code into more manageable chunks
        const lines = code.split('\\n');
        
        // Set a small delay to allow the form to be ready
        setTimeout(() => {
            // For a large block of code, we'll first show what we're executing 
            // by posting a comment to the terminal
            if (lines.length > 1) {
                // Post the code to the terminal as a comment first
                terminalInput.value = "# Running code from editor:";
                terminalForm.requestSubmit();
                
                // Add a short delay before sending the actual code
                setTimeout(() => {
                    // Set the input value to the code
                    terminalInput.value = code;
                    terminalForm.requestSubmit();
                }, 100);
            } else {
                // For a single line, just execute it directly
                terminalInput.value = code;
                terminalForm.requestSubmit();
            }
        }, 100);
    } else {
        // Terminal form not found, try using WebSocket directly
        useWebSocketFallback(code);
    }
}

function useWebSocketFallback(code) {
    // Fallback to using WebSocket directly
    if (!terminalSocket) {
        // Find the active WebSocket connection
        if (window.ws && window.ws.readyState === WebSocket.OPEN) {
            terminalSocket = window.ws;
        } else {
            // Create a function to check if the global WebSocket becomes available
            const checkInterval = setInterval(() => {
                if (window.ws && window.ws.readyState === WebSocket.OPEN) {
                    terminalSocket = window.ws;
                    clearInterval(checkInterval);
                    
                    // Now that we have the socket, send the code
                    sendCodeWithSocket(code);
                }
            }, 500);
            
            // Add notification
            document.getElementById('output-preview').innerHTML = '<div class="running-indicator">Connecting to terminal...</div>';
            return;
        }
    }
    
    sendCodeWithSocket(code);
}

function sendCodeWithSocket(code) {
    // Send code to terminal
    if (terminalSocket && terminalSocket.readyState === WebSocket.OPEN) {
        // Send a comment indicating code is from editor then the actual code
        terminalSocket.send("# Running code from editor\\n" + code);
        
        // Update output preview
        document.getElementById('output-preview').innerHTML = '<div class="running-indicator">Running code...</div>';
    } else {
        alert('Terminal connection is not available. Please refresh the page.');
    }
}

function saveCodeToFile() {
    // Get the code and language
    const code = editor.getValue();
    const language = document.getElementById('editor-language').value;
    
    // Default file extension based on language
    const extensions = {
        'python': '.py',
        'javascript': '.js',
        'solidity': '.sol',
        'html': '.html',
        'css': '.css'
    };
    
    // Prompt for filename
    let filename = prompt('Enter filename to save:', 'code' + (extensions[language] || '.txt'));
    
    if (!filename) return; // User cancelled
    
    // Add extension if not provided
    if (!filename.includes('.') && extensions[language]) {
        filename += extensions[language];
    }
    
    // Send to server for saving
    fetch('/save-code', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: filename,
            code: code
        }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // Show success message in output preview
            document.getElementById('output-preview').innerHTML = `
                <div class="success-message">File saved successfully: ${data.path}</div>
                <div class="file-info">
                    <strong>File:</strong> ${filename} 
                    <span class="language-badge">${language}</span>
                </div>
            `;
                
            // Also send to terminal if connected
            if (terminalSocket && terminalSocket.readyState === WebSocket.OPEN) {
                terminalSocket.send(`# File saved: ${data.path}`);
            }
        } else {
            // Show error message
            document.getElementById('output-preview').innerHTML = `
                <div class="error-message">Error saving file: ${data.error}</div>
                <div class="file-info">
                    <strong>File:</strong> ${filename} (not saved)
                    <span class="language-badge">${language}</span>
                </div>
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('output-preview').innerHTML = 
            `<div class="error-message">Error saving file: ${error.message}</div>`;
    });
}

document.addEventListener('DOMContentLoaded', initEditor);

// Add an event handler for HTMX websocket connections for better integration
document.addEventListener('htmx:wsOpen', function(event) {
    // Update the terminal WebSocket reference when a connection is established
    console.log('WebSocket connection established');
    if (!terminalSocket && window.ws) {
        terminalSocket = window.ws;
        
        // Update any waiting indicators
        const outputPreview = document.getElementById('output-preview');
        if (outputPreview && outputPreview.querySelector('.running-indicator')) {
            outputPreview.innerHTML = '<div class="running-indicator">Terminal connected! Ready to run code.</div>';
        }
    }
});
""")

def EditorToolbar():
    return Div(
        Div(
            Select(
                Option("Python", value="python", selected=True),
                Option("JavaScript", value="javascript"),
                Option("Solidity", value="solidity"),
                Option("HTML", value="html"),
                Option("CSS", value="css"),
                id="editor-language",
                cls="p-2 border rounded bg-gray-800 text-white border-gray-700"
            ),
            cls="mr-4"
        ),
        Div(
            Button(
                NotStr('<i class="fas fa-play mr-2"></i> Run (Ctrl+Enter)'),
                id="run-code",
                cls="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
            ),
            Button(
                NotStr('<i class="fas fa-save mr-2"></i> Save'),
                id="save-code",
                cls="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 ml-2"
            ),
            cls="flex items-center"
        ),
        cls="flex justify-between items-center p-4 bg-gray-900 border-b border-gray-700 rounded-t-lg"
    )

def CodeEditor():
    return (
        Div(
            EditorToolbar(),
            Div(
                id="code-editor",
                cls="w-full h-full",
                style="height: 500px;"
            ),
            Div(
                H3("Output", cls="text-lg font-semibold mb-2 text-white"),
                Div(
                    id="output-preview",
                    cls="p-4 bg-gray-800 rounded overflow-auto text-white font-mono text-sm",
                    style="height: 150px;"
                ),
                cls="mt-4 p-4 bg-gray-900 rounded-b-lg border-t border-gray-700"
            ),
            cls="code-editor-wrapper bg-gray-900 rounded-lg shadow-lg overflow-hidden"
        ),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/ace.js"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/ext-language_tools.js"),
        Style("""
        .running-indicator {
            color: #4CAF50;
            animation: pulse 1.5s infinite;
        }
        
        .success-message {
            color: #4CAF50;
            background-color: rgba(76, 175, 80, 0.1);
            padding: 8px 12px;
            border-radius: 4px;
            border-left: 3px solid #4CAF50;
        }
        
        .error-message {
            color: #F44336;
            background-color: rgba(244, 67, 54, 0.1);
            padding: 8px 12px;
            border-radius: 4px;
            border-left: 3px solid #F44336;
            margin-bottom: 10px;
        }
        
        .file-info {
            padding: 8px 12px;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
        }
        
        .language-badge {
            background-color: var(--primary-color);
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 8px;
        }
        
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        """),
        editor_script
    )