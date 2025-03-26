"""
IPython terminal integration for the Finite Monkey web interface.

This module provides an IPython console that can be embedded in the web interface,
allowing for interactive debugging and exploration of the agents with syntax highlighting.

The syntax highlighting features support:
1. Automatic language detection for Python, Solidity, Rust, and JavaScript
2. TreeSitter-based highlighting for more advanced syntax parsing
3. Pygments fallback for basic highlighting when TreeSitter is not available
4. Highlighting of code blocks in IPython output

Usage Example:
```python
# In your web application:
from finite_monkey.web.ui.ipython_terminal import AsyncIPythonBridge

# Initialize with your agent namespace
namespace = {
    'orchestrator': orchestrator,
    'researcher': researcher,
    'validator': validator
}

# Create the bridge
terminal = AsyncIPythonBridge(namespace)

# Start output polling
await terminal.start_polling()

# Run code with syntax highlighting
result = await terminal.run_code(
    'import json\\nprint(json.dumps({"hello": "world"}, indent=2))',
    highlight=True
)

# The result will contain HTML with syntax highlighting
```

For web interface integration, make sure to include the required CSS for your chosen
highlighting style. If using Pygments, you can generate the CSS with:
```python
from pygments.formatters import HtmlFormatter
css = HtmlFormatter(style='default').get_style_defs('.code-block')
```
"""

import asyncio
import sys
import io
import threading
import re
from typing import Dict, Any, Optional, Callable, List, Tuple
from queue import Queue

try:
    from pygments import highlight
    from pygments.lexers import PythonLexer, get_lexer_by_name
    from pygments.formatters import HtmlFormatter
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

# TreeSitter integration for advanced syntax highlighting
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
    
    # Initialize TreeSitter parsers
    # Path to compiled language libraries
    LANGUAGE_DIR = "./tree_sitter_languages"
    SUPPORTED_LANGUAGES = {
        "python": "py",
        "javascript": "js",
        "solidity": "sol",
        "rust": "rs",
    }
    
    # Load languages if directory exists
    import os
    if os.path.exists(LANGUAGE_DIR):
        TS_LANGUAGES = {}
        for lang_name, ext in SUPPORTED_LANGUAGES.items():
            try:
                lib_path = os.path.join(LANGUAGE_DIR, f"{lang_name}.so")
                if os.path.exists(lib_path):
                    TS_LANGUAGES[lang_name] = Language(lib_path, lang_name)
            except Exception:
                pass
    else:
        TS_LANGUAGES = {}
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TS_LANGUAGES = {}

class IPythonTerminal:
    """
    Terminal for running IPython in a separate thread with I/O redirection.
    
    This class allows running an IPython console in a separate thread and redirecting
    its I/O to the web interface.
    """
    
    def __init__(self, namespace: Optional[Dict[str, Any]] = None):
        """
        Initialize the IPython terminal.
        
        Args:
            namespace: The namespace to use for the IPython console.
        """
        self.namespace = namespace or {}
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.thread = None
        self.running = False
        
        # Add helper functions to namespace
        self.namespace.update({
            "terminal": self,
        })
        
        # Create a dummy file descriptor for prompt_toolkit
        self._dummy_fd = None
        
        # Add encoding attribute required by prompt_toolkit
        self.encoding = 'utf-8'
    
    def start(self):
        """Start the IPython terminal thread."""
        if self.thread is not None and self.thread.is_alive():
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_ipython)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the IPython terminal thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        # Clean up the file descriptor if it exists
        if self._dummy_fd is not None:
            import os
            try:
                os.close(self._dummy_fd)
            except OSError:
                pass
            self._dummy_fd = None
    
    def _run_ipython(self):
        """Run IPython in a separate thread."""
        # Save original stdin/stdout/stderr
        orig_stdin = sys.stdin
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        
        try:
            # Redirect stdin/stdout/stderr
            sys.stdin = self
            sys.stdout = self
            sys.stderr = self
            
            # Import IPython
            from IPython import start_ipython
            from traitlets.config.loader import Config
            import os
            
            # Set environment variables for proper color rendering
            os.environ['TERM'] = 'xterm-256color'
            os.environ['FORCE_COLOR'] = '1'
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            # Configure IPython to disable cursor position requests
            # and properly handle colors in web environment
            c = Config()
            c.TerminalInteractiveShell.term_title = False
            c.TerminalInteractiveShell.colors = 'linux'  # Use lowercase as recommended
            c.TerminalInteractiveShell.simple_prompt = True  # Disable fancy prompts
            c.TerminalInteractiveShell.display_completions = 'readlinelike'
            c.TerminalInteractiveShell.banner1 = 'Welcome to the Finite Monkey IPython Terminal!\nType Python commands to interact with the framework.\n'
            c.TerminalInteractiveShell.confirm_exit = False
            
            # App configuration
            c.TerminalIPythonApp.display_banner = False
            
            # Start IPython with configuration
            start_ipython(
                argv=['--no-banner'],
                user_ns=self.namespace,
                config=c
            )
        finally:
            # Restore original stdin/stdout/stderr
            sys.stdin = orig_stdin
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            
            # Mark as stopped
            self.running = False
    
    def write(self, data):
        """Write data to the output queue (used for stdout/stderr)."""
        if data:
            # Replace escape sequences that might confuse the terminal
            # But preserve ANSI color codes which will be handled by client-side JS
            clean_data = data
            
            # Remove cursor up/down/forward/backward commands
            ansi_cursor_pattern = re.compile(r'\x1b\[\d*[ABCDEFGJKST]')
            clean_data = ansi_cursor_pattern.sub('', clean_data)
            
            # Remove terminal title setting
            if '\x1b]0;' in clean_data:
                clean_data = re.sub(r'\x1b\]0;.*?\x07', '', clean_data)
            
            # Remove terminal clear screen
            clean_data = clean_data.replace('\x1b[2J', '')
            
            # Remove cursor position requests
            clean_data = re.sub(r'\x1b\[\d*n', '', clean_data)
            
            # Queue the clean data for output
            self.output_queue.put(clean_data)
            
        return len(data) if data else 0
    
    def flush(self):
        """Flush the output."""
        pass
    
    def readline(self):
        """Read a line from the input queue (used for stdin)."""
        if not self.running:
            return "\n"
        return self.input_queue.get()
    
    def isatty(self):
        """Return whether this is a TTY-like device."""
        return True
        
    def fileno(self):
        """
        Return a file descriptor for this terminal.
        
        This is required by prompt_toolkit. We create a pipe and return
        the write end as a dummy file descriptor.
        """
        if self._dummy_fd is None:
            import os
            # Create a pipe - we only need the write end
            r, w = os.pipe()
            os.close(r)  # Close the read end as we don't need it
            self._dummy_fd = w
        return self._dummy_fd
    
    def send_input(self, data):
        """Send input to the IPython console."""
        self.input_queue.put(data + "\n")
    
    def get_output(self):
        """Get output from the IPython console."""
        output = []
        while not self.output_queue.empty():
            output.append(self.output_queue.get_nowait())
        return "".join(output)
    
    def has_output(self):
        """Check if there is output available."""
        return not self.output_queue.empty()
    
    def highlight_code(self, code, language="python"):
        """
        Apply syntax highlighting to code.
        
        Args:
            code: The code to highlight
            language: The programming language
            
        Returns:
            Highlighted code as HTML
        """
        if not code:
            return code
            
        # Use TreeSitter for more advanced highlighting if available
        if TREE_SITTER_AVAILABLE and language in TS_LANGUAGES:
            try:
                parser = Parser()
                parser.set_language(TS_LANGUAGES[language])
                tree = parser.parse(bytes(code, "utf8"))
                
                # Generate HTML with TreeSitter nodes
                # This is a simplified implementation
                # A full implementation would walk the syntax tree
                # and apply CSS classes based on node types
                return f'<pre class="code-block language-{language}">{code}</pre>'
            except Exception:
                # Fall back to Pygments if TreeSitter fails
                pass
                
        # Use Pygments for basic highlighting
        if PYGMENTS_AVAILABLE:
            try:
                lexer = get_lexer_by_name(language, stripall=True)
                formatter = HtmlFormatter(linenos=False, cssclass="code-block")
                result = highlight(code, lexer, formatter)
                return result
            except Exception:
                # If Pygments fails, return the original code wrapped in pre tags
                return f'<pre class="code-block">{code}</pre>'
        
        # If no highlighter is available, return as-is with minimal formatting
        return f'<pre class="code-block">{code}</pre>'
    
    def detect_language(self, code):
        """
        Attempt to detect the programming language from code.
        
        Args:
            code: The code to analyze
            
        Returns:
            Detected language or "python" as default
        """
        # Simple language detection based on common patterns
        if re.search(r'(contract|pragma solidity|function\s+\w+\s*\([^)]*\)\s*(public|private|external|internal))', code):
            return "solidity"
        elif re.search(r'(fn\s+\w+|let\s+mut|impl|struct|enum)\s', code):
            return "rust"
        elif re.search(r'(import\s+{|export\s+const|=>|async\s+function)', code):
            return "javascript"
        # Default to Python for the IPython console
        return "python"
    
    def run_code(self, code, highlight=True):
        """
        Run code in the IPython console.
        
        Args:
            code: The code to run.
            highlight: Whether to apply syntax highlighting to the output.
            
        Returns:
            The output of the code, optionally with syntax highlighting.
        """
        # Clear the output queue
        while not self.output_queue.empty():
            self.output_queue.get_nowait()
        
        try:
            # Send the code to the console
            print(f"\n>> Executing: {code}")  # Print for debugging
            for line in code.strip().split("\n"):
                self.send_input(line)
            
            # Wait for the code to complete
            # This is a simple implementation that could be improved
            # by checking for IPython prompts or other indicators
            import time
            max_wait = 10.0  # Maximum wait time in seconds (increased for longer executions)
            wait_increment = 0.1
            total_wait = 0.0
            
            # Wait initially to let the code start running
            time.sleep(0.3)
            
            # Then check periodically if there's more output
            last_size = 0
            last_output_time = time.time()
            
            while total_wait < max_wait:
                # Get current output size
                current_output = self.get_output()
                current_size = len(current_output)
                
                # If output has grown, reset idle timer
                if current_size > last_size:
                    last_size = current_size
                    last_output_time = time.time()
                
                # Sleep between checks
                time.sleep(wait_increment)
                total_wait += wait_increment
                
                # If no output for a while and we have output, break
                idle_time = time.time() - last_output_time
                if current_size > 0 and idle_time >= 0.8:
                    break
            
            # One final delay to ensure all output is captured
            time.sleep(0.2)
            
            # Return the final output
            output = self.get_output()
            
            # Print the output for debugging
            print(f"Output length: {len(output)}")
            if output and len(output) < 100:
                print(f"Output preview: {output}")
                
            if not output:
                return ""
            
            # Apply syntax highlighting if requested
            if highlight:
                # Try to detect code blocks in the output
                code_blocks = self._extract_code_blocks(output)
                if code_blocks:
                    highlighted_output = output
                    for block, language in code_blocks:
                        highlighted_block = self.highlight_code(block, language)
                        # Replace the original block with highlighted version
                        # Using a unique replacement pattern to avoid conflicts
                        placeholder = f"__CODE_BLOCK_{id(block)}__"
                        highlighted_output = highlighted_output.replace(block, placeholder)
                        highlighted_output = highlighted_output.replace(placeholder, highlighted_block)
                    return highlighted_output
            
            return output
            
        except Exception as e:
            error_message = f"Error executing code: {str(e)}"
            print(error_message)  # Print for server-side debugging
            return error_message
    
    def _extract_code_blocks(self, text):
        """
        Extract code blocks from text output.
        
        Args:
            text: The text to extract code blocks from
            
        Returns:
            List of (code_block, language) tuples
        """
        # Look for code blocks delimited by common patterns
        # IPython output formatting, markdown code blocks, etc.
        blocks = []
        
        # Pattern for IPython output code blocks
        ipython_pattern = r'In \[\d+\]:(.*?)(?=Out\[\d+\]:|In \[\d+\]:|$)'
        matches = re.finditer(ipython_pattern, text, re.DOTALL)
        for match in matches:
            code = match.group(1).strip()
            if code:
                blocks.append((code, self.detect_language(code)))
        
        # Pattern for markdown-style code blocks
        markdown_pattern = r'```(\w*)\n(.*?)```'
        matches = re.finditer(markdown_pattern, text, re.DOTALL)
        for match in matches:
            language = match.group(1) or "python"
            code = match.group(2).strip()
            if code:
                blocks.append((code, language))
        
        return blocks


class AsyncIPythonBridge:
    """
    Bridge for connecting the IPython terminal to the web interface.
    
    This class provides an asynchronous bridge between the IPython terminal and
    the web interface, allowing for non-blocking interaction.
    """
    
    def __init__(self, namespace: Optional[Dict[str, Any]] = None):
        """
        Initialize the AsyncIPythonBridge.
        
        Args:
            namespace: The namespace to use for the IPython console.
        """
        self.terminal = IPythonTerminal(namespace)
        self.terminal.start()
        
        # Set up event for notifying when output is available
        self._output_event = asyncio.Event()
        self._output_callback = None
        
        # Start the output polling task
        self._polling_task = None
    
    async def start_polling(self):
        """Start polling the terminal for output."""
        if self._polling_task is not None:
            return
        
        self._polling_task = asyncio.create_task(self._poll_output())
    
    async def stop_polling(self):
        """Stop polling the terminal for output."""
        if self._polling_task is not None:
            self._polling_task.cancel()
            self._polling_task = None
    
    async def _poll_output(self):
        """Poll the terminal for output."""
        while True:
            try:
                # Check for output
                if self.terminal.has_output():
                    output = self.terminal.get_output()
                    if output and self._output_callback:
                        await self._output_callback(output)
                        # Print only in debug mode - commented out for production
                        # print(f"Broadcasting output: {len(output)} chars")
                
                # Wait a short time
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in output polling: {e}")
                # Continue polling even if there's an error
                await asyncio.sleep(1.0)
    
    async def run_code(self, code: str, highlight: bool = True) -> str:
        """
        Run code in the IPython console.
        
        Args:
            code: The code to run.
            highlight: Whether to apply syntax highlighting to the output.
            
        Returns:
            The output of the code, optionally with syntax highlighting.
        """
        # This is a blocking call, so run it in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.terminal.run_code(code, highlight=highlight)
        )
        return result
        
    async def highlight_code(self, code: str, language: str = "python") -> str:
        """
        Apply syntax highlighting to code asynchronously.
        
        Args:
            code: The code to highlight
            language: The programming language
            
        Returns:
            Highlighted code as HTML
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.terminal.highlight_code(code, language)
        )
        return result
    
    def get_output(self) -> str:
        """
        Get the current output from the terminal.
        
        Returns:
            The current output as a string
        """
        return self.terminal.get_output()
    
    def set_output_callback(self, callback: Callable[[str], None]):
        """
        Set the callback to be called when output is available.
        
        Args:
            callback: The callback function.
        """
        self._output_callback = callback
    
    def register_agent(self, name: str, agent: Any):
        """
        Register an agent with the IPython console.
        
        Args:
            name: The name to use for the agent.
            agent: The agent object.
        """
        self.terminal.namespace[name] = agent
        
    @staticmethod
    def get_syntax_highlighting_css(style: str = 'default') -> str:
        """
        Get CSS for syntax highlighting.
        
        Args:
            style: Pygments style name (e.g., 'default', 'monokai', 'github')
            
        Returns:
            CSS for syntax highlighting
        """
        if PYGMENTS_AVAILABLE:
            try:
                from pygments.formatters import HtmlFormatter
                from pygments.styles import get_style_by_name
                
                # Get the style class
                style_class = get_style_by_name(style)
                
                # Generate the CSS
                css = HtmlFormatter(style=style_class).get_style_defs('.code-block')
                
                # Add custom CSS for the code blocks
                css += """
                .code-block {
                    background-color: #f5f5f5;
                    border-radius: 5px;
                    padding: 1em;
                    margin: 1em 0;
                    overflow-x: auto;
                    font-family: 'Fira Code', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                }
                """
                
                return css
            except Exception as e:
                return f"/* Error generating CSS: {e} */"
        else:
            return """
            /* Basic styling without Pygments */
            .code-block {
                background-color: #f5f5f5;
                border-radius: 5px;
                padding: 1em;
                margin: 1em 0;
                overflow-x: auto;
                font-family: monospace;
                white-space: pre;
                font-size: 14px;
                line-height: 1.5;
            }
            """