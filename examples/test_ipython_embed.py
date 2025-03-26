#!/usr/bin/env python3
"""
Test script to verify IPython embedding works correctly.
This is a simple standalone script that demonstrates IPython.embed()
without the web interface.
"""

import os
import sys
import threading
import queue
import time
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config import Config

def main():
    """Test IPython embedding."""
    print("Testing IPython embedding...")
    
    # Create input/output queues
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    # Define redirected stdin/stdout
    class QueueStdin:
        def __init__(self, queue):
            self.queue = queue
            
        def readline(self):
            return self.queue.get() + '\n'
            
        def isatty(self):
            return True
    
    class QueueStdout:
        def __init__(self, queue):
            self.queue = queue
            
        def write(self, data):
            if data:
                self.queue.put(data)
            return len(data) if data else 0
            
        def flush(self):
            pass
    
    # Custom namespace
    namespace = {
        'test_var': 'Hello, IPython!',
        'sys': sys,
        'os': os,
    }
    
    # Configure IPython
    c = Config()
    c.InteractiveShell.colors = 'Linux'
    c.InteractiveShell.confirm_exit = False
    c.TerminalInteractiveShell.term_title = False
    
    # Create a thread to run IPython
    def ipython_thread():
        # Save original stdin/stdout/stderr
        orig_stdin = sys.stdin
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        
        # Set up redirects
        stdin = QueueStdin(input_queue)
        stdout = QueueStdout(output_queue)
        
        try:
            # Redirect stdin/stdout/stderr
            sys.stdin = stdin
            sys.stdout = stdout
            sys.stderr = stdout
            
            # Create and run embedded shell
            shell = InteractiveShellEmbed(
                config=c,
                user_ns=namespace,
                banner1="Test IPython Embed"
            )
            shell()
        finally:
            # Restore original stdin/stdout/stderr
            sys.stdin = orig_stdin
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            
        print("IPython thread exited")
    
    # Start the thread
    thread = threading.Thread(target=ipython_thread, daemon=True)
    thread.start()
    
    # Wait a moment for the thread to initialize
    time.sleep(0.5)
    
    # Send a command to IPython
    print("\nSending command: print(test_var)")
    input_queue.put("print(test_var)")
    
    # Get the output
    try:
        time.sleep(0.5)
        while not output_queue.empty():
            output = output_queue.get_nowait()
            print(f"Output: {output}")
    except queue.Empty:
        pass
    
    # Send another command
    print("\nSending command: import sys; print(sys.version)")
    input_queue.put("import sys; print(sys.version)")
    
    # Get the output
    try:
        time.sleep(0.5)
        while not output_queue.empty():
            output = output_queue.get_nowait()
            print(f"Output: {output}")
    except queue.Empty:
        pass
    
    # Exit
    print("\nSending exit command")
    input_queue.put("exit")
    
    # Wait for the thread to finish
    thread.join(timeout=1.0)
    print("Test completed")

if __name__ == "__main__":
    main()