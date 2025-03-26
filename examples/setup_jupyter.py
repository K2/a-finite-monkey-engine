#!/usr/bin/env python3
"""
Set up the Jupyter environment for the Finite Monkey Engine.

This script:
1. Installs the ipython kernel
2. Creates a kernel.json file with the proper configuration
3. Makes the kernel available to Jupyter
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def main():
    """Set up the Jupyter environment."""
    print("Setting up Jupyter environment for Finite Monkey Engine...")
    
    # Get the path to the Python executable
    python_executable = sys.executable
    
    # Get the path to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a kernels directory if it doesn't exist
    kernels_dir = os.path.join(project_dir, "kernels")
    os.makedirs(kernels_dir, exist_ok=True)
    
    # Create a kernel.json file
    kernel_json = {
        "argv": [
            python_executable,
            "-m", "ipykernel",
            "-f", "{connection_file}"
        ],
        "display_name": "Finite Monkey Engine",
        "language": "python",
        "env": {
            "PYTHONPATH": project_dir,
            "PYTHONIOENCODING": "utf-8",
            "TERM": "xterm-256color",
            "FORCE_COLOR": "1"
        }
    }
    
    # Write the kernel.json file
    kernel_dir = os.path.join(kernels_dir, "finite_monkey")
    os.makedirs(kernel_dir, exist_ok=True)
    
    with open(os.path.join(kernel_dir, "kernel.json"), "w") as f:
        json.dump(kernel_json, f, indent=4)
    
    print(f"Created kernel.json in {kernel_dir}")
    
    # Check if ipykernel is installed
    try:
        import ipykernel
    except ImportError:
        print("Installing ipykernel...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ipykernel"])
    
    # Install the kernel
    print("Installing the kernel...")
    subprocess.check_call([
        sys.executable, 
        "-m", "ipykernel", 
        "install", 
        "--user", 
        "--name", "finite_monkey", 
        "--display-name", "Finite Monkey Engine"
    ])
    
    print("Jupyter environment setup complete!")
    print("You can now start the FastHTML web interface with:")
    print("  ./run_jupyter.sh")

if __name__ == "__main__":
    main()