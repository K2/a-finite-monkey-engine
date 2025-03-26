#!/usr/bin/env python3
"""
LlamaIndex Diagnostic and Fix Script

This script diagnoses and fixes issues with llama-index integration in the Finite Monkey Engine.
It checks for:
1. Correct installation of llama-index packages
2. Version compatibility
3. Configuration issues
4. File integrity
5. Adapter initialization problems

Usage:
python scripts/fix_llama_index.py [--reinstall] [--debug]
"""

import os
import sys
import subprocess
import importlib
import importlib.metadata
import inspect
import json
import shutil
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Parse arguments
parser = argparse.ArgumentParser(description="Diagnose and fix LlamaIndex issues")
parser.add_argument("--reinstall", action="store_true", help="Force reinstall of llama-index packages")
parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")
args = parser.parse_args()

DEBUG = args.debug
REINSTALL = args.reinstall

# Set up logging
def log(message: str, level: str = "INFO") -> None:
    """Log a message with timestamp and level"""
    if level == "DEBUG" and not DEBUG:
        return
    print(f"[{level}] {message}")

log("Starting LlamaIndex diagnostic and fix script", "INFO")

# Get project root directory
project_root = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
log(f"Project root: {project_root}", "DEBUG")

# Check Python version
python_version = sys.version_info
log(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}", "INFO")
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
    log("Warning: Python version < 3.9 detected. LlamaIndex requires Python 3.9+", "WARNING")

# Function to check installed packages
def check_packages() -> Dict[str, str]:
    """Check installed versions of relevant packages"""
    packages = [
        "llama-index",
        "llama-index-core",
        "llama-index-embeddings-openai",
        "llama-index-llms-openai",
        "llama-index-readers-file",
        "openai",
        "pydantic"
    ]
    
    installed = {}
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            installed[package] = version
            log(f"Found {package} version {version}", "INFO")
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
            log(f"Package {package} not found", "WARNING")
    
    return installed

# Function to fix package installations
def fix_packages(installed: Dict[str, str]) -> bool:
    """Install or fix package installations as needed"""
    # Define compatible package versions
    requirements = {
        "llama-index-core": ">=0.10.0",
        "llama-index-embeddings-openai": ">=0.1.0",
        "llama-index-llms-openai": ">=0.1.0", 
        "llama-index-readers-file": ">=0.1.0",
        "openai": ">=1.0.0",
        "pydantic": ">=2.0.0,<3.0.0"
    }
    
    # Check if reinstall is needed
    needs_install = REINSTALL
    
    # Check for missing packages
    for package, required_version in requirements.items():
        if package not in installed or installed[package] is None:
            log(f"Package {package} is missing and will be installed", "INFO")
            needs_install = True
    
    # If no install needed and not forcing reinstall, return
    if not needs_install:
        log("All required packages are installed", "INFO")
        return True
    
    # Create requirements file for installation
    temp_requirements = project_root / "temp_requirements.txt"
    with open(temp_requirements, "w") as f:
        for package, version in requirements.items():
            f.write(f"{package}{version}\n")
    
    # Install using pip
    log("Installing required packages...", "INFO")
    try:
        pip_command = [sys.executable, "-m", "pip", "install", "-r", str(temp_requirements)]
        if REINSTALL:
            pip_command.append("--force-reinstall")
        
        log(f"Running: {' '.join(pip_command)}", "DEBUG")
        result = subprocess.run(
            pip_command,
            check=True,
            capture_output=True,
            text=True
        )
        log("Package installation completed successfully", "INFO")
        log(result.stdout, "DEBUG")
    except subprocess.CalledProcessError as e:
        log(f"Error installing packages: {e}", "ERROR")
        log(e.stderr, "ERROR")
        return False
    finally:
        # Clean up temporary file
        if temp_requirements.exists():
            temp_requirements.unlink()
    
    return True

# Function to check llama_index_adapter.py
def check_adapter_file() -> Tuple[bool, Optional[str]]:
    """Check if llama_index_adapter.py exists and is valid"""
    adapter_path = project_root / "finite_monkey" / "llm" / "llama_index_adapter.py"
    
    if not adapter_path.exists():
        log(f"LlamaIndex adapter file not found at {adapter_path}", "ERROR")
        return False, None
    
    with open(adapter_path, "r") as f:
        content = f.read()
    
    return True, content

# Function to validate the adapter's content
def validate_adapter(content: str) -> bool:
    """Validate the content of the adapter file"""
    # Check for basic indicators of valid adapter file
    required_imports = [
        "llama_index", 
        "Settings",
        "ServiceContext",
        "LLMPredictor"
    ]
    
    for imp in required_imports:
        if imp not in content:
            log(f"Adapter file missing required import: {imp}", "ERROR")
            return False
    
    # Check for specific classes/methods that should be present
    required_classes = ["LlamaIndexAdapter"]
    for cls in required_classes:
        if cls not in content:
            log(f"Adapter file missing required class: {cls}", "ERROR")
            return False
    
    return True

# Function to create a backup of the adapter file
def backup_adapter(content: str) -> bool:
    """Create a backup of the adapter file"""
    adapter_path = project_root / "finite_monkey" / "llm" / "llama_index_adapter.py"
    backup_path = project_root / "finite_monkey" / "llm" / "llama_index_adapter.py.bak"
    
    try:
        with open(backup_path, "w") as f:
            f.write(content)
        log(f"Created backup of adapter file at {backup_path}", "INFO")
        return True
    except Exception as e:
        log(f"Failed to create backup: {e}", "ERROR")
        return False

# Function to fix the adapter file
def fix_adapter() -> bool:
    """Fix the adapter file with correct implementation"""
    adapter_path = project_root / "finite_monkey" / "llm" / "llama_index_adapter.py"
    
    # Current best implementation of the adapter
    new_adapter_content = """from typing import Any, Dict, List, Optional, Tuple, Union, Callable, AsyncIterator
import asyncio
from concurrent.futures import Future
from functools import partial
import json
import os
import sys
from loguru import logger

# LlamaIndex 0.10.x imports
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.settings import Settings
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.llms.openai import OpenAI

class LlamaIndexAdapter:
    def __init__(self, 
                 model_name: str = None,
                 provider: str = None,
                 base_url: str = None,
                 api_key: str = None,
                 temperature: float = 0.7,
                 context_window: int = 8192,
                 max_tokens: int = None):
        # Configure the model
        self.model_name = model_name
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.context_window = context_window
        self.max_tokens = max_tokens
        
        # Initialize the LLM
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> BaseLLM:
        try:
            # Default to OpenAI
            if self.provider is None or self.provider.lower() == "openai":
                api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
                return OpenAI(
                    model=self.model_name or "gpt-4", 
                    temperature=self.temperature,
                    api_key=api_key,
                    max_tokens=self.max_tokens or 1024
                )
            else:
                # For other providers, we would add support here
                logger.warning(f"Provider {self.provider} not explicitly supported yet, falling back to OpenAI")
                return OpenAI(
                    model=self.model_name or "gpt-4", 
                    temperature=self.temperature,
                    max_tokens=self.max_tokens or 1024
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"Failed to initialize LLM: {e}")
    
    async def complete(self, prompt: str) -> str:
        try:
            response = await self.llm.acomplete(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error during completion: {e}")
            raise
    
    async def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            # Convert to ChatMessage objects
            chat_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                chat_messages.append(ChatMessage(role=role, content=content))
            
            # Generate response
            response = await self.llm.achat(chat_messages)
            return {
                "content": response.message.content,
                "role": response.message.role
            }
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            raise
    
    async def submit_json_prompt(self, prompt: str, schema: Dict[str, Any]) -> Future:
        future = asyncio.get_event_loop().create_future()
        
        try:
            # Augment the prompt to request JSON output
            json_prompt = f"{prompt}\\n\\nPlease provide your response as valid JSON matching the following schema: {json.dumps(schema)}"
            
            # Get the completion
            completion = await self.complete(json_prompt)
            
            # Try to extract and parse JSON from the completion
            try:
                # First try to parse the entire completion as JSON
                result = json.loads(completion)
                future.set_result(result)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown code blocks
                ...
            # Default to OpenAI
            if self.provider is None or self.provider.lower() == "openai":
                api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
                return OpenAI(
                    model=self.model_name or "gpt-4", 
                    temperature=self.temperature,
                    api_key=api_key,
                    max_tokens=self.max_tokens or 1024
                )
            else:
                # For other providers, we would add support here
                logger.warning(f"Provider {self.provider} not explicitly supported yet, falling back to OpenAI")
                return OpenAI(
                    model=self.model_name or "gpt-4", 
                    temperature=self.temperature,
                    max_tokens=self.max_tokens or 1024
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"Failed to initialize LLM: {e}")"""